use std::sync::Arc;
use tokio::sync::{broadcast, Mutex};
use tokio_stream::wrappers::BroadcastStream;
use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};

use crate::engine::{AlignmentEngine, EngineConfig};
use crate::proto::alignment::{
    alignment_engine_server::AlignmentEngine as AlignmentEngineService, AnchorInfo, AnchorList,
    AudioChunk, ChunkHistoryList, ChunkInfo, Empty, EngineConfig as ProtoEngineConfig, LineChange,
    LoadResult, SrtFile, SubtitleFile,
};

/// Converts an `anyhow::Error` into a gRPC `Status::internal`.
fn to_status(e: anyhow::Error) -> Status {
    Status::internal(e.to_string())
}

// ── Service state ─────────────────────────────────────────────────────────────

/// Shared state passed to each gRPC handler.
#[derive(Clone)]
pub struct AlignmentService {
    engine: Arc<Mutex<AlignmentEngine>>,
    /// Broadcast channel for pushing `LineChange` events to `WatchChanges` subscribers.
    watch_tx: broadcast::Sender<LineChange>,
}

impl AlignmentService {
    pub fn new() -> Self {
        let (watch_tx, _) = broadcast::channel(1024);
        AlignmentService {
            engine: Arc::new(Mutex::new(AlignmentEngine::new())),
            watch_tx,
        }
    }
}

// ── Proto ↔ domain conversions ────────────────────────────────────────────────

fn proto_config_to_engine(cfg: ProtoEngineConfig) -> EngineConfig {
    let defaults = EngineConfig::default();
    EngineConfig {
        quality_threshold: if cfg.quality_threshold != 0.0 {
            cfg.quality_threshold as f64
        } else {
            defaults.quality_threshold
        },
        split_penalty: if cfg.split_penalty != 0 {
            cfg.split_penalty as f64
        } else {
            defaults.split_penalty
        },
        min_anchor_spacing_ms: if cfg.min_anchor_spacing_ms != 0 {
            cfg.min_anchor_spacing_ms as i64
        } else {
            defaults.min_anchor_spacing_ms
        },
        max_drift_ms: if cfg.max_drift_ms != 0 {
            cfg.max_drift_ms as i64
        } else {
            defaults.max_drift_ms
        },
        split_threshold_ms: if cfg.split_threshold_ms != 0 {
            cfg.split_threshold_ms as i64
        } else {
            defaults.split_threshold_ms
        },
        sweep_step_ms: if cfg.sweep_step_ms != 0 {
            cfg.sweep_step_ms as i64
        } else {
            defaults.sweep_step_ms
        },
        coverage_buffer_ms: if cfg.coverage_buffer_ms != 0 {
            cfg.coverage_buffer_ms as i64
        } else {
            defaults.coverage_buffer_ms
        },
        stitch_tolerance_ms: if cfg.stitch_tolerance_ms != 0 {
            cfg.stitch_tolerance_ms as i64
        } else {
            defaults.stitch_tolerance_ms
        },
    }
}

fn engine_config_to_proto(cfg: &EngineConfig) -> ProtoEngineConfig {
    ProtoEngineConfig {
        quality_threshold: cfg.quality_threshold as f32,
        split_penalty: cfg.split_penalty as i32,
        min_anchor_spacing_ms: cfg.min_anchor_spacing_ms as i32,
        max_drift_ms: cfg.max_drift_ms as i32,
        split_threshold_ms: cfg.split_threshold_ms as i32,
        sweep_step_ms: cfg.sweep_step_ms as i32,
        coverage_buffer_ms: cfg.coverage_buffer_ms as i32,
        stitch_tolerance_ms: cfg.stitch_tolerance_ms as i32,
    }
}

// ── gRPC trait implementation ─────────────────────────────────────────────────

#[tonic::async_trait]
impl AlignmentEngineService for AlignmentService {
    // ── LoadSubtitle ──────────────────────────────────────────────────────────
    async fn load_subtitle(
        &self,
        request: Request<SubtitleFile>,
    ) -> Result<Response<LoadResult>, Status> {
        let req = request.into_inner();
        let mut engine = self.engine.lock().await;
        let (line_count, fmt) = engine
            .load_subtitle(&req.content, &req.format)
            .map_err(to_status)?;
        Ok(Response::new(LoadResult {
            line_count: line_count as i32,
            format: fmt,
        }))
    }

    // ── IngestAudio ───────────────────────────────────────────────────────────
    type IngestAudioStream =
        std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<LineChange, Status>> + Send>>;

    async fn ingest_audio(
        &self,
        request: Request<AudioChunk>,
    ) -> Result<Response<Self::IngestAudioStream>, Status> {
        let req = request.into_inner();
        let mut engine = self.engine.lock().await;
        let changes = engine
            .ingest_chunk(
                req.pcm_data,
                req.sample_rate,
                req.film_start_ms,
                req.film_end_ms,
            )
            .map_err(to_status)?;

        // Broadcast changes to WatchChanges subscribers.
        for change in &changes {
            let _ = self.watch_tx.send(change.clone());
        }

        let stream =
            tokio_stream::iter(changes.into_iter().map(Ok::<LineChange, Status>));
        Ok(Response::new(Box::pin(stream)))
    }

    // ── WatchChanges ──────────────────────────────────────────────────────────
    type WatchChangesStream =
        std::pin::Pin<Box<dyn futures_core::Stream<Item = Result<LineChange, Status>> + Send>>;

    async fn watch_changes(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::WatchChangesStream>, Status> {
        let rx = self.watch_tx.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(|item| match item {
            Ok(change) => Some(Ok(change)),
            Err(_) => None, // lagged; skip dropped messages
        });
        Ok(Response::new(Box::pin(stream)))
    }

    // ── GetSubtitles ──────────────────────────────────────────────────────────
    async fn get_subtitles(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<SubtitleFile>, Status> {
        let engine = self.engine.lock().await;
        let sub = engine
            .subtitle
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("no subtitle loaded"))?;
        let content = sub.export_corrected().map_err(to_status)?;
        let format = sub.format.clone();
        Ok(Response::new(SubtitleFile { content, format }))
    }

    // ── GetAnchors ────────────────────────────────────────────────────────────
    async fn get_anchors(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<AnchorList>, Status> {
        let engine = self.engine.lock().await;
        let anchors: Vec<AnchorInfo> = engine
            .anchors
            .iter()
            .map(|a| AnchorInfo {
                id: a.id.to_string(),
                film_position_ms: a.film_position_ms,
                delta_ms: a.delta_ms,
                confidence: a.confidence as f32,
                line_start: a.line_range.start as i32,
                line_end: a.line_range.end as i32,
                chunk_id: a.chunk_id.to_string(),
            })
            .collect();
        Ok(Response::new(AnchorList { anchors }))
    }

    // ── GetChunkHistory ───────────────────────────────────────────────────────
    async fn get_chunk_history(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ChunkHistoryList>, Status> {
        let engine = self.engine.lock().await;
        let chunks: Vec<ChunkInfo> = engine
            .chunk_history
            .iter()
            .map(|c| ChunkInfo {
                id: c.id.to_string(),
                film_start_ms: c.film_start_ms,
                film_end_ms: c.film_end_ms,
                anchor_ids: c.anchor_ids.iter().map(|id| id.to_string()).collect(),
            })
            .collect();
        Ok(Response::new(ChunkHistoryList { chunks }))
    }

    // ── Reset ─────────────────────────────────────────────────────────────────
    async fn reset(&self, _request: Request<Empty>) -> Result<Response<Empty>, Status> {
        let mut engine = self.engine.lock().await;
        engine.reset();
        Ok(Response::new(Empty {}))
    }

    // ── ExportSrt ─────────────────────────────────────────────────────────────
    async fn export_srt(&self, _request: Request<Empty>) -> Result<Response<SrtFile>, Status> {
        let engine = self.engine.lock().await;
        let sub = engine
            .subtitle
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("no subtitle loaded"))?;
        Ok(Response::new(SrtFile {
            content: sub.export_srt(),
        }))
    }

    // ── SetConfig ─────────────────────────────────────────────────────────────
    async fn set_config(
        &self,
        request: Request<ProtoEngineConfig>,
    ) -> Result<Response<Empty>, Status> {
        let cfg = proto_config_to_engine(request.into_inner());
        let mut engine = self.engine.lock().await;
        engine.set_config(cfg);
        Ok(Response::new(Empty {}))
    }

    // ── GetConfig ─────────────────────────────────────────────────────────────
    async fn get_config(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<ProtoEngineConfig>, Status> {
        let engine = self.engine.lock().await;
        Ok(Response::new(engine_config_to_proto(&engine.config)))
    }
}
