use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};

use crate::engine::{AlignmentEngine, EngineConfig};
use crate::proto;
use crate::proto::alignment_engine_server::AlignmentEngine as AlignmentEngineTrait;

pub struct AlignmentService {
    engine: Arc<Mutex<AlignmentEngine>>,
}

impl AlignmentService {
    pub fn new(engine: Arc<Mutex<AlignmentEngine>>) -> Self {
        Self { engine }
    }
}

#[tonic::async_trait]
impl AlignmentEngineTrait for AlignmentService {
    async fn load_subtitle(
        &self,
        request: Request<proto::SubtitleFile>,
    ) -> Result<Response<proto::LoadResult>, Status> {
        let msg = request.into_inner();
        let mut engine = self.engine.lock().await;
        let (count, format) = engine
            .load_subtitle(&msg.content, &msg.format)
            .map_err(|e| Status::invalid_argument(e))?;
        Ok(Response::new(proto::LoadResult {
            line_count: count,
            format,
        }))
    }

    type IngestAudioStream = ReceiverStream<Result<proto::LineChange, Status>>;

    async fn ingest_audio(
        &self,
        request: Request<proto::AudioChunk>,
    ) -> Result<Response<Self::IngestAudioStream>, Status> {
        let chunk = request.into_inner();
        let engine = self.engine.clone();

        let (tx, rx) = mpsc::channel(1024);

        tokio::task::spawn_blocking(move || {
            let rt = tokio::runtime::Handle::current();
            let changes = rt.block_on(async {
                let mut eng = engine.lock().await;
                eng.ingest_chunk(
                    chunk.pcm_data.to_vec(),
                    chunk.sample_rate,
                    chunk.film_start_ms,
                    chunk.film_end_ms,
                )
            });

            match changes {
                Ok(changes) => {
                    for c in changes {
                        let _ = rt.block_on(tx.send(Ok(c)));
                    }
                }
                Err(e) => {
                    let _ = rt.block_on(tx.send(Err(Status::internal(e))));
                }
            }
        })
        .await
        .map_err(|e| Status::internal(format!("task join: {}", e)))?;

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    type WatchChangesStream = ReceiverStream<Result<proto::LineChange, Status>>;

    async fn watch_changes(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<Self::WatchChangesStream>, Status> {
        let mut broadcast_rx = {
            let engine = self.engine.lock().await;
            engine.subscribe_changes()
        };

        let (tx, rx) = mpsc::channel(1024);

        tokio::spawn(async move {
            loop {
                match broadcast_rx.recv().await {
                    Ok(change) => {
                        if tx.send(Ok(change)).await.is_err() {
                            break;
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                        continue;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }

    async fn get_subtitles(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::SubtitleFile>, Status> {
        let engine = self.engine.lock().await;
        let (content, format) = engine
            .get_corrected_subtitle()
            .ok_or_else(|| Status::not_found("no subtitle loaded"))?;
        Ok(Response::new(proto::SubtitleFile { content, format }))
    }

    async fn get_anchors(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::AnchorList>, Status> {
        let engine = self.engine.lock().await;
        Ok(Response::new(proto::AnchorList {
            anchors: engine.get_anchors(),
        }))
    }

    async fn get_chunk_history(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::ChunkHistoryList>, Status> {
        let engine = self.engine.lock().await;
        Ok(Response::new(proto::ChunkHistoryList {
            chunks: engine.get_chunk_history(),
        }))
    }

    async fn reset(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::Empty>, Status> {
        let mut engine = self.engine.lock().await;
        engine.reset();
        Ok(Response::new(proto::Empty {}))
    }

    async fn export_srt(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::SrtFile>, Status> {
        let engine = self.engine.lock().await;
        let content = engine
            .export_srt()
            .ok_or_else(|| Status::not_found("no subtitle loaded"))?;
        Ok(Response::new(proto::SrtFile { content }))
    }

    async fn set_config(
        &self,
        request: Request<proto::EngineConfig>,
    ) -> Result<Response<proto::Empty>, Status> {
        let msg = request.into_inner();
        let mut engine = self.engine.lock().await;
        engine.config = EngineConfig::from(&msg);
        Ok(Response::new(proto::Empty {}))
    }

    async fn get_config(
        &self,
        _request: Request<proto::Empty>,
    ) -> Result<Response<proto::EngineConfig>, Status> {
        let engine = self.engine.lock().await;
        Ok(Response::new(proto::EngineConfig::from(&engine.config)))
    }
}
