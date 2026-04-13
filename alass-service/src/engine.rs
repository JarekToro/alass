use anyhow::{bail, Result};
use byteorder::{ByteOrder, LittleEndian};
use std::ops::Range;
use uuid::Uuid;
use webrtc_vad::{SampleRate, Vad, VadMode};

use alass_core::{
    align, align_nosplit, standard_scoring, NoProgressHandler,
    TimePoint as AlgTimePoint, TimeSpan as AlgTimeSpan,
};

use crate::proto::alignment::LineChange;
use crate::subtitle::SubtitleState;
use crate::warp::interpolate_delta;

// ── Configuration ────────────────────────────────────────────────────────────

/// Runtime-tunable parameters for the alignment engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Minimum nosplit score to accept a chunk (0.0–1.0).
    pub quality_threshold: f64,
    /// alass-core split penalty (useful range 4–20).
    pub split_penalty: f64,
    /// Minimum gap in ms between two adjacent anchors.
    pub min_anchor_spacing_ms: i64,
    /// Coarse sweep half-width in ms.
    pub max_drift_ms: i64,
    /// Delta jump (ms) that signals a new anchor group boundary.
    pub split_threshold_ms: i64,
    /// Coarse sweep step size in ms.
    pub sweep_step_ms: i64,
    /// Extra subtitle window buffer beyond chunk edges (ms).
    pub coverage_buffer_ms: i64,
    /// Chunk adjacency tolerance for stitching (ms).
    pub stitch_tolerance_ms: i64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            quality_threshold: 0.0,
            split_penalty: 7.0,
            min_anchor_spacing_ms: 30_000,
            max_drift_ms: 30_000,
            split_threshold_ms: 500,
            sweep_step_ms: 1_000,
            coverage_buffer_ms: 10_000,
            stitch_tolerance_ms: 5_000,
        }
    }
}

// ── Anchor ───────────────────────────────────────────────────────────────────

/// A control point in the piecewise-linear warp function.
#[derive(Debug, Clone)]
pub struct Anchor {
    pub id: Uuid,
    /// Absolute film position in ms (midpoint of subtitle group).
    pub film_position_ms: i64,
    /// Median correction delta in ms for this anchor.
    pub delta_ms: i64,
    /// Confidence score (nosplit alignment score).
    pub confidence: f64,
    /// Subtitle line indices covered by this anchor.
    pub line_range: Range<usize>,
    /// ID of the chunk that produced this anchor.
    pub chunk_id: Uuid,
}

// ── ProcessedChunk ────────────────────────────────────────────────────────────

/// A record of a chunk that has passed the full pipeline.
#[derive(Debug, Clone)]
pub struct ProcessedChunk {
    pub id: Uuid,
    pub film_start_ms: i64,
    pub film_end_ms: i64,
    /// Raw PCM bytes (16-bit signed LE, mono) stored for re-stitch.
    pub pcm_data: Vec<u8>,
    pub sample_rate: i32,
    /// IDs of anchors this chunk produced.
    pub anchor_ids: Vec<Uuid>,
}

// ── AlignmentEngine ──────────────────────────────────────────────────────────

/// The central state object for the alignment service.
pub struct AlignmentEngine {
    pub subtitle: Option<SubtitleState>,
    /// Sorted list of anchors (ascending by `film_position_ms`).
    pub anchors: Vec<Anchor>,
    /// History of successfully processed chunks.
    pub chunk_history: Vec<ProcessedChunk>,
    pub config: EngineConfig,
}

impl AlignmentEngine {
    pub fn new() -> Self {
        AlignmentEngine {
            subtitle: None,
            anchors: Vec::new(),
            chunk_history: Vec::new(),
            config: EngineConfig::default(),
        }
    }

    /// Load a new subtitle file, clearing all prior state.
    pub fn load_subtitle(&mut self, content: &str, format: &str) -> Result<(usize, String)> {
        let state = SubtitleState::parse(content, format)?;
        let line_count = state.len();
        let fmt = state.format.clone();
        self.subtitle = Some(state);
        self.anchors.clear();
        self.chunk_history.clear();
        Ok((line_count, fmt))
    }

    /// Reset to the original subtitle, clearing all anchors and chunk history.
    pub fn reset(&mut self) {
        if let Some(ref mut sub) = self.subtitle {
            sub.reset_corrections();
        }
        self.anchors.clear();
        self.chunk_history.clear();
    }

    /// Update the runtime config.
    pub fn set_config(&mut self, cfg: EngineConfig) {
        self.config = cfg;
    }

    /// Process one incoming audio chunk through the full 12-step pipeline.
    /// Returns the list of subtitle line changes that resulted.
    pub fn ingest_chunk(
        &mut self,
        pcm_bytes: Vec<u8>,
        sample_rate: i32,
        film_start_ms: i64,
        film_end_ms: i64,
    ) -> Result<Vec<LineChange>> {
        if self.subtitle.is_none() {
            bail!("no subtitle loaded");
        }
        if sample_rate != 8000 && sample_rate != 16000 {
            bail!("unsupported sample rate {}; must be 8000 or 16000", sample_rate);
        }
        if film_end_ms <= film_start_ms {
            bail!("film_end_ms must be greater than film_start_ms");
        }

        // ── Step 1: STITCH CHECK ──────────────────────────────────────────────
        let (pcm_bytes, film_start_ms, film_end_ms, sample_rate) =
            self.stitch_check(pcm_bytes, film_start_ms, film_end_ms, sample_rate);

        // ── Step 2: VAD ───────────────────────────────────────────────────────
        let vad_spans = run_vad(&pcm_bytes, sample_rate, film_start_ms)?;
        if vad_spans.is_empty() {
            return Ok(Vec::new());
        }

        let cfg = self.config.clone();
        let subtitle = self.subtitle.as_ref().unwrap();

        // ── Step 3: COARSE SWEEP ──────────────────────────────────────────────
        let coarse_window_start = film_start_ms - cfg.max_drift_ms;
        let coarse_window_end = film_end_ms + cfg.max_drift_ms;
        let (_coarse_sub_indices, coarse_sub_spans) =
            extract_subtitle_window(subtitle, coarse_window_start, coarse_window_end);

        if coarse_sub_spans.is_empty() {
            return Ok(Vec::new());
        }

        let (coarse_delta, _coarse_score) = align_nosplit(
            &vad_spans,
            &coarse_sub_spans,
            standard_scoring,
            NoProgressHandler,
        );
        let coarse_delta_ms = coarse_delta.as_i64();

        // ── Step 4: EXTRACT SUBTITLE WINDOW ──────────────────────────────────
        let win_start = film_start_ms - cfg.max_drift_ms + coarse_delta_ms - cfg.coverage_buffer_ms;
        let win_end = film_end_ms + cfg.max_drift_ms + coarse_delta_ms + cfg.coverage_buffer_ms;
        let (sub_indices, sub_spans) = extract_subtitle_window(subtitle, win_start, win_end);

        if sub_spans.is_empty() {
            return Ok(Vec::new());
        }

        // ── Step 5: ALIGN ─────────────────────────────────────────────────────
        let (alg_deltas, score) = align(
            &vad_spans,
            &sub_spans,
            cfg.split_penalty,
            None,
            standard_scoring,
            NoProgressHandler,
        );

        // ── Step 6: QUALITY GATE ──────────────────────────────────────────────
        if score < cfg.quality_threshold {
            return Ok(Vec::new());
        }

        // ── Step 7: EXTRACT ANCHORS ───────────────────────────────────────────
        let delta_ms_vec: Vec<i64> = alg_deltas.iter().map(|d| d.as_i64()).collect();
        let new_anchors = extract_anchors(
            &delta_ms_vec,
            &sub_indices,
            subtitle,
            score,
            cfg.split_threshold_ms,
        );

        if new_anchors.is_empty() {
            return Ok(Vec::new());
        }

        // ── Step 8: CONFLICT RESOLUTION ───────────────────────────────────────
        let chunk_id = Uuid::new_v4();
        let (accepted_anchors, evict_ids) =
            resolve_conflicts(new_anchors, &self.anchors, cfg.min_anchor_spacing_ms);

        if accepted_anchors.is_empty() {
            return Ok(Vec::new());
        }

        // Evict displaced anchors.
        for eid in &evict_ids {
            self.anchors.retain(|a| &a.id != eid);
        }

        // ── Step 9: INSERT ANCHORS ────────────────────────────────────────────
        let inserted_anchors: Vec<Anchor> = accepted_anchors
            .into_iter()
            .map(|mut a| {
                a.chunk_id = chunk_id;
                a
            })
            .collect();
        let inserted_anchor_ids: Vec<Uuid> = inserted_anchors.iter().map(|a| a.id).collect();
        for anchor in inserted_anchors {
            let pos = self
                .anchors
                .partition_point(|a| a.film_position_ms <= anchor.film_position_ms);
            self.anchors.insert(pos, anchor);
        }

        // ── Step 10: STORE CHUNK ──────────────────────────────────────────────
        self.chunk_history.push(ProcessedChunk {
            id: chunk_id,
            film_start_ms,
            film_end_ms,
            pcm_data: pcm_bytes,
            sample_rate,
            anchor_ids: inserted_anchor_ids,
        });

        // ── Step 11: RECOMPUTE AFFECTED LINES ────────────────────────────────
        let affected_range = compute_affected_range(&self.anchors, &sub_indices);
        let subtitle = self.subtitle.as_ref().unwrap();
        let n = subtitle.len();

        let old_corrected: Vec<(i64, i64)> = subtitle
            .lines()
            .iter()
            .enumerate()
            .filter(|(i, _)| affected_range.contains(i))
            .map(|(_, l)| (l.corrected_start_ms, l.corrected_end_ms))
            .collect();
        let affected_line_indices: Vec<usize> = (0..n).filter(|i| affected_range.contains(i)).collect();

        let subtitle = self.subtitle.as_mut().unwrap();
        for &idx in &affected_line_indices {
            let line = &mut subtitle.lines[idx];
            let midpoint = (line.original_start_ms + line.original_end_ms) / 2;
            let delta = interpolate_delta(&self.anchors, midpoint);
            line.corrected_start_ms = line.original_start_ms + delta;
            line.corrected_end_ms = line.original_end_ms + delta;
        }

        // ── Step 12: DIFF + EMIT ──────────────────────────────────────────────
        let subtitle = self.subtitle.as_ref().unwrap();
        let mut changes = Vec::new();
        for (order_idx, &line_idx) in affected_line_indices.iter().enumerate() {
            let (old_start, old_end) = old_corrected[order_idx];
            let line = &subtitle.lines[line_idx];
            if line.corrected_start_ms != old_start || line.corrected_end_ms != old_end {
                // Confidence from the anchor whose range covers this line (closest).
                let confidence = anchor_confidence_for_line(&self.anchors, line_idx);
                changes.push(LineChange {
                    line_index: line_idx as i32,
                    old_start_ms: old_start,
                    old_end_ms: old_end,
                    new_start_ms: line.corrected_start_ms,
                    new_end_ms: line.corrected_end_ms,
                    delta_change_ms: (line.corrected_start_ms - old_start),
                    confidence: confidence as f32,
                });
            }
        }

        Ok(changes)
    }

    // ── Stitch check (step 1) ─────────────────────────────────────────────────

    /// Scan chunk history for chunks adjacent to [film_start_ms, film_end_ms].
    /// Merge any found, evicting their anchors.  Returns the effective
    /// (pcm_bytes, film_start_ms, film_end_ms, sample_rate) to process.
    fn stitch_check(
        &mut self,
        pcm_bytes: Vec<u8>,
        film_start_ms: i64,
        film_end_ms: i64,
        sample_rate: i32,
    ) -> (Vec<u8>, i64, i64, i32) {
        let tolerance = self.config.stitch_tolerance_ms;

        // Iteratively find all adjacent chunks and merge them.
        let mut merged_pcm = pcm_bytes;
        let mut merged_start = film_start_ms;
        let mut merged_end = film_end_ms;
        let merged_sr = sample_rate;

        loop {
            // Find chunks adjacent to the current merged range.
            let adjacent: Vec<usize> = self
                .chunk_history
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.sample_rate == merged_sr
                        && ((p.film_end_ms - merged_start).abs() <= tolerance
                            || (merged_end - p.film_start_ms).abs() <= tolerance)
                })
                .map(|(i, _)| i)
                .collect();

            if adjacent.is_empty() {
                break;
            }

            // Collect adjacent chunks and evict their anchors.
            let mut pieces: Vec<(i64, Vec<u8>)> = adjacent
                .iter()
                .rev()
                .map(|&i| {
                    let p = self.chunk_history.remove(i);
                    for aid in &p.anchor_ids {
                        self.anchors.retain(|a| &a.id != aid);
                    }
                    (p.film_start_ms, p.pcm_data)
                })
                .collect();

            // Include the current merged chunk.
            pieces.push((merged_start, merged_pcm));

            // Sort by film start and concatenate PCM.
            pieces.sort_by_key(|(start, _)| *start);
            let new_start = pieces.first().map(|(s, _)| *s).unwrap_or(merged_start);
            let mut new_pcm = Vec::new();
            for (_, data) in pieces {
                new_pcm.extend_from_slice(&data);
            }

            merged_pcm = new_pcm;
            merged_start = new_start;
            // film_end_ms grows to cover all merged chunks.
            merged_end = self
                .chunk_history
                .iter()
                .map(|p| p.film_end_ms)
                .chain(std::iter::once(merged_end))
                .max()
                .unwrap_or(merged_end);
        }

        (merged_pcm, merged_start, merged_end, merged_sr)
    }
}

// ── VAD (step 2) ──────────────────────────────────────────────────────────────

/// Run WebRTC VAD on raw PCM bytes and return speech spans in absolute film
/// time milliseconds, expressed as `AlgTimeSpan` values ready for alass-core.
fn run_vad(
    pcm_bytes: &[u8],
    sample_rate: i32,
    film_start_ms: i64,
) -> Result<Vec<AlgTimeSpan>> {
    let vad_sample_rate = if sample_rate == 8000 {
        SampleRate::Rate8kHz
    } else {
        SampleRate::Rate16kHz
    };
    // Frame size: 10 ms worth of samples.
    let frame_samples = (sample_rate as usize) / 100;

    let mut vad = Vad::new_with_rate_and_mode(vad_sample_rate, VadMode::Quality);

    let samples_i16: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|b| LittleEndian::read_i16(b))
        .collect();

    let mut spans: Vec<AlgTimeSpan> = Vec::new();
    let mut in_speech = false;
    let mut speech_start_ms = 0i64;

    for (frame_idx, frame) in samples_i16.chunks(frame_samples).enumerate() {
        if frame.len() < frame_samples {
            break; // discard incomplete trailing frame
        }
        let frame_start_ms = film_start_ms + (frame_idx as i64) * 10;

        let is_voice = vad.is_voice_segment(frame).unwrap_or(false);
        if is_voice && !in_speech {
            speech_start_ms = frame_start_ms;
            in_speech = true;
        } else if !is_voice && in_speech {
            spans.push(AlgTimeSpan::new_safe(
                AlgTimePoint::from(speech_start_ms),
                AlgTimePoint::from(frame_start_ms),
            ));
            in_speech = false;
        }
    }
    if in_speech {
        let total_frames = samples_i16.len() / frame_samples;
        let end_ms = film_start_ms + (total_frames as i64) * 10;
        spans.push(AlgTimeSpan::new_safe(
            AlgTimePoint::from(speech_start_ms),
            AlgTimePoint::from(end_ms),
        ));
    }

    Ok(spans)
}

// ── Window extraction helper ──────────────────────────────────────────────────

/// Extract subtitle lines whose original midpoint falls within [start_ms, end_ms].
/// Returns the indices into `subtitle.lines` and the corresponding `AlgTimeSpan` slice.
fn extract_subtitle_window(
    subtitle: &SubtitleState,
    window_start_ms: i64,
    window_end_ms: i64,
) -> (Vec<usize>, Vec<AlgTimeSpan>) {
    let mut indices = Vec::new();
    let mut spans = Vec::new();

    for (i, line) in subtitle.lines().iter().enumerate() {
        if line.original_end_ms >= window_start_ms && line.original_start_ms <= window_end_ms {
            indices.push(i);
            spans.push(AlgTimeSpan::new_safe(
                AlgTimePoint::from(line.original_start_ms),
                AlgTimePoint::from(line.original_end_ms),
            ));
        }
    }

    (indices, spans)
}

// ── Anchor extraction (step 7) ────────────────────────────────────────────────

fn extract_anchors(
    delta_ms_vec: &[i64],
    sub_indices: &[usize],
    subtitle: &SubtitleState,
    score: f64,
    split_threshold_ms: i64,
) -> Vec<Anchor> {
    if delta_ms_vec.is_empty() || delta_ms_vec.len() != sub_indices.len() {
        return Vec::new();
    }

    let mut anchors = Vec::new();
    let mut group_start = 0usize;

    for i in 1..=delta_ms_vec.len() {
        let is_last = i == delta_ms_vec.len();
        let split = if is_last {
            true
        } else {
            (delta_ms_vec[i] - delta_ms_vec[i - 1]).abs() >= split_threshold_ms
        };

        if split {
            let group_indices = &sub_indices[group_start..i];
            let group_deltas = &delta_ms_vec[group_start..i];

            let median_delta = median(group_deltas);

            // Film position = midpoint of the corrected subtitle span for this group.
            let first_line = &subtitle.lines()[*group_indices.first().unwrap()];
            let last_line = &subtitle.lines()[*group_indices.last().unwrap()];
            let group_orig_start = first_line.original_start_ms;
            let group_orig_end = last_line.original_end_ms;
            let film_position_ms = (group_orig_start + group_orig_end) / 2 + median_delta;

            anchors.push(Anchor {
                id: Uuid::new_v4(),
                film_position_ms,
                delta_ms: median_delta,
                confidence: score,
                line_range: *group_indices.first().unwrap()..*group_indices.last().unwrap() + 1,
                chunk_id: Uuid::nil(), // will be set after conflict resolution
            });

            group_start = i;
        }
    }

    anchors
}

/// Compute the median of a slice of i64 values.
fn median(vals: &[i64]) -> i64 {
    if vals.is_empty() {
        return 0;
    }
    let mut v: Vec<i64> = vals.to_vec();
    v.sort_unstable();
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2
    } else {
        v[n / 2]
    }
}

// ── Conflict resolution (step 8) ─────────────────────────────────────────────

/// Check each new anchor against the existing sorted anchor list.
/// Returns the accepted anchors and the IDs of existing anchors to evict.
/// If any new anchor loses a conflict the entire batch is rejected (empty return).
fn resolve_conflicts(
    new_anchors: Vec<Anchor>,
    existing: &[Anchor],
    min_spacing_ms: i64,
) -> (Vec<Anchor>, Vec<Uuid>) {
    let mut to_evict: Vec<Uuid> = Vec::new();
    let mut accepted: Vec<Anchor> = Vec::new();

    for anchor in &new_anchors {
        let p = anchor.film_position_ms;
        let c = anchor.confidence;

        let prev = existing
            .iter()
            .rev()
            .find(|a| a.film_position_ms < p && (p - a.film_position_ms) < min_spacing_ms);
        let next = existing
            .iter()
            .find(|a| a.film_position_ms > p && (a.film_position_ms - p) < min_spacing_ms);

        let beats_prev = prev.map_or(true, |a| c > a.confidence);
        let beats_next = next.map_or(true, |a| c > a.confidence);

        match (prev, next) {
            (None, None) => {
                accepted.push(anchor.clone());
            }
            (Some(pr), None) => {
                if beats_prev {
                    to_evict.push(pr.id);
                    accepted.push(anchor.clone());
                } else {
                    // This anchor lost — discard the entire chunk's batch.
                    return (Vec::new(), Vec::new());
                }
            }
            (None, Some(nx)) => {
                if beats_next {
                    to_evict.push(nx.id);
                    accepted.push(anchor.clone());
                } else {
                    return (Vec::new(), Vec::new());
                }
            }
            (Some(pr), Some(nx)) => {
                if beats_prev && beats_next {
                    to_evict.push(pr.id);
                    to_evict.push(nx.id);
                    accepted.push(anchor.clone());
                } else {
                    return (Vec::new(), Vec::new());
                }
            }
        }
    }

    (accepted, to_evict)
}

// ── Recompute helpers (step 11) ───────────────────────────────────────────────

/// Determine the range of subtitle line indices that need recomputation after
/// inserting anchors covering `sub_indices`.
fn compute_affected_range(anchors: &[Anchor], sub_indices: &[usize]) -> Range<usize> {
    if sub_indices.is_empty() || anchors.is_empty() {
        return 0..0;
    }
    let first_idx = *sub_indices.first().unwrap();
    let last_idx = *sub_indices.last().unwrap();

    // Find the anchors that bound this region in the sorted list.
    let lo = anchors
        .iter()
        .rev()
        .find(|a| a.line_range.start <= first_idx)
        .map(|a| a.line_range.start)
        .unwrap_or(0);
    let hi = anchors
        .iter()
        .find(|a| a.line_range.end > last_idx)
        .map(|a| a.line_range.end)
        .unwrap_or_else(|| last_idx + 1);

    lo..hi
}

/// Get the confidence of the anchor whose range covers `line_idx`.
fn anchor_confidence_for_line(anchors: &[Anchor], line_idx: usize) -> f64 {
    anchors
        .iter()
        .filter(|a| a.line_range.contains(&line_idx))
        .map(|a| a.confidence)
        .next()
        .unwrap_or(0.0)
}
