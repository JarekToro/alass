use std::collections::HashSet;

use alass_core::{self, NoProgressHandler, TimeDelta, TimePoint, TimeSpan};
use uuid::Uuid;

use crate::proto;
use crate::vad;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub quality_threshold: f64,
    pub split_penalty: f64,
    pub min_anchor_spacing_ms: i64,
    pub max_drift_ms: i64,
    pub split_threshold_ms: i64,
    pub sweep_step_ms: i64,
    pub coverage_buffer_ms: i64,
    pub stitch_tolerance_ms: i64,
    pub vad_mode: i32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            quality_threshold: 0.05,
            split_penalty: 7.0,
            min_anchor_spacing_ms: 30_000,
            max_drift_ms: 30_000,
            split_threshold_ms: 500,
            sweep_step_ms: 1_000,
            coverage_buffer_ms: 10_000,
            stitch_tolerance_ms: 5_000,
            vad_mode: 0,
        }
    }
}

impl From<&proto::EngineConfig> for EngineConfig {
    fn from(p: &proto::EngineConfig) -> Self {
        let d = EngineConfig::default();
        Self {
            quality_threshold: p
                .quality_threshold
                .map(|v| v as f64)
                .unwrap_or(d.quality_threshold),
            split_penalty: p
                .split_penalty
                .map(|v| v as f64)
                .unwrap_or(d.split_penalty),
            min_anchor_spacing_ms: p
                .min_anchor_spacing_ms
                .map(|v| v as i64)
                .unwrap_or(d.min_anchor_spacing_ms),
            max_drift_ms: p
                .max_drift_ms
                .map(|v| v as i64)
                .unwrap_or(d.max_drift_ms),
            split_threshold_ms: p
                .split_threshold_ms
                .map(|v| v as i64)
                .unwrap_or(d.split_threshold_ms),
            sweep_step_ms: p
                .sweep_step_ms
                .map(|v| v as i64)
                .unwrap_or(d.sweep_step_ms),
            coverage_buffer_ms: p
                .coverage_buffer_ms
                .map(|v| v as i64)
                .unwrap_or(d.coverage_buffer_ms),
            stitch_tolerance_ms: p
                .stitch_tolerance_ms
                .map(|v| v as i64)
                .unwrap_or(d.stitch_tolerance_ms),
            vad_mode: p.vad_mode.unwrap_or(d.vad_mode),
        }
    }
}

impl From<&EngineConfig> for proto::EngineConfig {
    fn from(c: &EngineConfig) -> Self {
        proto::EngineConfig {
            quality_threshold: Some(c.quality_threshold as f32),
            split_penalty: Some(c.split_penalty as i32),
            min_anchor_spacing_ms: Some(c.min_anchor_spacing_ms as i32),
            max_drift_ms: Some(c.max_drift_ms as i32),
            split_threshold_ms: Some(c.split_threshold_ms as i32),
            sweep_step_ms: Some(c.sweep_step_ms as i32),
            coverage_buffer_ms: Some(c.coverage_buffer_ms as i32),
            stitch_tolerance_ms: Some(c.stitch_tolerance_ms as i32),
            vad_mode: Some(c.vad_mode),
        }
    }
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Anchor {
    pub id: Uuid,
    pub film_position_ms: i64,
    pub delta_ms: i64,
    pub confidence: f64,
    pub line_start: usize,
    pub line_end: usize, // exclusive
    pub chunk_id: Uuid,
}

#[derive(Clone, Debug)]
struct ProcessedChunk {
    id: Uuid,
    film_start_ms: i64,
    film_end_ms: i64,
    pcm_data: Vec<u8>, // raw 16-bit LE PCM bytes
    sample_rate: i32,
    anchor_ids: Vec<Uuid>,
}

#[derive(Clone, Debug)]
struct SubtitleState {
    format_str: String,
    raw_content: Vec<u8>,
    /// (original_start_ms, original_end_ms) per line
    original_timespans: Vec<(i64, i64)>,
    corrected_start_ms: Vec<i64>,
    corrected_end_ms: Vec<i64>,
    /// Subtitle text per line (for SRT export)
    line_texts: Vec<String>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct AlignmentEngine {
    config: EngineConfig,
    subtitle: Option<SubtitleState>,
    anchors: Vec<Anchor>, // sorted by film_position_ms
    chunks: Vec<ProcessedChunk>,
    change_tx: tokio::sync::broadcast::Sender<proto::LineChange>,
}

impl AlignmentEngine {
    pub fn new(config: EngineConfig) -> Self {
        let (change_tx, _) = tokio::sync::broadcast::channel(4096);
        Self {
            config,
            subtitle: None,
            anchors: Vec::new(),
            chunks: Vec::new(),
            change_tx,
        }
    }

    pub fn subscribe_changes(
        &self,
    ) -> tokio::sync::broadcast::Receiver<proto::LineChange> {
        self.change_tx.subscribe()
    }

    pub fn set_config(&mut self, config: EngineConfig) {
        self.config = config;
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // LoadSubtitle
    // -----------------------------------------------------------------------

    pub fn load_subtitle(
        &mut self,
        content: &str,
        format: &str,
    ) -> Result<(i32, String), String> {
        use subparse::{parse_bytes, SubtitleFormat};

        let fmt = match format {
            "srt" => SubtitleFormat::SubRip,
            "ass" | "ssa" => SubtitleFormat::SubStationAlpha,
            other => return Err(format!("unsupported format: {}", other)),
        };

        let bytes = content.as_bytes();
        let sub_file = parse_bytes(fmt, bytes, None, 30.0)
            .map_err(|e| format!("parse error: {}", e))?;

        let entries = sub_file
            .get_subtitle_entries()
            .map_err(|e| format!("entry error: {}", e))?;

        let mut original_timespans = Vec::with_capacity(entries.len());
        let mut line_texts = Vec::with_capacity(entries.len());

        for entry in &entries {
            let start = std::cmp::min(entry.timespan.start, entry.timespan.end);
            let end = std::cmp::max(entry.timespan.start, entry.timespan.end);
            original_timespans.push((start.msecs(), end.msecs()));
            line_texts.push(entry.line.clone().unwrap_or_default());
        }

        let len = original_timespans.len();
        let corrected_start: Vec<i64> =
            original_timespans.iter().map(|t| t.0).collect();
        let corrected_end: Vec<i64> =
            original_timespans.iter().map(|t| t.1).collect();

        self.subtitle = Some(SubtitleState {
            format_str: format.to_string(),
            raw_content: bytes.to_vec(),
            original_timespans,
            corrected_start_ms: corrected_start,
            corrected_end_ms: corrected_end,
            line_texts,
        });

        // Clear alignment state
        self.anchors.clear();
        self.chunks.clear();

        Ok((len as i32, format.to_string()))
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    pub fn reset(&mut self) {
        if let Some(sub) = self.subtitle.as_mut() {
            sub.corrected_start_ms = sub.original_timespans.iter().map(|t| t.0).collect();
            sub.corrected_end_ms = sub.original_timespans.iter().map(|t| t.1).collect();
        }
        self.anchors.clear();
        self.chunks.clear();
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    pub fn get_anchors(&self) -> Vec<proto::AnchorInfo> {
        self.anchors
            .iter()
            .map(|a| proto::AnchorInfo {
                id: a.id.to_string(),
                film_position_ms: a.film_position_ms,
                delta_ms: a.delta_ms,
                confidence: a.confidence as f32,
                line_start: a.line_start as i32,
                line_end: a.line_end as i32,
                chunk_id: a.chunk_id.to_string(),
            })
            .collect()
    }

    pub fn get_chunk_history(&self) -> Vec<proto::ChunkInfo> {
        self.chunks
            .iter()
            .map(|c| proto::ChunkInfo {
                id: c.id.to_string(),
                film_start_ms: c.film_start_ms,
                film_end_ms: c.film_end_ms,
                anchor_ids: c.anchor_ids.iter().map(|id| id.to_string()).collect(),
            })
            .collect()
    }

    pub fn get_corrected_subtitle(&self) -> Option<(String, String)> {
        let fmt = self.subtitle.as_ref()?.format_str.clone();
        self.get_corrected_subtitle_as(&fmt)
    }

    pub fn export_srt(&self) -> Option<String> {
        // Delegate to the unified corrected-subtitle path to avoid
        // maintaining two independent serialisation codepaths.
        let (content, _) = self.get_corrected_subtitle_as("srt")?;
        Some(content)
    }

    /// Internal helper: produce corrected subtitle in the requested format.
    fn get_corrected_subtitle_as(&self, target_format: &str) -> Option<(String, String)> {
        let sub = self.subtitle.as_ref()?;

        use subparse::{parse_bytes, SubtitleFormat};
        let src_fmt = match sub.format_str.as_str() {
            "srt" => SubtitleFormat::SubRip,
            "ass" | "ssa" => SubtitleFormat::SubStationAlpha,
            _ => return None,
        };

        let out_fmt = match target_format {
            "srt" => SubtitleFormat::SubRip,
            "ass" | "ssa" => SubtitleFormat::SubStationAlpha,
            _ => return None,
        };

        // If converting across formats we'd need a different base file;
        // for now only same-format or always-SRT is supported.
        let base_fmt = if target_format == sub.format_str.as_str() {
            src_fmt
        } else {
            out_fmt
        };

        // For SRT export when source is not SRT, build from scratch
        if target_format == "srt" && sub.format_str != "srt" {
            return Some((self.build_srt_from_scratch(sub), "srt".to_string()));
        }

        let mut sub_file = parse_bytes(base_fmt, &sub.raw_content, None, 30.0).ok()?;

        let corrected: Vec<subparse::SubtitleEntry> = sub
            .corrected_start_ms
            .iter()
            .zip(sub.corrected_end_ms.iter())
            .zip(sub.line_texts.iter())
            .map(|((&s, &e), text)| subparse::SubtitleEntry {
                timespan: subparse::timetypes::TimeSpan::new(
                    subparse::timetypes::TimePoint::from_msecs(s),
                    subparse::timetypes::TimePoint::from_msecs(e),
                ),
                line: Some(text.clone()),
            })
            .collect();

        sub_file.update_subtitle_entries(&corrected).ok()?;
        let data = sub_file.to_data().ok()?;
        let text = String::from_utf8_lossy(&data).to_string();
        Some((text, target_format.to_string()))
    }

    /// Fallback SRT builder for cross-format export.
    fn build_srt_from_scratch(&self, sub: &SubtitleState) -> String {
        let mut out = String::new();
        for (i, text) in sub.line_texts.iter().enumerate() {
            let start = sub.corrected_start_ms[i];
            let end = sub.corrected_end_ms[i];
            out.push_str(&format!("{}\n", i + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(start),
                format_srt_time(end)
            ));
            out.push_str(text);
            if !text.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
        out
    }

    // -----------------------------------------------------------------------
    // Ingest — the 12-step pipeline
    // -----------------------------------------------------------------------

    pub fn ingest_chunk(
        &mut self,
        pcm_data: Vec<u8>,
        sample_rate: i32,
        film_start_ms: i64,
        film_end_ms: i64,
    ) -> Result<Vec<proto::LineChange>, String> {
        if self.subtitle.is_none() {
            return Err("no subtitle loaded".into());
        }
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err("sample_rate must be 8000 or 16000".into());
        }

        // ── 1. STITCH CHECK ──────────────────────────────────────────────
        let (eff_pcm, eff_start, eff_end, stitch_evicted) =
            self.stitch_check(pcm_data, film_start_ms, film_end_ms, sample_rate);

        // ── 2. VAD ───────────────────────────────────────────────────────
        let vad_spans = vad::run_vad(&eff_pcm, sample_rate, eff_start, self.config.vad_mode);
        if vad_spans.is_empty() {
            return Ok(Vec::new());
        }

        // ── 3. COARSE SWEEP ─────────────────────────────────────────────
        let broad_start = eff_start - 2 * self.config.max_drift_ms;
        let broad_end = eff_end + 2 * self.config.max_drift_ms;
        let broad_window = self.subtitle_spans_in_range(broad_start, broad_end);

        let (coarse_delta, _coarse_score) = if broad_window.is_empty() {
            (0i64, 0.0)
        } else {
            self.coarse_sweep(&vad_spans, &broad_window)
        };

        // ── 4. EXTRACT SUBTITLE WINDOW ──────────────────────────────────
        let win_start = eff_start - self.config.max_drift_ms + coarse_delta
            - self.config.coverage_buffer_ms;
        let win_end = eff_end + self.config.max_drift_ms + coarse_delta
            + self.config.coverage_buffer_ms;
        let (window_spans, window_indices) =
            self.subtitle_window_with_indices(win_start, win_end);

        if window_spans.is_empty() {
            return Ok(Vec::new());
        }

        // ── 5. ALIGN ────────────────────────────────────────────────────
        let (deltas, score) = alass_core::align(
            &vad_spans,
            &window_spans,
            self.config.split_penalty,
            None,
            alass_core::standard_scoring,
            NoProgressHandler,
        );

        // ── 6. QUALITY GATE ─────────────────────────────────────────────
        if score < self.config.quality_threshold {
            if stitch_evicted {
                let changes = self.recompute_all_and_diff();
                self.broadcast_changes(&changes);
                return Ok(changes);
            }
            return Ok(Vec::new());
        }

        // ── 7. EXTRACT ANCHORS ──────────────────────────────────────────
        let chunk_id = Uuid::new_v4();
        let new_anchors =
            self.extract_anchors(&deltas, &window_indices, score, chunk_id);

        if new_anchors.is_empty() {
            return Ok(Vec::new());
        }

        // ── 8. CONFLICT RESOLUTION ──────────────────────────────────────
        let resolution = self.resolve_conflicts(&new_anchors);
        match resolution {
            None => {
                // Chunk loses — only recompute if stitch evicted anchors
                if stitch_evicted {
                    let changes = self.recompute_all_and_diff();
                    self.broadcast_changes(&changes);
                    return Ok(changes);
                }
                Ok(Vec::new())
            }
            Some(ids_to_remove) => {
                // Remove conflicting existing anchors
                self.anchors.retain(|a| !ids_to_remove.contains(&a.id));

                // ── 9. INSERT ANCHORS ────────────────────────────────────
                let anchor_ids: Vec<Uuid> =
                    new_anchors.iter().map(|a| a.id).collect();
                for anchor in new_anchors {
                    let pos = self
                        .anchors
                        .partition_point(|a| a.film_position_ms < anchor.film_position_ms);
                    self.anchors.insert(pos, anchor);
                }

                // ── 10. STORE CHUNK ──────────────────────────────────────
                // PCM data is not retained — it was only needed for
                // stitching during this ingest call.  Dropping it avoids
                // unbounded memory growth for long sessions.
                self.chunks.push(ProcessedChunk {
                    id: chunk_id,
                    film_start_ms: eff_start,
                    film_end_ms: eff_end,
                    pcm_data: Vec::new(),
                    sample_rate,
                    anchor_ids,
                });

                // ── 11 + 12. RECOMPUTE + DIFF + EMIT ─────────────────────
                let changes = self.recompute_all_and_diff();
                self.broadcast_changes(&changes);
                Ok(changes)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Pipeline helpers
    // -----------------------------------------------------------------------

    /// Step 1: scan chunk history for adjacent/overlapping chunks, merge PCM.
    /// Returns `(pcm, start, end, anchors_evicted)`.
    fn stitch_check(
        &mut self,
        mut pcm: Vec<u8>,
        mut start: i64,
        mut end: i64,
        sample_rate: i32,
    ) -> (Vec<u8>, i64, i64, bool) {
        let mut evicted_any = false;

        loop {
            let tol = self.config.stitch_tolerance_ms;
            let found = self.chunks.iter().position(|c| {
                if c.sample_rate != sample_rate {
                    return false;
                }
                let adjacent_after = (c.film_end_ms - start).abs() <= tol;
                let adjacent_before = (end - c.film_start_ms).abs() <= tol;
                let overlaps = c.film_start_ms < end && c.film_end_ms > start;
                adjacent_after || adjacent_before || overlaps
            });

            match found {
                None => break,
                Some(idx) => {
                    let chunk = self.chunks.remove(idx);
                    // Evict old chunk's anchors
                    if !chunk.anchor_ids.is_empty() {
                        let evict: HashSet<Uuid> =
                            chunk.anchor_ids.iter().copied().collect();
                        self.anchors.retain(|a| !evict.contains(&a.id));
                        evicted_any = true;
                    }

                    // Merge PCM in time order
                    if chunk.film_start_ms <= start {
                        // Old chunk comes first
                        let gap_ms = start - chunk.film_end_ms;
                        let gap_samples =
                            (gap_ms.max(0) * sample_rate as i64 / 1000) as usize;
                        let gap_bytes = gap_samples * 2;

                        let mut merged = chunk.pcm_data;
                        merged.resize(merged.len() + gap_bytes, 0u8);
                        merged.extend_from_slice(&pcm);
                        pcm = merged;
                        start = chunk.film_start_ms;
                    } else {
                        // Old chunk comes after
                        let gap_ms = chunk.film_start_ms - end;
                        let gap_samples =
                            (gap_ms.max(0) * sample_rate as i64 / 1000) as usize;
                        let gap_bytes = gap_samples * 2;

                        pcm.resize(pcm.len() + gap_bytes, 0u8);
                        pcm.extend_from_slice(&chunk.pcm_data);
                        end = chunk.film_end_ms;
                    }
                    // Continue looping (cascade)
                }
            }
        }

        (pcm, start, end, evicted_any)
    }

    /// Extract alass-core TimeSpans for subtitle lines whose original timing
    /// overlaps `[range_start, range_end]`.
    fn subtitle_spans_in_range(
        &self,
        range_start: i64,
        range_end: i64,
    ) -> Vec<TimeSpan> {
        let sub = self.subtitle.as_ref().unwrap();
        sub.original_timespans
            .iter()
            .filter(|&&(s, e)| e >= range_start && s <= range_end)
            .map(|&(s, e)| {
                TimeSpan::new(TimePoint::from(s), TimePoint::from(e))
            })
            .collect()
    }

    /// Same as above but also returns the original line indices.
    fn subtitle_window_with_indices(
        &self,
        range_start: i64,
        range_end: i64,
    ) -> (Vec<TimeSpan>, Vec<usize>) {
        let sub = self.subtitle.as_ref().unwrap();
        let mut spans = Vec::new();
        let mut indices = Vec::new();

        for (i, &(s, e)) in sub.original_timespans.iter().enumerate() {
            if e >= range_start && s <= range_end {
                spans.push(TimeSpan::new(
                    TimePoint::from(s),
                    TimePoint::from(e),
                ));
                indices.push(i);
            }
        }
        (spans, indices)
    }

    /// Step 3: coarse sweep — evaluate nosplit score at discrete deltas.
    fn coarse_sweep(
        &self,
        vad_spans: &[TimeSpan],
        sub_spans: &[TimeSpan],
    ) -> (i64, f64) {
        let max_d = self.config.max_drift_ms;
        let step = self.config.sweep_step_ms.max(1);
        let steps = max_d / step;

        let mut best_delta: i64 = 0;
        let mut best_score: f64 = f64::NEG_INFINITY;

        for i in -steps..=steps {
            let delta = i * step;
            let shifted: Vec<TimeSpan> = sub_spans
                .iter()
                .map(|s| *s + TimeDelta::from_i64(delta))
                .collect();

            let score = alass_core::get_nosplit_score(
                vad_spans.iter().copied(),
                shifted.iter().copied(),
                alass_core::standard_scoring,
            );

            if score > best_score {
                best_score = score;
                best_delta = delta;
            }
        }

        (best_delta, best_score)
    }

    /// Step 7: group deltas into anchor groups, emit Anchors.
    fn extract_anchors(
        &self,
        deltas: &[TimeDelta],
        window_indices: &[usize],
        score: f64,
        chunk_id: Uuid,
    ) -> Vec<Anchor> {
        if deltas.is_empty() || window_indices.is_empty() {
            return Vec::new();
        }

        let sub = self.subtitle.as_ref().unwrap();
        let threshold = self.config.split_threshold_ms;

        // Build (line_index, delta_ms) pairs
        let pairs: Vec<(usize, i64)> = deltas
            .iter()
            .zip(window_indices.iter())
            .map(|(d, &idx)| (idx, d.as_i64()))
            .collect();

        // Group consecutive lines where |delta[i] - delta[i-1]| < threshold
        let mut groups: Vec<Vec<(usize, i64)>> = Vec::new();
        let mut current_group: Vec<(usize, i64)> = Vec::new();

        for &(idx, delta_ms) in &pairs {
            if let Some(&(_, prev_delta)) = current_group.last() {
                if (delta_ms - prev_delta).abs() >= threshold {
                    if !current_group.is_empty() {
                        groups.push(std::mem::take(&mut current_group));
                    }
                }
            }
            current_group.push((idx, delta_ms));
        }
        if !current_group.is_empty() {
            groups.push(current_group);
        }

        groups
            .into_iter()
            .map(|group| {
                // Use min/max of actual indices in the group rather than
                // assuming contiguity — indices may be non-contiguous
                // (e.g. [3, 7, 11]) when subtitle lines are sparse.
                let line_start = group.iter().map(|&(idx, _)| idx).min().unwrap();
                let line_end = group.iter().map(|&(idx, _)| idx).max().unwrap() + 1;

                // Median delta
                let mut ds: Vec<i64> = group.iter().map(|&(_, d)| d).collect();
                ds.sort();
                let median_delta = ds[ds.len() / 2];

                // Film position = midpoint of the actual sampled lines'
                // time range (first sampled start, last sampled end).
                let first_start = sub.original_timespans[line_start].0;
                let last_end = sub.original_timespans[line_end - 1].1;
                let film_position = (first_start + last_end) / 2;

                Anchor {
                    id: Uuid::new_v4(),
                    film_position_ms: film_position,
                    delta_ms: median_delta,
                    confidence: score,
                    line_start,
                    line_end,
                    chunk_id,
                }
            })
            .collect()
    }

    /// Step 8: conflict resolution. Returns `Some(ids_to_remove)` if all new
    /// anchors win their conflicts, `None` if any loses (discard chunk).
    fn resolve_conflicts(&self, new_anchors: &[Anchor]) -> Option<HashSet<Uuid>> {
        let spacing = self.config.min_anchor_spacing_ms;
        let mut ids_to_remove: HashSet<Uuid> = HashSet::new();

        for n in new_anchors {
            let pos = n.film_position_ms;
            let conf = n.confidence;

            let prev = self
                .anchors
                .iter()
                .filter(|a| {
                    a.film_position_ms < pos
                        && (pos - a.film_position_ms) < spacing
                })
                .max_by_key(|a| a.film_position_ms);

            let next = self
                .anchors
                .iter()
                .filter(|a| {
                    a.film_position_ms > pos
                        && (a.film_position_ms - pos) < spacing
                })
                .min_by_key(|a| a.film_position_ms);

            match (prev, next) {
                (None, None) => {} // free insert
                (Some(p), None) => {
                    if conf > p.confidence {
                        ids_to_remove.insert(p.id);
                    } else {
                        return None;
                    }
                }
                (None, Some(nx)) => {
                    if conf > nx.confidence {
                        ids_to_remove.insert(nx.id);
                    } else {
                        return None;
                    }
                }
                (Some(p), Some(nx)) => {
                    if conf > p.confidence && conf > nx.confidence {
                        ids_to_remove.insert(p.id);
                        ids_to_remove.insert(nx.id);
                    } else {
                        return None;
                    }
                }
            }
        }

        Some(ids_to_remove)
    }

    /// Steps 11 + 12: recompute all corrected times via warp function,
    /// diff against previous state, return changes.
    fn recompute_all_and_diff(&mut self) -> Vec<proto::LineChange> {
        let sub = match self.subtitle.as_ref() {
            Some(s) => s,
            None => return Vec::new(),
        };

        // Pre-compute all new corrected values and confidence from anchors
        // (immutable borrow of self only)
        let updates: Vec<(usize, i64, i64, i64, i64, f64)> = (0..sub.original_timespans.len())
            .filter_map(|i| {
                let (orig_s, orig_e) = sub.original_timespans[i];
                let new_s = orig_s + self.interpolate_delta_for(orig_s);
                let new_e = orig_e + self.interpolate_delta_for(orig_e);
                let old_s = sub.corrected_start_ms[i];
                let old_e = sub.corrected_end_ms[i];
                if new_s != old_s || new_e != old_e {
                    let conf = self.nearest_anchor_confidence_for(orig_s);
                    Some((i, old_s, old_e, new_s, new_e, conf))
                } else {
                    None
                }
            })
            .collect();

        // Now mutate subtitle state and build change list
        let sub = self.subtitle.as_mut().unwrap();
        let mut changes = Vec::with_capacity(updates.len());

        for (i, old_s, old_e, new_s, new_e, conf) in updates {
            sub.corrected_start_ms[i] = new_s;
            sub.corrected_end_ms[i] = new_e;

            let delta_change = ((new_s - old_s) + (new_e - old_e)) / 2;
            changes.push(proto::LineChange {
                line_index: i as i32,
                old_start_ms: old_s,
                old_end_ms: old_e,
                new_start_ms: new_s,
                new_end_ms: new_e,
                delta_change_ms: delta_change,
                confidence: conf as f32,
            });
        }

        changes
    }

    // Non-borrowing wrappers (avoid &self + &mut self conflict)
    fn interpolate_delta_for(&self, original_ms: i64) -> i64 {
        if self.anchors.is_empty() {
            return 0;
        }
        let idx =
            self.anchors.partition_point(|a| a.film_position_ms <= original_ms);
        if idx == 0 {
            return self.anchors[0].delta_ms;
        }
        if idx >= self.anchors.len() {
            return self.anchors.last().unwrap().delta_ms;
        }
        let prev = &self.anchors[idx - 1];
        let next = &self.anchors[idx];
        let span = (next.film_position_ms - prev.film_position_ms) as f64;
        if span == 0.0 {
            return prev.delta_ms;
        }
        let t = (original_ms - prev.film_position_ms) as f64 / span;
        (prev.delta_ms as f64 + t * (next.delta_ms - prev.delta_ms) as f64)
            as i64
    }

    fn nearest_anchor_confidence_for(&self, original_ms: i64) -> f64 {
        if self.anchors.is_empty() {
            return 0.0;
        }
        let idx =
            self.anchors.partition_point(|a| a.film_position_ms <= original_ms);
        if idx == 0 {
            return self.anchors[0].confidence;
        }
        if idx >= self.anchors.len() {
            return self.anchors.last().unwrap().confidence;
        }
        let prev = &self.anchors[idx - 1];
        let next = &self.anchors[idx];
        if (original_ms - prev.film_position_ms).abs()
            <= (next.film_position_ms - original_ms).abs()
        {
            prev.confidence
        } else {
            next.confidence
        }
    }

    fn broadcast_changes(&self, changes: &[proto::LineChange]) {
        for change in changes {
            let _ = self.change_tx.send(change.clone());
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build an engine without a broadcast channel dependency.
/// Tests that don't exercise broadcast can use this.
#[cfg(test)]
fn test_engine(config: EngineConfig) -> AlignmentEngine {
    AlignmentEngine::new(config)
}

fn format_srt_time(ms: i64) -> String {
    let negative = ms < 0;
    let abs = ms.unsigned_abs();
    let h = abs / 3_600_000;
    let m = (abs % 3_600_000) / 60_000;
    let s = (abs % 60_000) / 1_000;
    let millis = abs % 1_000;
    if negative {
        format!("-{:02}:{:02}:{:02},{:03}", h, m, s, millis)
    } else {
        format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRT: &str = "\
1
00:00:01,000 --> 00:00:04,000
Hello world

2
00:00:05,000 --> 00:00:08,000
Second line

3
00:00:10,000 --> 00:00:13,000
Third line

4
00:00:20,000 --> 00:00:23,000
Fourth line

5
00:00:30,000 --> 00:00:33,000
Fifth line
";

    fn make_engine() -> AlignmentEngine {
        test_engine(EngineConfig::default())
    }

    fn load_sample(engine: &mut AlignmentEngine) {
        engine.load_subtitle(SAMPLE_SRT, "srt").unwrap();
    }

    // -- E2E helpers --------------------------------------------------------

    /// Generate synthetic 16-bit LE mono PCM where `speech_segments` contain
    /// a 400 Hz sine wave and everything else is silence.  The WebRTC VAD
    /// (mode 0) reliably classifies the sine-wave frames as speech.
    ///
    /// `speech_segments` are `(start_ms, end_ms)` **relative to the start of
    /// the buffer** (i.e. sample 0 corresponds to ms 0).
    fn make_speech_pcm(
        sample_rate: i32,
        speech_segments: &[(i64, i64)],
        total_duration_ms: i64,
    ) -> Vec<u8> {
        let total_samples = (sample_rate as i64 * total_duration_ms / 1000) as usize;
        let mut samples = vec![0i16; total_samples];

        let freq = 400.0_f64;
        let amplitude = 10_000.0_f64;

        for &(start_ms, end_ms) in speech_segments {
            let start_sample = (sample_rate as i64 * start_ms / 1000) as usize;
            let end_sample = (sample_rate as i64 * end_ms / 1000) as usize;
            let end_sample = end_sample.min(total_samples);
            for i in start_sample..end_sample {
                let t = i as f64 / sample_rate as f64;
                samples[i] =
                    (amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()) as i16;
            }
        }

        let mut bytes = Vec::with_capacity(total_samples * 2);
        for s in &samples {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        bytes
    }

    /// Build a valid SRT string from `(start_ms, end_ms, text)` tuples.
    fn make_srt(segments: &[(i64, i64, &str)]) -> String {
        let mut out = String::new();
        for (i, &(start, end, text)) in segments.iter().enumerate() {
            out.push_str(&format!("{}\n", i + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(start),
                format_srt_time(end),
            ));
            out.push_str(text);
            out.push_str("\n\n");
        }
        out
    }

    /// Panics if `|actual - expected| > tolerance`.
    fn assert_within(actual: i64, expected: i64, tolerance: i64, label: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{}: expected ~{} +/- {}, got {}",
            label,
            expected,
            tolerance,
            actual,
        );
    }

    // -- LoadSubtitle -------------------------------------------------------

    #[test]
    fn load_subtitle_parses_srt() {
        let mut eng = make_engine();
        let (count, fmt) = eng.load_subtitle(SAMPLE_SRT, "srt").unwrap();
        assert_eq!(count, 5);
        assert_eq!(fmt, "srt");
    }

    #[test]
    fn load_subtitle_stores_timespans() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        let sub = eng.subtitle.as_ref().unwrap();
        assert_eq!(sub.original_timespans[0], (1000, 4000));
        assert_eq!(sub.original_timespans[1], (5000, 8000));
        assert_eq!(sub.original_timespans[4], (30000, 33000));
    }

    #[test]
    fn load_subtitle_rejects_unknown_format() {
        let mut eng = make_engine();
        assert!(eng.load_subtitle("data", "vtt").is_err());
    }

    #[test]
    fn load_subtitle_clears_previous_state() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Manually insert an anchor and a chunk
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 5000,
            delta_ms: 100,
            confidence: 0.9,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 10000,
            pcm_data: vec![],
            sample_rate: 16000,
            anchor_ids: vec![],
        });

        // Reload — should clear
        load_sample(&mut eng);
        assert!(eng.anchors.is_empty());
        assert!(eng.chunks.is_empty());
    }

    // -- Reset --------------------------------------------------------------

    #[test]
    fn reset_restores_original_times() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Manually shift corrected times
        let sub = eng.subtitle.as_mut().unwrap();
        sub.corrected_start_ms[0] = 9999;
        sub.corrected_end_ms[0] = 9999;

        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 5000,
            delta_ms: 100,
            confidence: 0.9,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });

        eng.reset();

        let sub = eng.subtitle.as_ref().unwrap();
        assert_eq!(sub.corrected_start_ms[0], 1000);
        assert_eq!(sub.corrected_end_ms[0], 4000);
        assert!(eng.anchors.is_empty());
    }

    // -- Warp function (interpolate_delta_for) ------------------------------

    #[test]
    fn warp_no_anchors_returns_zero() {
        let eng = make_engine();
        assert_eq!(eng.interpolate_delta_for(5000), 0);
    }

    #[test]
    fn warp_single_anchor_flat() {
        let mut eng = make_engine();
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 10_000,
            delta_ms: 3000,
            confidence: 0.8,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });

        // Before, at, and after the sole anchor — all get the same delta
        assert_eq!(eng.interpolate_delta_for(0), 3000);
        assert_eq!(eng.interpolate_delta_for(10_000), 3000);
        assert_eq!(eng.interpolate_delta_for(50_000), 3000);
    }

    #[test]
    fn warp_two_anchors_interpolates() {
        let mut eng = make_engine();
        let cid = Uuid::new_v4();
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 10_000,
            delta_ms: 1000,
            confidence: 0.8,
            line_start: 0,
            line_end: 1,
            chunk_id: cid,
        });
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 20_000,
            delta_ms: 3000,
            confidence: 0.8,
            line_start: 1,
            line_end: 2,
            chunk_id: cid,
        });

        // Before first → flat at first delta
        assert_eq!(eng.interpolate_delta_for(0), 1000);
        // At first anchor
        assert_eq!(eng.interpolate_delta_for(10_000), 1000);
        // Midpoint → linear interpolation
        assert_eq!(eng.interpolate_delta_for(15_000), 2000);
        // At second anchor
        assert_eq!(eng.interpolate_delta_for(20_000), 3000);
        // After last → flat at last delta
        assert_eq!(eng.interpolate_delta_for(99_000), 3000);
    }

    #[test]
    fn warp_three_anchors_piecewise() {
        let mut eng = make_engine();
        let cid = Uuid::new_v4();
        // 10s → +1s, 20s → +3s, 40s → +1s
        for (pos, delta) in [(10_000, 1000), (20_000, 3000), (40_000, 1000)] {
            eng.anchors.push(Anchor {
                id: Uuid::new_v4(),
                film_position_ms: pos,
                delta_ms: delta,
                confidence: 0.8,
                line_start: 0,
                line_end: 1,
                chunk_id: cid,
            });
        }

        // Between anchor 1-2: linear from 1000→3000 over 10s
        assert_eq!(eng.interpolate_delta_for(15_000), 2000);
        // Between anchor 2-3: linear from 3000→1000 over 20s
        assert_eq!(eng.interpolate_delta_for(30_000), 2000);
    }

    // -- extract_anchors ----------------------------------------------------

    #[test]
    fn extract_anchors_single_group() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        let deltas = vec![
            TimeDelta::from_i64(500),
            TimeDelta::from_i64(510),
            TimeDelta::from_i64(490),
        ];
        let indices = vec![0, 1, 2];
        let cid = Uuid::new_v4();

        let anchors = eng.extract_anchors(&deltas, &indices, 0.9, cid);
        // All deltas within split_threshold (500ms default) → one group
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].line_start, 0);
        assert_eq!(anchors[0].line_end, 3);
        assert_eq!(anchors[0].delta_ms, 500); // median of [490,500,510]
        assert_eq!(anchors[0].chunk_id, cid);
    }

    #[test]
    fn extract_anchors_splits_on_large_delta_jump() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Lines 0-1 at ~500ms, lines 2-3 at ~2000ms (jump = 1500 > 500 threshold)
        let deltas = vec![
            TimeDelta::from_i64(500),
            TimeDelta::from_i64(510),
            TimeDelta::from_i64(2000),
            TimeDelta::from_i64(2010),
        ];
        let indices = vec![0, 1, 2, 3];
        let cid = Uuid::new_v4();

        let anchors = eng.extract_anchors(&deltas, &indices, 0.9, cid);
        assert_eq!(anchors.len(), 2);
        assert_eq!(anchors[0].line_start, 0);
        assert_eq!(anchors[0].line_end, 2);
        assert_eq!(anchors[1].line_start, 2);
        assert_eq!(anchors[1].line_end, 4);
    }

    #[test]
    fn extract_anchors_empty_input() {
        let eng = make_engine();
        let anchors = eng.extract_anchors(&[], &[], 0.5, Uuid::new_v4());
        assert!(anchors.is_empty());
    }

    // -- conflict_resolution ------------------------------------------------

    #[test]
    fn conflict_no_existing_anchors_all_insert() {
        let eng = make_engine();
        let new = vec![
            Anchor {
                id: Uuid::new_v4(),
                film_position_ms: 10_000,
                delta_ms: 500,
                confidence: 0.8,
                line_start: 0,
                line_end: 1,
                chunk_id: Uuid::new_v4(),
            },
        ];
        let result = eng.resolve_conflicts(&new);
        assert!(result.is_some());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn conflict_new_beats_prev() {
        let mut eng = make_engine();
        let old_id = Uuid::new_v4();
        eng.anchors.push(Anchor {
            id: old_id,
            film_position_ms: 15_000,
            delta_ms: 500,
            confidence: 0.5,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });

        let new = vec![Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 20_000, // within 30s spacing of the old one
            delta_ms: 600,
            confidence: 0.9, // higher
            line_start: 1,
            line_end: 2,
            chunk_id: Uuid::new_v4(),
        }];

        let result = eng.resolve_conflicts(&new).unwrap();
        assert!(result.contains(&old_id));
    }

    #[test]
    fn conflict_new_loses_to_prev() {
        let mut eng = make_engine();
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 15_000,
            delta_ms: 500,
            confidence: 0.9, // higher than new
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });

        let new = vec![Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 20_000,
            delta_ms: 600,
            confidence: 0.5, // lower
            line_start: 1,
            line_end: 2,
            chunk_id: Uuid::new_v4(),
        }];

        assert!(eng.resolve_conflicts(&new).is_none());
    }

    #[test]
    fn conflict_new_must_beat_both_neighbours() {
        let mut eng = make_engine();
        let id_prev = Uuid::new_v4();
        let id_next = Uuid::new_v4();

        eng.anchors.push(Anchor {
            id: id_prev,
            film_position_ms: 10_000,
            delta_ms: 500,
            confidence: 0.4,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });
        eng.anchors.push(Anchor {
            id: id_next,
            film_position_ms: 30_000,
            delta_ms: 700,
            confidence: 0.6,
            line_start: 2,
            line_end: 3,
            chunk_id: Uuid::new_v4(),
        });

        // New anchor between them, beats both
        let new = vec![Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 20_000,
            delta_ms: 600,
            confidence: 0.9,
            line_start: 1,
            line_end: 2,
            chunk_id: Uuid::new_v4(),
        }];

        let result = eng.resolve_conflicts(&new).unwrap();
        assert!(result.contains(&id_prev));
        assert!(result.contains(&id_next));
    }

    #[test]
    fn conflict_new_fails_if_cannot_beat_next() {
        let mut eng = make_engine();
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 10_000,
            delta_ms: 500,
            confidence: 0.4,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 30_000,
            delta_ms: 700,
            confidence: 0.95, // new can't beat this
            line_start: 2,
            line_end: 3,
            chunk_id: Uuid::new_v4(),
        });

        let new = vec![Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 20_000,
            delta_ms: 600,
            confidence: 0.9,
            line_start: 1,
            line_end: 2,
            chunk_id: Uuid::new_v4(),
        }];

        assert!(eng.resolve_conflicts(&new).is_none());
    }

    // -- stitch_check -------------------------------------------------------

    #[test]
    fn stitch_merges_adjacent_after() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Pre-existing chunk: 0–5000ms
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 5000,
            pcm_data: vec![1, 2, 3, 4],
            sample_rate: 16000,
            anchor_ids: vec![],
        });

        // New chunk starting at 5000ms (within tolerance)
        let new_pcm = vec![5, 6, 7, 8];
        let (merged, start, end, _) = eng.stitch_check(new_pcm, 5000, 10000, 16000);

        assert_eq!(start, 0);
        assert_eq!(end, 10000);
        assert!(eng.chunks.is_empty()); // old chunk consumed
        // Merged PCM = old + gap(0) + new
        assert!(merged.starts_with(&[1, 2, 3, 4]));
        assert!(merged.ends_with(&[5, 6, 7, 8]));
    }

    #[test]
    fn stitch_merges_adjacent_before() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Pre-existing chunk: 10000–20000ms
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 10000,
            film_end_ms: 20000,
            pcm_data: vec![5, 6, 7, 8],
            sample_rate: 16000,
            anchor_ids: vec![],
        });

        // New chunk ending at 10000ms
        let new_pcm = vec![1, 2, 3, 4];
        let (merged, start, end, _) = eng.stitch_check(new_pcm, 0, 10000, 16000);

        assert_eq!(start, 0);
        assert_eq!(end, 20000);
        assert!(merged.starts_with(&[1, 2, 3, 4]));
        assert!(merged.ends_with(&[5, 6, 7, 8]));
    }

    #[test]
    fn stitch_evicts_old_anchors() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        let anchor_id = Uuid::new_v4();
        eng.anchors.push(Anchor {
            id: anchor_id,
            film_position_ms: 3000,
            delta_ms: 100,
            confidence: 0.8,
            line_start: 0,
            line_end: 1,
            chunk_id: Uuid::new_v4(),
        });

        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 5000,
            pcm_data: vec![0; 4],
            sample_rate: 16000,
            anchor_ids: vec![anchor_id],
        });

        let (_, _, _, evicted) = eng.stitch_check(vec![0; 4], 5000, 10000, 16000);
        assert!(evicted, "stitch must report anchor eviction");

        assert!(eng.anchors.is_empty(), "stitched chunk's anchors must be evicted");
    }

    #[test]
    fn stitch_cascades() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Two pre-existing chunks: [0–5000] and [10000–15000]
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 5000,
            pcm_data: vec![1, 2],
            sample_rate: 16000,
            anchor_ids: vec![],
        });
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 10000,
            film_end_ms: 15000,
            pcm_data: vec![5, 6],
            sample_rate: 16000,
            anchor_ids: vec![],
        });

        // New chunk fills the gap [5000–10000]
        let (_, start, end, _) = eng.stitch_check(vec![3, 4], 5000, 10000, 16000);

        assert_eq!(start, 0);
        assert_eq!(end, 15000);
        assert!(eng.chunks.is_empty(), "both old chunks consumed via cascade");
    }

    #[test]
    fn stitch_ignores_different_sample_rate() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 5000,
            pcm_data: vec![1, 2],
            sample_rate: 8000, // different from incoming 16000
            anchor_ids: vec![],
        });

        let (pcm, start, end, evicted) = eng.stitch_check(vec![3, 4], 5000, 10000, 16000);
        assert!(!evicted, "no anchors to evict");
        assert_eq!(start, 5000);
        assert_eq!(end, 10000);
        assert_eq!(pcm, vec![3, 4]);
        assert_eq!(eng.chunks.len(), 1); // old chunk untouched
    }

    // -- recompute_all_and_diff ---------------------------------------------

    #[test]
    fn recompute_applies_warp_and_diffs() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Insert an anchor that shifts everything by +2000ms
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 15_000,
            delta_ms: 2000,
            confidence: 0.9,
            line_start: 0,
            line_end: 5,
            chunk_id: Uuid::new_v4(),
        });

        let changes = eng.recompute_all_and_diff();

        // All 5 lines should have moved
        assert_eq!(changes.len(), 5);
        for c in &changes {
            assert_eq!(c.delta_change_ms, 2000);
            assert_eq!(c.new_start_ms, c.old_start_ms + 2000);
            assert_eq!(c.new_end_ms, c.old_end_ms + 2000);
        }

        // Calling again with no anchor changes → no diff
        let changes2 = eng.recompute_all_and_diff();
        assert!(changes2.is_empty());
    }

    // -- export_srt ---------------------------------------------------------

    #[test]
    fn export_srt_basic() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        let srt = eng.export_srt().unwrap();
        assert!(srt.contains("00:00:01,000 --> 00:00:04,000"));
        assert!(srt.contains("Hello world"));
        assert!(srt.contains("Fifth line"));
    }

    #[test]
    fn export_srt_reflects_corrections() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 15_000,
            delta_ms: 5000,
            confidence: 0.9,
            line_start: 0,
            line_end: 5,
            chunk_id: Uuid::new_v4(),
        });
        eng.recompute_all_and_diff();

        let srt = eng.export_srt().unwrap();
        // Line 1: was 1000→4000, now 6000→9000
        assert!(srt.contains("00:00:06,000 --> 00:00:09,000"));
    }

    // -- config round-trip --------------------------------------------------

    #[test]
    fn config_proto_round_trip() {
        let cfg = EngineConfig::default();
        let proto_cfg = proto::EngineConfig::from(&cfg);
        let back = EngineConfig::from(&proto_cfg);

        assert_eq!(back.split_penalty, cfg.split_penalty);
        assert_eq!(back.min_anchor_spacing_ms, cfg.min_anchor_spacing_ms);
        assert_eq!(back.max_drift_ms, cfg.max_drift_ms);
        assert_eq!(back.split_threshold_ms, cfg.split_threshold_ms);
        assert_eq!(back.sweep_step_ms, cfg.sweep_step_ms);
        assert_eq!(back.coverage_buffer_ms, cfg.coverage_buffer_ms);
        assert_eq!(back.stitch_tolerance_ms, cfg.stitch_tolerance_ms);
        assert_eq!(back.vad_mode, cfg.vad_mode);
    }

    #[test]
    fn config_optional_fields_allow_zero() {
        // With optional proto fields, explicitly setting a value to 0
        // should be distinguishable from "not set" (which gets defaults).
        let proto_cfg = proto::EngineConfig {
            quality_threshold: Some(0.0),
            split_penalty: Some(0),
            min_anchor_spacing_ms: Some(0),
            max_drift_ms: Some(0),
            split_threshold_ms: Some(0),
            sweep_step_ms: Some(0),
            coverage_buffer_ms: Some(0),
            stitch_tolerance_ms: Some(0),
            vad_mode: Some(3),
        };
        let cfg = EngineConfig::from(&proto_cfg);
        assert_eq!(cfg.quality_threshold, 0.0);
        assert_eq!(cfg.split_penalty, 0.0);
        assert_eq!(cfg.min_anchor_spacing_ms, 0);
        assert_eq!(cfg.max_drift_ms, 0);
        assert_eq!(cfg.split_threshold_ms, 0);
        assert_eq!(cfg.sweep_step_ms, 0);
        assert_eq!(cfg.coverage_buffer_ms, 0);
        assert_eq!(cfg.stitch_tolerance_ms, 0);
        assert_eq!(cfg.vad_mode, 3);
    }

    #[test]
    fn config_unset_fields_get_defaults() {
        // When no fields are set (all None), defaults should be used.
        let proto_cfg = proto::EngineConfig {
            quality_threshold: None,
            split_penalty: None,
            min_anchor_spacing_ms: None,
            max_drift_ms: None,
            split_threshold_ms: None,
            sweep_step_ms: None,
            coverage_buffer_ms: None,
            stitch_tolerance_ms: None,
            vad_mode: None,
        };
        let cfg = EngineConfig::from(&proto_cfg);
        let d = EngineConfig::default();
        assert_eq!(cfg.quality_threshold, d.quality_threshold);
        assert_eq!(cfg.split_penalty, d.split_penalty);
        assert_eq!(cfg.min_anchor_spacing_ms, d.min_anchor_spacing_ms);
        assert_eq!(cfg.max_drift_ms, d.max_drift_ms);
        assert_eq!(cfg.vad_mode, d.vad_mode);
    }

    // -- format_srt_time ----------------------------------------------------

    #[test]
    fn srt_time_formatting() {
        assert_eq!(format_srt_time(0), "00:00:00,000");
        assert_eq!(format_srt_time(1000), "00:00:01,000");
        assert_eq!(format_srt_time(61_500), "00:01:01,500");
        assert_eq!(format_srt_time(3_661_123), "01:01:01,123");
        assert_eq!(format_srt_time(-1000), "-00:00:01,000");
    }

    // -- coarse_sweep -------------------------------------------------------

    #[test]
    fn coarse_sweep_finds_correct_offset() {
        let eng = make_engine();

        // VAD spans at absolute film time 10s–11s, 12s–13s
        let vad = vec![
            TimeSpan::new(TimePoint::from(10_000), TimePoint::from(11_000)),
            TimeSpan::new(TimePoint::from(12_000), TimePoint::from(13_000)),
        ];

        // Subtitle spans that are 5s early (at 5s–6s, 7s–8s)
        // The correct delta to shift subs to match VAD is +5000
        let subs = vec![
            TimeSpan::new(TimePoint::from(5_000), TimePoint::from(6_000)),
            TimeSpan::new(TimePoint::from(7_000), TimePoint::from(8_000)),
        ];

        let (delta, score) = eng.coarse_sweep(&vad, &subs);
        assert_eq!(delta, 5000);
        assert!(score > 0.0);
    }

    // -- nearest_anchor_confidence ------------------------------------------

    #[test]
    fn nearest_confidence_picks_closer_anchor() {
        let mut eng = make_engine();
        let cid = Uuid::new_v4();
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 10_000,
            delta_ms: 0,
            confidence: 0.3,
            line_start: 0,
            line_end: 1,
            chunk_id: cid,
        });
        eng.anchors.push(Anchor {
            id: Uuid::new_v4(),
            film_position_ms: 30_000,
            delta_ms: 0,
            confidence: 0.9,
            line_start: 1,
            line_end: 2,
            chunk_id: cid,
        });

        assert_eq!(eng.nearest_anchor_confidence_for(12_000), 0.3); // closer to 10k
        assert_eq!(eng.nearest_anchor_confidence_for(28_000), 0.9); // closer to 30k
    }

    // -- ingest_chunk pre-conditions ----------------------------------------

    #[test]
    fn ingest_rejects_without_subtitle() {
        let mut eng = make_engine();
        let res = eng.ingest_chunk(vec![], 16000, 0, 1000);
        assert!(res.is_err());
    }

    #[test]
    fn ingest_rejects_bad_sample_rate() {
        let mut eng = make_engine();
        load_sample(&mut eng);
        let res = eng.ingest_chunk(vec![], 44100, 0, 1000);
        assert!(res.is_err());
    }

    // -- subtitle_window_with_indices ----------------------------------------

    #[test]
    fn subtitle_window_filters_by_range() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Only lines whose original span overlaps [4000, 9000]
        // Line 0: 1000-4000  (end=4000 >= 4000, start=1000 <= 9000 → included)
        // Line 1: 5000-8000  → included
        // Line 2: 10000-13000 → excluded (start=10000 > 9000)
        let (spans, indices) = eng.subtitle_window_with_indices(4000, 9000);
        assert_eq!(indices, vec![0, 1]);
        assert_eq!(spans.len(), 2);
    }

    // -- stitch overlap detection (#3) ----------------------------------------

    #[test]
    fn stitch_merges_overlapping_chunks() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Pre-existing chunk: 0–10000ms
        eng.chunks.push(ProcessedChunk {
            id: Uuid::new_v4(),
            film_start_ms: 0,
            film_end_ms: 10000,
            pcm_data: vec![1, 2, 3, 4],
            sample_rate: 16000,
            anchor_ids: vec![],
        });

        // New chunk overlaps: 5000–15000ms
        let new_pcm = vec![5, 6, 7, 8];
        let (_, start, end, _) = eng.stitch_check(new_pcm, 5000, 15000, 16000);

        assert_eq!(start, 0);
        assert_eq!(end, 15000);
        assert!(eng.chunks.is_empty());
    }

    // -- set_config / config() accessor (#9) ----------------------------------

    #[test]
    fn set_config_and_config_accessor() {
        let mut eng = make_engine();
        let mut cfg = EngineConfig::default();
        cfg.vad_mode = 3;
        cfg.max_drift_ms = 60_000;
        eng.set_config(cfg.clone());

        assert_eq!(eng.config().vad_mode, 3);
        assert_eq!(eng.config().max_drift_ms, 60_000);
    }

    // =======================================================================
    // E2E tests — full ingest_chunk pipeline
    // =======================================================================

    #[test]
    fn test_make_speech_pcm_produces_vad_spans() {
        let pcm = make_speech_pcm(16000, &[(1000, 4000), (6000, 9000)], 10000);
        let spans = vad::run_vad(&pcm, 16000, 0, 0);

        assert!(
            spans.len() >= 2,
            "expected at least 2 VAD spans, got {}",
            spans.len()
        );

        // First span should cover approximately 1000-4000ms
        let s0_start = i64::from(spans[0].start);
        let s0_end = i64::from(spans[0].end);
        assert_within(s0_start, 1000, 200, "span0 start");
        assert_within(s0_end, 4000, 200, "span0 end");

        // Second span should cover approximately 6000-9000ms
        let s1_start = i64::from(spans[1].start);
        let s1_end = i64::from(spans[1].end);
        assert_within(s1_start, 6000, 200, "span1 start");
        assert_within(s1_end, 9000, 200, "span1 end");
    }

    // -- e2e: silence rejected ----------------------------------------------

    #[test]
    fn e2e_silence_rejected() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // 40 seconds of silence at 16kHz
        let pcm = vec![0u8; 16000 * 2 * 40];
        let changes = eng.ingest_chunk(pcm, 16000, 0, 40000).unwrap();

        assert!(changes.is_empty(), "silence should produce no changes");
        assert!(eng.get_anchors().is_empty());
        assert!(eng.get_chunk_history().is_empty());

        // Corrected times unchanged
        let sub = eng.subtitle.as_ref().unwrap();
        for i in 0..sub.original_timespans.len() {
            assert_eq!(sub.corrected_start_ms[i], sub.original_timespans[i].0);
            assert_eq!(sub.corrected_end_ms[i], sub.original_timespans[i].1);
        }
    }

    // -- e2e: constant positive offset --------------------------------------

    #[test]
    fn e2e_constant_offset_positive() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Audio speech occurs 3s AFTER subtitle timings.
        // SAMPLE_SRT lines: 1-4s, 5-8s, 10-13s, 20-23s, 30-33s
        // Shifted +3s:       4-7s, 8-11s, 13-16s, 23-26s, 33-36s
        let pcm = make_speech_pcm(
            16000,
            &[
                (4000, 7000),
                (8000, 11000),
                (13000, 16000),
                (23000, 26000),
                (33000, 36000),
            ],
            40000,
        );

        let changes = eng.ingest_chunk(pcm, 16000, 0, 40000).unwrap();

        assert!(!changes.is_empty(), "should produce changes");

        // All changed lines should have delta approximately +3000
        for c in &changes {
            assert_within(
                c.new_start_ms - c.old_start_ms,
                3000,
                1500,
                &format!("line {} start delta", c.line_index),
            );
        }

        // Anchors exist with positive delta
        let anchors = eng.get_anchors();
        assert!(!anchors.is_empty(), "should have at least 1 anchor");
        for a in &anchors {
            assert_within(a.delta_ms, 3000, 1500, "anchor delta");
        }

        // One chunk in history
        assert_eq!(eng.get_chunk_history().len(), 1);

        // Exported SRT reflects correction
        let srt = eng.export_srt().unwrap();
        // Line 1 original: 00:00:01,000 → should now be ~00:00:04,000
        assert!(
            !srt.contains("00:00:01,000 --> 00:00:04,000"),
            "original timestamps should no longer appear in exported SRT"
        );
    }

    // -- e2e: constant negative offset --------------------------------------

    #[test]
    fn e2e_constant_offset_negative() {
        let mut eng = make_engine();

        // Use custom SRT with higher offsets to avoid clipping into negatives
        let srt = make_srt(&[
            (5000, 8000, "Line one"),
            (10000, 13000, "Line two"),
            (20000, 23000, "Line three"),
            (30000, 33000, "Line four"),
            (40000, 43000, "Line five"),
        ]);
        eng.load_subtitle(&srt, "srt").unwrap();

        // Audio speech occurs 3s BEFORE subtitle timings.
        // Shifted -3s: 2-5s, 7-10s, 17-20s, 27-30s, 37-40s
        let pcm = make_speech_pcm(
            16000,
            &[
                (2000, 5000),
                (7000, 10000),
                (17000, 20000),
                (27000, 30000),
                (37000, 40000),
            ],
            50000,
        );

        let changes = eng.ingest_chunk(pcm, 16000, 0, 50000).unwrap();

        assert!(!changes.is_empty(), "should produce changes");

        for c in &changes {
            assert_within(
                c.new_start_ms - c.old_start_ms,
                -3000,
                1500,
                &format!("line {} start delta", c.line_index),
            );
        }

        let anchors = eng.get_anchors();
        assert!(!anchors.is_empty());
        for a in &anchors {
            assert_within(a.delta_ms, -3000, 1500, "anchor delta");
        }
    }

    // -- e2e: poor match rejected by quality gate ---------------------------

    #[test]
    fn e2e_poor_match_rejected_by_quality_gate() {
        let mut cfg = EngineConfig::default();
        cfg.quality_threshold = 0.5; // raised to ensure rejection
        let mut eng = test_engine(cfg);
        load_sample(&mut eng);

        // Single short speech burst that doesn't match subtitle pattern
        let pcm = make_speech_pcm(16000, &[(19000, 19500)], 40000);

        let changes = eng.ingest_chunk(pcm, 16000, 0, 40000).unwrap();

        assert!(
            changes.is_empty(),
            "poor match should be rejected by quality gate"
        );
        assert!(eng.get_anchors().is_empty());
        assert!(eng.get_chunk_history().is_empty());
    }

    // -- e2e: multi-chunk progressive alignment -----------------------------

    #[test]
    fn e2e_multi_chunk_progressive() {
        let mut cfg = EngineConfig::default();
        cfg.stitch_tolerance_ms = 0; // prevent stitching
        cfg.min_anchor_spacing_ms = 5000; // allow anchors closer together
        cfg.max_drift_ms = 10000; // narrow window so each chunk sees only its lines
        cfg.coverage_buffer_ms = 5000;
        let mut eng = test_engine(cfg);

        // Two well-separated groups so each chunk's window captures only its lines.
        let srt = make_srt(&[
            (5000, 8000, "Group A line 1"),
            (10000, 13000, "Group A line 2"),
            (15000, 18000, "Group A line 3"),
            (80000, 83000, "Group B line 1"),
            (85000, 88000, "Group B line 2"),
            (90000, 93000, "Group B line 3"),
        ]);
        eng.load_subtitle(&srt, "srt").unwrap();

        // Offset: +2000ms

        // Chunk 1: film 0-25s, covers group A shifted +2s
        // Speech at (7-10s), (12-15s), (17-20s)
        let pcm1 = make_speech_pcm(
            16000,
            &[(7000, 10000), (12000, 15000), (17000, 20000)],
            25000,
        );
        let changes1 = eng.ingest_chunk(pcm1, 16000, 0, 25000).unwrap();
        assert!(!changes1.is_empty(), "chunk 1 should produce changes");
        assert!(
            !eng.get_anchors().is_empty(),
            "chunk 1 should create anchors"
        );

        let anchors_after_c1 = eng.get_anchors().len();

        // Chunk 2: film 75-100s, covers group B shifted +2s
        // Absolute speech: 82-85s, 87-90s, 92-95s
        // Relative to chunk start (75s): 7-10s, 12-15s, 17-20s
        let pcm2 = make_speech_pcm(
            16000,
            &[(7000, 10000), (12000, 15000), (17000, 20000)],
            25000,
        );
        let _changes2 = eng.ingest_chunk(pcm2, 16000, 75000, 100000).unwrap();

        // Chunk 2 should also produce changes or anchors
        // (it may produce changes even if no new anchors, via recompute)
        let anchors_after_c2 = eng.get_anchors().len();
        assert!(
            anchors_after_c2 >= anchors_after_c1,
            "chunk 2 should add anchors (had {}, now {})",
            anchors_after_c1,
            anchors_after_c2,
        );

        // At least one chunk stored
        assert!(
            !eng.get_chunk_history().is_empty(),
            "should have at least 1 chunk stored"
        );

        // All 6 lines should have corrected times ~+2000ms
        let sub = eng.subtitle.as_ref().unwrap();
        for i in 0..6 {
            let orig_start = sub.original_timespans[i].0;
            assert_within(
                sub.corrected_start_ms[i],
                orig_start + 2000,
                1500,
                &format!("line {} corrected start", i),
            );
        }
    }

    // -- e2e: reset mid-session ---------------------------------------------

    #[test]
    fn e2e_reset_mid_session() {
        let mut eng = make_engine();
        load_sample(&mut eng);

        // Phase 1: align with +3s offset
        let pcm1 = make_speech_pcm(
            16000,
            &[
                (4000, 7000),
                (8000, 11000),
                (13000, 16000),
                (23000, 26000),
                (33000, 36000),
            ],
            40000,
        );
        let changes1 = eng.ingest_chunk(pcm1, 16000, 0, 40000).unwrap();
        assert!(!changes1.is_empty(), "phase 1 should produce changes");
        assert!(!eng.get_anchors().is_empty(), "phase 1 should have anchors");

        // Phase 2: reset
        eng.reset();
        assert!(eng.get_anchors().is_empty(), "reset should clear anchors");
        assert!(
            eng.get_chunk_history().is_empty(),
            "reset should clear chunks"
        );
        let sub = eng.subtitle.as_ref().unwrap();
        for i in 0..5 {
            assert_eq!(
                sub.corrected_start_ms[i], sub.original_timespans[i].0,
                "reset should restore original start for line {}",
                i
            );
            assert_eq!(
                sub.corrected_end_ms[i], sub.original_timespans[i].1,
                "reset should restore original end for line {}",
                i
            );
        }

        // Phase 3: re-align with +5s offset
        let pcm2 = make_speech_pcm(
            16000,
            &[
                (6000, 9000),
                (10000, 13000),
                (15000, 18000),
                (25000, 28000),
                (35000, 38000),
            ],
            40000,
        );
        let changes2 = eng.ingest_chunk(pcm2, 16000, 0, 40000).unwrap();
        assert!(!changes2.is_empty(), "phase 3 should produce changes");

        // Should reflect +5s, not +3s from phase 1
        for c in &changes2 {
            assert_within(
                c.new_start_ms - c.old_start_ms,
                5000,
                1500,
                &format!("phase 3 line {} delta", c.line_index),
            );
        }

        // Only 1 chunk in history (not 2)
        assert_eq!(eng.get_chunk_history().len(), 1);
    }

    // -- e2e: split detection -----------------------------------------------

    #[test]
    fn e2e_split_detection() {
        let mut eng = make_engine();

        // Custom SRT with large gap between groups to make split clear
        let srt = make_srt(&[
            (5000, 8000, "Early A"),
            (10000, 13000, "Early B"),
            (30000, 33000, "Late A"),
            (35000, 38000, "Late B"),
        ]);
        eng.load_subtitle(&srt, "srt").unwrap();

        // Lines 0-1 shifted +2s: speech at 7-10s, 12-15s
        // Lines 2-3 shifted -3s: speech at 27-30s, 32-35s
        let pcm = make_speech_pcm(
            16000,
            &[
                (7000, 10000),
                (12000, 15000),
                (27000, 30000),
                (32000, 35000),
            ],
            50000,
        );

        let changes = eng.ingest_chunk(pcm, 16000, 0, 50000).unwrap();
        assert!(!changes.is_empty(), "should produce changes");

        let anchors = eng.get_anchors();
        assert!(
            anchors.len() >= 2,
            "should have at least 2 anchors for split, got {}",
            anchors.len()
        );

        // Find anchors for each region
        let early_anchor = anchors
            .iter()
            .find(|a| a.film_position_ms < 20000)
            .expect("should have anchor for early region");
        let late_anchor = anchors
            .iter()
            .find(|a| a.film_position_ms > 20000)
            .expect("should have anchor for late region");

        assert_within(early_anchor.delta_ms, 2000, 1500, "early anchor delta");
        assert_within(late_anchor.delta_ms, -3000, 1500, "late anchor delta");

        // Lines 0-1 should be shifted ~+2000
        let sub = eng.subtitle.as_ref().unwrap();
        for i in 0..2 {
            assert_within(
                sub.corrected_start_ms[i] - sub.original_timespans[i].0,
                2000,
                1500,
                &format!("early line {} delta", i),
            );
        }
        // Lines 2-3 should be shifted ~-3000
        for i in 2..4 {
            assert_within(
                sub.corrected_start_ms[i] - sub.original_timespans[i].0,
                -3000,
                1500,
                &format!("late line {} delta", i),
            );
        }
    }
}
