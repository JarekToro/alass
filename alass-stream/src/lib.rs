//! Streaming subtitle alignment.
//!
//! The caller loads a full subtitle file once, then feeds timestamped audio
//! chunks (as PCM or as pre-computed VAD spans) as they arrive. On each
//! ingest the aligner runs `alass_core::align` once against the complete
//! accumulated VAD evidence, returning per-line changes vs. the previous
//! alignment state.
//!
//! The audio stream may be sparse: a handful of minutes of coverage against
//! a two-hour subtitle file is fine, and gaps in the middle of the stream
//! are fine. Alass handles constant shifts and per-group splits on whatever
//! evidence it has; as more evidence arrives the alignment refines.

use std::cmp::{max, min};

use alass_core::{align, standard_scoring, NoProgressHandler, TimeDelta, TimePoint, TimeSpan};
use subparse::timetypes::{TimeDelta as SubDelta, TimePoint as SubPoint, TimeSpan as SubSpan};
use subparse::{parse_bytes, SubtitleEntry, SubtitleFile, SubtitleFormat};

mod vad;
pub use vad::pcm_to_spans;

/// Tunables for the streaming aligner.
#[derive(Debug, Clone)]
pub struct Config {
    /// Alass split penalty (0-1000, typical 4-20, optimum around 7).
    pub split_penalty: f64,
    /// Alass speed-optimization parameter. `None` for maximum accuracy.
    pub speed_optimization: Option<f64>,
    /// WebRTC VAD aggressiveness (0 = Quality, 3 = VeryAggressive).
    pub vad_mode: u8,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            split_penalty: 7.0,
            // Mild: ~2-3× faster than None with no loss of accuracy on the
            // scenarios we care about. Higher values quantize the DP and can
            // pick tied shifts arbitrarily when VAD is sparse.
            speed_optimization: Some(1.0),
            vad_mode: 0,
        }
    }
}

/// A subtitle line change produced by an ingest call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineChange {
    pub line_index: usize,
    pub old_start_ms: i64,
    pub old_end_ms: i64,
    pub new_start_ms: i64,
    pub new_end_ms: i64,
    /// Change in this line's applied delta since the previous ingest
    /// (`new_delta_ms - old_delta_ms`). Not a re-statement of start/end deltas.
    pub delta_change_ms: i64,
}

/// A corrected subtitle line for read-only inspection.
#[derive(Debug, Clone)]
pub struct CorrectedLine {
    pub index: usize,
    pub original_start_ms: i64,
    pub original_end_ms: i64,
    pub start_ms: i64,
    pub end_ms: i64,
    pub delta_ms: i64,
    pub text: Option<String>,
}

#[derive(Debug)]
pub enum Error {
    UnknownFormat,
    Parse(String),
    InvalidSampleRate(u32),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::UnknownFormat => write!(f, "unknown subtitle format"),
            Error::Parse(msg) => write!(f, "subtitle parse error: {msg}"),
            Error::InvalidSampleRate(r) => write!(f, "unsupported sample rate {r} (want 8000 or 16000)"),
        }
    }
}

impl std::error::Error for Error {}

/// Streaming subtitle aligner.
///
/// Lifecycle: `new()` → `load_subtitle(...)` → repeatedly `ingest_*` →
/// `export()` / `corrected_lines()` whenever needed. `reset()` clears all
/// VAD evidence and applied deltas (but keeps the loaded subtitle).
pub struct StreamingAligner {
    config: Config,
    loaded: Option<Loaded>,
    vad: Vec<TimeSpan>,
    current_deltas: Vec<TimeDelta>,
}

struct Loaded {
    format: SubtitleFormat,
    file: SubtitleFile,
    original_spans: Vec<TimeSpan>,
    original_entries: Vec<SubtitleEntry>,
}

impl StreamingAligner {
    pub fn new() -> Self {
        Self::with_config(Config::default())
    }

    pub fn with_config(config: Config) -> Self {
        StreamingAligner {
            config,
            loaded: None,
            vad: Vec::new(),
            current_deltas: Vec::new(),
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn set_config(&mut self, config: Config) {
        self.config = config;
    }

    /// Load or replace the subtitle file. Clears all VAD evidence and
    /// applied deltas so the next ingest starts fresh.
    ///
    /// `format` is hinted explicitly; callers that don't know the format
    /// up-front should use `load_subtitle_bytes`.
    pub fn load_subtitle(&mut self, content: &str, format: SubtitleFormat) -> Result<usize, Error> {
        let file = parse_bytes(format, content.as_bytes(), None, 25.0)
            .map_err(|e| Error::Parse(e.to_string()))?;
        self.install(file, format)
    }

    /// Like `load_subtitle`, but detects format from the bytes (and optional
    /// filename-derived hint like `"srt"` / `"ass"`) and handles encoding.
    pub fn load_subtitle_bytes(
        &mut self,
        bytes: &[u8],
        extension_hint: Option<&str>,
    ) -> Result<usize, Error> {
        let format = match extension_hint {
            Some(ext) => subparse::get_subtitle_format(Some(std::ffi::OsStr::new(ext)), bytes)
                .ok_or(Error::UnknownFormat)?,
            None => subparse::get_subtitle_format(None, bytes).ok_or(Error::UnknownFormat)?,
        };
        let file = parse_bytes(format, bytes, None, 25.0)
            .map_err(|e| Error::Parse(e.to_string()))?;
        self.install(file, format)
    }

    fn install(&mut self, file: SubtitleFile, format: SubtitleFormat) -> Result<usize, Error> {
        let entries = file
            .get_subtitle_entries()
            .map_err(|e| Error::Parse(e.to_string()))?;

        let original_spans: Vec<TimeSpan> = entries
            .iter()
            .map(|e| sub_to_alg_span(e.timespan))
            .collect();

        let count = original_spans.len();
        self.current_deltas = vec![TimeDelta::zero(); count];
        self.vad.clear();
        self.loaded = Some(Loaded {
            format,
            file,
            original_spans,
            original_entries: entries,
        });
        Ok(count)
    }

    /// Ingest a chunk of raw 16-bit signed little-endian mono PCM whose first
    /// sample corresponds to `film_start_ms`. Runs WebRTC VAD, merges the
    /// resulting speech spans into accumulated evidence, and re-runs
    /// `alass_core::align`. Returns one `LineChange` per line whose applied
    /// delta moved.
    pub fn ingest_pcm(
        &mut self,
        pcm: &[u8],
        sample_rate: u32,
        film_start_ms: i64,
    ) -> Result<Vec<LineChange>, Error> {
        if sample_rate != 8000 && sample_rate != 16000 {
            return Err(Error::InvalidSampleRate(sample_rate));
        }
        let mode = vad::mode_from_u8(self.config.vad_mode);
        let spans = pcm_to_spans(pcm, sample_rate, film_start_ms, mode);
        Ok(self.ingest_vad_spans(&spans))
    }

    /// Ingest pre-computed VAD speech spans (absolute film-time, ms-metric).
    /// Useful for tests and for callers that run their own VAD or speech-
    /// activity detector. Spans may be in any order and may overlap with
    /// previously-ingested evidence; they will be merged.
    pub fn ingest_vad_spans(&mut self, new_spans: &[TimeSpan]) -> Vec<LineChange> {
        if !new_spans.is_empty() {
            self.vad.extend_from_slice(new_spans);
            self.vad = merge_spans(std::mem::take(&mut self.vad));
        }
        self.recompute_alignment()
    }

    fn recompute_alignment(&mut self) -> Vec<LineChange> {
        let loaded = match &self.loaded {
            Some(l) => l,
            None => return Vec::new(),
        };
        if self.vad.is_empty() || loaded.original_spans.is_empty() {
            return Vec::new();
        }

        let (new_deltas, _score) = align(
            &self.vad,
            &loaded.original_spans,
            self.config.split_penalty,
            self.config.speed_optimization,
            standard_scoring,
            NoProgressHandler,
        );

        let mut changes = Vec::new();
        for (i, (&new_delta, &old_delta)) in new_deltas.iter().zip(self.current_deltas.iter()).enumerate() {
            if new_delta == old_delta {
                continue;
            }
            let orig = loaded.original_spans[i];
            let old_start = (orig.start + old_delta).as_i64();
            let old_end = (orig.end + old_delta).as_i64();
            let new_start = (orig.start + new_delta).as_i64();
            let new_end = (orig.end + new_delta).as_i64();
            changes.push(LineChange {
                line_index: i,
                old_start_ms: old_start,
                old_end_ms: old_end,
                new_start_ms: new_start,
                new_end_ms: new_end,
                delta_change_ms: new_delta.as_i64() - old_delta.as_i64(),
            });
        }
        self.current_deltas = new_deltas;
        changes
    }

    /// Clear all VAD evidence and applied deltas, keeping the loaded
    /// subtitle. After reset, the next ingest starts alignment from scratch.
    pub fn reset(&mut self) {
        self.vad.clear();
        if let Some(loaded) = &self.loaded {
            self.current_deltas = vec![TimeDelta::zero(); loaded.original_spans.len()];
        } else {
            self.current_deltas.clear();
        }
    }

    /// Number of subtitle lines currently loaded.
    pub fn line_count(&self) -> usize {
        self.loaded.as_ref().map(|l| l.original_spans.len()).unwrap_or(0)
    }

    /// Read-only view of the accumulated VAD evidence (merged, sorted).
    pub fn vad_evidence(&self) -> &[TimeSpan] {
        &self.vad
    }

    /// Per-line current delta in ms. Zero for lines that have not been moved
    /// from their original timing.
    pub fn current_deltas_ms(&self) -> Vec<i64> {
        self.current_deltas.iter().map(|d| d.as_i64()).collect()
    }

    /// Snapshot the current corrected subtitle timeline.
    pub fn corrected_lines(&self) -> Vec<CorrectedLine> {
        let loaded = match &self.loaded {
            Some(l) => l,
            None => return Vec::new(),
        };
        loaded
            .original_spans
            .iter()
            .zip(self.current_deltas.iter())
            .zip(loaded.original_entries.iter())
            .enumerate()
            .map(|(i, ((orig, delta), entry))| CorrectedLine {
                index: i,
                original_start_ms: orig.start.as_i64(),
                original_end_ms: orig.end.as_i64(),
                start_ms: (orig.start + *delta).as_i64(),
                end_ms: (orig.end + *delta).as_i64(),
                delta_ms: delta.as_i64(),
                text: entry.line.clone(),
            })
            .collect()
    }

    /// Serialise the corrected subtitle back to its original format (as
    /// produced by subparse). Returns `None` if no subtitle is loaded.
    pub fn export(&self) -> Option<Vec<u8>> {
        let loaded = self.loaded.as_ref()?;
        let mut file = loaded.file.clone();
        let updated: Vec<SubtitleEntry> = loaded
            .original_entries
            .iter()
            .zip(self.current_deltas.iter())
            .map(|(entry, delta)| SubtitleEntry {
                timespan: shift_sub_span(entry.timespan, *delta),
                line: entry.line.clone(),
            })
            .collect();
        file.update_subtitle_entries(&updated).ok()?;
        file.to_data().ok()
    }

    /// Same as `export` but returns UTF-8 text. Returns `None` if the output
    /// isn't valid UTF-8 (shouldn't happen for srt/ass).
    pub fn export_text(&self) -> Option<String> {
        String::from_utf8(self.export()?).ok()
    }

    /// The loaded subtitle's detected format, if any.
    pub fn format(&self) -> Option<SubtitleFormat> {
        self.loaded.as_ref().map(|l| l.format)
    }
}

impl Default for StreamingAligner {
    fn default() -> Self {
        Self::new()
    }
}

fn sub_to_alg_span(ts: SubSpan) -> TimeSpan {
    let a = ts.start.msecs();
    let b = ts.end.msecs();
    TimeSpan::new(TimePoint::from(min(a, b)), TimePoint::from(max(a, b)))
}

fn shift_sub_span(ts: SubSpan, delta: TimeDelta) -> SubSpan {
    let d = SubDelta::from_msecs(delta.as_i64());
    let (s, e) = if ts.start <= ts.end { (ts.start, ts.end) } else { (ts.end, ts.start) };
    SubSpan::new(shift_point(s, d), shift_point(e, d))
}

fn shift_point(p: SubPoint, d: SubDelta) -> SubPoint {
    SubPoint::from_msecs(p.msecs() + d.msecs())
}

/// Merge a set of spans into a sorted, non-overlapping, non-adjacent list.
/// Spans with zero length are dropped; touching spans (end == next.start)
/// are coalesced.
fn merge_spans(mut spans: Vec<TimeSpan>) -> Vec<TimeSpan> {
    spans.retain(|s| s.end > s.start);
    spans.sort_by_key(|s| s.start);
    let mut out: Vec<TimeSpan> = Vec::with_capacity(spans.len());
    for s in spans {
        match out.last_mut() {
            Some(last) if s.start <= last.end => {
                if s.end > last.end {
                    *last = TimeSpan::new(last.start, s.end);
                }
            }
            _ => out.push(s),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(a: i64, b: i64) -> TimeSpan {
        TimeSpan::new(TimePoint::from(a), TimePoint::from(b))
    }

    #[test]
    fn merge_sorts_and_coalesces() {
        let merged = merge_spans(vec![ts(1000, 2000), ts(500, 1500), ts(3000, 4000), ts(4000, 4500)]);
        assert_eq!(merged, vec![ts(500, 2000), ts(3000, 4500)]);
    }

    #[test]
    fn merge_drops_zero_length() {
        let merged = merge_spans(vec![ts(100, 100), ts(200, 300)]);
        assert_eq!(merged, vec![ts(200, 300)]);
    }
}
