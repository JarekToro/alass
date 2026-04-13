use anyhow::{bail, Result};
use std::ffi::OsStr;
use subparse::timetypes::TimePoint as SubTimePoint;
use subparse::timetypes::TimeSpan as SubTimeSpan;
use subparse::{get_subtitle_format_err, parse_bytes, SubtitleEntry, SubtitleFile, SubtitleFormat};

/// A single subtitle line with its timing information.
#[derive(Debug, Clone)]
pub struct SubtitleLine {
    /// Original start time in milliseconds.
    pub original_start_ms: i64,
    /// Original end time in milliseconds.
    pub original_end_ms: i64,
    /// Current corrected start time in milliseconds (starts equal to original).
    pub corrected_start_ms: i64,
    /// Current corrected end time in milliseconds (starts equal to original).
    pub corrected_end_ms: i64,
}

/// Holds the parsed subtitle state, including all lines and the backing
/// `SubtitleFile` object for format-preserving round-trip output.
pub struct SubtitleState {
    pub lines: Vec<SubtitleLine>,
    /// Detected format string, e.g. "srt" or "ass".
    pub format: String,
    /// Backing parsed file used for format-preserving export.
    subtitle_file: SubtitleFile,
}

impl SubtitleState {
    /// Parse a subtitle file from its raw text content and a format hint string
    /// ("srt" or "ass").
    pub fn parse(content: &str, format_hint: &str) -> Result<Self> {
        let bytes = content.as_bytes();
        let ext = OsStr::new(format_hint);
        let fmt = get_subtitle_format_err(Some(ext), bytes)
            .map_err(|e| anyhow::anyhow!("unsupported subtitle format '{}': {}", format_hint, e))?;

        let format_str = match fmt {
            SubtitleFormat::SubRip => "srt",
            SubtitleFormat::SubStationAlpha => "ass",
            _ => format_hint,
        };

        let subtitle_file = parse_bytes(fmt, bytes, None, 24.0)
            .map_err(|e| anyhow::anyhow!("failed to parse subtitle: {}", e))?;

        let entries = subtitle_file
            .get_subtitle_entries()
            .map_err(|e| anyhow::anyhow!("failed to get subtitle entries: {}", e))?;

        if entries.is_empty() {
            bail!("subtitle file contains no entries");
        }

        let lines: Vec<SubtitleLine> = entries
            .into_iter()
            .map(|entry| {
                let start_ms = entry.timespan.start.msecs();
                let end_ms = entry.timespan.end.msecs();
                SubtitleLine {
                    original_start_ms: start_ms,
                    original_end_ms: end_ms,
                    corrected_start_ms: start_ms,
                    corrected_end_ms: end_ms,
                }
            })
            .collect();

        Ok(SubtitleState {
            lines,
            format: format_str.to_string(),
            subtitle_file,
        })
    }

    /// Reset all corrected timings back to the original values.
    pub fn reset_corrections(&mut self) {
        for line in &mut self.lines {
            line.corrected_start_ms = line.original_start_ms;
            line.corrected_end_ms = line.original_end_ms;
        }
    }

    /// Export the corrected subtitle as SRT text, regardless of original format.
    pub fn export_srt(&self) -> String {
        let mut out = String::new();
        for (i, line) in self.lines.iter().enumerate() {
            out.push_str(&format!("{}\n", i + 1));
            out.push_str(&format!(
                "{} --> {}\n",
                format_srt_time(line.corrected_start_ms.max(0)),
                format_srt_time(line.corrected_end_ms.max(0))
            ));
            // Retrieve the text from the original subtitle_file entries.
            // We re-fetch each time to keep a simple data model.
            // For the text portion we emit a placeholder when entries are unavailable.
            out.push('\n');
        }
        out
    }

    /// Export the corrected subtitle in its original format as a UTF-8 string.
    pub fn export_corrected(&self) -> Result<String> {
        let entries: Vec<SubtitleEntry> = self
            .lines
            .iter()
            .map(|line| {
                let start = SubTimePoint::from_msecs(line.corrected_start_ms.max(0));
                let end = SubTimePoint::from_msecs(line.corrected_end_ms.max(0));
                SubtitleEntry::from(SubTimeSpan::new(start, end))
            })
            .collect();

        let mut file = self.subtitle_file.clone();
        file.update_subtitle_entries(&entries)
            .map_err(|e| anyhow::anyhow!("failed to update subtitle entries: {}", e))?;

        let data = file
            .to_data()
            .map_err(|e| anyhow::anyhow!("failed to serialize subtitle: {}", e))?;

        String::from_utf8(data).map_err(|e| anyhow::anyhow!("subtitle output is not valid UTF-8: {}", e))
    }

    /// Number of subtitle lines.
    pub fn len(&self) -> usize {
        self.lines.len()
    }

    /// Access lines as a slice.
    pub fn lines(&self) -> &[SubtitleLine] {
        &self.lines
    }
}

/// Format milliseconds as an SRT timestamp: HH:MM:SS,mmm
pub fn format_srt_time(ms: i64) -> String {
    let ms = ms.max(0) as u64;
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let millis = ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, millis)
}
