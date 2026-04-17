//! End-to-end tests for the streaming aligner using synthetic VAD spans.
//!
//! The tests use varied subtitle line lengths (matching real subtitle files);
//! a uniform-length grid creates degenerate tie conditions that aren't
//! representative of real content.

use alass_core::{TimePoint, TimeSpan};
use alass_stream::{Config, LineChange, StreamingAligner};
use subparse::SubtitleFormat;

fn span(a: i64, b: i64) -> TimeSpan {
    TimeSpan::new(TimePoint::from(a), TimePoint::from(b))
}

/// Deterministic pseudo-random subtitle timeline: `count` lines with lengths
/// in [500, 3500) ms and gaps in [200, 2000) ms. Returns `(srt_text,
/// spans)`. `seed` controls the distribution.
fn synth_subs(count: usize, seed: u64) -> (String, Vec<TimeSpan>) {
    let mut rng = seed;
    let mut out = String::new();
    let mut spans = Vec::with_capacity(count);
    let mut t: i64 = 0;
    for i in 0..count {
        rng = lcg(rng);
        let len = 500 + (rng % 3000) as i64;
        rng = lcg(rng);
        let gap = 200 + (rng % 1800) as i64;
        let s = t;
        let e = t + len;
        out.push_str(&format!(
            "{}\n{} --> {}\nline {}\n\n",
            i + 1,
            fmt_srt(s),
            fmt_srt(e),
            i + 1,
        ));
        spans.push(span(s, e));
        t = e + gap;
    }
    (out, spans)
}

fn lcg(x: u64) -> u64 {
    x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

fn fmt_srt(ms: i64) -> String {
    let ms = ms.max(0);
    let h = ms / 3_600_000;
    let m = (ms % 3_600_000) / 60_000;
    let s = (ms % 60_000) / 1_000;
    let mm = ms % 1_000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, mm)
}

/// VAD evidence for the portion of the film where audio is covered.
/// Emits `sub + shift` for each sub whose shifted form falls entirely in
/// `[from_ms, to_ms)`. Partial overlaps are clipped.
fn vad_in_window(subs: &[TimeSpan], shift: i64, from_ms: i64, to_ms: i64) -> Vec<TimeSpan> {
    subs.iter()
        .filter_map(|s| {
            let start = s.start.as_i64() + shift;
            let end = s.end.as_i64() + shift;
            if end <= from_ms || start >= to_ms {
                return None;
            }
            let cs = start.max(from_ms);
            let ce = end.min(to_ms);
            if ce > cs { Some(span(cs, ce)) } else { None }
        })
        .collect()
}

fn fast_config() -> Config {
    Config {
        split_penalty: 7.0,
        // No speed optimisation so the tests get exact millisecond deltas
        // and can use equality assertions. Runtime stays well under 1s
        // thanks to these being synthetic.
        speed_optimization: None,
        vad_mode: 0,
    }
}

#[test]
fn detects_constant_shift_from_single_chunk() {
    // 200 varied-length subs (~17 min of content). Audio is shifted +5000 ms.
    // We ingest the first 60 s of audio. Alass should report +5000 for every
    // line (covered lines from matched VAD; uncovered lines from alass's
    // preference for no splits under the given penalty).
    let (srt, _) = synth_subs(200, 0xDEADBEEF);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    // Build VAD from the aligner's original spans so the synthetic VAD
    // exactly matches the loaded sub timeline.
    let originals: Vec<TimeSpan> = a.corrected_lines().iter().map(|l| span(l.original_start_ms, l.original_end_ms)).collect();
    let vad = vad_in_window(&originals, 5_000, 0, 60_000);
    assert!(vad.len() >= 5, "sanity: enough VAD to anchor the shift");

    let changes = a.ingest_vad_spans(&vad);
    assert!(!changes.is_empty());

    // Every line should land on +5000 ms.
    for (i, d) in a.current_deltas_ms().iter().enumerate() {
        assert_eq!(*d, 5_000, "line {i} delta = {d}");
    }
}

#[test]
fn incremental_ingest_refines_alignment_and_emits_only_changed_lines() {
    let (srt, _) = synth_subs(200, 0xC0FFEE);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    let originals: Vec<TimeSpan> = a.corrected_lines().iter().map(|l| span(l.original_start_ms, l.original_end_ms)).collect();

    // First chunk: 0..30s.
    let v1 = vad_in_window(&originals, 2_500, 0, 30_000);
    let c1 = a.ingest_vad_spans(&v1);
    assert!(!c1.is_empty(), "first ingest emits changes");
    for d in a.current_deltas_ms() { assert_eq!(d, 2_500); }

    // Second chunk: 30..60s with the same shift. Since the current
    // alignment is already correct, no new line changes are expected.
    let v2 = vad_in_window(&originals, 2_500, 30_000, 60_000);
    let c2 = a.ingest_vad_spans(&v2);
    assert!(c2.is_empty(), "re-aligning to the same shift emits no changes; got {}", c2.len());

    // Empty ingest is always a no-op.
    assert!(a.ingest_vad_spans(&[]).is_empty());
}

#[test]
fn gapped_coverage_still_solves_constant_shift() {
    // Two widely-separated coverage windows. The shift is the same in both.
    let (srt, _) = synth_subs(400, 0xBADF00D);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();
    let originals: Vec<TimeSpan> = a.corrected_lines().iter().map(|l| span(l.original_start_ms, l.original_end_ms)).collect();

    let mut vad = vad_in_window(&originals, 1_500, 0, 180_000);
    vad.extend(vad_in_window(&originals, 1_500, 600_000, 780_000));
    a.ingest_vad_spans(&vad);

    for d in a.current_deltas_ms() {
        assert_eq!(d, 1_500);
    }
}

#[test]
fn line_changes_reflect_delta_movement_only() {
    let (srt, orig) = synth_subs(50, 0xCAFEBABE);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    let vad = vad_in_window(&orig, 2_500, 0, 60_000);
    let changes = a.ingest_vad_spans(&vad);
    assert!(!changes.is_empty());

    for LineChange {
        line_index,
        old_start_ms,
        old_end_ms,
        new_start_ms,
        new_end_ms,
        delta_change_ms,
    } in changes
    {
        assert_eq!(old_start_ms, orig[line_index].start.as_i64());
        assert_eq!(old_end_ms, orig[line_index].end.as_i64());
        assert_eq!(new_start_ms - old_start_ms, delta_change_ms);
        assert_eq!(new_end_ms - old_end_ms, delta_change_ms);
    }
}

#[test]
fn reset_clears_evidence_and_deltas_but_keeps_subtitle() {
    let (srt, orig) = synth_subs(30, 42);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    let vad = vad_in_window(&orig, 3_000, 0, 60_000);
    a.ingest_vad_spans(&vad);
    assert!(a.current_deltas_ms().iter().any(|&d| d != 0));

    a.reset();
    assert!(a.vad_evidence().is_empty());
    assert!(a.current_deltas_ms().iter().all(|&d| d == 0));
    assert_eq!(a.line_count(), 30);
}

#[test]
fn export_produces_shifted_srt() {
    let (srt, orig) = synth_subs(10, 7);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    let vad = vad_in_window(&orig, 4_000, 0, 60_000);
    a.ingest_vad_spans(&vad);

    let out = a.export_text().expect("srt export");
    // Line 1's original start was orig[0].start; after +4000 shift it's
    // orig[0].start + 4000.
    let expected_ms = orig[0].start.as_i64() + 4_000;
    let expected = fmt_srt(expected_ms);
    assert!(out.contains(&expected), "exported srt missing {expected}:\n{out}");
}

#[test]
fn long_sparse_stream_matches_user_scenario() {
    // The spec case: ~2h40m of subtitles, audio arrives as a 1-minute clip,
    // a 2-minute gap, then a 3-minute clip — all with a constant +2000 ms
    // shift. The aligner must resolve +2000 on the subs that had evidence
    // and (via alass's no-split preference) on the rest as well.
    const SUB_COUNT: usize = 2400;
    let (srt, orig) = synth_subs(SUB_COUNT, 0xFEEDFACE);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    let chunk_a = vad_in_window(&orig, 2_000, 0, 60_000);
    let chunk_b = vad_in_window(&orig, 2_000, 180_000, 360_000);

    a.ingest_vad_spans(&chunk_a);
    a.ingest_vad_spans(&chunk_b);

    for d in a.current_deltas_ms() {
        assert_eq!(d, 2_000);
    }
}

#[test]
fn split_shift_is_recovered_when_evidence_supports_it() {
    // First half of the film has shift +1000, second half +4000. With
    // VAD in both halves, alass should split the deltas into two groups.
    let (srt, orig) = synth_subs(120, 0xABC123);
    let mut a = StreamingAligner::with_config(fast_config());
    a.load_subtitle(&srt, SubtitleFormat::SubRip).unwrap();

    // Split point: anything after line 60.
    let split = orig[60].start.as_i64();
    let first_half: Vec<TimeSpan> = orig.iter().take(60).copied().collect();
    let second_half: Vec<TimeSpan> = orig.iter().skip(60).copied().collect();

    let mut vad = vad_in_window(&first_half, 1_000, 0, split + 10_000);
    vad.extend(vad_in_window(&second_half, 4_000, split, i64::MAX));

    a.ingest_vad_spans(&vad);
    let deltas = a.current_deltas_ms();

    for (i, d) in deltas.iter().enumerate() {
        let expected = if i < 60 { 1_000 } else { 4_000 };
        assert_eq!(*d, expected, "line {i} delta = {d}, expected {expected}");
    }
}
