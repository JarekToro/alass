use alass_core::{TimePoint, TimeSpan};
use webrtc_vad::{SampleRate, Vad, VadMode};

pub fn mode_from_u8(mode: u8) -> VadMode {
    match mode {
        0 => VadMode::Quality,
        1 => VadMode::LowBitrate,
        2 => VadMode::Aggressive,
        _ => VadMode::VeryAggressive,
    }
}

/// Run WebRTC VAD over a block of 16-bit signed little-endian mono PCM and
/// return speech segments as absolute-time `TimeSpan`s (units: milliseconds).
///
/// `film_start_ms` is the absolute film position of the first sample in `pcm`.
/// A full 10 ms at the tail of `pcm` that doesn't fill a frame is dropped.
pub fn pcm_to_spans(pcm: &[u8], sample_rate: u32, film_start_ms: i64, mode: VadMode) -> Vec<TimeSpan> {
    let rate = match sample_rate {
        8000 => SampleRate::Rate8kHz,
        16000 => SampleRate::Rate16kHz,
        _ => return Vec::new(),
    };

    let mut vad = Vad::new_with_rate(rate);
    vad.set_mode(mode);

    let frame_samples = (sample_rate / 100) as usize; // 10 ms
    let ms_per_frame: i64 = 10;

    let samples: Vec<i16> = pcm
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let mut spans = Vec::new();
    let mut in_speech = false;
    let mut speech_start_ms: i64 = 0;

    for (i, frame) in samples.chunks_exact(frame_samples).enumerate() {
        let frame_start_ms = film_start_ms + (i as i64) * ms_per_frame;
        let is_voice = vad.is_voice_segment(frame).unwrap_or(false);

        if is_voice && !in_speech {
            speech_start_ms = frame_start_ms;
            in_speech = true;
        } else if !is_voice && in_speech {
            push_span(&mut spans, speech_start_ms, frame_start_ms);
            in_speech = false;
        }
    }

    if in_speech {
        let frames_processed = samples.len() / frame_samples;
        let end_ms = film_start_ms + (frames_processed as i64) * ms_per_frame;
        push_span(&mut spans, speech_start_ms, end_ms);
    }

    spans
}

fn push_span(spans: &mut Vec<TimeSpan>, start_ms: i64, end_ms: i64) {
    if end_ms > start_ms {
        spans.push(TimeSpan::new(TimePoint::from(start_ms), TimePoint::from(end_ms)));
    }
}
