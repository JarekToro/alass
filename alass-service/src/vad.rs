use alass_core::{TimePoint, TimeSpan};
use webrtc_vad::{SampleRate, Vad};

/// Run WebRTC VAD on raw PCM bytes and return speech segments as absolute-time TimeSpans.
///
/// `pcm_bytes`: 16-bit signed LE mono PCM
/// `sample_rate`: 8000 or 16000
/// `film_start_ms`: absolute film position of the first sample
pub fn run_vad(pcm_bytes: &[u8], sample_rate: i32, film_start_ms: i64) -> Vec<TimeSpan> {
    let rate = match sample_rate {
        8000 => SampleRate::Rate8kHz,
        16000 => SampleRate::Rate16kHz,
        _ => return Vec::new(),
    };

    let mut vad = Vad::new_with_rate(rate);

    // 10ms frame: 8kHz → 80 samples, 16kHz → 160 samples
    let frame_samples = (sample_rate / 100) as usize;
    let ms_per_frame: i64 = 10;

    let samples: Vec<i16> = pcm_bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect();

    let mut voice_flags: Vec<bool> = Vec::with_capacity(samples.len() / frame_samples);
    for frame in samples.chunks(frame_samples) {
        if frame.len() < frame_samples {
            break;
        }
        let is_voice = vad.is_voice_segment(frame).unwrap_or(false);
        voice_flags.push(is_voice);
    }

    // Consolidate consecutive voice frames into TimeSpans
    let mut spans = Vec::new();
    let mut in_speech = false;
    let mut speech_start_ms: i64 = 0;

    for (i, &is_voice) in voice_flags.iter().enumerate() {
        let frame_ms = film_start_ms + (i as i64) * ms_per_frame;
        if is_voice && !in_speech {
            speech_start_ms = frame_ms;
            in_speech = true;
        } else if !is_voice && in_speech {
            let speech_end_ms = frame_ms;
            if speech_end_ms > speech_start_ms {
                spans.push(TimeSpan::new(
                    TimePoint::from(speech_start_ms),
                    TimePoint::from(speech_end_ms),
                ));
            }
            in_speech = false;
        }
    }

    // Close trailing speech segment
    if in_speech {
        let speech_end_ms = film_start_ms + (voice_flags.len() as i64) * ms_per_frame;
        if speech_end_ms > speech_start_ms {
            spans.push(TimeSpan::new(
                TimePoint::from(speech_start_ms),
                TimePoint::from(speech_end_ms),
            ));
        }
    }

    spans
}
