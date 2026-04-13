use crate::engine::Anchor;

/// Interpolate the warp delta at a given film position using piecewise linear
/// interpolation over a sorted slice of anchors.
///
/// - Before the first anchor: flat extrapolation (hold first delta).
/// - After the last anchor: flat extrapolation (hold last delta).
/// - Between two anchors: linear interpolation.
pub fn interpolate_delta(anchors: &[Anchor], film_time_ms: i64) -> i64 {
    if anchors.is_empty() {
        return 0;
    }

    let first = &anchors[0];
    let last = &anchors[anchors.len() - 1];

    if film_time_ms <= first.film_position_ms {
        return first.delta_ms;
    }
    if film_time_ms >= last.film_position_ms {
        return last.delta_ms;
    }

    // Binary search for the first anchor whose position is strictly greater than film_time_ms.
    let idx = anchors.partition_point(|a| a.film_position_ms <= film_time_ms);
    let prev = &anchors[idx - 1];
    let next = &anchors[idx];

    let span = (next.film_position_ms - prev.film_position_ms) as f64;
    let t = (film_time_ms - prev.film_position_ms) as f64 / span;

    (prev.delta_ms as f64 + t * (next.delta_ms - prev.delta_ms) as f64).round() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Anchor;
    use uuid::Uuid;

    fn make_anchor(film_pos_ms: i64, delta_ms: i64) -> Anchor {
        Anchor {
            id: Uuid::new_v4(),
            film_position_ms: film_pos_ms,
            delta_ms,
            confidence: 1.0,
            line_range: 0..1,
            chunk_id: Uuid::new_v4(),
        }
    }

    #[test]
    fn empty_anchors_returns_zero() {
        assert_eq!(interpolate_delta(&[], 5000), 0);
    }

    #[test]
    fn single_anchor_flat() {
        let anchors = vec![make_anchor(10_000, 500)];
        assert_eq!(interpolate_delta(&anchors, 0), 500);
        assert_eq!(interpolate_delta(&anchors, 10_000), 500);
        assert_eq!(interpolate_delta(&anchors, 99_999), 500);
    }

    #[test]
    fn two_anchors_interpolation() {
        let anchors = vec![make_anchor(0, 0), make_anchor(10_000, 1_000)];
        assert_eq!(interpolate_delta(&anchors, 0), 0);
        assert_eq!(interpolate_delta(&anchors, 5_000), 500);
        assert_eq!(interpolate_delta(&anchors, 10_000), 1_000);
        assert_eq!(interpolate_delta(&anchors, 20_000), 1_000);
    }

    #[test]
    fn before_first_anchor_extrapolates_flat() {
        let anchors = vec![make_anchor(10_000, 200), make_anchor(20_000, 400)];
        assert_eq!(interpolate_delta(&anchors, 5_000), 200);
    }

    #[test]
    fn after_last_anchor_extrapolates_flat() {
        let anchors = vec![make_anchor(10_000, 200), make_anchor(20_000, 400)];
        assert_eq!(interpolate_delta(&anchors, 30_000), 400);
    }
}
