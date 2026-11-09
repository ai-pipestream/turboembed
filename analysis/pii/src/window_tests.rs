use std::cell::Cell;

use turbo_analysis::{Layer, Span, ViewId};
use turbo_analysis_layers::{Pii, PiiKind};

use super::{BOUNDARY_CHARS_VISITED, has_label};
use crate::{Pack, PiiDetector};

/// Regional indicator runs have `letter_or_digit == false` and must not
/// trigger quadratic backtracking in `context_before` when `words_before` and
/// `words_after` scan across them. Counts characters visited in
/// `context_before`, never wall-clock time (#1936).
#[test]
fn regional_indicator_runs_scan_in_linear_time() {
    let imei_window = PiiDetector::pack(Pack::Device).with_label_window(PiiKind::Imei, 4);
    let visited_for = |n: usize| {
        let flags = "\u{1F1E6}".repeat(n);
        let text = format!("{flags} 490154203237518 {flags}");
        let start = flags.len() as u32 + 1;
        BOUNDARY_CHARS_VISITED.with(|c| c.set(0));
        assert!(!has_label(&text, Span::new(start, start + 15), &["imei"], 4));
        let mut pii = Pii::new(ViewId::SOURCE);
        imei_window.detect(&text, &mut pii);
        assert_eq!(pii.len(), 0);
        BOUNDARY_CHARS_VISITED.with(Cell::get)
    };
    let (v1, v4) = (visited_for(1_000), visited_for(4_000));
    assert!(v1 > 0 && v4 <= v1 * 4, "context_before char visits must grow linearly: N={v1}, 4N={v4}");
    assert!(v4 <= 8 * 4_000 + 32, "expected O(1) context chars per boundary, visited {v4}");
}
