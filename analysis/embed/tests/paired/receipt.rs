//! Text receipts for the exact samples and allowance used by the gate.

use super::{WIDEST, window};

pub fn pairs(sets: &[Vec<(f64, f64)>]) -> String {
    sets.iter()
        .map(|set| set.iter().map(|(x, y)| format!("{x}/{y}")).collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join(" | ")
}

// Window values are fractions; sample Display retains the full f64 value.
pub fn own_sets(sets: &[Vec<(f64, f64)>]) -> String {
    sets.iter()
        .enumerate()
        .map(|(i, set)| {
            let w = window(set).map_or_else(|| format!("over {:.3}%", WIDEST * 100.0), |w| format!("{w}"));
            format!("attempt={} window={w} pairs={}", i + 1, pairs(std::slice::from_ref(set)))
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

pub fn allowance(reading: Option<(f64, f64, f64, &str)>) -> String {
    reading
        .map_or_else(|| "inconclusive".to_owned(), |(_, fixed, allowed, _)| format!("{:.3}%", 100.0 * allowed / fixed))
}

#[test]
fn pooled_receipt_reports_the_allowance_used_by_the_gate() {
    // The last retry could not form a quiet window, yet the pool is valid.
    assert_eq!(allowance(Some((0.1, 100.0, 2.5, "pooled"))), "2.500%");
    // Quiet readings report the same allowance used by their gate.
    assert_eq!(allowance(Some((0.1, 100.0, 0.5, "quiet"))), "0.500%");
    assert_eq!(allowance(None), "inconclusive");
}

#[test]
fn self_pair_receipt_keeps_every_attempt_and_its_window() {
    let quiet = vec![(100.0, 100.0); 6];
    let wild = vec![(10.0, 10.0), (20.0, 20.0), (30.0, 30.0), (40.0, 40.0), (50.0, 50.0), (60.0, 60.0)];
    let text = own_sets(&[quiet, wild]);
    assert!(text.starts_with("attempt=1 window=0.005 pairs="));
    assert!(text.contains(" | attempt=2 window=over 5.000% pairs="));
    assert_eq!(text.matches('/').count(), 12);
    assert!(text.ends_with("60/60"));
}
