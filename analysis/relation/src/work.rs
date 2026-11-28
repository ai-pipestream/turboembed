//! Deterministic work counters for [`crate::RelationAnnotator`].

/// Largest growth allowed relative to the expected linear work ratio, in
/// percent. Linear work sits at 100%; `n log n` binary-search probes sit near
/// 113% across a 4x step; superlinear `n^1.5` (200%) and quadratic `n^2`
/// (400%) exceed this bound.
const MAX_LINEAR_HEADROOM_PERCENT: u128 = 125;

/// Deterministic work counters for one [`crate::RelationAnnotator`] run.
///
/// The counters are recorded where the annotator probes words during head and
/// sentence resolution, visits same-sentence candidate pairs, reads parent
/// dependency arcs, and merges duplicate edges. Wall-clock time is not part of
/// these counters: CI runners share hosts, so scaling assertions use
/// deterministic work units instead of elapsed time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RelationWork {
    /// Word probes during entity-head binary searches, overlapping-word
    /// checks, and word-to-sentence lookups.
    pub head_probes: usize,
    /// Same-sentence `(subject, object)` entity pairs visited.
    pub pair_visits: usize,
    /// Parent-arc lookups performed while searching and building shortest
    /// dependency paths.
    pub arc_reads: usize,
    /// Edge insertions and per-pair duplicate-kind comparisons while building
    /// the deduplicated edge layer.
    pub edge_ops: usize,
}

impl RelationWork {
    /// Checks that every counter grows no faster than near-linearly with the
    /// expected work units across `samples`, given as `(work_units, work)` with
    /// strictly increasing `work_units`. Each counter is checked independently
    /// so a large linear counter cannot mask superlinear growth in another.
    pub fn check_linear_scaling(samples: &[(usize, Self)]) -> Result<(), String> {
        if samples.len() < 2 {
            return Err(format!("need at least two samples, got {}", samples.len()));
        }
        for pair in samples.windows(2) {
            let (u_prev, w_prev) = pair[0];
            let (u_curr, w_curr) = pair[1];
            if u_prev == 0 || u_curr <= u_prev {
                return Err(format!(
                    "work units must strictly increase from a non-zero base, got {u_prev} -> {u_curr}"
                ));
            }
            for (name, prev, curr) in [
                ("head_probes", w_prev.head_probes, w_curr.head_probes),
                ("pair_visits", w_prev.pair_visits, w_curr.pair_visits),
                ("arc_reads", w_prev.arc_reads, w_curr.arc_reads),
                ("edge_ops", w_prev.edge_ops, w_curr.edge_ops),
            ] {
                let lhs = (curr as u128) * (u_prev as u128) * 100;
                let rhs = (prev as u128) * (u_curr as u128) * MAX_LINEAR_HEADROOM_PERCENT;
                if lhs > rhs {
                    return Err(format!(
                        "{name} grew faster than linear ({u_prev} -> {u_curr} units): {prev} -> {curr}"
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::RelationWork;

    #[test]
    fn check_linear_scaling_checks_every_counter_independently() {
        let base = RelationWork { head_probes: 1_000, pair_visits: 1_000, arc_reads: 4_000, edge_ops: 1_000 };
        assert!(RelationWork::check_linear_scaling(&[(100, base)]).is_err());
        assert!(RelationWork::check_linear_scaling(&[(0, base), (100, base)]).is_err());
        assert!(RelationWork::check_linear_scaling(&[(100, base), (100, base)]).is_err());

        let four_x = RelationWork { head_probes: 4_500, pair_visits: 4_000, arc_reads: 16_000, edge_ops: 4_000 };
        RelationWork::check_linear_scaling(&[(100, base), (400, four_x)]).unwrap();

        for (field, bad) in [
            ("head_probes", RelationWork { head_probes: 16_000, ..four_x }),
            ("pair_visits", RelationWork { pair_visits: 16_000, ..four_x }),
            ("arc_reads", RelationWork { arc_reads: 32_000, ..four_x }),
            ("edge_ops", RelationWork { edge_ops: 16_000, ..four_x }),
        ] {
            let err = RelationWork::check_linear_scaling(&[(100, base), (400, bad)]).unwrap_err();
            assert!(err.contains(field), "expected {field} in error, got: {err}");
        }
    }
}
