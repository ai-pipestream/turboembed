use super::*;

#[test]
fn threshold_matches_sorted_across_beam_sizes_and_special_floats() {
    let samples: &[&[f64]] = &[
        &[0.5],
        &[0.1, 0.7, 0.2],
        &[0.25, 0.25, 0.25, 0.25],
        &[-0.0, 0.0, -1.0, 1.0, f64::NAN, f64::INFINITY],
        &[0.05, 0.15, 0.30, 0.10, 0.25, 0.08, 0.07],
    ];
    for &probs in samples {
        let mut full = probs.to_vec();
        full.sort_by(|a, b| compare_doubles(*a, *b));
        let mut scratch = vec![0.0; probs.len()];
        for size in 0..=(probs.len() + 2) {
            let got = threshold(probs, &mut scratch, size);
            let want = full[probs.len().saturating_sub(size.max(1))];
            assert_eq!(compare_doubles(got, want), Ordering::Equal, "size={size}, probs={probs:?}");
        }
    }
}

#[test]
fn compare_nodes_matches_full_path_comparison() {
    let outcomes: &[&str] = &["A", "B", "C", "D"];
    let nodes = vec![
        Node { distribution: 0, parent: ROOT, outcome: 1, prob: 0.5, score: -1.0, len: 1 },
        Node { distribution: 0, parent: 0, outcome: 0, prob: 0.5, score: -1.0, len: 2 },
        Node { distribution: 0, parent: 0, outcome: 2, prob: 0.5, score: -1.0, len: 2 },
        Node { distribution: 0, parent: 1, outcome: 3, prob: 0.5, score: -1.0, len: 3 },
        Node { distribution: 0, parent: 2, outcome: 0, prob: 0.5, score: -1.0, len: 3 },
    ];
    let reference = |a: u32, b: u32| -> Ordering {
        let (x, y) = (nodes[a as usize], nodes[b as usize]);
        let by_score = compare_doubles(y.score, x.score);
        if by_score != Ordering::Equal {
            return by_score;
        }
        let (mut pa, mut pb) = (Vec::new(), Vec::new());
        path(&nodes, a, outcomes, &mut pa);
        path(&nodes, b, outcomes, &mut pb);
        for (s, t) in pa.iter().zip(&pb) {
            let by_outcome = str::compare(s, t);
            if by_outcome != Ordering::Equal {
                return by_outcome;
            }
        }
        pa.len().cmp(&pb.len())
    };
    for a in 0..nodes.len() as u32 {
        for b in 0..nodes.len() as u32 {
            assert_eq!(compare_nodes(&nodes, outcomes, a, b), reference(a, b), "a={a}, b={b}");
        }
    }
}
