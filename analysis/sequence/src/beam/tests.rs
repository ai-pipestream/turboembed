use super::*;

fn path<'m, L: ?Sized>(nodes: &[Node], id: u32, outcomes: &[&'m L], out: &mut Vec<&'m L>) {
    out.clear();
    let mut at = id;
    while at != ROOT {
        let n = nodes[at as usize];
        out.push(outcomes[n.outcome as usize]);
        at = n.parent;
    }
    out.reverse();
}

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
fn compare_step_nodes_and_path_slots_match_full_path_reference() {
    let outcomes: &[&str] = &["A", "B", "C", "D"];
    // Step 1: two nodes at len 1 ("B" and "C")
    // Step 2: children of step 1 at len 2 ("B,A", "B,C", "C,A")
    // Step 3: children of step 2 at len 3 ("B,A,D", "B,C,A", "C,A,B")
    let mut nodes = vec![
        Node { parent: ROOT, parent_rank: 0, outcome: 1, prob: 0.5, score: -1.0, len: 1 },
        Node { parent: ROOT, parent_rank: 0, outcome: 2, prob: 0.5, score: -1.0, len: 1 },
    ];
    let mut rank_order = Vec::new();
    let mut prev_ranks = Vec::new();
    let step1 = [0u32, 1];
    assign_beam_ranks(&nodes, outcomes, &step1, 4, &mut rank_order, &mut prev_ranks);

    nodes.push(Node { parent: 0, parent_rank: prev_ranks[0], outcome: 0, prob: 0.5, score: -2.0, len: 2 });
    nodes.push(Node { parent: 0, parent_rank: prev_ranks[0], outcome: 2, prob: 0.5, score: -2.0, len: 2 });
    nodes.push(Node { parent: 1, parent_rank: prev_ranks[1], outcome: 0, prob: 0.5, score: -2.0, len: 2 });
    let step2 = [2u32, 3, 4];
    assign_beam_ranks(&nodes, outcomes, &step2, 4, &mut rank_order, &mut prev_ranks);

    nodes.push(Node { parent: 2, parent_rank: prev_ranks[0], outcome: 3, prob: 0.5, score: -3.0, len: 3 });
    nodes.push(Node { parent: 3, parent_rank: prev_ranks[1], outcome: 0, prob: 0.5, score: -3.0, len: 3 });
    nodes.push(Node { parent: 4, parent_rank: prev_ranks[2], outcome: 1, prob: 0.5, score: -3.0, len: 3 });
    let step3 = [5u32, 6, 7];

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

    for step in [&step1[..], &step2[..], &step3[..]] {
        for &a in step {
            for &b in step {
                assert_eq!(compare_step_nodes(&nodes, outcomes, a, b), reference(a, b), "a={a}, b={b}");
            }
        }
    }

    let mut slots = PathSlots::new(4, 3);
    let mut want = Vec::new();
    for step in [&[ROOT][..], &step1[..], &step2[..], &step3[..]] {
        slots.prepare(&nodes, step, outcomes);
        for (beam_idx, &id) in step.iter().enumerate() {
            path(&nodes, id, outcomes, &mut want);
            assert_eq!(slots.prior(beam_idx), want.as_slice(), "id={id}");
        }
    }
}
