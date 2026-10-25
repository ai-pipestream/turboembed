//! Tests for the graph cursors of #1198 part 3: bounded
//! breadth-first reachability over `Edges<L, L>` with reusable
//! scratch, and `Groups::groups_containing` reverse lookups.

use turbo_analysis::*;
use turbo_analysis_layers::Entities;

/// 0 -> 1 -> 2 -> 0 (a cycle), 1 -> 3, and 4 isolated.
fn graph() -> Edges<Entities, Entities> {
    let rels = TagSet::new("rel", &["r"]);
    let r = rels.tag("r").unwrap();
    let e = |a: u32, b: u32| Edge::new(Ix::new(a), Ix::new(b), r, 1.0);
    Edges::new(rels, [e(0, 1), e(1, 2), e(1, 3), e(2, 0)])
}

fn ids(out: &[Ix<Entities>]) -> Vec<usize> {
    out.iter().map(|i| i.get()).collect()
}

#[test]
fn reachable_from_walks_breadth_first_with_hop_bounds_and_cycles() {
    let g = graph();
    g.check().unwrap();
    let mut scratch = ReachScratch::new();
    let mut out = Vec::new();

    g.reachable_from(Ix::new(0), 0, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![0]);

    g.reachable_from(Ix::new(0), 1, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![0, 1]);

    g.reachable_from(Ix::new(0), 2, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![0, 1, 2, 3]);

    // The cycle back to 0 terminates; more hops change nothing, and
    // the isolated item is never reached.
    g.reachable_from(Ix::new(0), u32::MAX, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![0, 1, 2, 3]);

    g.reachable_from(Ix::new(4), u32::MAX, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![4]);

    // A second traversal through the same scratch starts clean.
    g.reachable_from(Ix::new(2), u32::MAX, &mut scratch, &mut out);
    assert_eq!(ids(&out), vec![2, 0, 1, 3]);
}

#[test]
fn reachable_from_agrees_with_repeated_outgoing_union() {
    let g = graph();
    let mut scratch = ReachScratch::new();
    let mut out = Vec::new();
    for hops in 0..5u32 {
        // Brute force: the closure of `outgoing` up to `hops` times.
        let mut seen: Vec<usize> = vec![1];
        let mut frontier = vec![1usize];
        for _ in 0..hops {
            let mut next = Vec::new();
            for &n in &frontier {
                for (_, e) in g.outgoing(Ix::new(n as u32)) {
                    if !seen.contains(&e.to.get()) {
                        seen.push(e.to.get());
                        next.push(e.to.get());
                    }
                }
            }
            frontier = next;
        }
        g.reachable_from(Ix::new(1), hops, &mut scratch, &mut out);
        let mut got = ids(&out);
        let mut want = seen.clone();
        got.sort_unstable();
        want.sort_unstable();
        assert_eq!(got, want, "hops {hops}");
        // Breadth-first order: the output is grouped by distance, so
        // sorting only inside each level is already the order here.
        assert_eq!(ids(&out)[0], 1);
    }
}

#[test]
fn groups_containing_reports_every_group_and_rank() {
    let ix = |i: u32| Ix::new(i);
    // Overlapping groups: 2 is in all three, 5 only in the last,
    // 6 in none; members need not be given sorted or unique.
    let groups =
        Groups::<Entities>::new([vec![ix(2), ix(0), ix(1), ix(2)], vec![ix(2), ix(3)], vec![ix(5), ix(2), ix(4)]]);
    groups.check().unwrap();
    let found: Vec<(usize, usize)> = groups.groups_containing(ix(2)).map(|(g, rank)| (g.get(), rank)).collect();
    assert_eq!(found, vec![(0, 2), (1, 0), (2, 0)]);
    assert_eq!(groups.groups_containing(ix(5)).collect::<Vec<_>>(), vec![(Ix::new(2), 2)]);
    assert_eq!(groups.groups_containing(ix(6)).count(), 0);
    // Brute-force agreement on every item.
    for item in 0..7u32 {
        let want: Vec<(usize, usize)> = groups
            .iter()
            .filter_map(|(g, members)| members.iter().position(|&m| m == ix(item)).map(|r| (g.get(), r)))
            .collect();
        let got: Vec<(usize, usize)> = groups.groups_containing(ix(item)).map(|(g, rank)| (g.get(), rank)).collect();
        assert_eq!(got, want, "item {item}");
    }
}
