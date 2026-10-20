//! Sequential [`Alignment`] edit-script composition (`A -> B` followed by `B -> C`).

use crate::span::Span;

use super::{Alignment, Edit};

pub(super) fn compose_alignments(ab: &Alignment, bc: &Alignment) -> Alignment {
    if ab.edits.is_empty() {
        return bc.clone();
    }
    if bc.edits.is_empty() {
        return ab.clone();
    }
    let ab_edits = split_pure_insertions(&ab.edits, &bc.edits);
    let bc_edits = split_pure_deletions(&bc.edits, &ab.edits);
    let out = merge_edits(&ab_edits, &bc_edits);
    Alignment { edits: out }
}

fn split_pure_insertions(ab: &[Edit], bc: &[Edit]) -> Vec<Edit> {
    let mut cuts: Vec<u32> = Vec::with_capacity(bc.len() * 2);
    for e in bc {
        cuts.push(e.parent.start);
        cuts.push(e.parent.end);
    }
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::with_capacity(ab.len());
    for &e in ab {
        if e.parent.is_empty() && !e.child.is_empty() {
            let mut cur = e.child.start;
            let lo = cuts.partition_point(|&p| p <= e.child.start);
            let hi = cuts.partition_point(|&p| p < e.child.end);
            for &p in &cuts[lo..hi] {
                out.push(Edit { parent: e.parent, child: Span::new(cur, p) });
                cur = p;
            }
            out.push(Edit { parent: e.parent, child: Span::new(cur, e.child.end) });
        } else {
            out.push(e);
        }
    }
    out
}

fn split_pure_deletions(bc: &[Edit], ab: &[Edit]) -> Vec<Edit> {
    let mut cuts: Vec<u32> = Vec::with_capacity(ab.len() * 2);
    for e in ab {
        cuts.push(e.child.start);
        cuts.push(e.child.end);
    }
    cuts.sort_unstable();
    cuts.dedup();
    let mut out = Vec::with_capacity(bc.len());
    for &e in bc {
        if e.child.is_empty() && !e.parent.is_empty() {
            let mut cur = e.parent.start;
            let lo = cuts.partition_point(|&p| p <= e.parent.start);
            let hi = cuts.partition_point(|&p| p < e.parent.end);
            for &p in &cuts[lo..hi] {
                out.push(Edit { parent: Span::new(cur, p), child: e.child });
                cur = p;
            }
            out.push(Edit { parent: Span::new(cur, e.parent.end), child: e.child });
        } else {
            out.push(e);
        }
    }
    out
}

fn merge_edits(ab: &[Edit], bc: &[Edit]) -> Vec<Edit> {
    let mut out = Vec::with_capacity(ab.len() + bc.len());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut a_at, mut b_at, mut c_at) = (0u32, 0u32, 0u32);

    while i < ab.len() || j < bc.len() {
        if i < ab.len() && (j == bc.len() || ab[i].child.end <= bc[j].parent.start) {
            let e1 = ab[i];
            let d = e1.child.start - b_at;
            let c_start = c_at + d;
            let c_end = c_start + e1.child.len();
            out.push(Edit { parent: e1.parent, child: Span::new(c_start, c_end) });
            a_at = e1.parent.end;
            b_at = e1.child.end;
            c_at = c_end;
            i += 1;
        } else if j < bc.len() && (i == ab.len() || bc[j].parent.end <= ab[i].child.start) {
            let e2 = bc[j];
            let d = e2.parent.start - b_at;
            let a_start = a_at + d;
            let a_end = a_start + e2.parent.len();
            out.push(Edit { parent: Span::new(a_start, a_end), child: e2.child });
            a_at = a_end;
            b_at = e2.parent.end;
            c_at = e2.child.end;
            j += 1;
        } else {
            let e1 = ab[i];
            let e2 = bc[j];
            let cluster_b_start = e1.child.start.min(e2.parent.start);
            let d = cluster_b_start - b_at;
            let cluster_a_start = a_at + d;
            let cluster_c_start = c_at + d;
            let mut cluster_b_end = e1.child.end.max(e2.parent.end);
            i += 1;
            j += 1;
            loop {
                if i < ab.len() && ab[i].child.start < cluster_b_end && cluster_b_start < ab[i].child.end {
                    cluster_b_end = cluster_b_end.max(ab[i].child.end);
                    i += 1;
                    continue;
                }
                if j < bc.len() && bc[j].parent.start < cluster_b_end && cluster_b_start < bc[j].parent.end {
                    cluster_b_end = cluster_b_end.max(bc[j].parent.end);
                    j += 1;
                    continue;
                }
                break;
            }
            let last_ab = ab[i - 1];
            let last_bc = bc[j - 1];
            let cluster_a_end = last_ab.parent.end + (cluster_b_end - last_ab.child.end);
            let cluster_c_end = last_bc.child.end + (cluster_b_end - last_bc.parent.end);
            a_at = cluster_a_end;
            b_at = cluster_b_end;
            c_at = cluster_c_end;
            if !(cluster_a_start == cluster_a_end && cluster_c_start == cluster_c_end) {
                out.push(Edit {
                    parent: Span::new(cluster_a_start, cluster_a_end),
                    child: Span::new(cluster_c_start, cluster_c_end),
                });
            }
        }
    }
    out
}
