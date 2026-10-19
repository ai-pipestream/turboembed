//! Sequential [`Alignment`] composition (`A -> B` followed by `B -> C`).

use crate::span::Span;

use super::{Alignment, Edit, Side, map};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Breakpoint {
    pub(super) pos: u32,
    pub(super) at_start: u32,
    pub(super) at_end: u32,
    pub(super) after: After,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum After {
    Shift { base: u32 },
    Constant { start: u32, end: u32 },
}

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
    if out.is_empty() {
        return Alignment::identity();
    }
    let forward = build_forward_breakpoints(ab, bc, &out);
    let backward = build_backward_breakpoints(ab, bc, &out);
    Alignment { edits: out, forward, backward }
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

fn build_forward_breakpoints(ab: &Alignment, bc: &Alignment, out_edits: &[Edit]) -> Vec<Breakpoint> {
    let mut pts: Vec<u32> = Vec::new();
    for e in out_edits {
        pts.push(e.parent.start);
        pts.push(e.parent.end);
    }
    for e in &ab.edits {
        pts.push(e.parent.start);
        pts.push(e.parent.end);
    }
    for bp in &ab.forward {
        pts.push(bp.pos);
    }
    for e in &bc.edits {
        for b in [e.parent.start, e.parent.end] {
            pts.push(ab.to_parent(b, Side::Start));
            pts.push(ab.to_parent(b, Side::End));
        }
    }
    for bp in &bc.forward {
        pts.push(ab.to_parent(bp.pos, Side::Start));
        pts.push(ab.to_parent(bp.pos, Side::End));
    }
    pts.sort_unstable();
    pts.dedup();

    let map_two = |x: u32, side: Side| bc.to_child(ab.to_child(x, side), side);
    let mut bps = Vec::with_capacity(pts.len());
    let mut differs = false;

    for (k, &p) in pts.iter().enumerate() {
        let at_start = map_two(p, Side::Start);
        let at_end = map_two(p, Side::End);
        if at_start != map(out_edits, p, Side::Start, |e| e.parent, |e| e.child)
            || at_end != map(out_edits, p, Side::End, |e| e.parent, |e| e.child)
        {
            differs = true;
        }
        let mid = p.saturating_add(1);
        let s_mid = map_two(mid, Side::Start);
        let e_mid = map_two(mid, Side::End);
        if s_mid != map(out_edits, mid, Side::Start, |e| e.parent, |e| e.child)
            || e_mid != map(out_edits, mid, Side::End, |e| e.parent, |e| e.child)
        {
            differs = true;
        }
        let in_edit = k + 1 < pts.len()
            && out_edits
                .iter()
                .any(|e| e.parent.start <= p && pts[k + 1] <= e.parent.end && e.parent.start < e.parent.end);
        let after = if in_edit {
            After::Constant { start: s_mid, end: e_mid }
        } else {
            After::Shift { base: s_mid.saturating_sub(1) }
        };
        bps.push(Breakpoint { pos: p, at_start, at_end, after });
    }
    if differs { bps } else { Vec::new() }
}

fn build_backward_breakpoints(ab: &Alignment, bc: &Alignment, out_edits: &[Edit]) -> Vec<Breakpoint> {
    let mut pts: Vec<u32> = Vec::new();
    for e in out_edits {
        pts.push(e.child.start);
        pts.push(e.child.end);
    }
    for e in &bc.edits {
        pts.push(e.child.start);
        pts.push(e.child.end);
    }
    for bp in &bc.backward {
        pts.push(bp.pos);
    }
    for e in &ab.edits {
        for b in [e.child.start, e.child.end] {
            pts.push(bc.to_child(b, Side::Start));
            pts.push(bc.to_child(b, Side::End));
        }
    }
    for bp in &ab.backward {
        pts.push(bc.to_child(bp.pos, Side::Start));
        pts.push(bc.to_child(bp.pos, Side::End));
    }
    pts.sort_unstable();
    pts.dedup();

    let map_two = |z: u32, side: Side| ab.to_parent(bc.to_parent(z, side), side);
    let mut bps = Vec::with_capacity(pts.len());
    let mut differs = false;

    for (k, &q) in pts.iter().enumerate() {
        let at_start = map_two(q, Side::Start);
        let at_end = map_two(q, Side::End);
        if at_start != map(out_edits, q, Side::Start, |e| e.child, |e| e.parent)
            || at_end != map(out_edits, q, Side::End, |e| e.child, |e| e.parent)
        {
            differs = true;
        }
        let mid = q.saturating_add(1);
        let s_mid = map_two(mid, Side::Start);
        let e_mid = map_two(mid, Side::End);
        if s_mid != map(out_edits, mid, Side::Start, |e| e.child, |e| e.parent)
            || e_mid != map(out_edits, mid, Side::End, |e| e.child, |e| e.parent)
        {
            differs = true;
        }
        let in_edit = k + 1 < pts.len()
            && out_edits.iter().any(|e| e.child.start <= q && pts[k + 1] <= e.child.end && e.child.start < e.child.end);
        let after = if in_edit {
            After::Constant { start: s_mid, end: e_mid }
        } else {
            After::Shift { base: s_mid.saturating_sub(1) }
        };
        bps.push(Breakpoint { pos: q, at_start, at_end, after });
    }
    if differs { bps } else { Vec::new() }
}

pub(super) fn eval_breakpoints(bps: &[Breakpoint], x: u32, side: Side) -> u32 {
    match bps.binary_search_by_key(&x, |b| b.pos) {
        Ok(i) => match side {
            Side::Start => bps[i].at_start,
            Side::End => bps[i].at_end,
        },
        Err(0) => x,
        Err(i) => {
            let prev = &bps[i - 1];
            match prev.after {
                After::Shift { base } => base + (x - prev.pos),
                After::Constant { start, end } => match side {
                    Side::Start => start,
                    Side::End => end,
                },
            }
        }
    }
}
