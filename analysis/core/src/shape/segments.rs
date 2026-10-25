use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn segment<'a>(
    window: &DocumentWindow,
    layers: impl IntoIterator<Item = (usize, TextSpans<'a>, Range<usize>)>,
) -> WindowShape {
    let (ws, we) = (window.span.start, window.span.end);
    // Clipped non-empty fragments as sweep events, plus zero-width
    // items by position.
    let mut starts: BTreeMap<u32, Vec<ShapeMember>> = BTreeMap::new();
    let mut ends: BTreeMap<u32, Vec<ShapeMember>> = BTreeMap::new();
    let mut cuts: Vec<u32> = vec![ws, we];
    let mut zero: BTreeMap<u32, BTreeSet<ShapeMember>> = BTreeMap::new();
    for (layer_ix, ts, items) in layers {
        for item in items {
            #[cfg(test)]
            super::ITEM_VISITS.with(|n| n.set(n.get() + 1));
            let member = ShapeMember { layer: layer_ix, item: item as u32 };
            if ts.frag_item_end.is_empty() {
                collect(ts.start[item], ts.end[item], member, ws, we, &mut starts, &mut ends, &mut cuts, &mut zero);
            } else {
                let r = ts.item_frag_range(item);
                let kept = prepared::ordered_range(&ts.frag_start[r.clone()], &ts.frag_end[r.clone()], window.span);
                for f in (r.start + kept.start)..(r.start + kept.end) {
                    #[cfg(test)]
                    super::FRAGMENT_VISITS.with(|n| n.set(n.get() + 1));
                    collect(
                        ts.frag_start[f],
                        ts.frag_end[f],
                        member,
                        ws,
                        we,
                        &mut starts,
                        &mut ends,
                        &mut cuts,
                        &mut zero,
                    );
                }
            }
        }
    }
    cuts.sort_unstable();
    cuts.dedup();

    let mut shape = WindowShape { window: *window, segments: Vec::new(), members: Vec::new() };
    let mut active: BTreeSet<ShapeMember> = BTreeSet::new();
    let mut zero_between = false;
    for (i, &at) in cuts.iter().enumerate() {
        if let Some(set) = zero.get(&at) {
            let from = shape.members.len();
            shape.members.extend(set.iter().copied());
            shape.segments.push(ShapeSegment { span: Span { start: at, end: at }, members: from..shape.members.len() });
            zero_between = true;
        }
        if let Some(gone) = ends.get(&at) {
            for m in gone {
                active.remove(m);
            }
        }
        if let Some(fresh) = starts.get(&at) {
            for m in fresh {
                active.insert(*m);
            }
        }
        if i + 1 < cuts.len() {
            let next = cuts[i + 1];
            let candidate: Vec<ShapeMember> = active.iter().copied().collect();
            let extends = !zero_between
                && shape
                    .segments
                    .last()
                    .is_some_and(|last| last.span.end == at && shape.members[last.members.clone()] == candidate[..]);
            if extends {
                shape.segments.last_mut().expect("checked above").span.end = next;
            } else {
                let from = shape.members.len();
                shape.members.extend(candidate);
                shape
                    .segments
                    .push(ShapeSegment { span: Span { start: at, end: next }, members: from..shape.members.len() });
            }
            zero_between = false;
        }
    }
    shape
}

#[allow(clippy::too_many_arguments)]
fn collect(
    start: u32,
    end: u32,
    member: ShapeMember,
    ws: u32,
    we: u32,
    starts: &mut BTreeMap<u32, Vec<ShapeMember>>,
    ends: &mut BTreeMap<u32, Vec<ShapeMember>>,
    cuts: &mut Vec<u32>,
    zero: &mut BTreeMap<u32, BTreeSet<ShapeMember>>,
) {
    if start == end {
        if (ws..=we).contains(&start) {
            zero.entry(start).or_default().insert(member);
            cuts.push(start);
        }
        return;
    }
    if end <= ws || start >= we {
        return;
    }
    let (s, e) = (start.max(ws), end.min(we));
    starts.entry(s).or_default().push(member);
    ends.entry(e).or_default().push(member);
    cuts.push(s);
    cuts.push(e);
}
