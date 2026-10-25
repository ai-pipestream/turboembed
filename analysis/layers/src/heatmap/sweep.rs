//! Event-updated item summaries; work does not enumerate active memberships.
use super::{Error, HeatmapLaneKind, HeatmapSegment, Item, KeyId, Span, invalid};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet},
};

#[cfg(test)]
thread_local! { pub(super) static QUERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

#[derive(Clone, Copy)]
pub(super) struct Event {
    pub at: u32,
    pub item: usize,
    pub delta: i32,
}

pub(super) struct Active<'a> {
    items: &'a [Item],
    counts: BTreeMap<usize, usize>,
    first: BTreeSet<(u32, usize)>,
    strongest: BTreeSet<(u32, Reverse<u32>, usize)>,
}
impl<'a> Active<'a> {
    pub(super) fn new(items: &'a [Item]) -> Self {
        Self { items, counts: BTreeMap::new(), first: BTreeSet::new(), strongest: BTreeSet::new() }
    }
    fn update(&mut self, i: usize, delta: i64) -> bool {
        let old = self.counts.get(&i).copied().unwrap_or(0);
        let new = if delta < 0 { old - (-delta) as usize } else { old + delta as usize };
        let item = &self.items[i];
        let score_key = (item.score.abs().to_bits(), Reverse(item.source), i);
        if new == 0 {
            self.counts.remove(&i);
            self.first.remove(&(item.source, i));
            self.strongest.remove(&score_key);
        } else {
            self.counts.insert(i, new);
            self.first.insert((item.source, i));
            self.strongest.insert(score_key);
        }
        (old == 0) != (new == 0)
    }
    pub(super) fn summary(&self, kind: HeatmapLaneKind) -> (f32, u32, Option<u32>) {
        #[cfg(test)]
        QUERIES.with(|n| n.set(n.get() + 1));
        let count = self.counts.len() as u32;
        if kind.is_density() {
            return (count as f32, count, self.first.first().map(|x| x.0));
        }
        self.strongest
            .last()
            .map_or((0.0, 0, None), |&(_, _, i)| (self.items[i].score, count, Some(self.items[i].source)))
    }
}

/// A zero delta is a point item, not part of the adjacent intervals.
fn apply(events: &[Event], active: &mut Active<'_>) -> (bool, Vec<usize>) {
    let mut changed = false;
    let mut points = Vec::new();
    let mut i = 0;
    while i < events.len() {
        let item = events[i].item;
        let mut delta = 0i64;
        while i < events.len() && events[i].item == item {
            if events[i].delta == 0 && item != usize::MAX {
                points.push(item);
            }
            delta += i64::from(events[i].delta);
            i += 1;
        }
        if delta != 0 {
            changed |= active.update(item, delta);
        }
    }
    points.dedup();
    (changed, points)
}

pub(super) fn events(key: KeyId, items: &[Item]) -> Result<Vec<Event>, Error> {
    let n = items
        .iter()
        .try_fold(2usize, |n, it| it.frags.len().checked_mul(2).and_then(|m| n.checked_add(m)))
        .ok_or_else(|| invalid(key, "heatmap event count exceeds usize".into()))?;
    let mut events = Vec::new();
    events.try_reserve_exact(n).map_err(|e| invalid(key, format!("cannot reserve heatmap events: {e}")))?;
    Ok(events)
}

pub(super) fn exact(
    text_len: u32,
    key: KeyId,
    kind: HeatmapLaneKind,
    items: &[Item],
) -> Result<Vec<HeatmapSegment>, Error> {
    let mut events = events(key, items)?;
    for (i, it) in items.iter().enumerate() {
        for &(s, e) in &it.frags {
            if s == e {
                events.push(Event { at: s, item: i, delta: 0 });
            } else {
                events.push(Event { at: s, item: i, delta: 1 });
                events.push(Event { at: e, item: i, delta: -1 });
            }
        }
    }
    for at in [0, text_len] {
        events.push(Event { at, item: usize::MAX, delta: 0 });
    }
    events.sort_unstable_by_key(|e| (e.at, e.item));
    let mut active = Active::new(items);
    let mut out: Vec<HeatmapSegment> = Vec::new();
    let mut i = 0;
    while i < events.len() {
        let at = events[i].at;
        let from = i;
        while i < events.len() && events[i].at == at {
            i += 1;
        }
        let (changed, points) = apply(&events[from..i], &mut active);
        if !points.is_empty() {
            let mut point = Active::new(items);
            for &p in &points {
                point.update(p, 1);
            }
            push(&mut out, key, Span::new(at, at), point.summary(kind))?;
        }
        if i < events.len() {
            let next = events[i].at;
            if !changed && points.is_empty() && out.last().is_some_and(|s| s.span.end == at && !s.span.is_empty()) {
                out.last_mut().expect("checked").span.end = next;
            } else {
                push(&mut out, key, Span::new(at, next), active.summary(kind))?;
            }
        }
    }
    Ok(out)
}

fn push(
    out: &mut Vec<HeatmapSegment>,
    key: KeyId,
    span: Span,
    (score, count, primary_item): (f32, u32, Option<u32>),
) -> Result<(), Error> {
    out.try_reserve(1).map_err(|e| invalid(key, format!("cannot reserve heatmap segments: {e}")))?;
    out.push(HeatmapSegment { span, score, count, primary_item });
    Ok(())
}

pub(super) fn buckets(
    edges: &[u32],
    kind: HeatmapLaneKind,
    items: &[Item],
    mut events: Vec<Event>,
    out: &mut Vec<HeatmapSegment>,
) {
    events.sort_unstable_by_key(|e| (e.at, e.item));
    let mut active = Active::new(items);
    let mut i = 0;
    for b in 0..edges.len() - 1 {
        let from = i;
        while i < events.len() && events[i].at as usize == b {
            i += 1;
        }
        let (_, points) = apply(&events[from..i], &mut active);
        let summary = if edges[b] == edges[b + 1] {
            let mut point = Active::new(items);
            for &p in &points {
                point.update(p, 1);
            }
            point.summary(kind)
        } else {
            for &p in &points {
                active.update(p, 1);
            }
            let summary = active.summary(kind);
            for &p in &points {
                active.update(p, -1);
            }
            summary
        };
        let (score, count, primary_item) = summary;
        out.push(HeatmapSegment { span: Span::new(edges[b], edges[b + 1]), score, count, primary_item });
    }
}

#[cfg(test)]
mod parity;
#[cfg(test)]
mod tests;
