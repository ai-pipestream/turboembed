//! Merging entity layers: what OpenNLP's `TokenNameFinderTool` does with
//! the names of several finders. It adds every finder's names to one
//! list, in finder order, and keeps what `NameFinderME.dropOverlappingSpans`
//! keeps:
//!
//! 1. The names are sorted, stably, by `Span.compareTo`: by first token,
//!    then longest first, then by type with `String.compareTo` (UTF-16
//!    order).
//! 2. Going through them in that order, a name that intersects the last
//!    name kept is dropped.
//!
//! `Span.intersects` counts containment, so an empty name at the end of a
//! kept name (`[3, 3)` after `[1, 3)`) is dropped as inside it, while two
//! names that only touch (`[1, 3)` and `[3, 5)`) are both kept. The same
//! name from two finders is kept once, from the first finder, with its
//! score.

use std::cmp::Ordering;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, FilterRule, Refs, Spans};
use turbo_analysis_layers::{Entities, Tag, TagSet};
use turbo_jvm::chars::compare_to;

/// `Span.compareTo` on token spans and types.
fn compare(a: (u32, u32, &str), b: (u32, u32, &str)) -> Ordering {
    a.0.cmp(&b.0).then(b.1.cmp(&a.1)).then_with(|| compare_to(a.2, b.2).cmp(&0))
}

/// `Span.intersects`.
fn intersects(a: (u32, u32), b: (u32, u32)) -> bool {
    let contains = |x: (u32, u32), y: (u32, u32)| x.0 <= y.0 && y.1 <= x.1;
    contains(a, b) || contains(b, a) || (a.0 <= b.0 && b.0 < a.1) || (b.0 <= a.0 && a.0 < b.1)
}

/// `NameFinderME.dropOverlappingSpans` on spans given as token start,
/// token end and type: the indices of the spans it keeps, in its order.
pub fn drop_overlapping(spans: &[(u32, u32, &str)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by(|&a, &b| compare(spans[a], spans[b]));
    let mut last: Option<(u32, u32)> = None;
    order.retain(|&i| {
        let (start, end, _) = spans[i];
        if last.is_some_and(|l| intersects(l, (start, end))) {
            return false;
        }
        last = Some((start, end));
        true
    });
    order
}

/// The entities of `layers` taken together, as `drop_overlapping` leaves
/// them, in its order. Every layer must be in the same view and made on
/// the same tokens; the types are the union of theirs, in order of first
/// appearance.
pub fn merge(layers: &[&Entities]) -> Result<Entities, String> {
    let Some(first) = layers.first() else {
        return Err("nothing to merge".into());
    };
    let view = first.spans.view();
    if layers.iter().any(|l| l.spans.view() != view) {
        return Err("entities in different views".into());
    }
    let labels: Vec<&str> =
        layers.iter().flat_map(|l| (0..l.kinds.len()).map(|i| l.kinds.label(Tag(i as u16)))).collect();
    let mut unique: Vec<&str> = Vec::with_capacity(labels.len());
    for label in labels {
        if !unique.contains(&label) {
            unique.push(label);
        }
    }
    if unique.len() > u16::MAX as usize + 1 {
        return Err("more than 65536 entity types".into());
    }
    let kinds = TagSet::new(&first.kinds.name, &unique);
    // (layer, entity) pairs and their spans, in finder order.
    let items: Vec<(usize, usize)> =
        layers.iter().enumerate().flat_map(|(l, layer)| (0..layer.spans.len()).map(move |e| (l, e))).collect();
    let spans: Vec<(u32, u32, &str)> = items
        .iter()
        .map(|&(l, e)| {
            let layer = layers[l];
            let r = layer.tokens[e];
            (r.start.get() as u32, r.end.get() as u32, layer.kinds.label(layer.kind[e]))
        })
        .collect();
    let kept = drop_overlapping(&spans);
    let mut out = Entities {
        spans: Spans::with_capacity(view, kept.len()),
        tokens: Vec::with_capacity(kept.len()),
        kind: Vec::with_capacity(kept.len()),
        kinds: Arc::clone(&kinds),
        score: Vec::with_capacity(kept.len()),
    };
    for (l, e) in kept.into_iter().map(|i| items[i]) {
        let layer = layers[l];
        let ix = turbo_analysis::Ix::new(e as u32);
        if layer.spans.is_discontinuous(ix) {
            let frags: Vec<turbo_analysis::Span> = layer.spans.fragments(ix).collect();
            out.spans.push_fragments(&frags);
        } else {
            out.spans.push(layer.spans.get(ix));
        }
        out.tokens.push(layer.tokens[e]);
        out.kind.push(kinds.tag(layer.kinds.label(layer.kind[e])).expect("every label is in the union"));
        out.score.push(layer.score[e]);
    }
    Ok(out)
}

/// Resolves overlapping or nested entities in an [`Entities`] layer into
/// a non-overlapping [`Entities`] layer in text order according to a
/// [`FilterRule`]. Bind the input and output instances with
/// [`turbo_analysis::Bound`].
#[derive(Copy, Clone, Debug)]
pub struct FilterEntities {
    pub rule: FilterRule,
}

impl FilterEntities {
    pub const fn new(rule: FilterRule) -> FilterEntities {
        FilterEntities { rule }
    }
}

impl Annotator for FilterEntities {
    const ID: &'static str = "turbo-namefind:filter";
    type Reads = (Entities,);
    type Writes = (Entities,);

    fn annotate(&self, (entities,): Refs<'_, Self>) -> Result<(Entities,), Error> {
        Ok((entities.filter(self.rule),))
    }

    fn config_hash(&self) -> u64 {
        match self.rule {
            FilterRule::LongestFirst => 1,
            FilterRule::HighestScore => 2,
            FilterRule::LeftmostLongest => 3,
        }
    }
}

/// Merges `N` instances of `opennlp:entities` into one, as
/// [`merge`] does. Bind the instances it reads with `Bound::read`, in
/// finder order: the first wins a tie.
#[derive(Copy, Clone, Debug, Default)]
pub struct MergeEntities<const N: usize>;

macro_rules! merge_annotator {
    ($n:literal, $($i:tt),+) => {
        impl Annotator for MergeEntities<$n> {
            const ID: &'static str = "turbo-namefind:merge";
            type Reads = ($(merge_annotator!(@entities $i),)+);
            type Writes = (Entities,);

            fn annotate(&self, input: Refs<'_, Self>) -> Result<(Entities,), Error> {
                let layers = [$(input.$i),+];
                merge(&layers).map(|e| (e,)).map_err(|detail| Error::Annotator { annotator: Self::ID, detail })
            }
        }
    };
    (@entities $i:tt) => { Entities };
}

merge_annotator!(2, 0, 1);
merge_annotator!(3, 0, 1, 2);
merge_annotator!(4, 0, 1, 2, 3);
merge_annotator!(5, 0, 1, 2, 3, 4);
merge_annotator!(6, 0, 1, 2, 3, 4, 5);

#[cfg(test)]
mod tests {
    use super::*;

    fn dropped(spans: &[(u32, u32, &str)]) -> Vec<(u32, u32, String)> {
        drop_overlapping(spans).into_iter().map(|i| (spans[i].0, spans[i].1, spans[i].2.to_string())).collect()
    }

    #[test]
    fn longest_first_then_type_order() {
        let got = dropped(&[(2, 3, "b"), (1, 3, "x"), (1, 4, "a"), (4, 5, "c"), (4, 5, "B")]);
        assert_eq!(got, [(1, 4, "a".into()), (4, 5, "B".into())]);
    }

    #[test]
    fn touching_names_stay_and_empty_ones_inside_go() {
        let got = dropped(&[(1, 3, "a"), (3, 5, "a"), (3, 3, "a"), (5, 5, "a"), (6, 6, "a")]);
        assert_eq!(got, [(1, 3, "a".into()), (3, 5, "a".into()), (6, 6, "a".into())]);
    }

    #[test]
    fn types_compare_by_utf16_units() {
        // U+1F600 sorts after U+FF21 by code point, before it by unit.
        let got = dropped(&[(0, 1, "\u{FF21}"), (0, 1, "\u{1F600}")]);
        assert_eq!(got, [(0, 1, "\u{1F600}".into())]);
    }
}
