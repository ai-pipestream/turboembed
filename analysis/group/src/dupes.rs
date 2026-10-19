//! Near-duplicate grouping and version links, inside one document as
//! layers and across a batch of documents as a result of its own.

use std::marker::PhantomData;
use std::sync::{Arc, OnceLock};

use turbo_analysis::{Annotator, Edge, Edges, Error, Groups, Ix, Layer, Refs, Tag, TagSet};

use crate::link::{DuplicateKind, Entry, Link, Thresholds, link};
use crate::sketch::Sketches;

/// The tag set of a duplicate edge layer's kinds,
/// `turbo:duplicate-kinds`: `exact`, `near-duplicate`, `version-edit`,
/// in `DuplicateKind` order. One shared set, so every edge layer of
/// duplicates carries the same one.
pub fn duplicate_kinds() -> Arc<TagSet> {
    static KINDS: OnceLock<Arc<TagSet>> = OnceLock::new();
    KINDS.get_or_init(|| TagSet::new("turbo:duplicate-kinds", &DuplicateKind::LABELS)).clone()
}

/// The entries of one document's sketched items, in item order. Refused
/// when the items are not in start order or have no spans.
fn entries<'s, L: Layer>(doc: u32, units: &L, sketches: &'s Sketches<L>) -> Result<Vec<Entry<'s>>, String> {
    let ts = units.text_spans().ok_or_else(|| format!("{} has no spans", L::NAME))?;
    if !ts.is_start_ordered() {
        return Err(format!("the items of {} are not in start order", L::NAME));
    }
    if sketches.words.len() != ts.start.len() {
        return Err(format!("{} sketches for {} items", sketches.words.len(), ts.start.len()));
    }
    sketches.check()?;
    Ok((0..ts.start.len())
        .map(|i| Entry {
            doc,
            start: ts.start[i],
            end: ts.end[i],
            words: sketches.words[i],
            exact: sketches.exact[i],
            row: sketches.row(Ix::new(i as u32)),
        })
        .collect())
}

/// Reads the items of `L` and their sketches; writes the near-duplicate
/// groups among them (`Groups<L>`) and the links between them
/// (`Edges<L, L>`, kinds `duplicate_kinds`).
///
/// Every link runs from an earlier item to a later one, so a version
/// link points from the older text to the newer, `Edges::outgoing` lists
/// an item's later copies and versions and `Edges::incoming` its earlier
/// ones, each by two binary searches. Exact copies link from the first
/// copy to each later one with score 1; near duplicates and versions link
/// first copies with their estimated Jaccard similarity. Items that
/// overlap (a block and a block nested in it, a chunk and its parent)
/// never link. A group holds the items joined by exact and near-duplicate
/// links; versions stay links. Time is linear in the items plus the
/// candidate pairs, which are the pairs that share one of the 16 bands
/// of their MinHash rows.
pub struct NearDuplicates<L> {
    pub thresholds: Thresholds,
    _of: PhantomData<fn() -> L>,
}

impl<L> NearDuplicates<L> {
    pub fn new(thresholds: Thresholds) -> NearDuplicates<L> {
        NearDuplicates { thresholds, _of: PhantomData }
    }
}

impl<L> Default for NearDuplicates<L> {
    fn default() -> NearDuplicates<L> {
        NearDuplicates::new(Thresholds::default())
    }
}

impl<L: Layer> Annotator for NearDuplicates<L> {
    const ID: &'static str = "turbo:near-duplicates";
    type Reads = (L, Sketches<L>);
    type Writes = (Groups<L>, Edges<L, L>);

    fn annotate(&self, (units, sketches): Refs<'_, Self>) -> Result<(Groups<L>, Edges<L, L>), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        self.thresholds.check().map_err(fail)?;
        let e = entries(0, units, sketches).map_err(fail)?;
        let (links, groups) = link(&e, &self.thresholds);
        let ix = |i: usize| Ix::<L>::new(i as u32);
        let rows = links.iter().map(|l| Edge::new(ix(l.from), ix(l.to), Tag(l.kind as u16), l.score));
        Ok((Groups::new(groups.iter().map(|g| g.iter().map(|&i| ix(i)))), Edges::new(duplicate_kinds(), rows)))
    }

    fn config_hash(&self) -> u64 {
        self.thresholds.hash()
    }
}

/// An item of one document of a batch.
pub struct Member<L> {
    /// The document's place in the batch.
    pub doc: u32,
    pub item: Ix<L>,
}

impl<L> Clone for Member<L> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<L> Copy for Member<L> {}
impl<L> PartialEq for Member<L> {
    fn eq(&self, other: &Self) -> bool {
        (self.doc, self.item) == (other.doc, other.item)
    }
}
impl<L> std::fmt::Debug for Member<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.doc, self.item.get())
    }
}

/// A link between two items of a batch, from the earlier (by document,
/// then item) to the later.
pub struct CrossLink<L> {
    pub from: Member<L>,
    pub to: Member<L>,
    pub kind: DuplicateKind,
    pub score: f64,
}

/// The duplicates of a batch: links in `(from, to)` order and groups in
/// the order of their first member.
pub struct CrossDuplicates<L> {
    pub links: Vec<CrossLink<L>>,
    pub groups: Vec<Vec<Member<L>>>,
}

impl<L> std::fmt::Debug for CrossLink<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} -> {:?} {} {}", self.from, self.to, self.kind.label(), self.score)
    }
}

impl<L> std::fmt::Debug for CrossDuplicates<L> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossDuplicates").field("links", &self.links).field("groups", &self.groups).finish()
    }
}

impl<L> CrossDuplicates<L> {
    /// The links out of `m`, to later items: two binary searches.
    pub fn outgoing(&self, m: Member<L>) -> &[CrossLink<L>] {
        let key = |l: &CrossLink<L>| (l.from.doc, l.from.item);
        let a = self.links.partition_point(|l| key(l) < (m.doc, m.item));
        let b = a + self.links[a..].partition_point(|l| key(l) == (m.doc, m.item));
        &self.links[a..b]
    }
}

/// The near duplicates and versions among the items of `L` across a
/// batch of documents, each given as its items and their sketches, by
/// the rules of `NearDuplicates` with the batch as one list of items in
/// (document, item) order: links run from earlier documents to later
/// ones, and pairs inside one document are found too. Refused when two
/// documents' sketches use different shingle widths, since their MinHash
/// rows would not be comparable.
pub fn cross_duplicates<L: Layer>(batch: &[(&L, &Sketches<L>)], t: &Thresholds) -> Result<CrossDuplicates<L>, String> {
    t.check()?;
    let Some(width) = batch.first().map(|(_, s)| s.shingle) else {
        return Ok(CrossDuplicates { links: Vec::new(), groups: Vec::new() });
    };
    let mut all = Vec::new();
    let mut at = Vec::new();
    for (d, (units, sketches)) in batch.iter().enumerate() {
        if sketches.shingle != width {
            return Err(format!("document {d} has shingles of {} words, document 0 of {width}", sketches.shingle));
        }
        let e = entries(d as u32, *units, sketches).map_err(|e| format!("document {d}: {e}"))?;
        at.extend((0..e.len() as u32).map(|i| (d as u32, i)));
        all.extend(e);
    }
    let (links, groups) = link(&all, t);
    let member = |i: usize| Member { doc: at[i].0, item: Ix::new(at[i].1) };
    Ok(CrossDuplicates {
        links: links
            .iter()
            .map(|&Link { from, to, kind, score }| CrossLink { from: member(from), to: member(to), kind, score })
            .collect(),
        groups: groups.iter().map(|g| g.iter().map(|&i| member(i)).collect()).collect(),
    })
}
