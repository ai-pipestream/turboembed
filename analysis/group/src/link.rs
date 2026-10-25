//! Linking sketched items: exact copies by their word sequence, near
//! duplicates and versions by MinHash banding, and the groups they make.
//! Both the one-document annotator and the batch function run this over
//! a flat list of items.

use crate::hash::{self, BANDS};
use turbo_analysis::Fnv1a;

/// How two linked items relate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DuplicateKind {
    /// The same words in the same order.
    Exact,
    /// Estimated Jaccard similarity at least `Thresholds::near`.
    NearDuplicate,
    /// Estimated Jaccard similarity at least `Thresholds::version` and
    /// below `Thresholds::near`: an edited version.
    VersionEdit,
}

impl DuplicateKind {
    /// The labels of the kinds, in tag order, for an edge layer's tag set.
    pub const LABELS: [&'static str; 3] = ["exact", "near-duplicate", "version-edit"];

    pub fn label(self) -> &'static str {
        Self::LABELS[self as usize]
    }
}

/// When two items link.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Thresholds {
    /// The least estimated Jaccard similarity of near duplicates: 0.9.
    pub near: f64,
    /// The least of versions: 0.5.
    pub version: f64,
    /// Items with fewer words are not linked: 1, so only empty items are
    /// left out.
    pub min_words: u32,
}

impl Default for Thresholds {
    fn default() -> Thresholds {
        Thresholds { near: 0.9, version: 0.5, min_words: 1 }
    }
}

impl Thresholds {
    pub(crate) fn check(&self) -> Result<(), String> {
        let unit = 0.0..=1.0;
        if !unit.contains(&self.near) || !unit.contains(&self.version) || self.version > self.near {
            return Err(format!(
                "thresholds version {} and near {} not in 0 <= version <= near <= 1",
                self.version, self.near
            ));
        }
        Ok(())
    }

    pub(crate) fn hash(&self) -> u64 {
        [self.near.to_bits(), self.version.to_bits(), u64::from(self.min_words)]
            .iter()
            .fold(Fnv1a::new(), |mut h, &x| {
                h.u64(x);
                h
            })
            .finish()
    }
}

/// One item to link: which document, and its envelope there, so items
/// that overlap in one document (a list item and its paragraph, a chunk
/// and its parent) are never linked to each other.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Entry<'s> {
    pub doc: u32,
    pub start: u32,
    pub end: u32,
    pub words: u32,
    pub exact: u128,
    pub row: &'s [u32],
}

/// A link from entry `from` to the later entry `to`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct Link {
    pub from: usize,
    pub to: usize,
    pub kind: DuplicateKind,
    pub score: f64,
}

fn overlap(a: &Entry, b: &Entry) -> bool {
    a.doc == b.doc && a.start < b.end && b.start < a.end
}

/// The links and groups among `entries`, which are in (document, start)
/// order: the callers refuse items out of start order.
///
/// Items of equal word sequence form an exact class; a copy that overlaps
/// a copy already in its class in the same document (nested blocks with
/// one text) is the same text, not a duplicate, and is dropped from the
/// class. The first copy of each class links to every later one as
/// `Exact`, score 1. The first copies are then banded on their MinHash
/// rows: any two that share a band and do not overlap are compared, and
/// linked from the earlier to the later as `NearDuplicate` or
/// `VersionEdit` with their estimated Jaccard similarity as the score.
/// A group is a set of at least two items joined by exact and near
/// duplicate links (not versions), listed in entry order; groups are in
/// the order of their first member.
pub(crate) fn link(entries: &[Entry], t: &Thresholds) -> (Vec<Link>, Vec<Vec<usize>>) {
    let n = entries.len();
    let mut links = Vec::new();
    let mut parent: Vec<usize> = (0..n).collect();
    // Exact classes: entries of equal sequence, in entry order.
    let mut order: Vec<usize> = (0..n).filter(|&i| entries[i].words >= t.min_words.max(1)).collect();
    order.sort_unstable_by_key(|&i| (entries[i].exact, i));
    let mut firsts = Vec::new();
    let mut class: Vec<usize> = Vec::new();
    for run in order.chunk_by(|&a, &b| entries[a].exact == entries[b].exact) {
        class.clear();
        for &i in run {
            // Kept copies of one document do not overlap and are in start
            // order, so only the last can overlap a later one.
            if !class.last().is_some_and(|&k| overlap(&entries[k], &entries[i])) {
                class.push(i);
            }
        }
        let first = class[0];
        firsts.push(first);
        for &i in &class[1..] {
            links.push(Link { from: first, to: i, kind: DuplicateKind::Exact, score: 1.0 });
            union(&mut parent, first, i);
        }
    }
    firsts.sort_unstable();
    // Candidate pairs among the first copies: those sharing a band.
    let mut keyed: Vec<(u64, u32)> = Vec::with_capacity(firsts.len() * BANDS);
    for (k, &i) in firsts.iter().enumerate() {
        for b in 0..BANDS {
            keyed.push((hash::band(entries[i].row, b), k as u32));
        }
    }
    keyed.sort_unstable();
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for run in keyed.chunk_by(|a, b| a.0 == b.0) {
        for (x, a) in run.iter().enumerate() {
            for b in &run[x + 1..] {
                if a.1 != b.1 {
                    pairs.push((a.1.min(b.1), a.1.max(b.1)));
                }
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();
    for (a, b) in pairs {
        let (i, j) = (firsts[a as usize], firsts[b as usize]);
        if overlap(&entries[i], &entries[j]) {
            continue;
        }
        let score = hash::jaccard(entries[i].row, entries[j].row);
        let kind = if score >= t.near {
            union(&mut parent, i, j);
            DuplicateKind::NearDuplicate
        } else if score >= t.version {
            DuplicateKind::VersionEdit
        } else {
            continue;
        };
        links.push(Link { from: i, to: j, kind, score });
    }
    links.sort_unstable_by_key(|l| (l.from, l.to, l.kind));
    (links, groups(&mut parent))
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

/// Joins two sets, keeping the smaller index as the root, so a set's
/// root is its first member.
fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra.max(rb)] = ra.min(rb);
    }
}

fn groups(parent: &mut [usize]) -> Vec<Vec<usize>> {
    let n = parent.len();
    let mut by_root: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut sizes = vec![0u32; n];
    for i in 0..n {
        sizes[find(parent, i)] += 1;
    }
    for i in 0..n {
        let r = find(parent, i);
        if sizes[r] > 1 {
            by_root[r].push(i);
        }
    }
    // A root is its set's first member, so roots in index order list the
    // groups by their first member.
    by_root.into_iter().filter(|g| !g.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::LANES;

    fn entry(doc: u32, start: u32, end: u32, exact: u128, row: &[u32]) -> Entry<'_> {
        Entry { doc, start, end, words: 5, exact, row }
    }

    #[test]
    fn nested_copies_of_one_text_are_not_duplicates() {
        let row = [7u32; LANES];
        // A list item and the paragraph inside it, then a real copy.
        let e = [entry(0, 0, 10, 1, &row), entry(0, 2, 10, 1, &row), entry(0, 20, 30, 1, &row)];
        let (links, groups) = link(&e, &Thresholds::default());
        assert_eq!(links, [Link { from: 0, to: 2, kind: DuplicateKind::Exact, score: 1.0 }]);
        assert_eq!(groups, [vec![0, 2]]);
    }

    #[test]
    fn near_links_join_groups_and_version_links_do_not() {
        let a = [1u32; LANES];
        let mut b = a;
        b[0] = 2; // 63 of 64 lanes agree
        let mut c = a;
        c[..24].fill(3); // 40 of 64 agree with a
        let e = [entry(0, 0, 1, 1, &a), entry(0, 2, 3, 2, &b), entry(1, 0, 1, 3, &c)];
        let (links, groups) = link(&e, &Thresholds::default());
        let kinds: Vec<_> = links.iter().map(|l| (l.from, l.to, l.kind)).collect();
        assert_eq!(
            kinds,
            [
                (0, 1, DuplicateKind::NearDuplicate),
                (0, 2, DuplicateKind::VersionEdit),
                (1, 2, DuplicateKind::VersionEdit)
            ]
        );
        assert_eq!(links[0].score, 63.0 / 64.0);
        assert_eq!(groups, [vec![0, 1]]);
    }

    #[test]
    fn empty_items_link_to_nothing() {
        let row = [u32::MAX; LANES];
        let mut e = [entry(0, 0, 0, 9, &row), entry(0, 1, 1, 9, &row)];
        e.iter_mut().for_each(|x| x.words = 0);
        assert_eq!(link(&e, &Thresholds::default()), (Vec::new(), Vec::new()));
    }

    #[test]
    fn thresholds_out_of_order_are_refused() {
        assert!(Thresholds { near: 0.4, version: 0.5, min_words: 1 }.check().is_err());
        assert!(Thresholds { near: f64::NAN, ..Thresholds::default() }.check().is_err());
        assert!(Thresholds::default().check().is_ok());
    }
}
