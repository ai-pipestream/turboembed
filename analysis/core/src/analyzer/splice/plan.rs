//! How one partition layer cuts the base and the new document: which new
//! partitions are copies of base partitions, which are dirty, and the
//! windows of dirty partitions the local steps run again on.

use std::collections::HashMap;
use std::ops::Range;

use crate::doc::Document;
use crate::layer::{Anchor, KeyId};
use crate::span::Span;
use crate::splice::{partition_ranges, partitions};
use crate::text::{Alignment, ViewId};

use super::Splicers;

/// A run of dirty new partitions, re-run on a document of their text.
pub(super) struct Window {
    /// The new partitions it covers.
    pub parts: Range<usize>,
    /// Its text in the new document.
    pub span: Span,
    /// The document the local steps run on: the window's text, seeded
    /// with the partition layer's slice.
    pub doc: Document,
}

pub(super) struct Plan {
    /// The base document's partitions, in text order.
    pub base: Vec<Span>,
    /// The new document's partitions, in text order.
    pub new: Vec<Span>,
    /// Per new partition, the base partition it is an untouched copy of.
    pub clean: Vec<Option<usize>>,
    /// Per new partition, its window when it is dirty.
    pub window_of: Vec<Option<usize>>,
    pub windows: Vec<Window>,
    /// Per layer on the base document, its items in each base partition.
    base_ranges: HashMap<KeyId, Option<Vec<Range<usize>>>>,
    /// Per layer on the new document, its items in each new partition.
    new_ranges: HashMap<KeyId, Option<Vec<Range<usize>>>>,
}

impl Plan {
    /// Cuts both documents by partition layer `p` and classifies the new
    /// partitions. The windows' documents hold the text only; the caller
    /// seeds them.
    pub fn build(
        p: KeyId,
        base: &Document,
        new: &Document,
        alignment: &Alignment,
        splicers: &Splicers,
    ) -> Result<Plan, String> {
        let sp = splicers.get(&p.ty).ok_or_else(|| format!("{p} has no splicer"))?;
        if sp.anchor() != Anchor::Text {
            return Err(format!("{p} is not anchored to text"));
        }
        let cut = |doc: &Document, what: &str| -> Result<Vec<Span>, String> {
            let value = doc.erased(p).ok_or_else(|| format!("{p} is not on the {what} document"))?;
            let ts = sp.text_spans(value).ok_or_else(|| format!("{p} has no spans"))?;
            if ts.view != ViewId::SOURCE {
                return Err(format!("{p} is in view {}, not the source", ts.view.0));
            }
            partitions(&ts).ok_or_else(|| format!("the items of {p} do not nest"))
        };
        let base_parts = cut(base, "base")?;
        let new_parts = cut(new, "new")?;
        let clean = classify(&base_parts, &new_parts, alignment);
        let text = new.get::<crate::doc::View>().expect("a source view").text();
        let mut windows = Vec::new();
        let mut window_of = vec![None; new_parts.len()];
        let mut j = 0;
        while j < new_parts.len() {
            if clean[j].is_some() {
                j += 1;
                continue;
            }
            let a = j;
            while j < new_parts.len() && clean[j].is_none() {
                window_of[j] = Some(windows.len());
                j += 1;
            }
            let span = Span::new(new_parts[a].start, new_parts[j - 1].end);
            windows.push(Window { parts: a..j, span, doc: Document::new(&text[span.range()]) });
        }
        Ok(Plan {
            base: base_parts,
            new: new_parts,
            clean,
            window_of,
            windows,
            base_ranges: HashMap::new(),
            new_ranges: HashMap::new(),
        })
    }

    /// The items of `key` on the base document in each base partition.
    pub fn base_ranges(&mut self, key: KeyId, base: &Document, splicers: &Splicers) -> Option<&[Range<usize>]> {
        if !self.base_ranges.contains_key(&key) {
            let r = ranges(base, key, &self.base, splicers);
            self.base_ranges.insert(key, r);
        }
        self.base_ranges.get(&key).and_then(|r| r.as_deref())
    }

    /// The items of `key` on the new document in each new partition. The
    /// layer must be complete on the new document when first asked for.
    pub fn new_ranges(&mut self, key: KeyId, new: &Document, splicers: &Splicers) -> Option<&[Range<usize>]> {
        if !self.new_ranges.contains_key(&key) {
            let r = ranges(new, key, &self.new, splicers);
            self.new_ranges.insert(key, r);
        }
        self.new_ranges.get(&key).and_then(|r| r.as_deref())
    }
}

/// The items of `key` on `doc` inside each of `parts`: by its own spans
/// for a text-anchored layer, by its anchor's for an aligned one. None
/// when the layer, or its anchor, is not spliceable by these partitions.
fn ranges(doc: &Document, key: KeyId, parts: &[Span], splicers: &Splicers) -> Option<Vec<Range<usize>>> {
    let sp = splicers.get(&key.ty)?;
    let value = doc.erased(key)?;
    match sp.anchor() {
        Anchor::Text => {
            let ts = sp.text_spans(value)?;
            if ts.view != ViewId::SOURCE {
                return None;
            }
            partition_ranges(&ts, parts)
        }
        Anchor::Aligned { .. } => {
            let anchor = *doc.made_on(key).first()?;
            ranges(doc, anchor, parts, splicers)
        }
        Anchor::Document => None,
    }
}

/// Per new partition, the base partition it is an untouched copy of. A
/// base partition is touched by an edit whose parent span shares a byte
/// with it, or by an insertion at any offset from its start to its end;
/// a touched partition dirties its neighbours too. An untouched base
/// partition is a copy only when its span, mapped through the
/// alignment, is exactly one new partition's span.
fn classify(base: &[Span], new: &[Span], alignment: &Alignment) -> Vec<Option<usize>> {
    let n = base.len();
    let mut touched = vec![false; n];
    for e in alignment.edits() {
        let (s, t) = (e.parent.start, e.parent.end);
        if s == t {
            let mut i = base.partition_point(|p| p.end < s);
            while i < n && base[i].start <= s {
                touched[i] = true;
                i += 1;
            }
        } else {
            let mut i = base.partition_point(|p| p.end <= s);
            while i < n && base[i].start < t {
                touched[i] = true;
                i += 1;
            }
        }
    }
    let mut clean = vec![None; new.len()];
    let mut cursor = alignment.cursor();
    let mut from = 0;
    for i in 0..n {
        if touched[i] || (i > 0 && touched[i - 1]) || (i + 1 < n && touched[i + 1]) {
            continue;
        }
        let s = cursor.span_to_child(base[i]);
        let j = from + new[from..].partition_point(|q| q.start < s.start);
        if j < new.len() && new[j] == s {
            clean[j] = Some(i);
            from = j + 1;
        }
    }
    clean
}
