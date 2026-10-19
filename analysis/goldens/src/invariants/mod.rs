//! The offset invariants every layer with spans must keep, checked as
//! properties over any document: on the fuzz corpus, on the golden
//! inputs, on anything an annotator produced.
//!
//! For each layer, its own contract (`Layer::check`: sentences and
//! subwords in text order and not overlapping, a sentence for every token
//! in sentence order, blocks a tree in text order, and so on) holds. For
//! each span of a text anchored layer, in the view its spans are in:
//!
//! - it is in bounds, ends at or after its start, and both ends are on
//!   character boundaries of the view's text;
//! - UTF-8 to UTF-16 to UTF-8 round trips through the view's
//!   `OffsetTable`, and so do code point offsets; the UTF-16 offsets agree
//!   with `turbo_unicode`'s own conversion, which is written apart from it;
//! - it maps to source bytes exactly through every normalizer between the
//!   view and the source: the source span is on character boundaries,
//!   mapping it forward again covers the span, mapping that back gives the
//!   same source span, and where the span touches no edit of the view's
//!   alignment the source bytes are the view's bytes (this last check
//!   needs the view's own alignment, so it runs for a view the document
//!   holds as a layer, not for one it holds only in its chain).
//!
//! `check_document` finds every instance of the shared layer types that
//! have text spans (`known_layers`); `check_spans` takes any span columns.
//! A `Violation` names the layer, the span and what failed, so a test can
//! print the list and stop.

#[cfg(test)]
mod tests;

use std::any::TypeId;
use std::fmt;

use turbo_analysis::{Document, Key, Layer, Span, View, ViewId};
use turbo_analysis_layers::{
    Chunks, Entities, Matches, Money, Numbers, PhraseChunks, Pii, Quantities, Sentences, Structure, Subwords,
    Temporals, Tokens,
};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

/// One broken invariant: the layer (as the document names it), the view
/// and span it was found on, and what failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Violation {
    pub layer: String,
    pub view: ViewId,
    pub index: usize,
    pub span: Span,
    pub what: String,
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} item {} at {}..{} in view {}: {}",
            self.layer, self.index, self.span.start, self.span.end, self.view.0, self.what
        )
    }
}

/// Checks the spans of one layer of `doc`: `starts[i]..ends[i]` in `view`.
pub fn check_spans(doc: &Document, layer: &str, view: ViewId, starts: &[u32], ends: &[u32]) -> Vec<Violation> {
    let mut out = Vec::new();
    let text = doc.text(view);
    let offsets = doc.offsets(view);
    let source = doc.source();
    // The view's own alignment, against its parent, when it is reachable.
    let own = own_view(doc, view);
    let edits: Option<Vec<Span>> = own.map(|v| v.alignment().edits().iter().map(|e| e.child).collect());
    if starts.len() != ends.len() {
        out.push(Violation {
            layer: layer.into(),
            view,
            index: 0,
            span: Span::new(0, 0),
            what: format!("{} starts and {} ends", starts.len(), ends.len()),
        });
        return out;
    }
    for (i, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        // Not `Span::new`: a span ending before its start is one of the
        // things reported here.
        let span = Span { start, end };
        let mut fail = |what: String| out.push(Violation { layer: layer.into(), view, index: i, span, what });
        if start > end || end as usize > text.len() {
            fail(format!("not inside the {} bytes of the view, or ends before it starts", text.len()));
            continue;
        }
        if !text.is_char_boundary(start as usize) || !text.is_char_boundary(end as usize) {
            fail("cuts a character".into());
            continue;
        }
        for (what, b) in [("start", start), ("end", end)] {
            let u = offsets.to_utf16(text, b);
            match offsets.from_utf16(text, u) {
                Ok(back) if back == b => {}
                Ok(back) => fail(format!("{what} {b} goes to UTF-16 {} and back to {back}", u.0)),
                Err(e) => fail(format!("{what} {b} goes to UTF-16 {} and does not come back: {e}", u.0)),
            }
            let independent = byte_to_utf16(text, b as usize);
            if independent != Some(u.0 as usize) {
                fail(format!("{what} {b}: UTF-16 {} by the offset table, {independent:?} by turbo_unicode", u.0));
            }
            if utf16_to_byte(text, u.0 as usize) != Some(b as usize) {
                fail(format!("{what} {b}: UTF-16 {} does not come back through turbo_unicode", u.0));
            }
            let cp = offsets.to_code_point(text, b);
            match offsets.from_code_point(text, cp) {
                Ok(back) if back == b => {}
                Ok(back) => fail(format!("{what} {b} goes to code point {} and back to {back}", cp.0)),
                Err(e) => fail(format!("{what} {b} goes to code point {} and does not come back: {e}", cp.0)),
            }
        }
        let (u16s, u16e) = offsets.span_to_utf16(text, span);
        if u16s.0 > u16e.0 || (u16e.0 - u16s.0) as usize != text[span.range()].encode_utf16().count() {
            fail(format!("the UTF-16 span {}..{} is not the text's length in units", u16s.0, u16e.0));
        }
        // Back to the source and forward again.
        let back = doc.to_source(view, span);
        if back.start > back.end || back.end as usize > source.len() {
            fail(format!("maps to {}..{}, outside the source", back.start, back.end));
            continue;
        }
        if !source.is_char_boundary(back.start as usize) || !source.is_char_boundary(back.end as usize) {
            fail(format!("maps to {}..{}, which cuts a source character", back.start, back.end));
            continue;
        }
        let forward = doc.from_source(view, back);
        if forward.start > start || forward.end < end {
            fail(format!(
                "maps to source {}..{}, which maps forward to {}..{}, not covering it",
                back.start, back.end, forward.start, forward.end
            ));
        }
        let again = doc.to_source(view, forward);
        if again != back {
            fail(format!(
                "maps to source {}..{}, forward to {}..{}, back to {}..{}",
                back.start, back.end, forward.start, forward.end, again.start, again.end
            ));
        }
        // A span touching no edit of the view's own alignment has the same
        // bytes in the parent view, whatever earlier links of the chain
        // changed.
        if let (Some(v), Some(edits)) = (own, &edits)
            && let Some(parent) = v.parent()
            && !touches_an_edit(edits, span)
        {
            let up = v.alignment().span_to_parent(span);
            let parent_text = doc.text(parent);
            if parent_text.get(up.range()) != Some(&text[span.range()]) {
                fail(format!(
                    "touches no edit, yet maps to {}..{} {:?} of its parent view, not its own text {:?}",
                    up.start,
                    up.end,
                    parent_text.get(up.range()),
                    &text[span.range()]
                ));
            }
        }
    }
    out
}

/// The view itself, with its alignment against its parent, when it is a
/// layer of the document. None for the source view, which has no parent,
/// and for a view the document holds only in its chain, whose alignment
/// is not reachable, so the text check is skipped for it.
fn own_view(doc: &Document, view: ViewId) -> Option<&View> {
    if view == ViewId::SOURCE {
        return None;
    }
    let target = Key::<View>::DEFAULT.id().ty;
    for k in doc.layers() {
        if k.ty == target
            && let Some(v) = doc.get_key(&Key::<View>::named(k.name))
            && v.id() == view
        {
            return Some(v);
        }
    }
    None
}

/// Whether `span` overlaps an edit's child bytes, or has a deletion (an
/// edit with no child bytes) strictly inside it: either makes its source
/// bytes differ from its own.
fn touches_an_edit(edits: &[Span], span: Span) -> bool {
    edits.iter().any(|e| {
        if e.start == e.end {
            span.start < e.start && e.start < span.end
        } else {
            e.start < span.end && e.end > span.start
        }
    })
}

/// Checks one instance of a layer type: its own contract, then its text
/// spans if it has them.
pub fn check_layer<L: Layer>(doc: &Document, key: &Key<L>) -> Vec<Violation> {
    let Some(layer) = doc.get_key(key) else { return Vec::new() };
    let mut out = check_contract(layer, &key.id().to_string());
    if let Some(ts) = layer.text_spans() {
        out.extend(check_spans(doc, &key.id().to_string(), ts.view, ts.start, ts.end));
        if !ts.frag_item_end.is_empty() {
            let mut frag_violations =
                check_spans(doc, &format!("{} (fragments)", key.id()), ts.view, ts.frag_start, ts.frag_end);
            for v in &mut frag_violations {
                let item = ts.frag_item_end.partition_point(|&end| (end as usize) <= v.index);
                if item < ts.frag_item_end.len() {
                    v.index = item;
                }
            }
            out.extend(frag_violations);
        }
    }
    out
}

/// The layer's own contract, `Layer::check`, as a violation when it does
/// not hold: the order of sentences, subwords and blocks, a sentence for
/// every token, and whatever else the layer type promises. `Document::add`
/// refuses a layer that breaks it, so this holds a document's layers to
/// what they were added as, and a layer built outside a document to the
/// same rule.
pub fn check_contract<L: Layer>(layer: &L, name: &str) -> Vec<Violation> {
    match layer.check() {
        Ok(()) => Vec::new(),
        Err(what) => {
            let view = layer.text_spans().map_or(ViewId::SOURCE, |ts| ts.view);
            vec![Violation {
                layer: name.into(),
                view,
                index: 0,
                span: Span::new(0, 0),
                what: format!("the layer's contract: {what}"),
            }]
        }
    }
}

type Check = fn(&Document, &'static str) -> Vec<Violation>;

fn checker<L: Layer>() -> (TypeId, Check) {
    (Key::<L>::DEFAULT.id().ty, |doc, name| check_layer(doc, &Key::<L>::named(name)))
}

/// The shared layer types with text spans, which `check_document` looks
/// for on a document.
fn known_layers() -> Vec<(TypeId, Check)> {
    vec![
        checker::<Sentences>(),
        checker::<Tokens>(),
        checker::<Subwords>(),
        checker::<Entities>(),
        checker::<PhraseChunks>(),
        checker::<Matches>(),
        checker::<Numbers>(),
        checker::<Money>(),
        checker::<Quantities>(),
        checker::<Temporals>(),
        checker::<Pii>(),
        checker::<Structure>(),
        checker::<Chunks>(),
    ]
}

/// Checks every instance, default or named, of every known layer type on
/// `doc`, and returns how many layers were checked with the violations.
pub fn check_document(doc: &Document) -> (usize, Vec<Violation>) {
    let known = known_layers();
    let mut checked = 0;
    let mut out = Vec::new();
    for k in doc.layers() {
        if let Some((_, check)) = known.iter().find(|(ty, _)| *ty == k.ty) {
            checked += 1;
            out.extend(check(doc, k.name));
        }
    }
    (checked, out)
}

/// Panics with every violation listed when there is one.
#[track_caller]
pub fn assert_invariants(doc: &Document, what: &str) -> usize {
    let (checked, violations) = check_document(doc);
    assert!(
        violations.is_empty(),
        "{what}: {} offset invariants broken:\n{}",
        violations.len(),
        violations.iter().map(|v| format!("  {v}")).collect::<Vec<_>>().join("\n")
    );
    checked
}
