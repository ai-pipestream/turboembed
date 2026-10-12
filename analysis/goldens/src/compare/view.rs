//! `GoldenCompare` for a normalized `View`. Its golden is a `turbo:view`
//! block: one row per edit, the source bytes `start..end` replaced by
//! `with`. The source with every edit applied is the normalized text, and
//! that is compared first, whole and exactly; then the edit lists.
//!
//! The JVM writes adjacent one-to-one replacements as separate rows, one
//! per character, as OpenNLP's mapping gives them, while a Rust alignment
//! merges edits that touch (`ViewBuilder::replace`). Both describe the
//! same mapping, so the golden's touching rows are merged the same way
//! before the lists are compared, and an item index in a report counts
//! merged edits.

use turbo_analysis::{View, ViewId};
use turbo_analysis_layers::Strs;

use super::{Cx, GoldenCompare, Items, Labels, Mismatch, Names, SpanCols, Tolerance, compare_items};
use crate::format::ColumnType::{Str, U32};
use crate::format::{ColumnSpec, GoldenAnchor, GoldenLayer};

/// An edit list by column: source spans and replacement texts.
#[derive(Default)]
struct Edits {
    start: Vec<u32>,
    end: Vec<u32>,
    with: Strs,
}

impl Edits {
    fn items(&self) -> Items<'_> {
        Items {
            len: self.start.len(),
            spans: Some(SpanCols { view: ViewId::SOURCE, start: &self.start, end: &self.end }),
            context: None,
            tokens: None,
            labels: Some(Labels::Strs(&self.with)),
            values: None,
        }
    }
}

impl GoldenCompare for View {
    const COLUMNS: &'static [ColumnSpec] =
        &[ColumnSpec::new("start", U32), ColumnSpec::new("end", U32), ColumnSpec::new("with", Str)];

    /// A view is a document layer in Rust, but its golden rows are edits
    /// over the source text.
    fn golden_anchor() -> GoldenAnchor {
        GoldenAnchor::Text
    }

    /// Only a view made directly from the source (or the source itself)
    /// is compared: the edits of a chain of views are each against their
    /// own parent, and composing them into edits against the source is
    /// not done here, so a chained view is reported as such.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        match self.parent() {
            None | Some(ViewId::SOURCE) => {}
            Some(_) => return Err(Mismatch::ChainedView { layer: cx.layer.clone() }),
        }
        let (g, golden_text) = apply(cx, golden)?;
        let rust_text = self.text();
        if rust_text != golden_text {
            let at = rust_text.bytes().zip(golden_text.bytes()).take_while(|(a, b)| a == b).count();
            return Err(Mismatch::Text {
                layer: cx.layer.clone(),
                at,
                rust: excerpt(rust_text, at),
                golden: excerpt(&golden_text, at),
            });
        }
        let mut r = Edits::default();
        for e in self.alignment().edits() {
            r.start.push(e.parent.start);
            r.end.push(e.parent.end);
            r.with.push(&rust_text[e.child.range()]);
        }
        compare_items(cx, &r.items(), &g.items(), Names { label: Some("with"), value: None }, tol)
    }
}

/// The golden's edits with touching rows merged, and the source text
/// with them applied. Rows that overlap, or that cut a character, are a
/// golden this crate cannot apply.
fn apply(cx: &Cx<'_>, golden: &GoldenLayer) -> Result<(Edits, String), Mismatch> {
    let source = cx.text();
    let (start, end) = (golden.u32s("start").expect("checked"), golden.u32s("end").expect("checked"));
    let with = golden.strs("with").expect("checked");
    let bad = |detail: String| Mismatch::BadGolden { layer: cx.layer.clone(), detail };
    let mut edits = Edits::default();
    let mut text = String::with_capacity(source.len());
    let mut merged = String::new();
    let mut at = 0usize;
    for i in 0..golden.len() {
        let (s, e) = (start[i] as usize, end[i] as usize);
        if s < at || s > e || e > source.len() {
            return Err(bad(format!("row {i}, the edit {s}..{e}, overlaps the row before it or leaves the text")));
        }
        if !source.is_char_boundary(s) || !source.is_char_boundary(e) {
            return Err(bad(format!("row {i}, the edit {s}..{e}, cuts a character")));
        }
        text.push_str(&source[at..s]);
        text.push_str(with.get(i));
        if i > 0 && s == at {
            // Touches the edit before it: one edit, as a Rust alignment has it.
            *edits.end.last_mut().expect("an edit before") = e as u32;
        } else {
            if i > 0 {
                edits.with.push(&merged);
                merged.clear();
            }
            edits.start.push(s as u32);
            edits.end.push(e as u32);
        }
        merged.push_str(with.get(i));
        at = e;
    }
    if !edits.start.is_empty() {
        edits.with.push(&merged);
    }
    text.push_str(&source[at..]);
    Ok((edits, text))
}

/// Up to 24 bytes either side of byte `at`, widened to whole characters.
fn excerpt(text: &str, at: usize) -> String {
    let mut a = at.saturating_sub(24).min(text.len());
    while !text.is_char_boundary(a) {
        a -= 1;
    }
    let mut b = (at + 24).min(text.len());
    while !text.is_char_boundary(b) {
        b += 1;
    }
    text[a..b].to_string()
}
