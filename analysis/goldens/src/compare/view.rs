//! `GoldenCompare` for a normalized `View`. Its golden is a `turbo:view`
//! block: one row per edit, the source bytes `start..end` replaced by
//! `with`. The source with every edit applied is the normalized text, and
//! that is compared first, whole and exactly; then the golden's rows are
//! compared one for one, in order, with the view's alignment edits, so
//! golden row i is Rust edit i. The JVM writes one row per character for
//! one-to-one replacements, so a normalizer matches it by making one edit
//! per character too.

use turbo_analysis::{View, ViewId};
use turbo_analysis_layers::Strs;

use super::{Cx, GoldenCompare, Items, Labels, Mismatch, Names, SpanCols, Tolerance, compare_items};
use crate::format::ColumnType::{Str, U32};
use crate::format::{ColumnSpec, GoldenAnchor, GoldenLayer};

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
        let start = golden.u32s("start").expect("a u32 column the shape checked");
        let end = golden.u32s("end").expect("a u32 column the shape checked");
        let with = golden.strs("with").expect("a str column the shape checked");
        let golden_text = apply(cx, start, end, with)?;
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
        // The alignment stores edits as structs; the comparison reads
        // columns, so the Rust side is laid out once per layer.
        let edits = self.alignment().edits();
        let mut r_start = Vec::with_capacity(edits.len());
        let mut r_end = Vec::with_capacity(edits.len());
        let mut r_with = Strs::default();
        for e in edits {
            r_start.push(e.parent.start);
            r_end.push(e.parent.end);
            r_with.push(&rust_text[e.child.range()]);
        }
        let rust = Items {
            len: edits.len(),
            spans: Some(SpanCols { view: ViewId::SOURCE, start: &r_start, end: &r_end }),
            context: None,
            tokens: None,
            labels: Some(Labels::Strs(&r_with)),
            values: None,
        };
        let golden = Items {
            len: golden.len(),
            spans: Some(SpanCols { view: ViewId::SOURCE, start, end }),
            context: None,
            tokens: None,
            labels: Some(Labels::Strs(with)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("with"), value: None }, tol)
    }
}

/// The source text with the golden's edits applied. Rows that overlap, or
/// that cut a character, are a golden this crate cannot apply.
fn apply(cx: &Cx<'_>, start: &[u32], end: &[u32], with: &Strs) -> Result<String, Mismatch> {
    let source = cx.text();
    let bad = |detail: String| Mismatch::BadGolden { layer: cx.layer.clone(), detail };
    let mut text = String::with_capacity(source.len());
    let mut at = 0usize;
    for i in 0..start.len() {
        let (s, e) = (start[i] as usize, end[i] as usize);
        if s < at || s > e || e > source.len() {
            return Err(bad(format!("row {i}, the edit {s}..{e}, overlaps the row before it or leaves the text")));
        }
        if !source.is_char_boundary(s) || !source.is_char_boundary(e) {
            return Err(bad(format!("row {i}, the edit {s}..{e}, cuts a character")));
        }
        text.push_str(&source[at..s]);
        text.push_str(with.get(i));
        at = e;
    }
    text.push_str(&source[at..]);
    Ok(text)
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
