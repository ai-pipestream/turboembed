//! The item by item comparison every layer type shares. A layer lays out
//! its columns as `Items` (borrowed, nothing copied), the golden's the
//! same way, and `compare_items` walks both. It allocates nothing unless
//! it finds a difference, and then only the report.

use turbo_analysis::{IxRange, Span, ViewId};
use turbo_analysis_layers::{Strs, Tag, TagSet, Tokens};

use super::{Cx, Item, Mismatch, Tolerance};

/// A layer's columns as the comparison sees them. A field the layer does
/// not have is `None`.
pub struct Items<'a> {
    pub len: usize,
    /// The items' own spans, compared.
    pub spans: Option<SpanCols<'a>>,
    /// For an aligned layer, its anchor's spans: shown in a report, not
    /// compared.
    pub context: Option<SpanCols<'a>>,
    /// An entity's tokens, compared.
    pub tokens: Option<TokenCols<'a>>,
    /// A label per item, compared as text.
    pub labels: Option<Labels<'a>>,
    /// A value per item, compared within the tolerance.
    pub values: Option<&'a [f64]>,
}

/// Span columns in one view. A golden's are in the source text.
#[derive(Copy, Clone)]
pub struct SpanCols<'a> {
    pub view: ViewId,
    pub start: &'a [u32],
    pub end: &'a [u32],
}

/// Token ranges: the Rust layer's, or the golden's two columns.
#[derive(Copy, Clone)]
pub enum TokenCols<'a> {
    Ranges(&'a [IxRange<Tokens>]),
    Columns { start: &'a [u32], end: &'a [u32] },
}

impl TokenCols<'_> {
    fn get(&self, i: usize) -> (u32, u32) {
        match self {
            TokenCols::Ranges(r) => (r[i].start.get() as u32, r[i].end.get() as u32),
            TokenCols::Columns { start, end } => (start[i], end[i]),
        }
    }
}

/// Labels: tags through their tag set, strings by column, or the one
/// label of a document layer.
#[derive(Copy, Clone)]
pub enum Labels<'a> {
    Tags(&'a TagSet, &'a [Tag]),
    Strs(&'a Strs),
    One(&'a str),
}

impl<'a> Labels<'a> {
    fn get(&self, i: usize) -> &'a str {
        match *self {
            Labels::Tags(set, tags) => set.label(tags[i]),
            Labels::Strs(s) => s.get(i),
            Labels::One(s) => s,
        }
    }
}

/// The names of the label and value columns, for reports.
#[derive(Copy, Clone)]
pub struct Names {
    pub label: Option<&'static str>,
    pub value: Option<&'static str>,
}

/// Compares two layers' items in order, stopping at the first
/// difference: an item's span, then its tokens, its label and its value.
/// When every item both sides have is the same and one side has more, the
/// counts differ.
pub fn compare_items(
    cx: &Cx<'_>,
    rust: &Items<'_>,
    golden: &Items<'_>,
    names: Names,
    tol: Tolerance,
) -> Result<(), Mismatch> {
    let n = rust.len.min(golden.len);
    for i in 0..n {
        let what = if let (Some(r), Some(g)) = (rust.spans, golden.spans)
            && span(cx, r, i) != span(cx, g, i)
        {
            Some("span")
        } else if let (Some(r), Some(g)) = (rust.tokens, golden.tokens)
            && r.get(i) != g.get(i)
        {
            Some("tokens")
        } else if let (Some(r), Some(g)) = (rust.labels, golden.labels)
            && r.get(i) != g.get(i)
        {
            names.label
        } else if let (Some(r), Some(g)) = (rust.values, golden.values)
            && !tol.accepts(r[i], g[i])
        {
            names.value
        } else {
            None
        };
        if let Some(what) = what {
            return Err(Mismatch::Item {
                layer: cx.layer.clone(),
                index: i,
                what,
                tolerance: tol,
                rust: Box::new(item(cx, rust, names, i)),
                golden: Box::new(item(cx, golden, names, i)),
            });
        }
    }
    if rust.len != golden.len {
        let longer = if rust.len > golden.len { rust } else { golden };
        return Err(Mismatch::Count {
            layer: cx.layer.clone(),
            rust: rust.len,
            golden: golden.len,
            extra: Box::new(item(cx, longer, names, n)),
        });
    }
    Ok(())
}

/// Span `i` in source bytes: a span of a derived view is mapped back
/// through its alignment.
fn span(cx: &Cx<'_>, s: SpanCols<'_>, i: usize) -> Span {
    let at = Span { start: s.start[i], end: s.end[i] };
    if s.view == ViewId::SOURCE || at.start > at.end { at } else { cx.doc.to_source(s.view, at) }
}

/// Item `i` for a report.
fn item(cx: &Cx<'_>, items: &Items<'_>, names: Names, i: usize) -> Item {
    let span = items.spans.or(items.context).filter(|s| i < s.start.len()).map(|s| span(cx, s, i));
    let text = span.and_then(|s| cx.text().get(s.start as usize..s.end as usize)).map(String::from);
    Item {
        span,
        text,
        tokens: items.tokens.map(|t| t.get(i)),
        label: items.labels.and_then(|l| Some((names.label?, l.get(i).to_string()))),
        value: items.values.and_then(|v| Some((names.value?, v[i]))),
    }
}
