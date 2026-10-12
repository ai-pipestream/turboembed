//! `GoldenCompare` for the shared layers with a golden shape. Each lays
//! its columns and the golden's side by side; the columns match the
//! table in docs/analysis.md, "Golden files".

use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::{Confidence, Entities, Language, Lemmas, PosTags, Sentences, Tokens, Utf16Spans};

use super::{CutCol, Items, Labels, Names, SpanCols, TokenCols, UnitCols, compare_items};
use super::{Cx, GoldenCompare, Mismatch, Tolerance};
use crate::format::ColumnType::{F64, Str, U32};
use crate::format::{ColumnSpec, GoldenAnchor, GoldenLayer};

const SPAN: [ColumnSpec; 2] = [ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];

// A golden layer's columns are checked against the type before `compare`
// runs, so a column the shape names is there with its type.
fn u32s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [u32] {
    g.u32s(name).expect("a u32 column the shape checked")
}

fn f64s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [f64] {
    g.f64s(name).expect("an f64 column the shape checked")
}

fn strs<'a>(g: &'a GoldenLayer, name: &str) -> Labels<'a> {
    Labels::Strs(g.strs(name).expect("a str column the shape checked"))
}

fn golden_spans(g: &GoldenLayer) -> SpanCols<'_> {
    SpanCols { view: ViewId::SOURCE, start: u32s(g, "start"), end: u32s(g, "end") }
}

/// The items of a layer that is nothing but spans.
fn spans_only<'a, L: Layer>(layer: &'a L) -> Items<'a> {
    let spans = layer.text_spans().map(|(view, start, end)| SpanCols { view, start, end });
    Items { len: layer.len(), spans, context: None, tokens: None, units: None, labels: None, values: None }
}

fn golden_spans_only(g: &GoldenLayer) -> Items<'_> {
    Items {
        len: g.len(),
        spans: Some(golden_spans(g)),
        context: None,
        tokens: None,
        units: None,
        labels: None,
        values: None,
    }
}

/// The spans of the golden layer an aligned golden layer is aligned to.
fn anchor_spans<'a>(cx: &Cx<'a>, g: &GoldenLayer) -> Option<SpanCols<'a>> {
    match &g.anchor {
        GoldenAnchor::Aligned(of) => cx.golden_spans(of),
        _ => None,
    }
}

const NO_NAMES: Names = Names { label: None, value: None };

impl GoldenCompare for Sentences {
    const COLUMNS: &'static [ColumnSpec] = &SPAN;

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        compare_items(cx, &spans_only(self), &golden_spans_only(golden), NO_NAMES, tol)
    }
}

impl GoldenCompare for Tokens {
    const COLUMNS: &'static [ColumnSpec] = &SPAN;

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        compare_items(cx, &spans_only(self), &golden_spans_only(golden), NO_NAMES, tol)
    }
}

/// A confidence per item of `L`, for any `L` with a golden shape: the
/// sentence detector's and the tokenizer's.
impl<L: GoldenCompare> GoldenCompare for Confidence<L> {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("value", F64)];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.value.len(),
            spans: None,
            context: cx.rust_anchor::<L>(),
            tokens: None,
            units: None,
            labels: None,
            values: Some(&self.value),
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: None,
            labels: None,
            values: Some(f64s(golden, "value")),
        };
        compare_items(cx, &rust, &golden, Names { label: None, value: Some("value") }, tol)
    }
}

/// The exact UTF-16 span of each item of `L` and whether it cuts a
/// character, as the JVM computed it, for a tokenizer or sentence detector
/// that splits a character outside the Basic Multilingual Plane. Every
/// column is compared exactly; the tolerance has nothing to apply to.
impl<L: GoldenCompare> GoldenCompare for Utf16Spans<L> {
    const COLUMNS: &'static [ColumnSpec] =
        &[ColumnSpec::new("start16", U32), ColumnSpec::new("end16", U32), ColumnSpec::new("cut", U32)];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let cut = u32s(golden, "cut");
        if let Some(i) = cut.iter().position(|&c| c > 1) {
            return Err(Mismatch::BadGolden {
                layer: cx.layer.clone(),
                detail: format!("row {i} has cut {}, expected 0 or 1", cut[i]),
            });
        }
        let rust = Items {
            len: self.start16.len(),
            spans: None,
            context: cx.rust_anchor::<L>(),
            tokens: None,
            units: Some(UnitCols { start: &self.start16, end: &self.end16, cut: CutCol::Bools(&self.cut) }),
            labels: None,
            values: None,
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: Some(UnitCols {
                start: u32s(golden, "start16"),
                end: u32s(golden, "end16"),
                cut: CutCol::U32s(cut),
            }),
            labels: None,
            values: None,
        };
        compare_items(cx, &rust, &golden, NO_NAMES, tol)
    }
}

impl GoldenCompare for PosTags {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("tag", Str), ColumnSpec::new("prob", F64)];

    /// The golden always has probabilities. A Rust layer without them is
    /// a difference, not a pass on the tags alone.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let Some(prob) = &self.prob else {
            return Err(Mismatch::NoValues { layer: cx.layer.clone(), column: "prob" });
        };
        let rust = Items {
            len: self.tags.len(),
            spans: None,
            context: cx.rust_anchor::<Tokens>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Tags(&self.tagset, &self.tags)),
            values: Some(prob),
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: None,
            labels: Some(strs(golden, "tag")),
            values: Some(f64s(golden, "prob")),
        };
        compare_items(cx, &rust, &golden, Names { label: Some("tag"), value: Some("prob") }, tol)
    }
}

impl GoldenCompare for Lemmas {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("lemma", Str)];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.lemma.len(),
            spans: None,
            context: cx.rust_anchor::<Tokens>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&self.lemma)),
            values: None,
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: None,
            labels: Some(strs(golden, "lemma")),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("lemma"), value: None }, tol)
    }
}

impl GoldenCompare for Entities {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("kind", Str),
        ColumnSpec::new("prob", F64),
    ];

    /// The span, then the token range, the kind (its label through the
    /// tag set) and the probability, which the Rust layer calls `score`.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let (start, end) = self.spans.columns();
        let rust = Items {
            len: self.spans.len(),
            spans: Some(SpanCols { view: self.spans.view(), start, end }),
            context: None,
            tokens: Some(TokenCols::Ranges(&self.tokens)),
            units: None,
            labels: Some(Labels::Tags(&self.kinds, &self.kind)),
            values: Some(&self.score),
        };
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: Some(TokenCols::Columns { start: u32s(golden, "token_start"), end: u32s(golden, "token_end") }),
            units: None,
            labels: Some(strs(golden, "kind")),
            values: Some(f64s(golden, "prob")),
        };
        compare_items(cx, &rust, &golden, Names { label: Some("kind"), value: Some("prob") }, tol)
    }
}

impl GoldenCompare for Language {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("code", Str), ColumnSpec::new("confidence", F64)];

    /// Row by row, best first: the code, then the confidence (the Rust
    /// layer's `score`), so the ranking must be the golden's.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.code.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&self.code)),
            values: Some(&self.score),
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(strs(golden, "code")),
            values: Some(f64s(golden, "confidence")),
        };
        compare_items(cx, &rust, &golden, Names { label: Some("code"), value: Some("confidence") }, tol)
    }
}
