//! `GoldenCompare` for the shared layers with a golden shape. Each lays
//! its columns and the golden's side by side; the columns match the
//! table in docs/analysis.md, "Golden files".

mod extraction;
mod sentiment;
mod tagging;
#[cfg(test)]
mod tests;

use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::{Confidence, Sentences, Tokens, Utf16Spans};

use super::{CutCol, Items, Labels, Names, SpanCols, UnitCols, compare_items};
use super::{Cx, GoldenCompare, Mismatch, Tolerance};
use crate::format::ColumnType::{F64, U32};
use crate::format::{ColumnSpec, GoldenAnchor, GoldenLayer};

pub(super) const SPAN: [ColumnSpec; 2] = [ColumnSpec::new("start", U32), ColumnSpec::new("end", U32)];

// A golden layer's columns are checked against the type before `compare`
// runs, so a column the shape names is there with its type.
pub(super) fn u32s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [u32] {
    g.u32s(name).expect("a u32 column the shape checked")
}

pub(super) fn f64s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [f64] {
    g.f64s(name).expect("an f64 column the shape checked")
}

pub(super) fn strs<'a>(g: &'a GoldenLayer, name: &str) -> Labels<'a> {
    Labels::Strs(g.strs(name).expect("a str column the shape checked"))
}

pub(super) fn golden_spans(g: &GoldenLayer) -> SpanCols<'_> {
    SpanCols { view: ViewId::SOURCE, start: u32s(g, "start"), end: u32s(g, "end") }
}

/// Checks that a golden `u32` column encoding a boolean contains only `0` or `1`.
pub(super) fn check_bool_u32(cx: &Cx<'_>, col: &[u32], name: &str) -> Result<(), Mismatch> {
    if let Some(i) = col.iter().position(|&c| c > 1) {
        return Err(Mismatch::BadGolden {
            layer: cx.layer.clone(),
            detail: format!("row {i} has {name} {}, expected 0 or 1", col[i]),
        });
    }
    Ok(())
}

/// The items of a layer that is nothing but spans.
pub(super) fn spans_only<'a, L: Layer>(layer: &'a L) -> Items<'a> {
    let spans = layer.text_spans().map(|ts| SpanCols { view: ts.view, start: ts.start, end: ts.end });
    Items { len: layer.len(), spans, context: None, tokens: None, units: None, labels: None, values: None }
}

pub(super) fn golden_spans_only(g: &GoldenLayer) -> Items<'_> {
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
pub(super) fn anchor_spans<'a>(cx: &Cx<'a>, g: &GoldenLayer) -> Option<SpanCols<'a>> {
    match &g.anchor {
        GoldenAnchor::Aligned(of) => cx.golden_spans(of),
        _ => None,
    }
}

pub(super) const NO_NAMES: Names = Names { label: None, value: None };

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
        check_bool_u32(cx, cut, "cut")?;
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
