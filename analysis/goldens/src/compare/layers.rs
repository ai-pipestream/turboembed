//! `GoldenCompare` for the shared layers with a golden shape. Each lays
//! its columns and the golden's side by side; the columns match the
//! table in docs/analysis.md, "Golden files".

use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::{
    Categories, ChunkTags, Confidence, DocumentDate, Entities, Language, Lemmas, Money, Numbers, PhraseChunks, PosTags,
    Quantities, Sentences, Strs, Subwords, Temporals, Tokens, Utf16Spans,
};

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
    let spans = layer.text_spans().map(|ts| SpanCols { view: ts.view, start: ts.start, end: ts.end });
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

impl GoldenCompare for ChunkTags {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("tag", Str), ColumnSpec::new("prob", F64)];

    /// As `PosTags`: the tag's label through the tag set, then the
    /// probability. A layer without probabilities compares them as absent.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.tags.len(),
            spans: None,
            context: cx.rust_anchor::<Tokens>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Tags(&self.tagset, &self.tags)),
            values: self.prob.as_deref(),
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

impl GoldenCompare for PhraseChunks {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("start", U32),
        ColumnSpec::new("end", U32),
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("kind", Str),
    ];

    /// The span, then the token range and the kind (its label through the
    /// tag set). A phrase has no probability of its own.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let (start, end) = self.spans.columns();
        let rust = Items {
            len: self.spans.len(),
            spans: Some(SpanCols { view: self.spans.view(), start, end }),
            context: None,
            tokens: Some(TokenCols::Ranges(&self.tokens)),
            units: None,
            labels: Some(Labels::Tags(&self.kinds, &self.kind)),
            values: None,
        };
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: Some(TokenCols::Columns { start: u32s(golden, "token_start"), end: u32s(golden, "token_end") }),
            units: None,
            labels: Some(strs(golden, "kind")),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("kind"), value: None }, tol)
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

impl GoldenCompare for Categories {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("category", Str), ColumnSpec::new("score", F64)];

    /// Row by row, best first: the category, then its score, so the
    /// ranking must be the golden's.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let rust = Items {
            len: self.category.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&self.category)),
            values: Some(&self.score),
        };
        let golden = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(strs(golden, "category")),
            values: Some(f64s(golden, "score")),
        };
        compare_items(cx, &rust, &golden, Names { label: Some("category"), value: Some("score") }, tol)
    }
}

impl GoldenCompare for Subwords {
    const COLUMNS: &'static [ColumnSpec] = &[SPAN[0], SPAN[1], ColumnSpec::new("id", U32)];

    /// The span, then the vocabulary id, compared as decimal text so a
    /// report shows both. The golden has no token column: the pieces'
    /// word tokens are this crate's own alignment, not the JVM's.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for id in &self.ids {
            ours.push(&id.to_string());
        }
        let mut theirs = Strs::default();
        for id in u32s(golden, "id") {
            theirs.push(&id.to_string());
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let mut golden = golden_spans_only(golden);
        golden.labels = Some(Labels::Strs(&theirs));
        compare_items(cx, &rust, &golden, Names { label: Some("id"), value: None }, tol)
    }
}

impl GoldenCompare for Numbers {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("numeral_end", U32),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
    ];

    /// The span first, then the rest of each row as one label: the value as
    /// `BigDecimal.toPlainString` writes it, where the numeral ends, the sign
    /// and the coefficient and scale. A report shows both rows whole.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = turbo_analysis::Ix::new(i as u32);
            line.clear();
            self.value(ix).write_plain(&mut line);
            let rest = format!(
                " numeral_end {} negative {} coefficient {} scale {}",
                self.numeral_end[i],
                u32::from(self.negative[i]),
                self.coefficient.get(i),
                self.scale[i]
            );
            line.push_str(&rest);
            ours.push(&line);
        }
        let (numeral_end, negative, scale) =
            (u32s(golden, "numeral_end"), u32s(golden, "negative"), u32s(golden, "scale"));
        let (coefficient, plain) = (golden.strs("coefficient"), golden.strs("plain"));
        let (coefficient, plain) = (coefficient.expect("a str column"), plain.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} numeral_end {} negative {} coefficient {} scale {}",
                plain.get(i),
                numeral_end[i],
                negative[i],
                coefficient.get(i),
                scale[i]
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Money {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
        ColumnSpec::new("currency", Str),
    ];

    /// The span first, then the rest of each row as one label: the amount
    /// as `BigDecimal.toPlainString` writes it, the sign, the coefficient
    /// and scale, and the currency. The goldens write an amount with a
    /// scale below zero (`2E+1`) with scale 0, so ours is written the same
    /// way: the coefficient followed by that many zeros.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = turbo_analysis::Ix::new(i as u32);
            let amount = self.amount(ix);
            line.clear();
            amount.write_plain(&mut line);
            line.push_str(" negative ");
            line.push_str(if amount.negative { "1" } else { "0" });
            line.push_str(" coefficient ");
            line.push_str(amount.digits);
            let scale = if amount.scale < 0 {
                if amount.digits != "0" {
                    line.extend(std::iter::repeat_n('0', amount.scale.unsigned_abs() as usize));
                }
                0
            } else {
                amount.scale
            };
            line.push_str(&format!(" scale {scale} currency {}", self.currency[i]));
            ours.push(&line);
        }
        let (negative, scale) = (u32s(golden, "negative"), u32s(golden, "scale"));
        let (coefficient, plain, currency) =
            (golden.strs("coefficient"), golden.strs("plain"), golden.strs("currency"));
        let (coefficient, plain, currency) =
            (coefficient.expect("a str column"), plain.expect("a str column"), currency.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} negative {} coefficient {} scale {} currency {}",
                plain.get(i),
                negative[i],
                coefficient.get(i),
                scale[i],
                currency.get(i)
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Quantities {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
        ColumnSpec::new("unit", Str),
    ];

    /// The span first, then the rest of each row as one label: the value
    /// as `BigDecimal.toPlainString` writes it, the sign, the coefficient
    /// and scale, and the unit. As for money, a value with a scale below
    /// zero is written with scale 0, the coefficient followed by that many
    /// zeros; a value read from text has none.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = turbo_analysis::Ix::new(i as u32);
            let value = self.value(ix);
            line.clear();
            value.write_plain(&mut line);
            line.push_str(" negative ");
            line.push_str(if value.negative { "1" } else { "0" });
            line.push_str(" coefficient ");
            line.push_str(value.digits);
            let scale = if value.scale < 0 {
                if value.digits != "0" {
                    line.extend(std::iter::repeat_n('0', value.scale.unsigned_abs() as usize));
                }
                0
            } else {
                value.scale
            };
            line.push_str(&format!(" scale {scale} unit {}", self.unit(ix)));
            ours.push(&line);
        }
        let (negative, scale) = (u32s(golden, "negative"), u32s(golden, "scale"));
        let (coefficient, plain, unit) = (golden.strs("coefficient"), golden.strs("plain"), golden.strs("unit"));
        let (coefficient, plain, unit) =
            (coefficient.expect("a str column"), plain.expect("a str column"), unit.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} negative {} coefficient {} scale {} unit {}",
                plain.get(i),
                negative[i],
                coefficient.get(i),
                scale[i],
                unit.get(i)
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Temporals {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("value", Str),
        ColumnSpec::new("granularity", Str),
        ColumnSpec::new("origin", Str),
    ];

    /// The span first, then the rest of each row as one label: the value
    /// as OpenNLP's `TemporalExpression.value()` writes it, written here
    /// from the layer's calendar fields, then the granularity and origin
    /// by their Java names.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = turbo_analysis::Ix::new(i as u32);
            ours.push(&format!("{} {} {}", self.value(ix), self.granularity[i].name(), self.origin[i].name()));
        }
        let (value, granularity, origin) = (golden.strs("value"), golden.strs("granularity"), golden.strs("origin"));
        let (value, granularity, origin) =
            (value.expect("a str column"), granularity.expect("a str column"), origin.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("{} {} {}", value.get(i), granularity.get(i), origin.get(i)));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let mut golden = golden_spans_only(golden);
        golden.labels = Some(Labels::Strs(&theirs));
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for DocumentDate {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("date", Str), ColumnSpec::new("mention", U32)];

    /// No row, or one: the date as `LocalDate.toString` writes it and the
    /// index of the temporal mention it was read from in the file's
    /// `turbo:temporal` layer, compared as one label. A date with no
    /// mention behind it, which only a caller can give, has none to write
    /// and is compared with `-`.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        if let Some(date) = self.date {
            ours.push(&format!("{date} mention {}", self.mention.map_or("-".into(), |m| m.get().to_string())));
        }
        let (date, mention) = (golden.strs("date").expect("a str column"), u32s(golden, "mention"));
        let mut theirs = Strs::default();
        for (i, m) in mention.iter().enumerate().take(golden.len()) {
            theirs.push(&format!("{} mention {m}", date.get(i)));
        }
        let items =
            |len, labels| Items { len, spans: None, context: None, tokens: None, units: None, labels, values: None };
        let rust = items(self.len(), Some(Labels::Strs(&ours)));
        let golden = items(golden.len(), Some(Labels::Strs(&theirs)));
        compare_items(cx, &rust, &golden, Names { label: Some("date"), value: None }, tol)
    }
}
