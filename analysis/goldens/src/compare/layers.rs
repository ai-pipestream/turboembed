//! `GoldenCompare` for the shared layers with a golden shape. Each lays
//! its columns and the golden's side by side; the columns match the
//! table in docs/analysis.md, "Golden files".

use turbo_analysis::{Ix, Layer, ViewId};
use turbo_analysis_layers::{
    Aspects, Categories, ChunkTags, Confidence, DocumentDate, Entities, Incivility, IncivilityLabel, Language, Lemmas,
    Money, Numbers, Opinions, PhraseChunks, Pii, PosTags, Pragmatics, Quantities, Sentences, Strs, Subwords, Temporals,
    Tokens, Utf16Spans,
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

impl GoldenCompare for Pii {
    const COLUMNS: &'static [ColumnSpec] =
        &[SPAN[0], SPAN[1], ColumnSpec::new("type", Str), ColumnSpec::new("normalized", Str)];

    /// The span first, then the kind by its OpenNLP type name and the
    /// normalized form as one label. OpenNLP has no confidence, so the
    /// goldens hold none.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for m in self.mentions() {
            ours.push(&format!("{} {}", m.kind.name(), m.normalized));
        }
        let (kind, normalized) = (golden.strs("type"), golden.strs("normalized"));
        let (kind, normalized) = (kind.expect("a str column"), normalized.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("{} {}", kind.get(i), normalized.get(i)));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let mut golden = golden_spans_only(golden);
        golden.labels = Some(Labels::Strs(&theirs));
        compare_items(cx, &rust, &golden, Names { label: Some("mention"), value: None }, tol)
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

/// Compares two layers that have multiple `f64` columns in row-major order
/// (`0..n`), checking tokens and labels first and then each `(name, rust,
/// golden)` float slice under `tol`.
fn compare_multi_values<'a, 'b>(
    cx: &Cx<'_>,
    mut rust: Items<'a>,
    mut golden: Items<'b>,
    label_name: Option<&'static str>,
    value_cols: &[(&'static str, &'a [f64], &'b [f64])],
    tol: Tolerance,
) -> Result<(), Mismatch> {
    let n = rust.len.min(golden.len);
    let mut chosen = value_cols[0];
    'rows: for i in 0..n {
        for &(col_name, r, g) in value_cols {
            if !tol.accepts(r[i], g[i]) {
                chosen = (col_name, r, g);
                break 'rows;
            }
        }
    }
    rust.values = Some(chosen.1);
    golden.values = Some(chosen.2);
    compare_items(cx, &rust, &golden, Names { label: label_name, value: Some(chosen.0) }, tol)
}

impl GoldenCompare for Opinions {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("polarity", F64),
        ColumnSpec::new("intensity", F64),
        ColumnSpec::new("negated", U32),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let negated = u32s(golden, "negated");
        if let Some(i) = negated.iter().position(|&c| c > 1) {
            return Err(Mismatch::BadGolden {
                layer: cx.layer.clone(),
                detail: format!("row {i} has negated {}, expected 0 or 1", negated[i]),
            });
        }
        let mut ours = Strs::default();
        for &neg in &self.negated {
            ours.push(if neg { "1" } else { "0" });
        }
        let mut theirs = Strs::default();
        for &neg in negated {
            theirs.push(if neg != 0 { "1" } else { "0" });
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: Some(TokenCols::Ranges(&self.tokens)),
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: Some(TokenCols::Columns { start: u32s(golden, "token_start"), end: u32s(golden, "token_end") }),
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            ("polarity", self.polarity.as_slice(), f64s(golden, "polarity")),
            ("intensity", self.intensity.as_slice(), f64s(golden, "intensity")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("negated"), &cols, tol)
    }
}

impl From<&Opinions> for GoldenLayer {
    fn from(layer: &Opinions) -> GoldenLayer {
        use crate::format::{Column, ColumnData};
        let token_start = layer.tokens.iter().map(|r| r.start.get() as u32).collect();
        let token_end = layer.tokens.iter().map(|r| r.end.get() as u32).collect();
        let negated = layer.negated.iter().map(|&b| u32::from(b)).collect();
        GoldenLayer {
            id: Opinions::ID.into(),
            anchor: Opinions::golden_anchor(),
            columns: vec![
                Column { name: "token_start".into(), data: ColumnData::U32(token_start) },
                Column { name: "token_end".into(), data: ColumnData::U32(token_end) },
                Column { name: "polarity".into(), data: ColumnData::F64(layer.polarity.clone()) },
                Column { name: "intensity".into(), data: ColumnData::F64(layer.intensity.clone()) },
                Column { name: "negated".into(), data: ColumnData::U32(negated) },
            ],
        }
    }
}

impl GoldenCompare for Aspects {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("token_start", U32),
        ColumnSpec::new("token_end", U32),
        ColumnSpec::new("sentence", Str),
        ColumnSpec::new("category", Str),
        ColumnSpec::new("positive", F64),
        ColumnSpec::new("negative", F64),
        ColumnSpec::new("neutral", F64),
        ColumnSpec::new("opinions", Str),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let (ts, te) = self.tokens[i].map_or((0, 0), |r| (r.start.get() as u32, r.end.get() as u32));
            let sent = self.sentence[i].map_or_else(String::new, |s| s.get().to_string());
            let cat = self.category_label(ix).unwrap_or("");
            let ops = self.opinions_of(ix).iter().map(|o| o.get().to_string()).collect::<Vec<_>>().join(",");
            ours.push(&format!("tokens {ts}..{te} sentence {sent:?} category {cat:?} opinions [{ops}]"));
        }
        let (token_start, token_end) = (u32s(golden, "token_start"), u32s(golden, "token_end"));
        let sentence = golden.strs("sentence").expect("a str column");
        let category = golden.strs("category").expect("a str column");
        let opinions = golden.strs("opinions").expect("a str column");
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "tokens {}..{} sentence {:?} category {:?} opinions [{}]",
                token_start[i],
                token_end[i],
                sentence.get(i),
                category.get(i),
                opinions.get(i)
            ));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            ("positive", self.positive.as_slice(), f64s(golden, "positive")),
            ("negative", self.negative.as_slice(), f64s(golden, "negative")),
            ("neutral", self.neutral.as_slice(), f64s(golden, "neutral")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("aspect"), &cols, tol)
    }
}

impl From<&Aspects> for GoldenLayer {
    fn from(layer: &Aspects) -> GoldenLayer {
        use crate::format::{Column, ColumnData};
        let mut token_start = Vec::with_capacity(layer.len());
        let mut token_end = Vec::with_capacity(layer.len());
        let mut sentence = Strs::default();
        let mut category = Strs::default();
        let mut opinions = Strs::default();
        for i in 0..layer.len() {
            let ix = Ix::new(i as u32);
            let (ts, te) = layer.tokens[i].map_or((0, 0), |r| (r.start.get() as u32, r.end.get() as u32));
            token_start.push(ts);
            token_end.push(te);
            match layer.sentence[i] {
                Some(s) => sentence.push(&s.get().to_string()),
                None => sentence.push(""),
            }
            category.push(layer.category_label(ix).unwrap_or(""));
            let ops = layer.opinions_of(ix).iter().map(|o| o.get().to_string()).collect::<Vec<_>>().join(",");
            opinions.push(&ops);
        }
        GoldenLayer {
            id: Aspects::ID.into(),
            anchor: Aspects::golden_anchor(),
            columns: vec![
                Column { name: "token_start".into(), data: ColumnData::U32(token_start) },
                Column { name: "token_end".into(), data: ColumnData::U32(token_end) },
                Column { name: "sentence".into(), data: ColumnData::Str(sentence) },
                Column { name: "category".into(), data: ColumnData::Str(category) },
                Column { name: "positive".into(), data: ColumnData::F64(layer.positive.clone()) },
                Column { name: "negative".into(), data: ColumnData::F64(layer.negative.clone()) },
                Column { name: "neutral".into(), data: ColumnData::F64(layer.neutral.clone()) },
                Column { name: "opinions".into(), data: ColumnData::Str(opinions) },
            ],
        }
    }
}

impl GoldenCompare for Incivility {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("target_kind", Str),
        ColumnSpec::new("target_index", U32),
        ColumnSpec::new("insult", F64),
        ColumnSpec::new("profanity", F64),
        ColumnSpec::new("threat", F64),
        ColumnSpec::new("identity_attack", F64),
        ColumnSpec::new("intolerance", F64),
        ColumnSpec::new("severe_toxicity", F64),
        ColumnSpec::new("evidence", Str),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let t = self.target[i];
            let ev = self
                .evidence_of(ix)
                .map(|(l, r)| format!("{}:{}..{}", l.name(), r.start.get(), r.end.get()))
                .collect::<Vec<_>>()
                .join(",");
            ours.push(&format!("{}:{} evidence [{ev}]", t.kind_name(), t.raw_index()));
        }
        let target_kind = golden.strs("target_kind").expect("a str column");
        let target_index = u32s(golden, "target_index");
        let evidence = golden.strs("evidence").expect("a str column");
        let mut theirs = Strs::default();
        for (i, &idx) in target_index.iter().enumerate().take(golden.len()) {
            theirs.push(&format!("{}:{idx} evidence [{}]", target_kind.get(i), evidence.get(i)));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: None,
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        let cols = [
            (IncivilityLabel::Insult.name(), self.insult.as_slice(), f64s(golden, "insult")),
            (IncivilityLabel::Profanity.name(), self.profanity.as_slice(), f64s(golden, "profanity")),
            (IncivilityLabel::Threat.name(), self.threat.as_slice(), f64s(golden, "threat")),
            (IncivilityLabel::IdentityAttack.name(), self.identity_attack.as_slice(), f64s(golden, "identity_attack")),
            (IncivilityLabel::Intolerance.name(), self.intolerance.as_slice(), f64s(golden, "intolerance")),
            (IncivilityLabel::SevereToxicity.name(), self.severe_toxicity.as_slice(), f64s(golden, "severe_toxicity")),
        ];
        compare_multi_values(cx, rust, golden_items, Some("target"), &cols, tol)
    }
}

impl From<&Incivility> for GoldenLayer {
    fn from(layer: &Incivility) -> GoldenLayer {
        use crate::format::{Column, ColumnData};
        let mut target_kind = Strs::default();
        let mut target_index = Vec::with_capacity(layer.len());
        let mut evidence = Strs::default();
        for i in 0..layer.len() {
            let ix = Ix::new(i as u32);
            let t = layer.target[i];
            target_kind.push(t.kind_name());
            target_index.push(t.raw_index());
            let ev = layer
                .evidence_of(ix)
                .map(|(l, r)| format!("{}:{}..{}", l.name(), r.start.get(), r.end.get()))
                .collect::<Vec<_>>()
                .join(",");
            evidence.push(&ev);
        }
        GoldenLayer {
            id: Incivility::ID.into(),
            anchor: Incivility::golden_anchor(),
            columns: vec![
                Column { name: "target_kind".into(), data: ColumnData::Str(target_kind) },
                Column { name: "target_index".into(), data: ColumnData::U32(target_index) },
                Column { name: "insult".into(), data: ColumnData::F64(layer.insult.clone()) },
                Column { name: "profanity".into(), data: ColumnData::F64(layer.profanity.clone()) },
                Column { name: "threat".into(), data: ColumnData::F64(layer.threat.clone()) },
                Column { name: "identity_attack".into(), data: ColumnData::F64(layer.identity_attack.clone()) },
                Column { name: "intolerance".into(), data: ColumnData::F64(layer.intolerance.clone()) },
                Column { name: "severe_toxicity".into(), data: ColumnData::F64(layer.severe_toxicity.clone()) },
                Column { name: "evidence".into(), data: ColumnData::Str(evidence) },
            ],
        }
    }
}

impl GoldenCompare for Pragmatics {
    const COLUMNS: &'static [ColumnSpec] = &[
        ColumnSpec::new("score", F64),
        ColumnSpec::new("pos_start", U32),
        ColumnSpec::new("pos_end", U32),
        ColumnSpec::new("neg_start", U32),
        ColumnSpec::new("neg_end", U32),
    ];

    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for ev in &self.evidence {
            let (ps, pe, ns, ne) = ev.map_or((0, 0, 0, 0), |inc| {
                (
                    inc.positive.start.get() as u32,
                    inc.positive.end.get() as u32,
                    inc.negative.start.get() as u32,
                    inc.negative.end.get() as u32,
                )
            });
            ours.push(&format!("pos {ps}..{pe} neg {ns}..{ne}"));
        }
        let (pos_start, pos_end) = (u32s(golden, "pos_start"), u32s(golden, "pos_end"));
        let (neg_start, neg_end) = (u32s(golden, "neg_start"), u32s(golden, "neg_end"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("pos {}..{} neg {}..{}", pos_start[i], pos_end[i], neg_start[i], neg_end[i]));
        }
        let rust = Items {
            len: self.len(),
            spans: None,
            context: cx.rust_anchor::<Sentences>(),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&ours)),
            values: Some(&self.score),
        };
        let golden_items = Items {
            len: golden.len(),
            spans: None,
            context: anchor_spans(cx, golden),
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: Some(f64s(golden, "score")),
        };
        compare_items(cx, &rust, &golden_items, Names { label: Some("evidence"), value: Some("score") }, tol)
    }
}

impl From<&Pragmatics> for GoldenLayer {
    fn from(layer: &Pragmatics) -> GoldenLayer {
        use crate::format::{Column, ColumnData};
        let mut pos_start = Vec::with_capacity(layer.len());
        let mut pos_end = Vec::with_capacity(layer.len());
        let mut neg_start = Vec::with_capacity(layer.len());
        let mut neg_end = Vec::with_capacity(layer.len());
        for ev in &layer.evidence {
            let (ps, pe, ns, ne) = ev.map_or((0, 0, 0, 0), |inc| {
                (
                    inc.positive.start.get() as u32,
                    inc.positive.end.get() as u32,
                    inc.negative.start.get() as u32,
                    inc.negative.end.get() as u32,
                )
            });
            pos_start.push(ps);
            pos_end.push(pe);
            neg_start.push(ns);
            neg_end.push(ne);
        }
        GoldenLayer {
            id: Pragmatics::ID.into(),
            anchor: Pragmatics::golden_anchor(),
            columns: vec![
                Column { name: "score".into(), data: ColumnData::F64(layer.score.clone()) },
                Column { name: "pos_start".into(), data: ColumnData::U32(pos_start) },
                Column { name: "pos_end".into(), data: ColumnData::U32(pos_end) },
                Column { name: "neg_start".into(), data: ColumnData::U32(neg_start) },
                Column { name: "neg_end".into(), data: ColumnData::U32(neg_end) },
            ],
        }
    }
}
