//! `GoldenCompare` for tagging, chunking, classification, and subword layers.

use turbo_analysis_layers::{
    Categories, ChunkTags, Entities, Language, Lemmas, PhraseChunks, PosTags, Strs, Subwords, Tag, Tokens,
};

use super::super::{Cx, GoldenCompare, Items, Labels, Mismatch, Names, SpanCols, TokenCols, Tolerance, compare_items};
use super::{SPAN, anchor_spans, f64s, golden_spans, golden_spans_only, spans_only, strs, u32s};
use crate::format::ColumnType::{F64, Str, U32};
use crate::format::{ColumnSpec, GoldenLayer};

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
        let (tags, scores): (Vec<Tag>, Vec<f64>) = match self.get(0) {
            Some(row) => row.ranked_tags().into_iter().map(|t| (t, row.scores[t.0 as usize])).unzip(),
            None => (Vec::new(), Vec::new()),
        };
        let rust = Items {
            len: tags.len(),
            spans: None,
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Tags(&self.categories, &tags)),
            values: Some(&scores),
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
