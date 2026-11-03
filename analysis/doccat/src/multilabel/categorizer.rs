//! The multi-label categorizer at prediction: `categorize` gives every
//! label's independent sigmoid score; the thresholded prediction is
//! every label whose score reaches its own threshold (and `min_score`),
//! capped by `max_labels`. As annotators, [`MultiLabelCategorizer`]
//! writes the document's `turbo:categories` and
//! [`MultiLabelSpanCategorizer`] each span's `Categories<L>`; an item
//! nothing reached is written with best category [`Tag::NONE`], the
//! clean abstention, never a forced argmax.

use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, ModelRef, Refs, Tag, View};
use turbo_analysis_layers::{Categories, Tokens};

use super::model::MultiLabelDoccatModel;
use crate::layer::{SpanCategorizable, extract_tokens, validate_spans};

/// Scores documents with a [`MultiLabelDoccatModel`] and emits every
/// label clearing its threshold.
#[derive(Clone, Debug)]
pub struct MultiLabelCategorizer {
    model: MultiLabelDoccatModel,
}

impl MultiLabelCategorizer {
    /// Creates a `MultiLabelCategorizer` from a [`MultiLabelDoccatModel`].
    pub fn new(model: MultiLabelDoccatModel) -> Self {
        Self { model }
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &MultiLabelDoccatModel {
        &self.model
    }

    /// Returns a [`MultiLabelSpanCategorizer`] categorizing each span of
    /// layer `L` (`Sentences`, `Structure`, or `Chunks`).
    pub fn for_spans<L: SpanCategorizable>(&self) -> MultiLabelSpanCategorizer<L> {
        MultiLabelSpanCategorizer::new(self.model.clone())
    }

    /// Categorizes tokenized `text` with empty extra information: every
    /// label's independent sigmoid score in `0.0..=1.0`, in label order.
    /// The scores do not sum to 1.0.
    pub fn categorize(&self, text: &[&str]) -> Vec<f64> {
        let extra = BTreeMap::new();
        let mut out = Vec::with_capacity(self.model.num_labels());
        let mut scratch = Vec::new();
        self.model.scores_into(text, &extra, &mut scratch, &mut out);
        out
    }

    /// Categorizes tokenized `text` and `extra_information`, returning
    /// every label's sigmoid score in label order.
    pub fn categorize_with_extra(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.model.num_labels());
        let mut scratch = Vec::new();
        self.model.scores_into(text, extra_information, &mut scratch, &mut out);
        out
    }

    /// Categorizes into a pre-allocated `out` buffer
    /// (`out.len() == self.number_of_categories()`), reusing
    /// `feature_scratch` to avoid repeated vector allocations.
    pub fn categorize_into(
        &self,
        text: &[&str],
        extra_information: &BTreeMap<String, String>,
        feature_scratch: &mut Vec<String>,
        out: &mut Vec<f64>,
    ) {
        self.model.scores_into(text, extra_information, feature_scratch, out);
    }

    /// The labels `scores` emits, in label order: every label whose
    /// score reaches its threshold and the model's `min_score`, at most
    /// `max_labels` of them. Empty is clean abstention.
    pub fn predicted_tags(&self, scores: &[f64]) -> Vec<Tag> {
        self.model.predicted_tags(scores)
    }

    /// The labels `text` emits, in label order, each with its score.
    pub fn predicted_labels(&self, text: &[&str]) -> Vec<(&str, f64)> {
        let scores = self.categorize(text);
        self.model
            .predicted_tags(&scores)
            .into_iter()
            .map(|t| (self.model.label(t.0 as usize).unwrap_or(""), scores[t.0 as usize]))
            .collect()
    }

    /// The number of labels the model knows.
    pub fn number_of_categories(&self) -> usize {
        self.model.num_labels()
    }

    /// The label name at `index`, or `None` past the end.
    pub fn get_category(&self, index: usize) -> Option<&str> {
        self.model.label(index)
    }

    /// The index of `category`, or `None` when the model has no such label.
    pub fn get_index(&self, category: &str) -> Option<usize> {
        self.model.index(category)
    }
}

impl Annotator for MultiLabelCategorizer {
    const ID: &'static str = "turbo-doccat:multilabel";
    type Reads = (View, Tokens);
    type Writes = (Categories,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        let tok_strs = extract_tokens(Self::ID, tokens, view.text())?;
        let scores = self.categorize(&tok_strs);
        let predicted = self.model.predicted_tags(&scores);
        let best = best_predicted(&scores, &predicted);
        Ok((Categories::document_selected(Arc::clone(self.model.categories()), best, &predicted, &scores),))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        multilabel_config_hash(&self.model)
    }
}

fn multilabel_config_hash(model: &MultiLabelDoccatModel) -> u64 {
    let mut h = crate::model::context_generator_config_hash(model.context_generator());
    let mix = |h: &mut u64, bytes: &[u8]| {
        *h = turbo_analysis::Fnv1a::with_seed(*h).bytes(bytes).finish();
    };
    mix(&mut h, model.language().as_bytes());
    mix(&mut h, &(model.num_labels() as u64).to_le_bytes());
    for i in 0..model.num_labels() {
        let label = model.categories().label(Tag(i as u16));
        mix(&mut h, &(label.len() as u64).to_le_bytes());
        mix(&mut h, label.as_bytes());
    }
    for &t in model.thresholds() {
        mix(&mut h, &t.to_bits().to_le_bytes());
    }
    mix(&mut h, &(model.max_labels().map_or(-1i64, |v| v as i64)).to_le_bytes());
    mix(&mut h, &(model.min_labels().map_or(-1i64, |v| v as i64)).to_le_bytes());
    mix(&mut h, &model.min_score().to_bits().to_le_bytes());
    h
}

/// The best category of a scored document: the highest-scoring emitted
/// label, ties toward the lower index as `getBestCategory` breaks them;
/// [`Tag::NONE`] when no label cleared its threshold, the clean
/// abstention.
fn best_predicted(scores: &[f64], predicted: &[Tag]) -> Tag {
    let mut best = Tag::NONE;
    let mut best_score = 0.0;
    for &t in predicted {
        let s = scores[t.0 as usize];
        if best == Tag::NONE || s > best_score {
            best = t;
            best_score = s;
        }
    }
    best
}

/// Categorizes each span of a positional layer `L` (`Sentences`,
/// `Structure`, or `Chunks`) with the tokens contained within that span,
/// writing `Categories<L>` (`turbo:categories<L>`): every label clearing
/// its threshold, or [`Tag::NONE`] where none does.
#[derive(Debug)]
pub struct MultiLabelSpanCategorizer<L: SpanCategorizable> {
    categorizer: MultiLabelCategorizer,
    _of: PhantomData<fn() -> L>,
}

impl<L: SpanCategorizable> Clone for MultiLabelSpanCategorizer<L> {
    fn clone(&self) -> Self {
        Self { categorizer: self.categorizer.clone(), _of: PhantomData }
    }
}

impl<L: SpanCategorizable> MultiLabelSpanCategorizer<L> {
    /// Creates a `MultiLabelSpanCategorizer<L>` from a model.
    pub fn new(model: MultiLabelDoccatModel) -> Self {
        Self { categorizer: MultiLabelCategorizer::new(model), _of: PhantomData }
    }

    /// The underlying [`MultiLabelCategorizer`].
    pub fn categorizer(&self) -> &MultiLabelCategorizer {
        &self.categorizer
    }
}

impl<L: SpanCategorizable> Annotator for MultiLabelSpanCategorizer<L> {
    const ID: &'static str = "turbo-doccat:multilabel-spans";
    type Reads = (View, Tokens, L);
    type Writes = (Categories<L>,);

    fn annotate(&self, (view, tokens, units): Refs<'_, Self>) -> Result<(Categories<L>,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        if units.layer_spans().view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "spans are not in the given view".into() });
        }
        let text = view.text();
        validate_spans(Self::ID, "token", &tokens.spans, text)?;
        validate_spans(Self::ID, "unit", units.layer_spans(), text)?;
        let model = self.categorizer.model();
        let n_units = units.len();
        let n_tokens = tokens.spans.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let tokens_ordered = tokens.spans.is_start_ordered();
        let mut out = Categories::<L>::with_capacity(Arc::clone(model.categories()), n_units);
        let mut scores = Vec::with_capacity(model.num_labels());
        let mut scratch = Vec::new();
        let mut tok_strs = Vec::new();
        let extra = BTreeMap::new();

        for i in 0..n_units {
            tok_strs.clear();
            for frag in units.layer_spans().fragments(Ix::new(i as u32)) {
                if tokens_ordered {
                    let a = tok_starts.partition_point(|&s| s < frag.start);
                    for t in a..n_tokens {
                        if tok_starts[t] > frag.end {
                            break;
                        }
                        if tok_ends[t] <= frag.end {
                            tok_strs.push(&text[tok_starts[t] as usize..tok_ends[t] as usize]);
                        }
                    }
                } else {
                    for t in 0..n_tokens {
                        if tok_starts[t] >= frag.start && tok_ends[t] <= frag.end {
                            tok_strs.push(&text[tok_starts[t] as usize..tok_ends[t] as usize]);
                        }
                    }
                }
            }
            model.scores_into(&tok_strs, &extra, &mut scratch, &mut scores);
            let predicted = model.predicted_tags(&scores);
            out.push_selected(best_predicted(&scores, &predicted), &predicted, &scores);
        }

        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.model().model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::with_seed(self.categorizer.config_hash()).bytes(L::ID.as_bytes()).finish()
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Key, Layer, Provenance, Span, Spans, ViewId};
    use turbo_analysis_layers::{Block, Sentences, Structure};

    use super::*;
    use crate::features::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator, DoccatContextGenerator};
    use crate::multilabel::LinearHead;

    fn spans<L>(items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(start, end) in items {
            s.push(Span::new(start, end));
        }
        s
    }

    fn test_categorizer() -> MultiLabelCategorizer {
        // With empty tokens (&[]), "pos" has bias +1.0 (sigmoid ~0.73 > 0.5) and
        // "neg" has bias -1.0 (sigmoid ~0.27 < 0.5), so "pos" wins. With an empty
        // token (&[""]), "bow=" fires "neg" (+4.0) and suppresses "pos" (-4.0).
        let pos_head =
            LinearHead::new(1.0, vec![("bow=".into(), -4.0), ("bow=bad".into(), -4.0), ("bow=good".into(), 3.0)])
                .unwrap();
        let neg_head =
            LinearHead::new(-1.0, vec![("bow=".into(), 4.0), ("bow=bad".into(), 4.0), ("bow=good".into(), -3.0)])
                .unwrap();
        let cg = DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::new(),
        )]);
        let model =
            MultiLabelDoccatModel::new("en", &["pos", "neg"], vec![pos_head, neg_head], vec![0.5, 0.5], cg).unwrap();
        MultiLabelCategorizer::new(model)
    }

    #[test]
    fn document_categorizer_rejects_invalid_token_spans() {
        let cat = test_categorizer();
        let doc = Document::new("é");
        let view = doc.get::<View>().unwrap();
        let oob = Tokens { spans: spans(&[(0, 99)]), sentence: None };
        assert!(cat.annotate((view, &oob)).is_err());

        let bad_utf8 = Tokens { spans: spans(&[(0, 1)]), sentence: None };
        assert!(cat.annotate((view, &bad_utf8)).is_err());
    }

    #[test]
    fn span_categorizer_validates_spans_and_handles_empty_and_unordered_tokens() {
        let cat = test_categorizer();
        assert_eq!(cat.predicted_labels(&[])[0].0, "pos");
        assert_eq!(cat.predicted_labels(&[""])[0].0, "neg");
        let span_cat: MultiLabelSpanCategorizer<Sentences> = cat.for_spans();

        // Empty token [0..0] inside empty sentence [0..0] must be included (yielding "neg", not "pos").
        let mut doc_empty = Document::new("good");
        doc_empty
            .add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 0)]), sentence: None }, Provenance::by("test"))
            .unwrap();
        doc_empty.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 0)]) }, Provenance::by("test")).unwrap();
        let analyzer = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Sentences>::DEFAULT)
            .add(span_cat.clone())
            .build()
            .unwrap();
        let mut docs = [doc_empty];
        for r in analyzer.run(&mut docs) {
            r.unwrap();
        }
        let cats = docs[0].get::<Categories<Sentences>>().unwrap();
        assert_eq!(cats.best_category(0), "neg");

        // Start-ordered overlapping tokens keep the fast path.
        let overlap_spans = spans(&[(0, 3), (0, 3)]);
        assert!(overlap_spans.is_start_ordered() && !overlap_spans.is_ordered());
        let doc_overlap = Document::new("bad");
        let overlap_tokens = Tokens { spans: overlap_spans, sentence: None };
        let overlap_sents = Sentences { spans: spans(&[(0, 3)]) };
        let (cats_overlap,) =
            span_cat.annotate((doc_overlap.get::<View>().unwrap(), &overlap_tokens, &overlap_sents)).unwrap();
        assert_eq!(cats_overlap.best_category(0), "neg");

        // Unordered token spans: "good" at 4..8 listed before "bad" at 0..3.
        let doc_unordered = Document::new("bad good");
        let unordered_tokens = Tokens { spans: spans(&[(4, 8), (0, 3)]), sentence: None };
        let sents = Sentences { spans: spans(&[(0, 3), (4, 8)]) };
        let (cats_unordered,) =
            span_cat.annotate((doc_unordered.get::<View>().unwrap(), &unordered_tokens, &sents)).unwrap();
        assert_eq!(cats_unordered.best_category(0), "neg");
        assert_eq!(cats_unordered.best_category(1), "pos");

        // Invalid unit span is rejected without panicking.
        let doc_bad_unit = Document::new("good");
        let valid_tokens = Tokens { spans: spans(&[(0, 4)]), sentence: None };
        let bad_sents = Sentences { spans: spans(&[(0, 99)]) };
        assert!(span_cat.annotate((doc_bad_unit.get::<View>().unwrap(), &valid_tokens, &bad_sents)).is_err());

        // Discontinuous Structure block excludes gap tokens and validates fragments.
        let struct_cat: MultiLabelSpanCategorizer<Structure> = cat.for_spans();
        let doc_gap = Document::new("good bad good");
        let gap_tokens = Tokens { spans: spans(&[(0, 4), (5, 8), (9, 13)]), sentence: None };
        let mut st_spans = Spans::new(ViewId::SOURCE);
        st_spans.push_fragments(&[Span::new(0, 4), Span::new(9, 13)]);
        let structure = Structure { spans: st_spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] };
        let (st_cats,) = struct_cat.annotate((doc_gap.get::<View>().unwrap(), &gap_tokens, &structure)).unwrap();
        assert_eq!(st_cats.len(), 1);
        assert_eq!(st_cats.get(0).unwrap().best, "pos");
        assert_eq!(st_cats.row_scores(0), cat.categorize(&["good", "good"]).as_slice());
        assert_ne!(st_cats.row_scores(0), cat.categorize(&["good", "bad", "good"]).as_slice());
    }
}
