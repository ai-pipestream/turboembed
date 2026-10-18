//! Document categorizer inference (`DocumentCategorizerME` and
//! `SpanCategorizer`), ported from `opennlp.tools.doccat.DocumentCategorizerME`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Ix, ModelRef, Refs, Tag, View};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_maxent::{MaxentModel, best_outcome};

use crate::features::DoccatContextGenerator;
use crate::layer::{SpanCategorizable, extract_tokens, validate_spans};
use crate::model::DoccatModel;

/// Maximum-entropy (or Perceptron / Naive Bayes) document categorizer, ported
/// from `opennlp.tools.doccat.DocumentCategorizerME`.
///
/// As an [`Annotator`], `DocumentCategorizerME` reads `(View, Tokens)` and
/// writes the document's `Categories` (`turbo:categories`): one item, the
/// best category with every category's probability. To categorize sentences (`Sentences`),
/// structural blocks (`Structure`), or chunks (`Chunks`), use
/// [`DocumentCategorizerME::for_spans`] (`SpanCategorizer<L>`).
#[derive(Clone, Debug)]
pub struct DocumentCategorizerME {
    model: DoccatModel,
}

impl DocumentCategorizerME {
    /// Creates a `DocumentCategorizerME` from a [`DoccatModel`].
    pub fn new(model: DoccatModel) -> Self {
        Self { model }
    }

    /// Returns a [`SpanCategorizer`] that categorizes each span of layer `L`
    /// (`Sentences`, `Structure`, or `Chunks`) using the tokens inside that
    /// span, writing `Categories<L>`.
    pub fn for_spans<L: SpanCategorizable>(&self) -> SpanCategorizer<L> {
        SpanCategorizer::new(self.model.clone())
    }

    /// Returns a reference to the underlying [`DoccatModel`].
    pub fn model(&self) -> &DoccatModel {
        &self.model
    }

    /// Returns the underlying [`MaxentModel`].
    pub fn maxent(&self) -> &MaxentModel {
        self.model.maxent()
    }

    /// Returns the context generator used to extract document features.
    pub fn context_generator(&self) -> &DoccatContextGenerator {
        self.model.context_generator()
    }

    /// Categorizes tokenized `text` with empty extra information and returns
    /// the per-category probability distribution in model outcome order.
    pub fn categorize(&self, text: &[&str]) -> Vec<f64> {
        let extra = BTreeMap::new();
        self.categorize_with_extra(text, &extra)
    }

    /// Categorizes tokenized `text` and `extra_information`, returning the
    /// per-category probability distribution in model outcome order.
    pub fn categorize_with_extra(&self, text: &[&str], extra_information: &BTreeMap<String, String>) -> Vec<f64> {
        let mut probs = vec![0.0; self.model.num_categories()];
        let mut scratch = Vec::new();
        self.categorize_into(text, extra_information, &mut scratch, &mut probs);
        probs
    }

    /// Categorizes tokenized `text` and `extra_information` into a pre-allocated
    /// `probs` buffer (`probs.len() == self.number_of_categories()`), reusing
    /// `feature_scratch` to avoid repeated vector allocations.
    pub fn categorize_into(
        &self,
        text: &[&str],
        extra_information: &BTreeMap<String, String>,
        feature_scratch: &mut Vec<String>,
        probs: &mut [f64],
    ) {
        self.model.context_generator().extract_features(text, extra_information, feature_scratch);
        let refs: Vec<&str> = feature_scratch.iter().map(String::as_str).collect();
        self.model.maxent().eval(&refs, probs);
    }

    /// Returns the winning category name from a computed `outcome` probability
    /// array (`DocumentCategorizerME.getBestCategory`).
    ///
    /// Ties break toward the lowest outcome index, matching
    /// `AbstractModel.getBestOutcome` (`outcome[i] > outcome[best]`).
    pub fn best_category(&self, outcome: &[f64]) -> &str {
        let idx = best_outcome(outcome);
        self.get_category(idx).unwrap_or("")
    }

    /// Returns the winning category outcome index from `outcome`.
    pub fn best_index(&self, outcome: &[f64]) -> usize {
        best_outcome(outcome)
    }

    /// Returns the outcome index of `category`, or `None` if `category` is not
    /// in the model (`DocumentCategorizerME.getIndex`).
    pub fn get_index(&self, category: &str) -> Option<usize> {
        self.model.index(category)
    }

    /// Returns the category name at `index`, or `None` if `index` is out of
    /// bounds (`DocumentCategorizerME.getCategory`).
    pub fn get_category(&self, index: usize) -> Option<&str> {
        self.model.category(index)
    }

    /// Returns the number of categories in the model
    /// (`DocumentCategorizerME.getNumberOfCategories`).
    pub fn number_of_categories(&self) -> usize {
        self.model.num_categories()
    }

    /// Formats all category outcomes and their probabilities as
    /// `cat0[0.1234]  cat1[0.8766]`, matching `DocumentCategorizerME.getAllResults`
    /// (`AbstractModel.getAllOutcomes`).
    pub fn all_results(&self, results: &[f64]) -> String {
        if results.len() != self.number_of_categories() || results.is_empty() {
            return String::new();
        }
        let mut sb = String::new();
        for (i, &p) in results.iter().enumerate() {
            if i > 0 {
                sb.push_str("  ");
            }
            let name = self.get_category(i).unwrap_or("");
            let _ = write!(sb, "{name}[{p:.4}]");
        }
        sb
    }

    /// Categorizes `text` and returns a map from category name to score
    /// (`DocumentCategorizerME.scoreMap`).
    pub fn score_map(&self, text: &[&str]) -> BTreeMap<String, f64> {
        let scores = self.categorize(text);
        let mut map = BTreeMap::new();
        for (i, &score) in scores.iter().enumerate() {
            if let Some(cat) = self.get_category(i) {
                map.insert(cat.to_owned(), score);
            }
        }
        map
    }

    /// Categorizes `text` and returns scores sorted in ascending order, each
    /// paired with the set of categories sharing that exact score
    /// (`DocumentCategorizerME.sortedScoreMap`).
    pub fn sorted_score_map(&self, text: &[&str]) -> Vec<(f64, BTreeSet<String>)> {
        let scores = self.categorize(text);
        let mut by_bits: BTreeMap<u64, (f64, BTreeSet<String>)> = BTreeMap::new();
        for (i, &score) in scores.iter().enumerate() {
            if let Some(cat) = self.get_category(i) {
                let normalized = if score == 0.0 { 0.0 } else { score };
                by_bits
                    .entry(normalized.to_bits())
                    .or_insert_with(|| (normalized, BTreeSet::new()))
                    .1
                    .insert(cat.to_owned());
            }
        }
        let mut out: Vec<(f64, BTreeSet<String>)> = by_bits.into_values().collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out
    }
}

impl Annotator for DocumentCategorizerME {
    const ID: &'static str = "opennlp:doccat";
    type Reads = (View, Tokens);
    type Writes = (Categories,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        let tok_strs = extract_tokens(Self::ID, tokens, view.text())?;
        let mut probs = vec![0.0; self.number_of_categories()];
        let mut scratch = Vec::new();
        let extra = BTreeMap::new();
        self.categorize_into(&tok_strs, &extra, &mut scratch, &mut probs);
        let best = Tag(self.best_index(&probs) as u16);
        Ok((Categories::document(Arc::clone(self.model.categories()), best, &probs),))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }
}

/// Categorizes each span of a positional layer `L` (`Sentences`, `Structure`,
/// or `Chunks`) using the tokens contained within that span
/// (`token.start >= unit.start && token.end <= unit.end`, including empty
/// tokens at either boundary), writing `Categories<L>` (`turbo:categories<L>`).
#[derive(Debug)]
pub struct SpanCategorizer<L: SpanCategorizable> {
    categorizer: DocumentCategorizerME,
    _of: PhantomData<fn() -> L>,
}

impl<L: SpanCategorizable> Clone for SpanCategorizer<L> {
    fn clone(&self) -> Self {
        Self { categorizer: self.categorizer.clone(), _of: PhantomData }
    }
}

impl<L: SpanCategorizable> SpanCategorizer<L> {
    /// Creates a `SpanCategorizer<L>` from a [`DoccatModel`].
    pub fn new(model: DoccatModel) -> Self {
        Self { categorizer: DocumentCategorizerME::new(model), _of: PhantomData }
    }

    /// Returns the underlying [`DocumentCategorizerME`].
    pub fn categorizer(&self) -> &DocumentCategorizerME {
        &self.categorizer
    }
}

impl<L: SpanCategorizable> Annotator for SpanCategorizer<L> {
    const ID: &'static str = "opennlp:doccat-spans";
    const VERSION: &'static str = "1";
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
        let n_units = units.len();
        let n_tokens = tokens.spans.len();
        let (tok_starts, tok_ends) = tokens.spans.columns();
        let tokens_ordered = tokens.spans.is_start_ordered();
        let mut out = Categories::<L>::with_capacity(Arc::clone(self.categorizer.model().categories()), n_units);
        let mut probs = vec![0.0; self.categorizer.number_of_categories()];
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
            self.categorizer.categorize_into(&tok_strs, &extra, &mut scratch, &mut probs);
            let best_idx = self.categorizer.best_index(&probs);
            out.push(Tag(best_idx as u16), &probs);
        }

        Ok((out,))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.categorizer.model().model_ref().clone())
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Key, Layer, Provenance, Span, Spans, ViewId};
    use turbo_analysis_layers::{Block, Sentences, Structure};
    use turbo_train::{Event, Key as Param, TrainingConfiguration, TrainingParameters, trainer};

    use super::*;
    use crate::features::{BagOfWordsFeatureGenerator, BuiltinFeatureGenerator};

    fn spans<L>(items: &[(u32, u32)]) -> Spans<L> {
        let mut s = Spans::new(ViewId::SOURCE);
        for &(start, end) in items {
            s.push(Span::new(start, end));
        }
        s
    }

    fn test_categorizer() -> DocumentCategorizerME {
        // Outcome 0 is "pos", so categorize(&[]) picks "pos", while "bow=" is
        // associated with "neg", so categorize(&[""]) picks "neg".
        let events = vec![
            Event::from_strs("pos", &["bow=good"]),
            Event::from_strs("pos", &["bow=good", "bow=great"]),
            Event::from_strs("neg", &["bow=bad", "bow="]),
            Event::from_strs("neg", &["bow=bad", "bow=awful"]),
        ];
        let mut params = TrainingParameters::new();
        params.put(Param::ALGORITHM, "MAXENT");
        params.put(Param::ITERATIONS, 50);
        params.put(Param::CUTOFF, 0);
        let (trained, _) = trainer::train_events(&events, &params, &mut TrainingConfiguration::default()).unwrap();
        let maxent = MaxentModel::read(&trained.write().unwrap()).unwrap();
        let cg = DoccatContextGenerator::from_builtins(vec![BuiltinFeatureGenerator::BagOfWords(
            BagOfWordsFeatureGenerator::new(),
        )]);
        DocumentCategorizerME::new(DoccatModel::from_maxent("en", maxent, cg).unwrap())
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
        assert_eq!(cat.best_category(&cat.categorize(&[])), "pos");
        assert_eq!(cat.best_category(&cat.categorize(&[""])), "neg");
        let span_cat: SpanCategorizer<Sentences> = cat.for_spans();

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
        assert_eq!(docs[0].provenance(Key::<Categories<Sentences>>::DEFAULT.id()).unwrap().version, "1");

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
        let struct_cat: SpanCategorizer<Structure> = cat.for_spans();
        let doc_gap = Document::new("good bad good");
        let gap_tokens = Tokens { spans: spans(&[(0, 4), (5, 8), (9, 13)]), sentence: None };
        let mut st_spans = Spans::new(ViewId::SOURCE);
        st_spans.push_fragments(&[Span::new(0, 4), Span::new(9, 13)]);
        let structure = Structure { spans: st_spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] };
        let (st_cats,) = struct_cat.annotate((doc_gap.get::<View>().unwrap(), &gap_tokens, &structure)).unwrap();
        assert_eq!(st_cats.len(), 1);
        assert_eq!(st_cats.get(0).unwrap().best, "pos");
    }
}
