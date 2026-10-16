//! The categorizer, OpenNLP's `DocumentCategorizerSVM` at prediction:
//! `categorize`, `getBestCategory`, `getIndex`, `getCategory`,
//! `getNumberOfCategories` and `getAllResults`, and the annotator that
//! writes `turbo:categories`.

use std::cell::RefCell;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, View};
use turbo_analysis_layers::{Categories, Strs, Tokens};
use turbo_jvm::text::double_to_string;

use super::error::ModelError;
use super::model::SvmDoccatModel;
use super::vector::{Scratch, vectorize};
use crate::java::double_compare;

/// `DocumentCategorizerSVM` over an [`SvmDoccatModel`].
#[derive(Debug)]
pub struct SvmCategorizer {
    model: SvmDoccatModel,
}

impl SvmCategorizer {
    /// A categorizer over `model`, which must have probability estimates:
    /// `categorize` asks LIBSVM for them, and OpenNLP throws on every
    /// document of a model without them ([`ModelError::NoProbabilityEstimates`]).
    pub fn new(model: SvmDoccatModel) -> Result<SvmCategorizer, ModelError> {
        if !model.libsvm().has_probability() {
            return Err(ModelError::NoProbabilityEstimates);
        }
        Ok(SvmCategorizer { model })
    }

    pub fn model(&self) -> &SvmDoccatModel {
        &self.model
    }

    /// `categorize(String[])`: the probability of each category, by
    /// category index. A category the LIBSVM model has no label for
    /// scores 0.
    pub fn categorize<'s>(&self, tokens: &[&str], s: &'s mut Scratch) -> &'s [f64] {
        let svm = self.model.libsvm();
        vectorize(self.model.sidecar(), tokens, s);
        svm.predict_probability(&s.index, &s.value, &mut s.predict);
        s.outcome.clear();
        s.outcome.resize(self.num_categories(), 0.0);
        for (&label, &p) in svm.labels().iter().zip(s.predict.probabilities()) {
            // The model's labels were checked to be category indexes.
            s.outcome[label as usize] = p;
        }
        &s.outcome
    }

    /// `getBestCategory`: the category of the first highest score.
    pub fn best_category(&self, outcome: &[f64]) -> &str {
        let mut best = 0;
        for i in 1..outcome.len() {
            if outcome[i] > outcome[best] {
                best = i;
            }
        }
        self.category(best)
    }

    /// `getIndex`: `None` where OpenNLP returns -1.
    pub fn index(&self, category: &str) -> Option<usize> {
        self.model.categories().iter().position(|c| c == category)
    }

    /// `getCategory`.
    pub fn category(&self, index: usize) -> &str {
        &self.model.categories()[index]
    }

    /// `getNumberOfCategories`.
    pub fn num_categories(&self) -> usize {
        self.model.categories().len()
    }

    /// `getAllResults`: each category and its score in brackets, as
    /// `Double.toString` writes it, separated by spaces.
    pub fn all_results(&self, outcome: &[f64]) -> String {
        let mut out = String::new();
        for (i, &p) in outcome.iter().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            out.push_str(self.category(i));
            out.push('[');
            out.push_str(&double_to_string(p));
            out.push(']');
        }
        out
    }

    /// The categories best first: by score as `Double.compare` orders,
    /// ties in category order, so the first is `getBestCategory`'s.
    pub fn rank(&self, outcome: &[f64], order: &mut Vec<usize>) {
        order.clear();
        order.extend(0..outcome.len());
        order.sort_by(|&a, &b| double_compare(outcome[b], outcome[a]));
    }
}

thread_local! {
    /// The annotator's buffers, one set per thread.
    static SCRATCH: RefCell<(Scratch, Vec<usize>)> = RefCell::new((Scratch::new(), Vec::new()));
}

/// Reads a view's tokens and writes `turbo:categories`: every category of
/// the model, best first, with its probability.
impl Annotator for SvmCategorizer {
    const ID: &'static str = "turbo-doccat:svm";
    type Reads = (View, Tokens);
    type Writes = (Categories,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Categories,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        let text = view.text();
        let words: Vec<&str> = tokens.spans.iter().map(|(_, s)| &text[s.range()]).collect();
        SCRATCH.with_borrow_mut(|(scratch, order)| {
            let outcome = self.categorize(&words, scratch);
            self.rank(outcome, order);
            let mut category = Strs::default();
            let mut score = Vec::with_capacity(order.len());
            for &i in order.iter() {
                category.push(self.category(i));
                score.push(outcome[i]);
            }
            Ok((Categories { category, score },))
        })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }
}
