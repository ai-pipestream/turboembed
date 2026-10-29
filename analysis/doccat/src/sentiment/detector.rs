//! `SentimentME` at prediction, and the annotator that writes its
//! outcomes as `turbo:categories`.

use std::cell::RefCell;
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Key, Layer, ModelRef, Refs, Tag, TagSet, View};
use turbo_analysis_layers::{Categories, Tokens};
use turbo_maxent::best_outcome;

use super::model::SentimentModel;
use crate::java::double_compare;

/// The instance of `turbo:categories` [`SentimentMe::bound`] writes, so
/// that sentiment sits beside a topic categorizer's default instance.
pub const SENTIMENT: Key<Categories> = Key::named("sentiment");

/// The buffers one call of [`SentimentMe::probabilities`] reuses.
#[derive(Debug, Default)]
pub struct Scratch {
    extra: Vec<String>,
    probabilities: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// `SentimentME` over a [`SentimentModel`].
#[derive(Debug, Clone)]
pub struct SentimentMe {
    model: SentimentModel,
    /// The model's outcomes by index, for the layer.
    outcomes: Arc<TagSet>,
}

impl SentimentMe {
    pub fn new(model: SentimentModel) -> SentimentMe {
        let names: Vec<&str> = (0..model.maxent().num_outcomes()).map(|i| model.label(i)).collect();
        let outcomes = TagSet::new(<Categories>::ID, &names);
        SentimentMe { model, outcomes }
    }

    pub fn model(&self) -> &SentimentModel {
        &self.model
    }

    /// `probabilities(tokens)`: the maxent model's probability of each
    /// outcome, by outcome index, for the factory's context of `tokens`.
    pub fn probabilities<'s>(&self, tokens: &[&str], s: &'s mut Scratch) -> &'s [f64] {
        let maxent = self.model.maxent();
        s.probabilities.resize(maxent.num_outcomes(), 0.0);
        let context = self.model.factory().context(tokens, &mut s.extra);
        maxent.eval(&context, &mut s.probabilities);
        &s.probabilities
    }

    /// `getBestSentiment`: the label of the first highest probability.
    pub fn best_sentiment(&self, probabilities: &[f64]) -> &str {
        self.model.label(best_outcome(probabilities))
    }

    /// `predict(String[])`: the best label for `tokens`.
    pub fn predict(&self, tokens: &[&str], s: &mut Scratch) -> &str {
        let i = best_outcome(self.probabilities(tokens, s));
        self.model.label(i)
    }

    /// The outcomes best first: by probability as `Double.compare` orders,
    /// ties in outcome order, so the first is `getBestSentiment`'s.
    pub fn rank(&self, probabilities: &[f64], order: &mut Vec<usize>) {
        order.clear();
        order.extend(0..probabilities.len());
        order.sort_by(|&a, &b| double_compare(probabilities[b], probabilities[a]));
    }

    /// The `turbo:categories` layer of `tokens`: the best outcome
    /// (`getBestSentiment`) and every outcome's probability.
    pub fn categories(&self, tokens: &[&str], s: &mut Scratch) -> Categories {
        let probabilities = self.probabilities(tokens, s);
        let best = Tag(best_outcome(probabilities) as u16);
        Categories::document(Arc::clone(&self.outcomes), best, probabilities)
    }

    /// This annotator bound to write the [`SENTIMENT`] instance of
    /// `turbo:categories`.
    pub fn bound(self) -> Bound<SentimentMe> {
        Bound::new(self).write(0, SENTIMENT.name)
    }
}

thread_local! {
    /// The annotator's buffers, one set per thread.
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

/// Reads a view's tokens and writes `turbo:categories`: the best outcome
/// and every outcome's probability. The default instance is
/// written unless the step is bound otherwise ([`SentimentMe::bound`]).
impl Annotator for SentimentMe {
    const ID: &'static str = "turbo-doccat:sentiment";
    type Reads = (View, Tokens);
    type Writes = (Categories,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Categories,), Error> {
        if tokens.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "tokens are not in the given view".into() });
        }
        let words = crate::layer::extract_tokens(Self::ID, tokens, view.text())?;
        Ok((SCRATCH.with_borrow_mut(|s| self.categories(&words, s)),))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }
}
