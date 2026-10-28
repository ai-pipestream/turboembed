//! Training a span classifier: the loop of `fit` over sequences with
//! gold spans, scored on dev by exact span F1 through the core's own
//! classifier (`turbo::head::SpanClassifier`), decoded as at inference.

use turbo::head::SpanClassifier;
use turbo::head::span::SpanWork;
use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, Report, check_config, train};
use crate::rng::Rng;
use crate::span::{GoldSpan, Spans, Work};

/// One training sequence: per-word features [words, hidden] and its gold
/// spans, none labelled `O`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpanExample {
    pub features: Vec<f32>,
    pub spans: Vec<GoldSpan>,
}

/// Exact span F1 of `head` on `examples`: a span counts when its words
/// and label are a gold span's. 1 when neither side has a span.
pub fn span_f1(head: &SpanClassifier, examples: &[SpanExample]) -> Result<f64> {
    let hidden = head.hidden();
    let (mut right, mut found, mut gold) = (0usize, 0usize, 0usize);
    let (mut scores, mut kept, mut w, mut sw) = (Vec::new(), Vec::new(), CoreWork::default(), SpanWork::default());
    for (i, e) in examples.iter().enumerate() {
        if hidden == 0 || !e.features.len().is_multiple_of(hidden) {
            return Err(invalid(format!("trainer: example {i} is not rows of {hidden} features")));
        }
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        head.scores(&e.features, &words, None, &mut scores, &mut w, &mut sw)?;
        head.decode(&scores, n, &mut kept)?;
        right += kept
            .iter()
            .filter(|s| e.spans.contains(&GoldSpan { first: s.first, last: s.last, label: s.label }))
            .count();
        found += kept.len();
        gold += e.spans.len();
    }
    Ok(if found + gold == 0 { 1.0 } else { 2.0 * right as f64 / (found + gold) as f64 })
}

impl Model for Spans<f32> {
    type Example = SpanExample;
    type Work = Work<f32>;
    fn params(&self) -> &[Vec<f32>] {
        Spans::params(self)
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        Spans::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        Spans::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        Spans::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        Spans::decays(self)
    }
    fn loss(&self, e: &SpanExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        Spans::loss(self, &e.features, &e.spans, grads, w)
    }
    fn score(&self, dev: &[SpanExample]) -> Result<f64> {
        span_f1(&self.to_core()?, dev)
    }
}

/// Trains a span classifier for `spec` over features `hidden` wide on
/// `train_set`, keeping the weights with the best dev span F1.
pub fn fit_spans(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[SpanExample],
    dev: &[SpanExample],
    c: &Config,
) -> Result<(Spans<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let model = Spans::<f32>::new(spec, hidden, &mut rng)?;
    // Each example's spans are checked by the loss before any gradient
    // moves; check dev's the same way now, so a bad dev set fails first.
    let mut w = Work::default();
    let mut grads = model.zero_grads();
    for e in dev {
        model.loss(&e.features, &e.spans, &mut grads, &mut w)?;
    }
    train(model, &mut rng, train_set, dev, c, &mut |_, _, _| Ok(()))
}
