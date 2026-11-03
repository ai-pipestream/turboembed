//! Training an open-label span typing head (`HEAD_OPEN_SPAN`): the loop of
//! `fit` over sequences with caller label vectors and gold spans, scored on
//! dev by exact span F1 through the core's own matcher
//! (`turbo::head::OpenSpanMatcher`), decoded as at inference.

use turbo::head::OpenSpanMatcher;
use turbo::head::open_span::OpenSpanWork;
use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::adamw::AdamW;
use crate::fit::{Config, Model, Report, check_config, train};
use crate::open_span::{OpenSpans, Work};
use crate::rng::Rng;
use crate::span::GoldSpan;

/// One training or evaluation sequence: per-word features `[words, hidden]`,
/// caller label vectors `[num_labels, hidden]`, and gold spans whose `label`
/// indexes `0..num_labels`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenSpanExample {
    pub features: Vec<f32>,
    pub label_features: Vec<f32>,
    pub spans: Vec<GoldSpan>,
}

/// Exact span F1 of `head` on `examples`: a span counts when its words and
/// label index match a gold span's. 1 when neither side has a span.
pub fn open_span_f1(head: &OpenSpanMatcher, examples: &[OpenSpanExample]) -> Result<f64> {
    let hidden = head.hidden();
    let (mut right, mut found, mut gold) = (0usize, 0usize, 0usize);
    let (mut scores, mut kept, mut w, mut sw) = (Vec::new(), Vec::new(), CoreWork::default(), OpenSpanWork::default());
    for (i, e) in examples.iter().enumerate() {
        if hidden == 0 || !e.features.len().is_multiple_of(hidden) {
            return Err(invalid(format!("trainer: example {i} is not rows of {hidden} features")));
        }
        if e.label_features.is_empty() || !e.label_features.len().is_multiple_of(hidden) {
            return Err(invalid(format!("trainer: example {i} has invalid label features for hidden {hidden}")));
        }
        let n = e.features.len() / hidden;
        let num_labels = e.label_features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        head.scores(&e.features, &words, None, &e.label_features, &mut scores, &mut w, &mut sw)?;
        head.decode(&scores, n, num_labels, &mut kept)?;
        right += kept
            .iter()
            .filter(|s| e.spans.contains(&GoldSpan { first: s.first, last: s.last, label: s.label }))
            .count();
        found += kept.len();
        gold += e.spans.len();
    }
    Ok(if found + gold == 0 { 1.0 } else { 2.0 * right as f64 / (found + gold) as f64 })
}

impl Model for OpenSpans<f32> {
    type Example = OpenSpanExample;
    type Work = Work<f32>;
    fn params(&self) -> Vec<Vec<f32>> {
        OpenSpans::params(self).to_vec()
    }
    fn set_params(&mut self, params: Vec<Vec<f32>>) -> Result<()> {
        OpenSpans::set_params(self, params)
    }
    fn step(&mut self, opt: &mut AdamW<f32>, grads: &mut [Vec<f32>], decay: &[bool]) -> Result<f64> {
        OpenSpans::step(self, opt, grads, decay)
    }
    fn zero_grads(&self) -> Vec<Vec<f32>> {
        OpenSpans::zero_grads(self)
    }
    fn decays(&self) -> Vec<bool> {
        OpenSpans::decays(self)
    }
    fn loss(&self, e: &OpenSpanExample, grads: &mut [Vec<f32>], w: &mut Work<f32>) -> Result<(f64, usize)> {
        OpenSpans::loss(self, &e.features, &e.label_features, &e.spans, grads, w)
    }
    fn score(&self, dev: &[OpenSpanExample]) -> Result<f64> {
        open_span_f1(&self.to_core()?, dev)
    }
}

/// Trains an open-label span matcher for `spec` over features `hidden` wide
/// on `train_set`, keeping the weights with the best dev span F1.
pub fn fit_open_spans(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[OpenSpanExample],
    dev: &[OpenSpanExample],
    c: &Config,
) -> Result<(OpenSpans<f32>, Report)> {
    fit_open_spans_with(spec, hidden, train_set, dev, c, |_, _, _| Ok(()))
}

/// [`fit_open_spans`], calling `each_epoch(&model, epoch, dev_f1)` at the
/// end of every epoch.
pub fn fit_open_spans_with(
    spec: &HeadSpec,
    hidden: usize,
    train_set: &[OpenSpanExample],
    dev: &[OpenSpanExample],
    c: &Config,
    mut each_epoch: impl FnMut(&OpenSpans<f32>, usize, f64) -> Result<()>,
) -> Result<(OpenSpans<f32>, Report)> {
    check_config(c, train_set.len())?;
    let mut rng = Rng::new(c.seed);
    let mut model = OpenSpans::<f32>::new(spec, hidden, &mut rng)?;
    if let Some((cx, cq)) = empirical_centers(train_set, hidden) {
        model.set_centers(&cx, &cq);
    }
    let mut w = Work::default();
    let mut grads = model.zero_grads();
    for e in dev {
        model.loss(&e.features, &e.label_features, &e.spans, &mut grads, &mut w)?;
    }
    train(model, &mut rng, train_set, dev, c, &mut each_epoch)
}

fn empirical_centers(train_set: &[OpenSpanExample], hidden: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    if hidden == 0 {
        return None;
    }
    let mut sum_x = vec![0.0f64; hidden];
    let mut count_x = 0usize;
    let mut sum_q = vec![0.0f64; hidden];
    let mut count_q = 0usize;
    for e in train_set {
        if !e.features.len().is_multiple_of(hidden) || !e.label_features.len().is_multiple_of(hidden) {
            continue;
        }
        let n = e.features.len() / hidden;
        for sp in &e.spans {
            let (first, last) = (sp.first as usize, sp.last as usize);
            if first <= last && last < n {
                for i in first..=last {
                    let row = &e.features[i * hidden..(i + 1) * hidden];
                    let norm = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt().max(1e-6);
                    for (s, &v) in sum_x.iter_mut().zip(row) {
                        *s += (v as f64) / norm;
                    }
                    count_x += 1;
                }
            }
        }
    }
    if let Some(first_ex) = train_set.first()
        && first_ex.label_features.len().is_multiple_of(hidden)
    {
        for row in first_ex.label_features.chunks_exact(hidden) {
            let norm = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt().max(1e-6);
            for (s, &v) in sum_q.iter_mut().zip(row) {
                *s += (v as f64) / norm;
            }
            count_q += 1;
        }
    }
    if count_x == 0 || count_q == 0 {
        return None;
    }
    let cx: Vec<f32> = sum_x.iter().map(|&s| (s / count_x as f64) as f32).collect();
    let cq: Vec<f32> = sum_q.iter().map(|&s| (s / count_q as f64) as f32).collect();
    let norm_x = cx.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    let norm_q = cq.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    (norm_x > 0.5 && norm_q > 0.5).then_some((cx, cq))
}
