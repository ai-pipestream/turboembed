use crate::{DependencyGraph, DependencyParser, DependencySample, Result, error};
use turbo_analysis::Ix;
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AttachmentScores {
    pub words: u64,
    pub heads: u64,
    pub labels: u64,
}
impl AttachmentScores {
    pub fn uas(&self) -> f64 {
        if self.words == 0 { 0.0 } else { self.heads as f64 / self.words as f64 }
    }
    pub fn las(&self) -> f64 {
        if self.words == 0 { 0.0 } else { self.labels as f64 / self.words as f64 }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DependencyEvaluator {
    pub all: AttachmentScores,
    pub without_punctuation: AttachmentScores,
}
impl DependencyEvaluator {
    pub fn evaluate<A: WordOrTokenAnchor, P: DependencyParser<A>>(
        &mut self,
        parser: &P,
        sample: &DependencySample<A>,
    ) -> Result<DependencyGraph<A>> {
        self.evaluate_with_punctuation(parser, sample, |tag| tag == "PUNCT")
    }
    pub fn evaluate_with_punctuation<A: WordOrTokenAnchor, P: DependencyParser<A>>(
        &mut self,
        parser: &P,
        sample: &DependencySample<A>,
        punct: impl Fn(&str) -> bool,
    ) -> Result<DependencyGraph<A>> {
        let tokens: Vec<_> = sample.tokens().iter().map(String::as_str).collect();
        let tags: Vec<_> = sample.tags().iter().map(String::as_str).collect();
        let predicted = parser.parse(&tokens, &tags)?;
        self.score_with_punctuation(sample, &predicted, punct)?;
        Ok(predicted)
    }
    pub fn score<A: WordOrTokenAnchor>(
        &mut self,
        gold: &DependencySample<A>,
        predicted: &DependencyGraph<A>,
    ) -> Result<()> {
        self.score_with_punctuation(gold, predicted, |tag| tag == "PUNCT")
    }
    pub fn score_with_punctuation<A: WordOrTokenAnchor>(
        &mut self,
        gold: &DependencySample<A>,
        predicted: &DependencyGraph<A>,
        punct: impl Fn(&str) -> bool,
    ) -> Result<()> {
        if gold.graph().len() != predicted.len() {
            return Err(error("prediction length differs from reference"));
        }
        predicted.check_tree()?;
        let mut next = *self;
        for i in 0..predicted.len() {
            let gold_arc = gold.graph().arc(Ix::new(i as u32))?;
            let pred = predicted.arc(Ix::new(i as u32))?;
            let head = gold_arc.head == pred.head;
            let label = head && gold_arc.relation == pred.relation;
            let add = |s: &mut AttachmentScores| -> Result<()> {
                s.words = s.words.checked_add(1).ok_or_else(|| error("score counter overflow"))?;
                s.heads = s.heads.checked_add(u64::from(head)).ok_or_else(|| error("score counter overflow"))?;
                s.labels = s.labels.checked_add(u64::from(label)).ok_or_else(|| error("score counter overflow"))?;
                Ok(())
            };
            add(&mut next.all)?;
            if !punct(&gold.tags()[i]) {
                add(&mut next.without_punctuation)?;
            }
        }
        *self = next;
        Ok(())
    }
}
/// Round-robin held-out folds, matching CrossValidationPartitioner. Each sample
/// is evaluated once; totals are token-weighted rather than fold-averaged.
#[derive(Default)]
pub struct DependencyCrossValidator {
    pub scores: DependencyEvaluator,
}
impl DependencyCrossValidator {
    pub fn evaluate<A: WordOrTokenAnchor, P: DependencyParser<A>>(
        &mut self,
        samples: &[DependencySample<A>],
        folds: usize,
        mut trainer: impl FnMut(&[&DependencySample<A>]) -> Result<P>,
    ) -> Result<()> {
        self.evaluate_with_punctuation(samples, folds, &mut trainer, |tag| tag == "PUNCT")
    }
    pub fn evaluate_with_punctuation<A: WordOrTokenAnchor, P: DependencyParser<A>>(
        &mut self,
        samples: &[DependencySample<A>],
        folds: usize,
        mut trainer: impl FnMut(&[&DependencySample<A>]) -> Result<P>,
        punct: impl Fn(&str) -> bool,
    ) -> Result<()> {
        if folds < 2 || folds > samples.len() {
            return Err(error("fold count must be between 2 and sample count"));
        }
        let mut scores = self.scores;
        for fold in 0..folds {
            let training: Vec<_> =
                samples.iter().enumerate().filter_map(|(i, s)| (i % folds != fold).then_some(s)).collect();
            let parser = trainer(&training)?;
            for (i, s) in samples.iter().enumerate() {
                if i % folds == fold {
                    scores.evaluate_with_punctuation(&parser, s, &punct)?;
                }
            }
        }
        self.scores = scores;
        Ok(())
    }
}
