//! Held-out accounting: failed predictions remain in every relevant denominator.
use crate::{DependencyEvaluator, DependencyGraph, DependencySample, Result, error};
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvaluationSubset {
    pub sentences: u64,
    pub failed: u64,
    pub scores: DependencyEvaluator,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeldOutEvaluation {
    pub all: EvaluationSubset,
    pub projective: EvaluationSubset,
    pub nonprojective: EvaluationSubset,
}
impl HeldOutEvaluation {
    /// Missing/invalid predictions get zero credit, never a smaller denominator.
    /// Returns false when the prediction was missing or not a valid aligned tree.
    pub fn record<A: WordOrTokenAnchor>(
        &mut self,
        gold: &DependencySample<A>,
        predicted: Option<&DependencyGraph<A>>,
    ) -> Result<bool> {
        gold.graph().check_tree()?;
        let mut one = DependencyEvaluator::default();
        let valid = predicted.is_some_and(|graph| one.score(gold, graph).is_ok());
        if !valid {
            one.all.words = gold.tokens().len() as u64;
            one.without_punctuation.words = gold.tags().iter().filter(|tag| *tag != "PUNCT").count() as u64;
        }
        let mut next = *self;
        add(&mut next.all, one, !valid)?;
        let subset = if gold.graph().is_projective() { &mut next.projective } else { &mut next.nonprojective };
        add(subset, one, !valid)?;
        *self = next;
        Ok(valid)
    }
    /// Caller-selected token-weighted floors on all syntactic words. Empty
    /// populations and any failed parse refuse, even if the numeric floors pass.
    pub fn passes(&self, minimum_uas: f64, minimum_las: f64) -> Result<bool> {
        if [minimum_uas, minimum_las].iter().any(|n| !n.is_finite() || !(0.0..=1.0).contains(n)) {
            return Err(error("UAS/LAS floors must be finite fractions in [0,1]"));
        }
        Ok(self.all.scores.all.words > 0
            && self.all.failed == 0
            && self.all.scores.all.uas() >= minimum_uas
            && self.all.scores.all.las() >= minimum_las)
    }
    pub fn write_tsv(&self) -> String {
        let mut out = String::from("subset\tpunctuation\tsentences\tfailed\twords\theads\tlabels\tuas\tlas\n");
        for (name, subset) in
            [("all", self.all), ("projective", self.projective), ("nonprojective", self.nonprojective)]
        {
            for (punct, scores) in [("included", subset.scores.all), ("excluded", subset.scores.without_punctuation)] {
                out.push_str(&format!(
                    "{name}\t{punct}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\n",
                    subset.sentences,
                    subset.failed,
                    scores.words,
                    scores.heads,
                    scores.labels,
                    scores.uas(),
                    scores.las()
                ));
            }
        }
        out
    }
}
fn add(to: &mut EvaluationSubset, scores: DependencyEvaluator, failed: bool) -> Result<()> {
    let plus = |a: &mut u64, b| -> Result<()> {
        *a = a.checked_add(b).ok_or_else(|| error("evaluation counter overflow"))?;
        Ok(())
    };
    plus(&mut to.sentences, 1)?;
    plus(&mut to.failed, u64::from(failed))?;
    for (target, source) in
        [(&mut to.scores.all, scores.all), (&mut to.scores.without_punctuation, scores.without_punctuation)]
    {
        plus(&mut target.words, source.words)?;
        plus(&mut target.heads, source.heads)?;
        plus(&mut target.labels, source.labels)?;
    }
    Ok(())
}
