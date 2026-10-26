use super::FeedforwardDependencyModel;
use super::sentence::SentenceIds;
use crate::{ArcStandardState, DependencyGraph, DependencyParser, ParserInput, Result, error};
use std::sync::Arc;
use turbo_analysis::ModelRef;
use turbo_analysis_layers::WordOrTokenAnchor;

#[derive(Clone)]
pub struct FeedforwardDependencyParser {
    model: Arc<FeedforwardDependencyModel>,
    beam_size: usize,
    identity: ModelRef,
}
impl FeedforwardDependencyParser {
    pub fn new(model: FeedforwardDependencyModel, beam_size: usize) -> Result<Self> {
        if beam_size == 0 {
            return Err(error("beam size must be positive"));
        }
        let identity = model.identity()?;
        Ok(Self { model: Arc::new(model), beam_size, identity })
    }
    pub fn model(&self) -> &FeedforwardDependencyModel {
        &self.model
    }
    pub fn beam_size(&self) -> usize {
        self.beam_size
    }
    pub fn identity(&self) -> ModelRef {
        self.identity.clone()
    }
    pub(crate) fn config_hash(&self) -> u64 {
        self.beam_size as u64
    }
    pub fn parse<A: WordOrTokenAnchor>(&self, words: &[&str], pos: &[&str]) -> Result<DependencyGraph<A>> {
        ParserInput::check(words, pos)?;
        // Each step reads its feature ids from the sentence's resolved rows,
        // into buffers reused across steps.
        let sentence = SentenceIds::new(&self.model, words, pos);
        let (mut positions, mut ids) = (Vec::new(), Vec::new());
        if self.beam_size == 1 {
            let mut state = ArcStandardState::<A>::new(words.len())?;
            while !state.is_terminal() {
                sentence.fill(&self.model, &state, &mut positions, &mut ids);
                let scores = self.model.score(&ids)?;
                let mut best = None;
                for (i, t) in self.model.transitions.iter().enumerate() {
                    if state.can_apply(t) && best.is_none_or(|b: usize| scores[i] > scores[b]) {
                        best = Some(i);
                    }
                }
                let best = best.ok_or_else(|| error("no applicable neural transition"))?;
                state.apply(&self.model.transitions[best])?;
            }
            return state.to_graph()?.deprojectivize();
        }
        let mut beam = vec![(ArcStandardState::<A>::new(words.len())?, 0.0)];
        while !beam.iter().all(|(s, _)| s.is_terminal()) {
            let mut candidates = Vec::new();
            for (index, (state, score)) in beam.iter().enumerate() {
                if state.is_terminal() {
                    candidates.push((*score, index, None));
                    continue;
                }
                sentence.fill(&self.model, state, &mut positions, &mut ids);
                let scores = self.model.score(&ids)?;
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_z = scores.iter().map(|s| (s - max).exp()).sum::<f64>().ln();
                for (outcome, t) in self.model.transitions.iter().enumerate() {
                    if state.can_apply(t) {
                        candidates.push((score + (scores[outcome] - max) - log_z, index, Some(outcome)));
                    }
                }
            }
            // Stable order preserves the Java beam/outcome tie break.
            candidates.sort_by(|a, b| b.0.total_cmp(&a.0));
            candidates.truncate(self.beam_size);
            if candidates.is_empty() {
                return Err(error("no applicable neural transition"));
            }
            let mut next = Vec::with_capacity(candidates.len());
            for (score, index, outcome) in candidates {
                let mut state = beam[index].0.clone();
                if let Some(outcome) = outcome {
                    state.apply(&self.model.transitions[outcome])?;
                }
                next.push((state, score));
            }
            beam = next;
        }
        beam[0].0.to_graph()?.deprojectivize()
    }
}
impl<A: WordOrTokenAnchor> DependencyParser<A> for FeedforwardDependencyParser {
    fn parse(&self, words: &[&str], tags: &[&str]) -> Result<DependencyGraph<A>> {
        self.parse(words, tags)
    }
}
