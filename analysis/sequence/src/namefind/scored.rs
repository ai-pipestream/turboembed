//! Optional full token distributions along the decoded best prefix.
use super::*;
impl NameFinder<'_> {
    /// Find names and return one full model distribution per token, in
    /// model outcome order, conditioned on the decoded prefix. These are
    /// local tag probabilities, not CRF marginals or joint sequence sets.
    /// Adaptive state advances exactly once, as in `find`.
    pub fn find_scored(&mut self, tokens: &[&str]) -> (Vec<FoundName>, Vec<Vec<f64>>) {
        self.context.set_additional_context(&[]);
        let Some((best, rows)) = self.search.best_sequence_scored(tokens, &mut self.context, &self.codec) else {
            self.probs.clear();
            return (Vec::new(), Vec::new());
        };
        self.context.update_adaptive_data(tokens, best.outcomes());
        self.probs.clear();
        self.probs.extend_from_slice(best.probs());
        let names = self
            .codec
            .decode(best.outcomes())
            .into_iter()
            .map(|span| {
                let prob = best.probs()[span.start..span.end].iter().sum::<f64>() / span.len() as f64;
                FoundName { span, prob }
            })
            .collect();
        (names, rows)
    }
}
