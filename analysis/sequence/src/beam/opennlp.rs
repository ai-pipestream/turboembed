use super::{BeamSearch, ContextGenerator, OutcomeLabel, Sequence, SequenceValidator};

/// One caller's OpenNLP beam evaluation state, with no evaluation cache.
///
/// QN starts each evaluation with the previous probabilities, including
/// across sentences. GIS, perceptron and Naive Bayes overwrite them. Keep
/// one session per independent stream (one per worker to match Java's
/// thread-local state). Calls require exclusive access; no global or
/// thread-local Rust state is used. The borrowed search fixes the model
/// and outcome order for this session's entire lifetime.
///
/// Native [`BeamSearch`] methods always start each evaluation fresh.
pub struct OpenNlpBeamSearch<'s, 'm, L: OutcomeLabel + ?Sized = str> {
    search: &'s BeamSearch<'m, L>,
    probs: Vec<f64>,
}

impl<'m, L: OutcomeLabel + ?Sized> BeamSearch<'m, L> {
    /// Starts an independent OpenNLP compatibility session with zero sums.
    /// Retain it across calls to reproduce QN's state across sentences.
    pub fn opennlp(&self) -> OpenNlpBeamSearch<'_, 'm, L> {
        OpenNlpBeamSearch { search: self, probs: vec![0.0; self.outcomes.len()] }
    }
}

impl<'m, L: OutcomeLabel + ?Sized> OpenNlpBeamSearch<'_, 'm, L> {
    /// Drops carried probabilities, as closing Java's beam search does for
    /// the current thread. This does not clear a context's adaptive data.
    pub fn reset(&mut self) {
        self.probs.fill(0.0);
    }

    /// The best sequence using this session's carried QN probabilities.
    pub fn best_sequence<T>(
        &mut self,
        sequence: &[T],
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Option<Sequence<'m, L>> {
        self.best_sequences(1, sequence, BeamSearch::<L>::DEFAULT_MIN_SCORE, context, validator).pop()
    }

    /// Up to `count` best sequences, with the same ordering and threshold
    /// as [`BeamSearch::best_sequences`]. Even when no result is requested
    /// or every path is rejected, completed evaluations update QN state,
    /// matching the JVM. An empty input performs no evaluation.
    pub fn best_sequences<T>(
        &mut self,
        count: usize,
        sequence: &[T],
        min_score: f64,
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Vec<Sequence<'m, L>> {
        self.search.best_sequences_inner::<true, T>(count, sequence, min_score, context, validator, &mut self.probs)
    }
}
