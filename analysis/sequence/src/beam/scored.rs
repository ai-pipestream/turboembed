//! Optional probability capture, separate from the ordinary search hot path.
use super::*;
impl<'m, L: OutcomeLabel + ?Sized> BeamSearch<'m, L> {
    /// Best sequence with the full distributions already evaluated along
    /// its prefix. Ordinary search uses a separate path without capture.
    pub fn best_sequence_scored<T>(
        &self,
        sequence: &[T],
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Option<(Sequence<'m, L>, Vec<Vec<f64>>)> {
        self.best_sequence_scored_flat(sequence, context, validator)
            .map(|(s, rows)| (s, rows.chunks(self.outcomes.len().max(1)).map(<[f64]>::to_vec).collect()))
    }

    /// Best sequence with row-major scores in model outcome order.
    /// This avoids allocating a separate vector for each token.
    pub fn best_sequence_scored_flat<T>(
        &self,
        sequence: &[T],
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Option<(Sequence<'m, L>, Vec<f64>)> {
        let (mut seq, mut rows) = self.search::<true, T>(1, sequence, Self::DEFAULT_MIN_SCORE, context, validator);
        seq.pop().map(|s| (s, rows.pop().unwrap()))
    }

    #[allow(clippy::type_complexity)]
    fn search<const SCORED: bool, T>(
        &self,
        count: usize,
        sequence: &[T],
        min_score: f64,
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> (Vec<Sequence<'m, L>>, Vec<Vec<f64>>) {
        let mut distributions = Vec::new();
        let mut node_distributions = Vec::new();
        let model = self.model;
        let outcomes = &self.outcomes;
        let n = outcomes.len();
        let mut probs = vec![0.0; n];
        let mut sorted = vec![0.0; n];
        let mut features = Features::new();
        let mut path_slots: PathSlots<'m, L> = PathSlots::new(sequence.len());

        let mut nodes: Vec<Node> = Vec::new();
        // The partial sequences of the last step, best first. The empty
        // sequence starts the search.
        let mut prev: Vec<u32> = vec![ROOT];
        let mut prev_ranks: Vec<u32> = vec![0];
        let mut rank_order: Vec<usize> = Vec::new();
        let mut next: Vec<u32> = Vec::new();
        context.start_sequence();

        for i in 0..sequence.len() {
            let width = self.size.min(prev.len());
            path_slots.prepare(&nodes, &prev[..width], outcomes);
            for (beam_idx, &top) in prev[..width].iter().enumerate() {
                let parent_rank = prev_ranks[beam_idx];
                let prior = path_slots.prior(beam_idx);
                features.clear();
                context.context(i, sequence, prior, &mut features);
                model.eval_features(features.iter(), &mut probs);

                let distribution = if SCORED {
                    let ix = u32::try_from(distributions.len()).expect("fewer than 2^32 contexts");
                    distributions.extend_from_slice(&probs);
                    ix
                } else {
                    0
                };
                let min = threshold(&probs, &mut sorted, self.size);

                let (top_score, top_len) =
                    if top == ROOT { (0.0, 0) } else { (nodes[top as usize].score, nodes[top as usize].len) };
                let mut extend = |p: usize, nodes: &mut Vec<Node>, next: &mut Vec<u32>| {
                    if validator.valid_sequence(i, sequence, prior, outcomes[p]) {
                        let score = top_score + strict_math::log(probs[p]);
                        if score > min_score {
                            let id = u32::try_from(nodes.len()).expect("fewer than 2^32 nodes");
                            nodes.push(Node {
                                parent: top,
                                parent_rank,
                                outcome: p as u32,
                                prob: probs[p],
                                score,
                                len: top_len + 1,
                            });
                            node_distributions.push(distribution);
                            next.push(id);
                        }
                    }
                };
                for (p, &prob) in probs.iter().enumerate() {
                    if prob >= min {
                        extend(p, &mut nodes, &mut next);
                    }
                }
                // When nothing of the whole step so far survived, every
                // valid outcome of this partial sequence is tried.
                if next.is_empty() {
                    for p in 0..n {
                        extend(p, &mut nodes, &mut next);
                    }
                }
            }
            next.sort_by(|&a, &b| compare_step_nodes(&nodes, outcomes, a, b));
            std::mem::swap(&mut prev, &mut next);
            next.clear();
            assign_beam_ranks(&nodes, outcomes, &prev, self.size, &mut rank_order, &mut prev_ranks);
        }

        let mut result = Vec::with_capacity(count.min(prev.len()));
        let mut result_rows = Vec::new();
        for &id in prev.iter().take(count) {
            if id == ROOT {
                result.push(Sequence { outcomes: Vec::new(), probs: Vec::new(), score: 0.0 });
                if SCORED {
                    result_rows.push(Vec::new());
                }
                continue;
            }
            let node = nodes[id as usize];
            let mut seq = Sequence {
                outcomes: Vec::with_capacity(node.len as usize),
                probs: Vec::with_capacity(node.len as usize),
                score: node.score,
            };
            let mut rows = vec![0.0; node.len as usize * outcomes.len()];
            let mut at = id;
            while at != ROOT {
                let n = nodes[at as usize];
                if SCORED {
                    let start = node_distributions[at as usize] as usize;
                    let dest = (n.len as usize - 1) * outcomes.len();
                    rows[dest..dest + outcomes.len()].copy_from_slice(&distributions[start..start + outcomes.len()]);
                }
                seq.outcomes.push(outcomes[n.outcome as usize]);
                seq.probs.push(n.prob);
                at = n.parent;
            }
            seq.outcomes.reverse();
            seq.probs.reverse();
            result.push(seq);
            if SCORED {
                result_rows.push(rows);
            }
        }
        (result, result_rows)
    }
}
