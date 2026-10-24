//! `BeamSearch`: the most probable outcome sequences of a maxent model
//! over a sequence of items, one decision per item, keeping the best
//! `size` partial sequences at each step.
//!
//! This is the search of `opennlp.tools.ml.BeamSearch` at the commit this
//! crate follows, operation for operation, so it returns the same
//! sequences with the same probabilities and bit-identical scores (for
//! GIS and perceptron models; QN uses an explicit [`OpenNlpBeamSearch`] session):
//!
//! - A partial sequence's score is the sum of the `StrictMath.log` of its
//!   outcome probabilities, added in sequence order from zero.
//! - At each step the best `size` partial sequences are extended, best
//!   first. For each, the model is evaluated on the context the
//!   [`ContextGenerator`] builds, and each outcome whose probability is at
//!   least the `size`-th largest (ties included) and that the
//!   [`SequenceValidator`] accepts makes a new partial sequence, kept when
//!   its score is above the minimum (`-100000` by default). When no
//!   extension of any partial sequence so far in the step was kept, every
//!   valid outcome of the current one is tried instead.
//! - Partial sequences are ordered by score, highest first, with
//!   `Double.compare`; equal scores are ordered by their outcomes, compared
//!   one by one with `String.compareTo`, which orders by UTF-16 code unit
//!   (not by code point, as `str` does), and then by length.
//!
//! Java keeps the candidates in a `PriorityQueue`; here they are sorted,
//! which removes them in the same order. OpenNLP's optional cache of
//! evaluations is left out. Compatibility sessions match the default JVM
//! cache size of zero; QN evaluation depends on earlier calls.
//!
//! The outcomes a search hands its context generator and validator, and
//! returns in a [`Sequence`], are labels of type `L`, an [`OutcomeLabel`]:
//! `str` for the taggers whose outcomes are text (tags, chunk tags, name
//! codes), and [`JavaString`] for the lemmatizer, whose classes are edit
//! scripts over UTF-16 units and can hold half of a surrogate pair. A
//! search over `str` refuses a model with such an outcome
//! ([`Error::OutcomeNotText`]).

mod opennlp;
mod paths;
mod scored;

pub use opennlp::OpenNlpBeamSearch;
use paths::{PathSlots, assign_beam_ranks, compare_step_nodes};

use std::convert::Infallible;

use std::cmp::Ordering;
use turbo_jvm::chars::compare as compare_java_strings;
use turbo_jvm::strict_math;
use turbo_maxent::{JavaString, MaxentModel, ModelFile};

use crate::error::{Error, Result};
use crate::features::Features;
use crate::java::{compare_doubles, parse_int};

/// The manifest property holding a model's beam size.
pub const BEAM_SIZE_PROPERTY: &str = "BeamSize";

/// The beam size the model's manifest gives, or `default` when it gives
/// none, as the taggers read it. A value `Integer.parseInt` rejects, or a
/// negative one, is an error.
pub fn manifest_beam_size(model: &ModelFile, default: usize) -> Result<usize> {
    match model.manifest().get(BEAM_SIZE_PROPERTY) {
        None => Ok(default),
        Some(value) => parse_int(value)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| Error::Manifest { property: BEAM_SIZE_PROPERTY.into(), value: value.to_owned() }),
    }
}

/// The type a search hands a model's outcomes out as: `str` when every
/// outcome is text, or [`JavaString`] for any outcome.
pub trait OutcomeLabel {
    /// Why a model's outcomes cannot be labels of this type; `Infallible`
    /// for [`JavaString`].
    type Unfit;

    /// The model's outcomes in index order as this type.
    fn labels(model: &MaxentModel) -> std::result::Result<Vec<&Self>, Self::Unfit>;

    /// `String.compareTo`.
    fn compare(a: &Self, b: &Self) -> Ordering;
}

impl OutcomeLabel for str {
    type Unfit = Error;

    /// The outcomes as text; [`Error::OutcomeNotText`] names the first
    /// that holds half of a surrogate pair.
    fn labels(model: &MaxentModel) -> Result<Vec<&str>> {
        model.outcome_texts().map_err(|o| Error::OutcomeNotText(o.clone()))
    }

    fn compare(a: &str, b: &str) -> Ordering {
        compare_java_strings(a, b)
    }
}

impl OutcomeLabel for JavaString {
    type Unfit = Infallible;

    fn labels(model: &MaxentModel) -> std::result::Result<Vec<&JavaString>, Infallible> {
        Ok(model.outcomes().collect())
    }

    fn compare(a: &JavaString, b: &JavaString) -> Ordering {
        a.cmp(b)
    }
}

/// Builds the context (the features) of one item, given the outcomes
/// decided for the items before it: `BeamSearchContextGenerator`.
pub trait ContextGenerator<T, L: OutcomeLabel + ?Sized = str> {
    /// Called once at the start of each search, before any
    /// [`context`](Self::context) for its sequence. Generators that keep
    /// per-sentence state reset it here (see the
    /// [`featuregen`](crate::featuregen) module).
    fn start_sequence(&mut self) {}

    /// Appends the features of `sequence[index]` to `out`. `prior` holds
    /// the outcomes of `sequence[..index]`.
    fn context(&mut self, index: usize, sequence: &[T], prior: &[&L], out: &mut Features);
}

/// Decides whether an outcome may follow the outcomes before it:
/// `SequenceValidator`.
pub trait SequenceValidator<T, L: OutcomeLabel + ?Sized = str> {
    /// Whether `outcome` is allowed for `sequence[index]` after `prior`,
    /// the outcomes of `sequence[..index]`.
    fn valid_sequence(&self, index: usize, sequence: &[T], prior: &[&L], outcome: &L) -> bool;
}

/// A validator that accepts every outcome.
#[derive(Copy, Clone, Debug, Default)]
pub struct AcceptAll;

impl<T, L: OutcomeLabel + ?Sized> SequenceValidator<T, L> for AcceptAll {
    fn valid_sequence(&self, _index: usize, _sequence: &[T], _prior: &[&L], _outcome: &L) -> bool {
        true
    }
}

/// An outcome sequence with the probability of each outcome and the
/// sequence's score: `opennlp.tools.util.Sequence`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence<'m, L: ?Sized = str> {
    outcomes: Vec<&'m L>,
    probs: Vec<f64>,
    score: f64,
}

impl<'m, L: ?Sized> Sequence<'m, L> {
    /// The outcomes, one per item.
    pub fn outcomes(&self) -> &[&'m L] {
        &self.outcomes
    }

    /// The probability the model gave each outcome.
    pub fn probs(&self) -> &[f64] {
        &self.probs
    }

    /// The sum of the `StrictMath.log` of the probabilities, in order.
    pub fn score(&self) -> f64 {
        self.score
    }

    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }
}

/// A partial sequence: its last outcome and the node before it.
#[derive(Clone, Copy)]
struct Node {
    /// The node before this one; `u32::MAX` for the first outcome.
    parent: u32,
    /// Lexicographical rank of `parent`'s outcome path among the active
    /// beam prefixes at the step that created this node.
    parent_rank: u32,
    outcome: u32,
    prob: f64,
    score: f64,
    len: u32,
}

const ROOT: u32 = u32::MAX;

/// The beam search over one model, handing outcomes out as `L`.
pub struct BeamSearch<'m, L: OutcomeLabel + ?Sized = str> {
    size: usize,
    model: &'m MaxentModel,
    /// The model's outcomes as labels, by outcome index.
    outcomes: Vec<&'m L>,
}

impl<'m, L: OutcomeLabel + ?Sized> BeamSearch<'m, L> {
    /// The minimum score of `bestSequences` without one (`ZERO_LOG`).
    pub const DEFAULT_MIN_SCORE: f64 = -100000.0;

    /// A search keeping `size` partial sequences (`BeamSize` in a model's
    /// manifest; the taggers' defaults are 3 for part of speech and names
    /// and 10 for chunks). Over `str`, a model with an outcome that is not
    /// text is refused; over [`JavaString`] the error is `Infallible`, so
    /// `let Ok(search) = BeamSearch::new(..)` opens it.
    pub fn new(size: usize, model: &'m MaxentModel) -> std::result::Result<BeamSearch<'m, L>, L::Unfit> {
        Ok(BeamSearch { size, model, outcomes: L::labels(model)? })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn model(&self) -> &'m MaxentModel {
        self.model
    }

    /// The best sequence, or `None` when no sequence scored above the
    /// minimum: `bestSequence`.
    pub fn best_sequence<T>(
        &self,
        sequence: &[T],
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Option<Sequence<'m, L>> {
        self.best_sequences(1, sequence, Self::DEFAULT_MIN_SCORE, context, validator).pop()
    }

    /// Up to `count` best sequences, best first, each scoring above
    /// `min_score`: `bestSequences`. An empty `sequence` gives one empty
    /// sequence.
    pub fn best_sequences<T>(
        &self,
        count: usize,
        sequence: &[T],
        min_score: f64,
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
    ) -> Vec<Sequence<'m, L>> {
        let mut probs = vec![0.0; self.outcomes.len()];
        self.best_sequences_inner::<false, T>(count, sequence, min_score, context, validator, &mut probs)
    }

    fn best_sequences_inner<const SEEDED: bool, T>(
        &self,
        count: usize,
        sequence: &[T],
        min_score: f64,
        context: &mut (impl ContextGenerator<T, L> + ?Sized),
        validator: &(impl SequenceValidator<T, L> + ?Sized),
        probs: &mut [f64],
    ) -> Vec<Sequence<'m, L>> {
        let model = self.model;
        let outcomes = &self.outcomes;
        let n = outcomes.len();
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
                if SEEDED {
                    model.eval_features_seeded(features.iter(), probs);
                } else {
                    model.eval_features(features.iter(), probs);
                }

                let min = threshold(probs, &mut sorted, self.size);

                let (top_score, top_len) =
                    if top == ROOT { (0.0, 0) } else { (nodes[top as usize].score, nodes[top as usize].len) };
                let extend = |p: usize, nodes: &mut Vec<Node>, next: &mut Vec<u32>| {
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
        for &id in prev.iter().take(count) {
            if id == ROOT {
                result.push(Sequence { outcomes: Vec::new(), probs: Vec::new(), score: 0.0 });
                continue;
            }
            let node = nodes[id as usize];
            let mut seq = Sequence {
                outcomes: Vec::with_capacity(node.len as usize),
                probs: Vec::with_capacity(node.len as usize),
                score: node.score,
            };
            let mut at = id;
            while at != ROOT {
                let n = nodes[at as usize];
                seq.outcomes.push(outcomes[n.outcome as usize]);
                seq.probs.push(n.prob);
                at = n.parent;
            }
            seq.outcomes.reverse();
            seq.probs.reverse();
            result.push(seq);
        }
        result
    }
}

/// Returns the `size`-th largest probability in `probs` under
/// `Double.compare` (`sorted[probs.len().saturating_sub(size)]`).
#[inline]
fn threshold(probs: &[f64], sorted: &mut [f64], size: usize) -> f64 {
    let n = probs.len();
    if size <= 1 {
        probs.iter().copied().max_by(|a, b| compare_doubles(*a, *b)).expect("at least one outcome")
    } else if size >= n {
        probs.iter().copied().min_by(|a, b| compare_doubles(*a, *b)).expect("at least one outcome")
    } else {
        sorted.copy_from_slice(probs);
        let (_, &mut min, _) = sorted.select_nth_unstable_by(n - size, |a, b| compare_doubles(*a, *b));
        min
    }
}

#[cfg(test)]
mod tests;
