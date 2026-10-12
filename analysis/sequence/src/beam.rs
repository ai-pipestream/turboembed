//! `BeamSearch`: the most probable outcome sequences of a maxent model
//! over a sequence of items, one decision per item, keeping the best
//! `size` partial sequences at each step.
//!
//! This is the search of `opennlp.tools.ml.BeamSearch` at the commit this
//! crate follows, operation for operation, so it returns the same
//! sequences with the same probabilities and bit-identical scores (for
//! GIS and perceptron models; see the crate documentation for QN):
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
//! evaluations is left out: it returns what evaluating again would.

use std::cmp::Ordering;

use turbo_jvm::chars::compare as compare_java_strings;
use turbo_jvm::strict_math;
use turbo_maxent::{MaxentModel, ModelFile};

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

/// Builds the context (the features) of one item, given the outcomes
/// decided for the items before it: `BeamSearchContextGenerator`.
pub trait ContextGenerator<T> {
    /// Called once at the start of each search, before any
    /// [`context`](Self::context) for its sequence. Generators that keep
    /// per-sentence state reset it here (see the
    /// [`featuregen`](crate::featuregen) module).
    fn start_sequence(&mut self) {}

    /// Appends the features of `sequence[index]` to `out`. `prior` holds
    /// the outcomes of `sequence[..index]`.
    fn context(&mut self, index: usize, sequence: &[T], prior: &[&str], out: &mut Features);
}

/// Decides whether an outcome may follow the outcomes before it:
/// `SequenceValidator`.
pub trait SequenceValidator<T> {
    /// Whether `outcome` is allowed for `sequence[index]` after `prior`,
    /// the outcomes of `sequence[..index]`.
    fn valid_sequence(&self, index: usize, sequence: &[T], prior: &[&str], outcome: &str) -> bool;
}

/// A validator that accepts every outcome.
#[derive(Copy, Clone, Debug, Default)]
pub struct AcceptAll;

impl<T> SequenceValidator<T> for AcceptAll {
    fn valid_sequence(&self, _index: usize, _sequence: &[T], _prior: &[&str], _outcome: &str) -> bool {
        true
    }
}

/// An outcome sequence with the probability of each outcome and the
/// sequence's score: `opennlp.tools.util.Sequence`.
#[derive(Debug, Clone, PartialEq)]
pub struct Sequence<'m> {
    outcomes: Vec<&'m str>,
    probs: Vec<f64>,
    score: f64,
}

impl<'m> Sequence<'m> {
    /// The outcomes, one per item.
    pub fn outcomes(&self) -> &[&'m str] {
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
    outcome: u32,
    prob: f64,
    score: f64,
    len: u32,
}

const ROOT: u32 = u32::MAX;

/// The beam search over one model.
pub struct BeamSearch<'m> {
    size: usize,
    model: &'m MaxentModel,
}

impl<'m> BeamSearch<'m> {
    /// The minimum score of `bestSequences` without one (`ZERO_LOG`).
    pub const DEFAULT_MIN_SCORE: f64 = -100000.0;

    /// A search keeping `size` partial sequences (`BeamSize` in a model's
    /// manifest; the taggers' defaults are 3 for part of speech and names
    /// and 10 for chunks).
    pub fn new(size: usize, model: &'m MaxentModel) -> BeamSearch<'m> {
        BeamSearch { size, model }
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
        context: &mut (impl ContextGenerator<T> + ?Sized),
        validator: &(impl SequenceValidator<T> + ?Sized),
    ) -> Option<Sequence<'m>> {
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
        context: &mut (impl ContextGenerator<T> + ?Sized),
        validator: &(impl SequenceValidator<T> + ?Sized),
    ) -> Vec<Sequence<'m>> {
        let model = self.model;
        let outcomes: Vec<&'m str> = model.outcomes().collect();
        let n = outcomes.len();
        let mut probs = vec![0.0; n];
        let mut sorted = vec![0.0; n];
        let mut features = Features::new();
        let mut prior: Vec<&'m str> = Vec::with_capacity(sequence.len());

        let mut nodes: Vec<Node> = Vec::new();
        // The partial sequences of the last step, best first. The empty
        // sequence starts the search.
        let mut prev: Vec<u32> = vec![ROOT];
        let mut next: Vec<u32> = Vec::new();
        context.start_sequence();

        for i in 0..sequence.len() {
            let width = self.size.min(prev.len());
            for &top in &prev[..width] {
                path(&nodes, top, &outcomes, &mut prior);
                features.clear();
                context.context(i, sequence, &prior, &mut features);
                model.eval_features(features.iter(), &mut probs);

                sorted.copy_from_slice(&probs);
                sorted.sort_by(|a, b| compare_doubles(*a, *b));
                let min = sorted[n.saturating_sub(self.size)];

                let (top_score, top_len) =
                    if top == ROOT { (0.0, 0) } else { (nodes[top as usize].score, nodes[top as usize].len) };
                let extend = |p: usize, nodes: &mut Vec<Node>, next: &mut Vec<u32>| {
                    if validator.valid_sequence(i, sequence, &prior, outcomes[p]) {
                        let score = top_score + strict_math::log(probs[p]);
                        if score > min_score {
                            let id = u32::try_from(nodes.len()).expect("fewer than 2^32 nodes");
                            nodes.push(Node {
                                parent: top,
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
            next.sort_by(|&a, &b| compare_nodes(&nodes, &outcomes, a, b));
            std::mem::swap(&mut prev, &mut next);
            next.clear();
        }

        let mut result = Vec::with_capacity(count.min(prev.len()));
        for &id in prev.iter().take(count) {
            if id == ROOT {
                result.push(Sequence { outcomes: Vec::new(), probs: Vec::new(), score: 0.0 });
                continue;
            }
            let node = nodes[id as usize];
            let mut seq = Sequence {
                outcomes: vec![""; node.len as usize],
                probs: vec![0.0; node.len as usize],
                score: node.score,
            };
            let mut at = id;
            for j in (0..node.len as usize).rev() {
                let n = nodes[at as usize];
                seq.outcomes[j] = outcomes[n.outcome as usize];
                seq.probs[j] = n.prob;
                at = n.parent;
            }
            result.push(seq);
        }
        result
    }
}

/// Fills `out` with the outcomes of the partial sequence ending at `id`,
/// first to last.
fn path<'m>(nodes: &[Node], id: u32, outcomes: &[&'m str], out: &mut Vec<&'m str>) {
    out.clear();
    let mut at = id;
    while at != ROOT {
        let n = nodes[at as usize];
        out.push(outcomes[n.outcome as usize]);
        at = n.parent;
    }
    out.reverse();
}

/// `SearchNode.compareTo`: higher scores first, then by outcomes in
/// `String.compareTo` order, then shorter first.
fn compare_nodes(nodes: &[Node], outcomes: &[&str], a: u32, b: u32) -> Ordering {
    let (x, y) = (nodes[a as usize], nodes[b as usize]);
    let by_score = compare_doubles(y.score, x.score);
    if by_score != Ordering::Equal {
        return by_score;
    }
    let (mut pa, mut pb) = (Vec::new(), Vec::new());
    path(nodes, a, outcomes, &mut pa);
    path(nodes, b, outcomes, &mut pb);
    for (s, t) in pa.iter().zip(&pb) {
        let by_outcome = compare_java_strings(s, t);
        if by_outcome != Ordering::Equal {
            return by_outcome;
        }
    }
    pa.len().cmp(&pb.len())
}
