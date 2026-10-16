//! Constituency parser evaluator and cross-validator.
//!
//! Mirrors `opennlp.tools.parser.ParserEvaluator` and
//! `opennlp.tools.parser.ParserCrossValidator`:
//! - [`ParserEvaluator::get_constituent_spans`] extracts the labeled spans of
//!   the nodes below the root that are not POS tag nodes, in the order of
//!   OpenNLP's `getConstituencySpans`.
//! - [`ParserEvaluator`] runs a [`Parser`] on the tokens of each reference
//!   [`Parse`] and updates an [`FMeasure`] over labeled spans.
//! - [`ParserCrossValidator`] partitions a corpus with [`CrossValidation`],
//!   trains either a [`chunking::Parser`] or [`treeinsert::Parser`] per fold,
//!   and accumulates the held-out [`FMeasure`].

use std::fmt;
use std::sync::Arc;

use crate::bottom_up::Parser;
use crate::chunking::{self, ParserTrainingParameters};
use crate::model::{HeadRulesArtifact, ParserModelError, ParserType};
use crate::parse::{Parse, ParseError};
use crate::treeinsert;

/// A span with an optional constituent type, compared as `opennlp.tools.util.Span.equals`
/// compares: by start, end, and type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LabeledSpan<'t> {
    pub start: u32,
    pub end: u32,
    pub kind: Option<&'t str>,
}

impl<'t> LabeledSpan<'t> {
    /// Creates a [`LabeledSpan`] at `[start..end)` with optional label `kind`.
    pub fn new(start: u32, end: u32, kind: Option<&'t str>) -> LabeledSpan<'t> {
        LabeledSpan { start, end, kind }
    }

    /// `Span.toString`: `[start..end)`, then a space and the type when there is one.
    pub fn java_string(&self) -> String {
        match self.kind {
            Some(k) => format!("[{}..{}) {k}", self.start, self.end),
            None => format!("[{}..{})", self.start, self.end),
        }
    }
}

impl fmt::Display for LabeledSpan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.java_string())
    }
}

/// Precision, recall, and F-measure over all evaluated samples (`opennlp.tools.util.eval.FMeasure`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct FMeasure {
    selected: u64,
    target: u64,
    true_positives: u64,
}

impl FMeasure {
    /// `updateScores`: adds the true positives of `predictions` against `references`, the number
    /// of predictions, and the number of references.
    pub fn update<T: PartialEq>(&mut self, references: &[T], predictions: &[T]) {
        self.true_positives += count_true_positives(references, predictions);
        self.selected += predictions.len() as u64;
        self.target += references.len() as u64;
    }

    /// `mergeInto`: adds the counts of `other` to these.
    pub fn merge(&mut self, other: &FMeasure) {
        self.selected += other.selected;
        self.target += other.target;
        self.true_positives += other.true_positives;
    }

    /// The true positives over the predictions, or `0.0` with no predictions.
    pub fn precision(&self) -> f64 {
        if self.selected > 0 { self.true_positives as f64 / self.selected as f64 } else { 0.0 }
    }

    /// The true positives over the references, or `0.0` with no references.
    pub fn recall(&self) -> f64 {
        if self.target > 0 { self.true_positives as f64 / self.target as f64 } else { 0.0 }
    }

    /// `2 * (precision * recall) / (precision + recall)`, or `-1.0` when precision and recall add
    /// up to `0.0` or less, as Java returns.
    pub fn f_measure(&self) -> f64 {
        let (p, r) = (self.precision(), self.recall());
        if p + r > 0.0 { 2.0 * (p * r) / (p + r) } else { -1.0 }
    }

    /// The number of predictions.
    pub fn selected(&self) -> u64 {
        self.selected
    }

    /// The number of references.
    pub fn target(&self) -> u64 {
        self.target
    }

    /// The number of true positives.
    pub fn true_positives(&self) -> u64 {
        self.true_positives
    }
}

fn count_true_positives<T: PartialEq>(references: &[T], predictions: &[T]) -> u64 {
    let mut remaining: Vec<&T> = predictions.iter().collect();
    let mut true_positives = 0;
    let mut matched: Option<&T> = None;
    for reference in references {
        for &prediction in &remaining {
            if reference == prediction {
                matched = Some(prediction);
                true_positives += 1;
            }
        }
        if let Some(m) = matched
            && let Some(at) = remaining.iter().position(|&p| p == m)
        {
            remaining.remove(at);
        }
    }
    true_positives
}

/// One fold of n-fold cross-validation (`opennlp.tools.util.eval.CrossValidationPartitioner`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Fold {
    /// The fold's index from `0`.
    pub index: usize,
    pub training: Vec<usize>,
    pub test: Vec<usize>,
}

impl Fold {
    /// The training samples of `samples`.
    pub fn training_of<'s, T>(&self, samples: &'s [T]) -> impl Iterator<Item = &'s T> {
        self.training.iter().map(move |&i| &samples[i])
    }

    /// The test samples of `samples`.
    pub fn test_of<'s, T>(&self, samples: &'s [T]) -> impl Iterator<Item = &'s T> {
        self.test.iter().map(move |&i| &samples[i])
    }
}

/// The folds over `len` samples in `partitions` partitions (`CrossValidationPartitioner`).
#[derive(Clone, Debug)]
pub struct CrossValidation {
    len: usize,
    partitions: usize,
    next: usize,
}

impl CrossValidation {
    /// Creates a new partitioner over `len` samples in `partitions` folds.
    pub fn new(len: usize, partitions: usize) -> CrossValidation {
        CrossValidation { len, partitions, next: 0 }
    }
}

impl Iterator for CrossValidation {
    type Item = Fold;

    fn next(&mut self) -> Option<Fold> {
        if self.next >= self.partitions {
            return None;
        }
        let (n, t, len) = (self.partitions, self.next, self.len);
        self.next += 1;
        let mut training = Vec::new();
        let mut at = 0;
        loop {
            if at % n == t {
                at += 1;
            }
            if at >= len {
                break;
            }
            training.push(at);
            at += 1;
        }
        let test = (t..len).step_by(n).collect();
        Some(Fold { index: t, training, test })
    }
}

/// Evaluates a constituency [`Parser`] against gold-standard [`Parse`] trees,
/// accumulating labeled constituent precision, recall, and F1 in [`FMeasure`].
///
/// Mirrors `opennlp.tools.parser.ParserEvaluator`.
pub struct ParserEvaluator<'a> {
    parser: &'a dyn Parser,
    fmeasure: FMeasure,
}

impl<'a> ParserEvaluator<'a> {
    /// Creates a new evaluator wrapping `parser`.
    pub fn new(parser: &'a dyn Parser) -> Self {
        Self { parser, fmeasure: FMeasure::default() }
    }

    /// Returns the accumulated [`FMeasure`].
    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }

    /// Extracts the constituent spans of `parse` as OpenNLP's `getConstituencySpans` does: every
    /// node below the root that is not a POS tag node, whatever the root's type (an `INC` guess
    /// included), each as `LabeledSpan::new(span.start, span.end, Some(kind))`.
    ///
    /// The order is OpenNLP's too, a stack of the root's children: each node comes before its
    /// children and the last child of a node first. [`FMeasure::update`] counts true positives
    /// in an order-dependent way when spans repeat, so the order is part of the score.
    pub fn get_constituent_spans(parse: &Parse) -> Vec<LabeledSpan<'_>> {
        let mut stack: Vec<&Parse> = parse.children().iter().collect();
        let mut out = Vec::new();
        while let Some(node) = stack.pop() {
            if !node.is_pos_tag() {
                let sp = node.span();
                out.push(LabeledSpan::new(sp.start, sp.end, Some(node.kind())));
                stack.extend(node.children());
            }
        }
        out
    }

    /// Evaluates a single reference [`Parse`] sample, updating [`Self::fmeasure`]
    /// and returning the predicted [`Parse`] (if any).
    ///
    /// Mirrors `ParserEvaluator.processSample(Parse reference)`: extracts the
    /// token strings from `reference.tag_nodes()`, builds a fresh flat parse via
    /// `Parse::from_tokens` (identical to `ParserTool.parseLine`), and compares
    /// [`Self::get_constituent_spans`] on reference and prediction. An error is
    /// where the parser would throw.
    pub fn evaluate_sample(&mut self, reference: &Parse) -> Result<Option<Parse>, ParseError> {
        let tag_nodes = reference.tag_nodes();
        let mut words: Vec<&str> = Vec::with_capacity(tag_nodes.len());
        for tok in &tag_nodes {
            words.push(tok.covered_text());
        }
        let Ok(flat) = Parse::from_tokens(&words) else {
            return Ok(None);
        };
        let predicted = self.parser.parse(&flat)?;
        let ref_spans = Self::get_constituent_spans(reference);
        match &predicted {
            Some(pred) => {
                let pred_spans = Self::get_constituent_spans(pred);
                self.fmeasure.update(&ref_spans, &pred_spans);
            }
            None => {
                let empty: [LabeledSpan<'_>; 0] = [];
                self.fmeasure.update(&ref_spans, &empty);
            }
        }
        Ok(predicted)
    }

    /// Evaluates a slice of gold-standard [`Parse`] trees.
    pub fn evaluate(&mut self, samples: &[Parse]) -> Result<(), ParseError> {
        for sample in samples {
            self.evaluate_sample(sample)?;
        }
        Ok(())
    }
}

/// K-fold cross-validator for constituency parsing, mirroring
/// `opennlp.tools.parser.ParserCrossValidator`.
#[derive(Debug, Clone)]
pub struct ParserCrossValidator {
    language_code: String,
    params: ParserTrainingParameters,
    parser_type: ParserType,
    fmeasure: FMeasure,
}

impl ParserCrossValidator {
    /// Creates a new cross-validator for `language_code`, `params`, and `parser_type`.
    pub fn new(language_code: impl Into<String>, params: ParserTrainingParameters, parser_type: ParserType) -> Self {
        Self { language_code: language_code.into(), params, parser_type, fmeasure: FMeasure::default() }
    }

    /// Runs `n_folds`-fold cross-validation over `samples` using `rules`.
    pub fn evaluate(
        &mut self,
        samples: &[Parse],
        n_folds: usize,
        rules: &HeadRulesArtifact,
    ) -> Result<(), ParserModelError> {
        for fold in CrossValidation::new(samples.len(), n_folds) {
            let train_samples: Vec<Parse> = fold.training_of(samples).cloned().collect();
            let model = match self.parser_type {
                ParserType::Chunking => {
                    chunking::Parser::train(&self.language_code, &train_samples, rules.clone(), &self.params)?
                }
                ParserType::TreeInsert => {
                    treeinsert::Parser::train(&self.language_code, &train_samples, rules.clone(), &self.params)?
                }
            };
            let model = Arc::new(model);

            match self.parser_type {
                ParserType::Chunking => {
                    let parser = chunking::Parser::new(model);
                    let mut evaluator = ParserEvaluator::new(&parser);
                    for test_sample in fold.test_of(samples) {
                        evaluator.evaluate_sample(test_sample)?;
                    }
                    self.fmeasure.merge(evaluator.fmeasure());
                }
                ParserType::TreeInsert => {
                    let parser = treeinsert::Parser::new(model);
                    let mut evaluator = ParserEvaluator::new(&parser);
                    for test_sample in fold.test_of(samples) {
                        evaluator.evaluate_sample(test_sample)?;
                    }
                    self.fmeasure.merge(evaluator.fmeasure());
                }
            }
        }
        Ok(())
    }

    /// Returns the aggregated [`FMeasure`] across all evaluated folds.
    pub fn fmeasure(&self) -> &FMeasure {
        &self.fmeasure
    }
}
