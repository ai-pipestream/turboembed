//! Constituency parser evaluator and cross-validator.
//!
//! Mirrors `opennlp.tools.parser.ParserEvaluator` and
//! `opennlp.tools.parser.ParserCrossValidator`:
//! - [`ParserEvaluator::get_constituent_spans`] extracts labeled constituent
//!   spans (excluding `TOP`, `TK`, and POS tag nodes) in character/byte
//!   coordinates matching OpenNLP's `getConstituencySpans`.
//! - [`ParserEvaluator`] runs a [`Parser`] on the tokens of each reference
//!   [`Parse`] and updates an [`FMeasure`] over labeled spans.
//! - [`ParserCrossValidator`] partitions a corpus with [`CrossValidation`],
//!   trains either a [`chunking::Parser`] or [`treeinsert::Parser`] per fold,
//!   and accumulates the held-out [`FMeasure`].

use std::sync::Arc;

use turbo_formats::evaluate::{CrossValidation, FMeasure, LabeledSpan};

use crate::bottom_up::Parser;
use crate::chunking::{self, ParserTrainingParameters};
use crate::model::{HeadRulesArtifact, ParserModelError, ParserType};
use crate::parse::{Parse, ParseError, TOK_NODE, TOP_NODE};
use crate::treeinsert;

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

    /// Extracts all constituent spans from `parse` in the same order and
    /// coordinates as `opennlp.tools.parser.ParserEvaluator.getConstituencySpans`.
    ///
    /// Specifically, performs a pre-order stack walk (pushing children in
    /// reverse so the leftmost child is visited first), skips `TOP`, `TK`, and
    /// POS-tag nodes, and records `LabeledSpan::new(span.start, span.end, Some(kind))`.
    pub fn get_constituent_spans(parse: &Parse) -> Vec<LabeledSpan<'_>> {
        let mut stack: Vec<&Parse> = Vec::new();
        if parse.kind() == TOP_NODE {
            for child in parse.children().iter().rev() {
                stack.push(child);
            }
        } else {
            stack.push(parse);
        }

        let mut out = Vec::new();
        while let Some(node) = stack.pop() {
            if !node.is_pos_tag() && node.kind() != TOK_NODE && node.kind() != TOP_NODE {
                let sp = node.span();
                out.push(LabeledSpan::new(sp.start, sp.end, Some(node.kind())));
            }
            for child in node.children().iter().rev() {
                stack.push(child);
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
