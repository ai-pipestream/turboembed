//! Routing unpunctuated scripts (`Thai`, `Lao`, `Khmer`) from
//! [`RuleSplitter`](turbo_sentdetect::RuleSplitter) and
//! [`RuleSentenceDetector`](turbo_sentdetect::RuleSentenceDetector) to
//! [`SatAnnotator`].

use std::fmt;
use std::path::Path;
use std::sync::Arc;

use turbo::status::Result as CoreResult;
use turbo_analysis::{Annotator, Error, Execution, ModelRef, Span};
use turbo_sentdetect::{RuleSentenceDetector, RuleSplitter, UnpunctuatedSplitter};

use crate::annotator::{SatAnnotator, SatOptions};

impl fmt::Debug for SatAnnotator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SatAnnotator")
            .field("model", &Annotator::model(self))
            .field("execution", &Annotator::execution(self))
            .field("block_size", &self.block_size())
            .field("stride", &self.stride())
            .field("config_hash", &Annotator::config_hash(self))
            .finish()
    }
}

impl UnpunctuatedSplitter for SatAnnotator {
    fn split_batch(&self, texts: &[&str]) -> Vec<Result<Vec<Span>, Error>> {
        self.scores(texts)
            .into_iter()
            .zip(texts)
            .map(|(res, text)| {
                let (spans, scores) = res?;
                let mut out = Vec::new();
                self.sentences(text, &spans, &scores, &mut out);
                Ok(out)
            })
            .collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Annotator::model(self)
    }

    fn execution(&self) -> Option<Execution> {
        Annotator::execution(self)
    }

    fn config_hash(&self) -> u64 {
        Annotator::config_hash(self)
    }
}

impl SatAnnotator {
    /// Builds a [`RuleSplitter`] for `tag` that routes candidate spans
    /// containing unpunctuated scripts (`Thai`, `Lao`, `Khmer`) to `self`.
    pub fn into_rule_splitter(self, tag: &str) -> RuleSplitter {
        RuleSplitter::new(tag).with_unpunctuated_fallback(Arc::new(self))
    }

    /// Builds a [`RuleSentenceDetector`] for `tag` that routes candidate spans
    /// containing unpunctuated scripts (`Thai`, `Lao`, `Khmer`) to `self`.
    pub fn into_rule_detector(self, tag: &str) -> RuleSentenceDetector {
        RuleSentenceDetector::new(tag).with_unpunctuated_fallback(Arc::new(self))
    }
}

/// Builds a [`RuleSplitter`] for `tag`, routing unpunctuated scripts (`Thai`,
/// `Lao`, `Khmer`) to the SaT bundle at `sat_bundle` when `Some`.
pub fn rule_splitter_with_sat(tag: &str, sat_bundle: Option<(&Path, &SatOptions)>) -> CoreResult<RuleSplitter> {
    let splitter = RuleSplitter::new(tag);
    match sat_bundle {
        Some((dir, opts)) => Ok(splitter.with_unpunctuated_fallback(Arc::new(SatAnnotator::open(dir, opts)?))),
        None => Ok(splitter),
    }
}

/// Builds a [`RuleSentenceDetector`] for `tag`, routing unpunctuated scripts
/// (`Thai`, `Lao`, `Khmer`) to the SaT bundle at `sat_bundle` when `Some`.
pub fn rule_detector_with_sat(tag: &str, sat_bundle: Option<(&Path, &SatOptions)>) -> CoreResult<RuleSentenceDetector> {
    Ok(RuleSentenceDetector { splitter: rule_splitter_with_sat(tag, sat_bundle)? })
}
