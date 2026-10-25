//! The YAKE extractor (Campos et al., "YAKE! Keyword extraction from
//! single documents using multiple local features", Information Sciences
//! 509 (2020) 257-289): five per-term statistics over the candidate
//! vocabulary, n-gram candidates scored by `S(kw) = prod(S(w)) / (KF *
//! (1 + sum(S(w))))`, lower better, then deduplication.
//!
//! The default parameter values follow Spark NLP's YakeParams
//! (com/johnsnowlabs/nlp/annotators/keyword/yake/YakeParams.scala),
//! Apache-2.0; the scoring formulas follow the paper. Three deliberate
//! departures from that implementation, all toward the paper, live in
//! `features`: the `ln(ln(3 + median))` position, `DR` from the right
//! window's own dispersion, and the frequency moments over the candidate
//! vocabulary rather than every token.

pub(crate) mod candidates;
pub(crate) mod features;

use turbo_analysis::{Annotator, Error, Key, Layer};
use turbo_analysis_layers::{Sentences, Terms, Tokens};

use crate::ranked::RankedKeyword;
use crate::{Dedup, Stopwords};

/// Extracts YAKE keyphrases from one document's layers. `extract` is the
/// whole algorithm; the annotator is a thin layer-writing wrapper.
#[derive(Clone, Debug)]
pub struct YakeExtractor {
    stopwords: Stopwords,
    window_size: usize,
    min_ngram: usize,
    max_ngram: usize,
    top_n: usize,
    threshold: f64,
    dedup: Dedup,
}

impl YakeExtractor {
    /// The paper-and-Spark-NLP defaults: a three-token window, 1..=3
    /// grams, the top 30, no score cut, token-overlap dedup at 0.8.
    pub fn new(stopwords: Stopwords) -> YakeExtractor {
        YakeExtractor {
            stopwords,
            window_size: 3,
            min_ngram: 1,
            max_ngram: 3,
            top_n: 30,
            threshold: f64::MAX,
            dedup: Dedup::default(),
        }
    }

    pub fn stopwords(&self) -> &Stopwords {
        &self.stopwords
    }

    pub fn window_size(&self) -> usize {
        self.window_size
    }

    pub fn ngrams(&self) -> (usize, usize) {
        (self.min_ngram, self.max_ngram)
    }

    pub fn top_n(&self) -> usize {
        self.top_n
    }

    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    pub fn dedup(&self) -> Dedup {
        self.dedup
    }

    pub fn with_window_size(mut self, window_size: usize) -> YakeExtractor {
        self.window_size = window_size;
        self
    }

    pub fn with_ngrams(mut self, min_ngram: usize, max_ngram: usize) -> YakeExtractor {
        self.min_ngram = min_ngram;
        self.max_ngram = max_ngram;
        self
    }

    pub fn with_top_n(mut self, top_n: usize) -> YakeExtractor {
        self.top_n = top_n;
        self
    }

    /// Keeps candidates whose raw score is at most `threshold` (the raw
    /// YAKE score, lower better). The default keeps every survivor.
    pub fn with_threshold(mut self, threshold: f64) -> YakeExtractor {
        self.threshold = threshold;
        self
    }

    pub fn with_dedup(mut self, dedup: Dedup) -> YakeExtractor {
        self.dedup = dedup;
        self
    }

    /// Every ranked keyphrase of the document, best first. An empty
    /// token or sentence layer yields an empty ranking, never an error;
    /// a configuration that cannot run fails the document.
    pub fn extract(&self, sentences: &Sentences, tokens: &Tokens, terms: &Terms) -> Result<Vec<RankedKeyword>, Error> {
        if self.window_size == 0 {
            return Err(config("window_size must be at least 1"));
        }
        if self.min_ngram == 0 || self.max_ngram < self.min_ngram {
            return Err(config("ngrams must hold 1 <= min <= max"));
        }
        if self.top_n == 0 {
            return Err(config("top_n must be at least 1"));
        }
        if !self.threshold.is_finite() && self.threshold != f64::MAX {
            return Err(config("threshold must be positive, or f64::MAX for no cut"));
        }
        if self.threshold <= 0.0 {
            return Err(config("threshold must be positive, or f64::MAX for no cut"));
        }
        if !(self.dedup.threshold > 0.0 && self.dedup.threshold <= 1.0) {
            return Err(config("dedup threshold must be in (0, 1]"));
        }
        // The layers are indexed column by column below; their own
        // invariants run first, so ragged columns refuse instead of
        // panicking.
        sentences.check().map_err(invalid::<Sentences>)?;
        tokens.check().map_err(invalid::<Tokens>)?;
        terms.check().map_err(invalid::<Terms>)?;
        if tokens.spans.is_empty() {
            return Ok(Vec::new());
        }
        let fold = features::fold(sentences, tokens, terms, self.window_size)?;
        let tscore = features::score_terms(&fold, &self.stopwords);
        let accums = candidates::enumerate(&fold, &self.stopwords, self.min_ngram, self.max_ngram);
        let scored = candidates::score(&fold, &tscore, accums, &self.stopwords);
        Ok(candidates::rank(&fold, scored, self.dedup, self.threshold, self.top_n))
    }

    pub(crate) fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new()
            .u64(self.window_size as u64)
            .u64(self.min_ngram as u64)
            .u64(self.max_ngram as u64)
            .u64(self.top_n as u64)
            .f64(self.threshold)
            .u64(self.dedup.config_hash())
            .u64(self.stopwords.config_hash())
            .finish()
    }
}

fn config(detail: &'static str) -> Error {
    Error::Annotator { annotator: crate::YakeAnnotator::ID, detail: format!("invalid configuration: {detail}") }
}

/// A layer whose own `check` refused it, as the step's error.
fn invalid<L: Layer>(detail: String) -> Error {
    Error::Invalid { layer: Key::<L>::DEFAULT.id(), detail }
}
