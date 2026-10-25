//! Zero-shot document classification framed as natural
//! language inference: for each caller-supplied candidate
//! label, an NLI scorer judges the pair (premise = the text,
//! hypothesis = the template with the label substituted),
//! and the entailment evidence becomes the label's score.
//! This module owns the framing, the scoring modes, and
//! long-premise windowing; the model itself sits behind
//! [`NliScorer`], so no pair-encoding runtime is assumed.

use std::fmt;
use std::sync::Arc;

use turbo_analysis::{Execution, ModelRef, TagSet};

mod annotate;

pub use annotate::{ZeroShotAnnotator, ZeroShotSpanAnnotator};

/// A sequence-pair NLI model: given (premise, hypothesis)
/// pairs, the raw logit of every outcome per pair.
pub trait NliScorer {
    /// The model's outcome names, in logit order: either
    /// entailment/neutral/contradiction or
    /// entailment/not_entailment, resolved case-insensitively
    /// when a categorizer is built over this scorer.
    fn outcome_labels(&self) -> &[String];

    /// The model scoring every pair, recorded as layer
    /// provenance by the annotators.
    fn model_ref(&self) -> ModelRef;

    /// The hardware execution provenance (`backend`, `device`, `precision`)
    /// recorded by the annotators when backed by a runtime session.
    fn execution(&self) -> Option<Execution> {
        None
    }

    /// The model's token budget for one pair, special tokens
    /// included: the most tokens one encoded pair may hold.
    fn max_pair_tokens(&self) -> u32;

    /// How many special tokens the model's pair template
    /// adds around the premise and hypothesis tokens: 3 for
    /// BERT-style `[CLS] p [SEP] h [SEP]`, 4 for RoBERTa-style
    /// `<s> p </s></s> h </s>`. Window packing reserves this
    /// headroom so a full window plus a full hypothesis plus
    /// the specials never exceeds `max_pair_tokens` and the
    /// hypothesis is never truncated.
    fn pair_special_tokens(&self) -> usize;

    /// How many of the model's tokens `text` encodes to on
    /// its own (no special tokens). Used only to pack premise
    /// windows; it must not shrink as text is appended.
    fn count_tokens(&self, text: &str) -> usize;

    /// One logit row per pair, each of `outcome_labels().len()`
    /// finite values, in input order.
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, ZeroShotError>;
}

/// How per-label entailment evidence becomes category scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum ZeroShotMode {
    /// Softmax over the entailment logits across all labels:
    /// exactly one label wins, scores sum to 1.
    #[default]
    SingleLabel,
    /// An independent two-way softmax over (entailment,
    /// contradiction) per label: every label gets its own
    /// probability in [0, 1].
    MultiLabel,
}

/// How per-window logits of one long premise combine, applied
/// to the entailment and contradiction logits separately,
/// before the mode's softmax runs once.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum WindowAggregation {
    /// The strongest window decides.
    #[default]
    Max,
    /// The unweighted mean over windows.
    Mean,
    /// The mean weighted by each window's premise token count;
    /// when the scorer counts no tokens in any window, the
    /// unweighted mean (the weights carry no information).
    LengthWeightedMean,
}

/// The framing of one zero-shot categorizer.
#[derive(Clone, Debug, PartialEq)]
pub struct ZeroShotConfig {
    /// The hypothesis template, with exactly one `{}`
    /// placeholder for the label.
    pub template: String,
    pub mode: ZeroShotMode,
    pub aggregation: WindowAggregation,
    /// Words of overlap between consecutive premise windows.
    pub overlap_words: usize,
}

impl Default for ZeroShotConfig {
    fn default() -> Self {
        ZeroShotConfig {
            template: "This example is {}.".into(),
            mode: ZeroShotMode::SingleLabel,
            aggregation: WindowAggregation::Max,
            overlap_words: 0,
        }
    }
}

/// Why a zero-shot categorizer could not be built or run.
#[derive(Debug)]
pub enum ZeroShotError {
    /// No candidate labels were given.
    NoLabels,
    /// The candidate label at this index is blank.
    BlankLabel(usize),
    /// This candidate label appears twice.
    DuplicateLabel(String),
    /// More candidate labels than a category tag can name.
    TooManyLabels(usize),
    /// The template does not hold exactly one `{}` with text
    /// around it.
    InvalidTemplate(String),
    /// The scorer's outcomes are not an NLI head this module
    /// can read.
    UnsupportedOutcomes(Vec<String>),
    /// The text to categorize is blank.
    EmptyPremise,
    /// The hypothesis for this label already fills the pair
    /// budget; no premise token would fit beside it.
    HypothesisTooLong { label: String, tokens: usize, specials: usize, max_pair_tokens: u32 },
    /// One whitespace-delimited word of the premise exceeds
    /// the premise budget on its own.
    WordTooLong { tokens: usize, budget: usize },
    /// The configured overlap would keep a window from
    /// advancing.
    OverlapTooLarge { overlap_words: usize, window_words: usize },
    /// The scorer returned the wrong shape or a non-finite
    /// logit.
    BadScorerOutput(String),
    /// The scorer itself failed.
    Scorer(String),
}

impl fmt::Display for ZeroShotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZeroShotError::NoLabels => write!(f, "zero-shot classification needs at least one candidate label"),
            ZeroShotError::BlankLabel(i) => write!(f, "candidate label {i} is blank"),
            ZeroShotError::DuplicateLabel(l) => write!(f, "candidate label {l:?} appears twice"),
            ZeroShotError::TooManyLabels(n) => write!(f, "{n} candidate labels exceed the 65536 a category tag names"),
            ZeroShotError::InvalidTemplate(t) => {
                write!(f, "hypothesis template {t:?} must contain exactly one {{}} placeholder and other text")
            }
            ZeroShotError::UnsupportedOutcomes(names) => {
                write!(
                    f,
                    "NLI outcomes {names:?} are neither entailment/neutral/contradiction nor entailment/not_entailment"
                )
            }
            ZeroShotError::EmptyPremise => write!(f, "the text to categorize is blank"),
            ZeroShotError::HypothesisTooLong { label, tokens, specials, max_pair_tokens } => write!(
                f,
                "the hypothesis for {label:?} takes {tokens} tokens plus {specials} pair specials of the \
                 {max_pair_tokens} pair tokens, leaving no premise token"
            ),
            ZeroShotError::WordTooLong { tokens, budget } => {
                write!(f, "one premise word takes {tokens} tokens, over the {budget}-token premise budget")
            }
            ZeroShotError::OverlapTooLarge { overlap_words, window_words } => {
                write!(f, "overlap of {overlap_words} words does not advance a {window_words}-word window")
            }
            ZeroShotError::BadScorerOutput(detail) => write!(f, "bad NLI scorer output: {detail}"),
            ZeroShotError::Scorer(detail) => write!(f, "NLI scorer failed: {detail}"),
        }
    }
}

impl std::error::Error for ZeroShotError {}

fn fold(s: &str) -> String {
    s.trim().chars().flat_map(char::to_lowercase).collect()
}

/// A zero-shot categorizer over the scorer `S`: candidate
/// labels and a hypothesis template, validated once.
pub struct ZeroShotCategorizer<S: NliScorer> {
    scorer: S,
    labels: Vec<String>,
    hypotheses: Vec<String>,
    premise_budget: usize,
    entail: usize,
    contra: usize,
    categories: Arc<TagSet>,
    config: ZeroShotConfig,
}

impl<S: NliScorer> ZeroShotCategorizer<S> {
    pub fn new(scorer: S, labels: Vec<String>, config: ZeroShotConfig) -> Result<Self, ZeroShotError> {
        if labels.is_empty() {
            return Err(ZeroShotError::NoLabels);
        }
        if labels.len() > u16::MAX as usize + 1 {
            return Err(ZeroShotError::TooManyLabels(labels.len()));
        }
        for (i, label) in labels.iter().enumerate() {
            if label.trim().is_empty() {
                return Err(ZeroShotError::BlankLabel(i));
            }
            if labels[..i].contains(label) {
                return Err(ZeroShotError::DuplicateLabel(label.clone()));
            }
        }
        let parts: Vec<&str> = config.template.split("{}").collect();
        if parts.len() != 2 || (parts[0].trim().is_empty() && parts[1].trim().is_empty()) {
            return Err(ZeroShotError::InvalidTemplate(config.template.clone()));
        }
        let names: Vec<String> = scorer.outcome_labels().iter().map(|n| fold(n)).collect();
        let find = |want: &str| names.iter().position(|n| n == want);
        let (Some(entail), contra) = (find("entailment"), find("contradiction").or_else(|| find("not_entailment")))
        else {
            return Err(ZeroShotError::UnsupportedOutcomes(scorer.outcome_labels().to_vec()));
        };
        let three = names.len() == 3 && find("neutral").is_some() && find("contradiction").is_some();
        let two = names.len() == 2 && find("not_entailment").is_some();
        if !three && !two {
            return Err(ZeroShotError::UnsupportedOutcomes(scorer.outcome_labels().to_vec()));
        }
        let hypotheses: Vec<String> = labels.iter().map(|l| config.template.replacen("{}", l, 1)).collect();
        let max_pair = scorer.max_pair_tokens();
        let specials = scorer.pair_special_tokens();
        let mut worst = 0usize;
        for (label, hypothesis) in labels.iter().zip(&hypotheses) {
            let tokens = scorer.count_tokens(hypothesis);
            if tokens + specials >= max_pair as usize {
                return Err(ZeroShotError::HypothesisTooLong {
                    label: label.clone(),
                    tokens,
                    specials,
                    max_pair_tokens: max_pair,
                });
            }
            worst = worst.max(tokens);
        }
        let categories = TagSet::new("category", &labels.iter().map(String::as_str).collect::<Vec<_>>());
        Ok(ZeroShotCategorizer {
            scorer,
            labels,
            hypotheses,
            // What remains for premise tokens once the longest
            // hypothesis and the pair template's special
            // tokens are reserved: a full window plus a full
            // hypothesis plus the specials never exceeds
            // max_pair_tokens.
            premise_budget: (max_pair as usize).saturating_sub(worst + specials),
            entail,
            contra: contra.unwrap(),
            categories,
            config,
        })
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn config(&self) -> &ZeroShotConfig {
        &self.config
    }

    pub fn scorer(&self) -> &S {
        &self.scorer
    }

    pub fn categories(&self) -> Arc<TagSet> {
        Arc::clone(&self.categories)
    }

    /// The hypothesis scored for `label`.
    pub fn hypothesis(&self, label: &str) -> String {
        self.config.template.replacen("{}", label, 1)
    }

    /// The premise windows for `text`: word ranges packed to
    /// the premise budget (the pair budget less the longest
    /// hypothesis, so every window carries a full hypothesis),
    /// advancing by the window size less the configured
    /// overlap. Whitespace inside a window is normalized to
    /// single spaces.
    pub fn premise_windows(&self, text: &str) -> Result<Vec<String>, ZeroShotError> {
        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Err(ZeroShotError::EmptyPremise);
        }
        let budget = self.premise_budget;
        let mut windows = Vec::new();
        let mut start = 0usize;
        loop {
            // The largest end whose joined text fits: token
            // counts do not shrink as words are appended.
            // Gallop outward from one word (1, 2, 4, ... times
            // the words already known to fit) to bracket that
            // end near the window itself, then binary-search
            // inside the bracket, so the text handed to
            // count_tokens stays proportional to the window,
            // never to the whole remaining document.
            let fits = |end: usize| self.scorer.count_tokens(&words[start..end].join(" ")) <= budget;
            if !fits(start + 1) {
                return Err(ZeroShotError::WordTooLong { tokens: self.scorer.count_tokens(words[start]), budget });
            }
            let mut lo = start + 1;
            let mut hi = words.len();
            let mut step = 1usize;
            while lo < hi {
                let probe = lo.saturating_add(step).min(hi);
                if fits(probe) {
                    lo = probe;
                    step = step.saturating_mul(2);
                } else {
                    hi = probe - 1;
                    break;
                }
            }
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if fits(mid) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            windows.push(words[start..lo].join(" "));
            if lo == words.len() {
                return Ok(windows);
            }
            let size = lo - start;
            if self.config.overlap_words >= size {
                return Err(ZeroShotError::OverlapTooLarge {
                    overlap_words: self.config.overlap_words,
                    window_words: size,
                });
            }
            start += size - self.config.overlap_words;
        }
    }

    /// The score of every candidate label for `text`, in
    /// label order: probabilities summing to 1 in single-label
    /// mode, one independent probability per label in
    /// multi-label mode.
    pub fn categorize(&self, text: &str) -> Result<Vec<f64>, ZeroShotError> {
        if text.trim().is_empty() {
            return Err(ZeroShotError::EmptyPremise);
        }
        let windows = self.premise_windows(text)?;
        let weights: Vec<f64> = windows.iter().map(|w| self.scorer.count_tokens(w) as f64).collect();
        let mut pairs: Vec<(&str, &str)> = Vec::with_capacity(windows.len() * self.labels.len());
        for window in &windows {
            for hypothesis in &self.hypotheses {
                pairs.push((window.as_str(), hypothesis.as_str()));
            }
        }
        let rows = self.scorer.score_pairs(&pairs)?;
        if rows.len() != pairs.len() {
            return Err(ZeroShotError::BadScorerOutput(format!("{} rows for {} pairs", rows.len(), pairs.len())));
        }
        let outcomes = self.scorer.outcome_labels().len();
        for row in &rows {
            if row.len() != outcomes || row.iter().any(|v| !v.is_finite()) {
                return Err(ZeroShotError::BadScorerOutput(format!(
                    "a row of {} values for {outcomes} outcomes, or a non-finite logit",
                    row.len()
                )));
            }
        }
        let aggregate = |values: &dyn Fn(usize) -> f64| -> f64 {
            match self.config.aggregation {
                WindowAggregation::Max => (0..windows.len()).map(values).fold(f64::NEG_INFINITY, f64::max),
                WindowAggregation::Mean => (0..windows.len()).map(values).sum::<f64>() / windows.len() as f64,
                WindowAggregation::LengthWeightedMean => {
                    let total: f64 = weights.iter().sum();
                    if total == 0.0 {
                        // The scorer counts no tokens in any
                        // window (they exist only as words its
                        // tokenizer strips): weights carry no
                        // information, so average unweighted
                        // rather than dividing 0 by 0 into a NaN
                        // the scorer never produced.
                        (0..windows.len()).map(values).sum::<f64>() / windows.len() as f64
                    } else {
                        (0..windows.len()).map(|w| weights[w] * values(w)).sum::<f64>() / total
                    }
                }
            }
        };
        let (mut entails, mut contras) = (Vec::with_capacity(self.labels.len()), Vec::with_capacity(self.labels.len()));
        for (label, _) in self.labels.iter().enumerate() {
            let at = |w: usize, outcome: usize| f64::from(rows[w * self.labels.len() + label][outcome]);
            entails.push(aggregate(&|w| at(w, self.entail)));
            contras.push(aggregate(&|w| at(w, self.contra)));
        }
        let scores = match self.config.mode {
            ZeroShotMode::SingleLabel => softmax(&entails),
            ZeroShotMode::MultiLabel => entails
                .iter()
                .zip(&contras)
                .map(|(e, c)| {
                    let max = e.max(*c);
                    let e = (e - max).exp();
                    e / (e + (c - max).exp())
                })
                .collect(),
        };
        if scores.iter().any(|s| !s.is_finite() || !(0.0..=1.0).contains(s)) {
            return Err(ZeroShotError::BadScorerOutput(format!("scores outside [0, 1]: {scores:?}")));
        }
        Ok(scores)
    }

    /// The best-scoring label for `text` and its score.
    pub fn best(&self, text: &str) -> Result<(&str, f64), ZeroShotError> {
        let scores = self.categorize(text)?;
        let i = best_index(&scores);
        Ok((&self.labels[i], scores[i]))
    }

    pub(super) fn config_hash(&self) -> u64 {
        let mut hash = turbo_analysis::Fnv1a::new();
        hash.str("turbo:zero-shot")
            .str(&self.config.template)
            .bytes(&[self.config.mode as u8, self.config.aggregation as u8])
            .u64(self.config.overlap_words as u64)
            // The resolved scoring shape: the window budget and
            // which outcome columns are read as entailment and
            // contradiction.
            .u64(self.premise_budget as u64)
            .u64(self.entail as u64)
            .u64(self.contra as u64);
        for label in &self.labels {
            hash.str(label);
        }
        hash.finish()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let total: f64 = exp.iter().sum();
    exp.iter().map(|e| e / total).collect()
}

/// The first index of the highest score.
pub(super) fn best_index(scores: &[f64]) -> usize {
    let mut best = 0;
    for (i, &s) in scores.iter().enumerate().skip(1) {
        if s > scores[best] {
            best = i;
        }
    }
    best
}
