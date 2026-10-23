//! Multi-label emotion annotator (`#855`, part of `#318`): the 28 GoEmotions
//! labels of `turbo:emotion` ([`EmotionLayer`]) for each sentence or chunk.
//!
//! [`EmotionAnnotator`] (`turbo-doccat:emotion`) scores a unit in one of
//! three stages ([`EmotionStage`]):
//! - **Lexicon**: the seed cue words and emoji of `lexicon.rs`. Each cue
//!   adds its weight to its label by a noisy-or, unless a negator ("not",
//!   "never", ...) stands in the three tokens before it, and records the
//!   cue's token as evidence. Neutral is one minus the strongest emotion.
//! - **Learned**: an [`EmotionVectorScorer`] such as [`EmotionVectorHead`]
//!   over the unit's vector in `Vectors<Sentences>` or `Vectors<Chunks>`;
//!   a unit with no vector fails the document, here and in the hybrid
//!   stage. A positive emotion's evidence is the unit's
//!   token range, the text the head read.
//! - **Hybrid**: each emotion the noisy-or of the two, its evidence the
//!   lexicon's cues and, where the lexicon found none, the unit's token
//!   range; neutral the smaller of the head's neutral and one minus the
//!   strongest fused emotion.
//!
//! Neutral never carries evidence: it is the absence of a cue. The stage,
//! the threshold and the head are in the model id and configuration hash.

use std::sync::Arc;

use turbo_analysis::{IxRange, ModelRef};
use turbo_analysis_layers::{EmotionDimension, EmotionScores, Tokens};

mod annotate;
mod head;
mod lexicon;
mod metrics;
#[cfg(test)]
mod tests;

pub use head::{EmotionHeadError, EmotionVectorHead, EmotionVectorScorer};
pub use metrics::{EmotionMetrics, GoldEmotionUnit, evaluate_emotion_corpus};

/// The score at or above which a label is active (`0.5`).
pub const DEFAULT_EMOTION_THRESHOLD: f64 = 0.5;

/// Which signals [`EmotionAnnotator`] scores with.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionStage {
    #[default]
    Lexicon = 0,
    Learned = 1,
    Hybrid = 2,
}

/// Which unit layer [`EmotionAnnotator`] scores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum EmotionUnit {
    #[default]
    Sentences = 0,
    Chunks = 1,
}

/// See the module comment.
#[derive(Clone)]
pub struct EmotionAnnotator {
    pub unit: EmotionUnit,
    pub stage: EmotionStage,
    pub vector_scorer: Option<Arc<dyn EmotionVectorScorer>>,
    /// The score at or above which a label is active and, for an emotion,
    /// carries evidence.
    pub threshold: f64,
}

impl Default for EmotionAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

/// One `(label, token range)` evidence pair.
pub type EmotionEvidence = (EmotionDimension, IxRange<Tokens>);

impl EmotionAnnotator {
    /// The lexicon stage over sentences at the default threshold.
    pub fn new() -> Self {
        EmotionAnnotator {
            unit: EmotionUnit::Sentences,
            stage: EmotionStage::Lexicon,
            vector_scorer: None,
            threshold: DEFAULT_EMOTION_THRESHOLD,
        }
    }

    pub fn with_unit(mut self, unit: EmotionUnit) -> Self {
        self.unit = unit;
        self
    }

    pub fn with_stage(mut self, stage: EmotionStage) -> Self {
        self.stage = stage;
        self
    }

    pub fn with_vector_scorer(mut self, scorer: Arc<dyn EmotionVectorScorer>) -> Self {
        self.vector_scorer = Some(scorer);
        self
    }

    /// Sets the threshold; scoring fails unless it is finite and in
    /// `0.05..=0.99`.
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Scores the unit of tokens `start..end`, `keys` being every token's
    /// lookup form (see `annotate.rs`), and `vector` the unit's vector.
    pub fn score_unit(
        &self,
        keys: &[String],
        start: usize,
        end: usize,
        vector: Option<&[f32]>,
    ) -> Result<(EmotionScores, Vec<EmotionEvidence>), String> {
        if !(self.threshold.is_finite() && (0.05..=0.99).contains(&self.threshold)) {
            return Err("emotion threshold must be finite and in 0.05..=0.99".into());
        }
        if start > end || end > keys.len() || end >= u32::MAX as usize {
            return Err("unit token range is reversed or outside the tokens".into());
        }
        let (mut lex, mut evidence) = (EmotionScores::default(), Vec::new());
        if self.stage != EmotionStage::Learned {
            for t in start..end {
                let negated = keys[t.saturating_sub(lexicon::NEGATION_WINDOW).max(start)..t]
                    .iter()
                    .any(|k| lexicon::NEGATORS.contains(&k.as_str()));
                for (d, w) in lexicon::cues(&keys[t]) {
                    if !negated {
                        lex.set(d, 1.0 - (1.0 - lex.get(d)) * (1.0 - w));
                        evidence.push((d, IxRange::new(t as u32, t as u32 + 1)));
                    }
                }
            }
        }
        let strongest = |s: &EmotionScores| {
            EmotionDimension::ALL
                .into_iter()
                .filter(|&d| d != EmotionDimension::Neutral)
                .map(|d| s.get(d))
                .fold(0.0, f64::max)
        };
        let head = match (self.stage, &self.vector_scorer, vector) {
            (EmotionStage::Lexicon, _, _) => None,
            (_, Some(s), Some(v)) => {
                let h = s.score_emotions(v)?;
                if !h.is_valid() {
                    return Err("emotion head scores must be finite and in 0..=1".into());
                }
                Some(h)
            }
            (_, None, _) => return Err("the learned and hybrid stages need a vector scorer".into()),
            (_, Some(_), None) => return Err("the learned and hybrid stages need a vector for every unit".into()),
        };
        let mut scores = lex;
        match (self.stage, head) {
            (EmotionStage::Learned, Some(h)) => scores = h,
            (EmotionStage::Hybrid, Some(h)) => {
                for d in EmotionDimension::ALL {
                    if d != EmotionDimension::Neutral {
                        scores.set(d, 1.0 - (1.0 - lex.get(d)) * (1.0 - h.get(d)));
                    }
                }
                scores.set(EmotionDimension::Neutral, h.get(EmotionDimension::Neutral).min(1.0 - strongest(&scores)));
            }
            _ => scores.set(EmotionDimension::Neutral, 1.0 - strongest(&scores)),
        }
        // A positive emotion with no cue in the unit points at the unit.
        if start < end {
            for d in scores.active(self.threshold) {
                if d != EmotionDimension::Neutral && !evidence.iter().any(|&(l, _)| l == d) {
                    evidence.push((d, IxRange::new(start as u32, end as u32)));
                }
            }
        }
        evidence.retain(|&(d, _)| scores.get(d) >= self.threshold);
        evidence.sort_unstable_by_key(|&(l, r)| (l, r.start, r.end));
        evidence.dedup();
        Ok((scores, evidence))
    }

    fn model_ref(&self) -> ModelRef {
        let stage = match self.stage {
            EmotionStage::Lexicon => "lexicon",
            EmotionStage::Learned => "learned",
            EmotionStage::Hybrid => "hybrid",
        };
        let head = self.vector_scorer.as_ref().filter(|_| self.stage != EmotionStage::Lexicon);
        match head.and_then(|h| h.model()) {
            Some(m) => ModelRef { id: format!("{}:emotion-{stage}", m.id), revision: m.revision, sha256: m.sha256 },
            None => ModelRef {
                id: format!("goemotions-seed:{stage}"),
                revision: "1".into(),
                sha256: turbo_maxent::sha256(&self.config().to_le_bytes()).to_hex(),
            },
        }
    }

    fn config(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let head = self.vector_scorer.as_ref().filter(|_| self.stage != EmotionStage::Lexicon);
        let words = lexicon::WORDS.iter().chain(lexicon::EMOJI).map(|(w, d, x)| (w.as_bytes(), *d as u8, x.to_bits()));
        let mut mix = |b: u64| h = (h ^ b).wrapping_mul(0x0100_0000_01b3);
        for (w, d, x) in words {
            w.iter().for_each(|&b| mix(u64::from(b)));
            mix(u64::from(d));
            mix(x);
        }
        mix(self.unit as u64);
        mix(self.stage as u64);
        mix(self.threshold.to_bits());
        mix(head.map_or(0, |s| s.config_hash()));
        h
    }
}
