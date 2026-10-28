//! Hashed character n-gram linear language classifier with script priors.

use std::collections::HashMap;

use turbo_analysis::ModelRef;

use crate::error::ModelError;

/// The classifier's entry in the model archive.
pub const MODEL_ENTRY: &str = "langdetect.model";

/// A language identification model: hashed character n-gram features and
/// a linear softmax classifier over language codes.
#[derive(Debug, Clone)]
pub struct NgramModel {
    languages: Vec<String>,
    weights: Vec<f32>,
    bias: Vec<f32>,
    ngram_range: (u8, u8),
    buckets: u32,
    model_ref: ModelRef,
}

impl NgramModel {
    /// Creates a model from its parts.
    pub fn new(
        languages: Vec<String>,
        weights: Vec<f32>,
        bias: Vec<f32>,
        ngram_range: (u8, u8),
        buckets: u32,
        model_ref: ModelRef,
    ) -> Result<NgramModel, ModelError> {
        if languages.is_empty() {
            return Err(ModelError::EmptyOutcome);
        }
        if weights.len() != languages.len() * buckets as usize {
            return Err(ModelError::Weights { expected: languages.len() * buckets as usize, found: weights.len() });
        }
        if bias.len() != languages.len() {
            return Err(ModelError::Bias { expected: languages.len(), found: bias.len() });
        }
        Ok(NgramModel { languages, weights, bias, ngram_range, buckets, model_ref })
    }

    /// The language codes the model can answer.
    pub fn supported_languages(&self) -> impl Iterator<Item = &str> {
        self.languages.iter().map(String::as_str)
    }

    /// What layers record as their model.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Creates a model from the output of `turbo_trainer::langid::train`.
    pub fn from_trained(
        languages: Vec<String>,
        weights: Vec<f32>,
        bias: Vec<f32>,
        ngram_range: (u8, u8),
        buckets: u32,
        model_ref: ModelRef,
    ) -> Result<NgramModel, ModelError> {
        Self::new(languages, weights, bias, ngram_range, buckets, model_ref)
    }

    /// Hashes a byte string into a bucket.
    fn hash(bytes: &[u8], buckets: u32) -> u32 {
        let mut h: u32 = 2166136261;
        for &b in bytes {
            h ^= b as u32;
            h = h.wrapping_mul(16777619);
        }
        h % buckets
    }

    /// Extracts hashed character n-gram features from `text`.
    fn features(&self, text: &str) -> HashMap<u32, f32> {
        let bytes = text.as_bytes();
        let mut features = HashMap::new();
        let (min_n, max_n) = self.ngram_range;
        for n in min_n..=max_n {
            let n = n as usize;
            if bytes.len() < n {
                continue;
            }
            for window in bytes.windows(n) {
                let bucket = Self::hash(window, self.buckets);
                *features.entry(bucket).or_insert(0.0) += 1.0;
            }
        }
        let total: f32 = features.values().sum();
        if total > 0.0 {
            for count in features.values_mut() {
                *count /= total;
            }
        }
        features
    }

    /// Predicts the language of `text`, returning scores for each language.
    pub fn scores(&self, text: &str) -> Vec<f32> {
        let features = self.features(text);
        let mut scores = self.bias.clone();
        for (bucket, count) in features {
            for (i, score) in scores.iter_mut().enumerate() {
                *score += self.weights[i * self.buckets as usize + bucket as usize] * count;
            }
        }
        self.apply_script_priors(text, &mut scores);
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += *score;
        }
        if sum > 0.0 {
            for score in scores.iter_mut() {
                *score /= sum;
            }
        }
        scores
    }

    /// Applies a small prior for languages whose script matches the text.
    fn apply_script_priors(&self, text: &str, scores: &mut [f32]) {
        let script = Script::of(text);
        for (i, lang) in self.languages.iter().enumerate() {
            if script.matches(lang) {
                scores[i] += 0.1;
            }
        }
    }

    /// Predicts the most likely language of `text`.
    pub fn predict(&self, text: &str) -> Option<&str> {
        let scores = self.scores(text);
        scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| self.languages[i].as_str())
    }
}

/// A coarse script class for language priors.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Script {
    Latin,
    Cyrillic,
    Arabic,
    Devanagari,
    Han,
    Hiragana,
    Katakana,
    Hangul,
    Other,
}

impl Script {
    fn of(text: &str) -> Script {
        let mut counts = [0usize; 9];
        for c in text.chars() {
            let i = match c {
                '\u{0040}'..='\u{024F}' => 0,
                '\u{0400}'..='\u{04FF}' => 1,
                '\u{0600}'..='\u{06FF}' => 2,
                '\u{0900}'..='\u{097F}' => 3,
                '\u{4E00}'..='\u{9FFF}' => 4,
                '\u{3040}'..='\u{309F}' => 5,
                '\u{30A0}'..='\u{30FF}' => 6,
                '\u{AC00}'..='\u{D7AF}' => 7,
                _ => 8,
            };
            counts[i] += 1;
        }
        let max = counts.iter().enumerate().max_by_key(|(_, c)| *c).map(|(i, _)| i).unwrap_or(8);
        match max {
            0 => Script::Latin,
            1 => Script::Cyrillic,
            2 => Script::Arabic,
            3 => Script::Devanagari,
            4 => Script::Han,
            5 => Script::Hiragana,
            6 => Script::Katakana,
            7 => Script::Hangul,
            _ => Script::Other,
        }
    }

    fn matches(self, lang: &str) -> bool {
        match self {
            Script::Latin => matches!(lang, "en" | "de" | "fr" | "es" | "it" | "pt" | "nl" | "pl" | "tr" | "vi"),
            Script::Cyrillic => matches!(lang, "ru" | "uk" | "bg" | "sr" | "mk"),
            Script::Arabic => matches!(lang, "ar" | "fa" | "ur"),
            Script::Devanagari => matches!(lang, "hi" | "mr" | "ne"),
            Script::Han => matches!(lang, "zh" | "ja"),
            Script::Hiragana | Script::Katakana => lang == "ja",
            Script::Hangul => lang == "ko",
            Script::Other => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> NgramModel {
        NgramModel::new(
            vec!["en".into(), "de".into()],
            vec![0.0; 2 * 16],
            vec![0.0; 2],
            (1, 3),
            16,
            ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
        )
        .unwrap()
    }

    #[test]
    fn predicts_with_bias_only() {
        let m = model();
        let scores = m.scores("hello");
        assert_eq!(scores.len(), 2);
        assert!((scores[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn supported_languages_iterates() {
        let m = model();
        let langs: Vec<_> = m.supported_languages().collect();
        assert_eq!(langs, ["en", "de"]);
    }
}
