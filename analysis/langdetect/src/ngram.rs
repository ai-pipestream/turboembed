//! Hashed character n-gram linear language classifier with script priors.

use std::collections::BTreeMap;

use turbo_analysis::ModelRef;

use crate::error::ModelError;

/// The native classifier's entry in a model archive.
pub const MODEL_ENTRY: &str = "langid.ngram";

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
        if languages.is_empty() || languages.iter().any(String::is_empty) {
            return Err(ModelError::EmptyOutcome);
        }
        for (i, language) in languages.iter().enumerate() {
            if languages[..i].contains(language) {
                return Err(ModelError::DuplicateOutcome(language.clone()));
            }
        }
        if buckets == 0 || ngram_range.0 == 0 || ngram_range.0 > ngram_range.1 {
            return Err(ModelError::Format("invalid n-gram configuration".into()));
        }
        if weights.len() != languages.len() * buckets as usize {
            return Err(ModelError::Weights { expected: languages.len() * buckets as usize, found: weights.len() });
        }
        if bias.len() != languages.len() {
            return Err(ModelError::Bias { expected: languages.len(), found: bias.len() });
        }
        Ok(NgramModel { languages, weights, bias, ngram_range, buckets, model_ref })
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

    /// The language codes the model can answer.
    pub fn supported_languages(&self) -> impl Iterator<Item = &str> {
        self.languages.iter().map(String::as_str)
    }

    /// Number of supported language outcomes.
    pub fn num_languages(&self) -> usize {
        self.languages.len()
    }

    /// Language code at `index`.
    pub fn language(&self, index: usize) -> &str {
        &self.languages[index]
    }

    /// Deterministic FNV-1a configuration hash of the serialized model.
    pub fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new().bytes(&self.to_bytes()).finish()
    }

    /// What layers record as their model.
    pub fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// The n-gram configuration the trainer and runtime share.
    pub fn ngram_range(&self) -> (u8, u8) {
        self.ngram_range
    }

    /// The number of feature buckets.
    pub fn buckets(&self) -> u32 {
        self.buckets
    }

    /// Serializes the trained classifier for [`MODEL_ENTRY`].
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"TNL1");
        put_u32(&mut out, self.languages.len() as u32);
        put_u32(&mut out, self.buckets);
        out.extend_from_slice(&[self.ngram_range.0, self.ngram_range.1, 0, 0]);
        for language in &self.languages {
            put_u32(&mut out, language.len() as u32);
            out.extend_from_slice(language.as_bytes());
        }
        for value in self.weights.iter().chain(&self.bias) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Loads a classifier serialized by [`NgramModel::to_bytes`].
    pub fn from_bytes(bytes: &[u8], model_ref: ModelRef) -> Result<NgramModel, ModelError> {
        let mut at = 0usize;
        if take(bytes, &mut at, 4)? != b"TNL1" {
            return Err(ModelError::Format("not a native language model".into()));
        }
        let language_count = read_u32(bytes, &mut at)? as usize;
        let buckets = read_u32(bytes, &mut at)?;
        let range = take(bytes, &mut at, 4)?;
        let ngram_range = (range[0], range[1]);
        // Every count is checked against the bytes left before anything is
        // allocated, so a forged header cannot ask for more memory than the
        // file could hold.
        let float_count = (buckets as usize)
            .checked_add(1)
            .and_then(|row| row.checked_mul(language_count))
            .and_then(|count| count.checked_mul(4))
            .ok_or_else(|| ModelError::Format("size overflow".into()))?;
        if language_count.saturating_mul(4).saturating_add(float_count) > bytes.len() - at {
            return Err(ModelError::Format("truncated model".into()));
        }
        let mut languages = Vec::with_capacity(language_count);
        for _ in 0..language_count {
            let len = read_u32(bytes, &mut at)? as usize;
            let text = std::str::from_utf8(take(bytes, &mut at, len)?)
                .map_err(|_| ModelError::Format("language code is not UTF-8".into()))?;
            languages.push(text.to_owned());
        }
        let weight_count =
            language_count.checked_mul(buckets as usize).ok_or_else(|| ModelError::Format("size overflow".into()))?;
        let weights = read_floats(bytes, &mut at, weight_count)?;
        let bias = read_floats(bytes, &mut at, language_count)?;
        if at != bytes.len() {
            return Err(ModelError::Format("trailing model bytes".into()));
        }
        Self::new(languages, weights, bias, ngram_range, buckets, model_ref)
    }

    /// Predicts scores for every language in model order.
    pub fn scores(&self, text: &str) -> Vec<f32> {
        let features = features(text, self.ngram_range, self.buckets);
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

    fn apply_script_priors(&self, text: &str, scores: &mut [f32]) {
        let script = Script::of(text);
        for (i, language) in self.languages.iter().enumerate() {
            if script.matches(language) {
                scores[i] += 0.1;
            }
        }
    }

    /// Predicts the most likely language, or `None` for blank text.
    pub fn predict(&self, text: &str) -> Option<&str> {
        self.best(text).map(|(language, _)| language)
    }

    /// Predicts the most likely language when its score meets `threshold`.
    pub fn predict_with_threshold(&self, text: &str, threshold: f32) -> Option<&str> {
        let (language, score) = self.best(text)?;
        (score >= threshold).then_some(language)
    }

    /// Returns the best language and its confidence.
    pub fn best(&self, text: &str) -> Option<(&str, f32)> {
        if text.trim().is_empty() {
            return None;
        }
        self.scores(text)
            .iter()
            .copied()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, score)| (self.languages[i].as_str(), score))
    }
}

/// Extracts hashed Unicode scalar n-grams in deterministic bucket order.
pub fn features(text: &str, ngram_range: (u8, u8), buckets: u32) -> BTreeMap<u32, f32> {
    let chars: Vec<char> = text.chars().collect();
    let mut features = BTreeMap::new();
    for n in ngram_range.0..=ngram_range.1 {
        for window in chars.windows(n as usize) {
            let mut hash = 2166136261u32;
            for &c in window {
                let mut encoded = [0u8; 4];
                for &b in c.encode_utf8(&mut encoded).as_bytes() {
                    hash ^= b as u32;
                    hash = hash.wrapping_mul(16777619);
                }
            }
            *features.entry(hash % buckets).or_insert(0.0) += 1.0;
        }
    }
    features
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn take<'a>(bytes: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8], ModelError> {
    let end = at.checked_add(len).ok_or_else(|| ModelError::Format("size overflow".into()))?;
    let part = bytes.get(*at..end).ok_or_else(|| ModelError::Format("truncated model".into()))?;
    *at = end;
    Ok(part)
}

fn read_u32(bytes: &[u8], at: &mut usize) -> Result<u32, ModelError> {
    Ok(u32::from_le_bytes(take(bytes, at, 4)?.try_into().expect("four bytes")))
}

fn read_floats(bytes: &[u8], at: &mut usize, count: usize) -> Result<Vec<f32>, ModelError> {
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(f32::from_le_bytes(take(bytes, at, 4)?.try_into().expect("four bytes")));
    }
    Ok(values)
}

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
            Script::Latin => matches!(
                lang,
                "en" | "eng"
                    | "de"
                    | "deu"
                    | "fr"
                    | "fra"
                    | "es"
                    | "spa"
                    | "it"
                    | "ita"
                    | "pt"
                    | "por"
                    | "nl"
                    | "nld"
                    | "pl"
                    | "pol"
                    | "tr"
                    | "tur"
                    | "vi"
                    | "vie"
            ),
            Script::Cyrillic => {
                matches!(lang, "ru" | "rus" | "uk" | "ukr" | "bg" | "bul" | "sr" | "srp" | "mk" | "mkd")
            }
            Script::Arabic => matches!(lang, "ar" | "ara" | "fa" | "fas" | "ur" | "urd"),
            Script::Devanagari => matches!(lang, "hi" | "hin" | "mr" | "mar" | "ne" | "nep"),
            Script::Han => matches!(lang, "zh" | "zho" | "cmn" | "ja" | "jpn"),
            Script::Hiragana | Script::Katakana => matches!(lang, "ja" | "jpn"),
            Script::Hangul => matches!(lang, "ko" | "kor"),
            Script::Other => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> NgramModel {
        NgramModel::new(
            vec!["eng".into(), "deu".into()],
            vec![0.0; 2 * 16],
            vec![0.0; 2],
            (1, 3),
            16,
            ModelRef { id: "test".into(), revision: "1".into(), sha256: "0".into() },
        )
        .unwrap()
    }

    #[test]
    fn scores_and_serializes_deterministically() {
        let model = model();
        assert_eq!(model.scores("hello"), vec![0.5, 0.5]);
        let loaded = NgramModel::from_bytes(&model.to_bytes(), model.model_ref.clone()).unwrap();
        assert_eq!(loaded.scores("hello"), model.scores("hello"));
        assert_eq!(loaded.supported_languages().collect::<Vec<_>>(), ["eng", "deu"]);
    }

    #[test]
    fn features_count_unicode_scalars_in_bucket_order() {
        let features = features("éé", (1, 1), 16);
        assert_eq!(features.len(), 1);
        assert_eq!(features.values().copied().collect::<Vec<_>>(), [2.0]);
        assert_eq!(features, turbo_trainer::langid::features("éé", (1, 1), 16));
    }

    #[test]
    fn forged_header_counts_are_refused_before_allocation() {
        let model = model();
        let mut header = b"TNL1".to_vec();
        put_u32(&mut header, u32::MAX);
        put_u32(&mut header, u32::MAX);
        header.extend_from_slice(&[1, 3, 0, 0]);
        let error = NgramModel::from_bytes(&header, model.model_ref.clone()).unwrap_err();
        assert!(matches!(error, ModelError::Format(_)), "{error}");
        let mut bytes = model.to_bytes();
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        let error = NgramModel::from_bytes(&bytes, model.model_ref.clone()).unwrap_err();
        assert!(matches!(error, ModelError::Format(_)), "{error}");
        let mut bytes = model.to_bytes();
        bytes.pop();
        assert!(NgramModel::from_bytes(&bytes, model.model_ref.clone()).is_err());
        assert!(NgramModel::from_bytes(&model.to_bytes(), model.model_ref.clone()).is_ok());
    }

    #[test]
    fn blank_text_has_no_prediction() {
        assert_eq!(model().predict(" \n"), None);
        assert_eq!(model().best(""), None);
    }
}
