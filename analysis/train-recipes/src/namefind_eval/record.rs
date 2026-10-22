use serde::{Deserialize, Serialize};
use turbo_formats::evaluate::FMeasure;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Mention {
    pub start: u32,
    pub end: u32,
    pub label: Option<String>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub mention: Mention,
    pub probability: f64,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub tokens: u32,
    pub clear_adaptive_data: bool,
    pub gold: Vec<Mention>,
    pub predicted: Vec<Prediction>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub format_version: u32,
    /// The producer and, for a reference run, its source revision.
    pub engine: String,
    pub model_sha256: String,
    pub corpus_sha256: String,
    pub types: String,
    pub samples: Vec<Sample>,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub sentences: usize,
    pub tokens: u64,
    pub gold: u64,
    pub predicted: u64,
    pub correct: u64,
    pub precision: f64,
    pub recall: f64,
    /// OpenNLP returns -1 when precision and recall are both zero.
    pub f1: f64,
}

fn spans<'a>(mentions: impl Iterator<Item = &'a Mention>, tokens: u32) -> Result<(), String> {
    let mut previous = 0;
    for m in mentions {
        if m.start < previous || m.start >= m.end || m.end > tokens {
            return Err("empty, overlapping, unsorted or out-of-range mention".into());
        }
        previous = m.end;
    }
    Ok(())
}
impl Record {
    /// Validate decoded records before comparison or scoring. Duplicates and
    /// overlaps cannot inflate true positives. Empty corpora are not evidence.
    pub fn validate(&self) -> Result<(), String> {
        if self.format_version != 1 || self.engine.trim().is_empty() || self.samples.is_empty() {
            return Err("record version, engine or empty corpus".into());
        }
        super::entity_types(&self.types)?;
        for digest in [&self.model_sha256, &self.corpus_sha256] {
            if digest.len() != 64 || !digest.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)) {
                return Err("record SHA-256 must be 64 lowercase hexadecimal characters".into());
            }
        }
        for (i, s) in self.samples.iter().enumerate() {
            if s.tokens == 0 {
                return Err(format!("sample {i}: no tokens"));
            }
            spans(s.gold.iter(), s.tokens).map_err(|e| format!("sample {i} gold: {e}"))?;
            spans(s.predicted.iter().map(|p| &p.mention), s.tokens).map_err(|e| format!("sample {i}: {e}"))?;
            if s.predicted.iter().any(|p| !p.probability.is_finite() || !(0.0..=1.0).contains(&p.probability)) {
                return Err(format!("sample {i}: probability must be finite and in [0,1]"));
            }
        }
        Ok(())
    }

    pub fn summary(&self) -> Result<Summary, String> {
        self.validate()?;
        let mut counts = FMeasure::default();
        let mut tokens = 0u64;
        for s in &self.samples {
            tokens = tokens.checked_add(u64::from(s.tokens)).ok_or("token count overflow")?;
            let predicted: Vec<_> = s.predicted.iter().map(|p| &p.mention).collect();
            let gold: Vec<_> = s.gold.iter().collect();
            counts.update(&gold, &predicted);
        }
        Ok(Summary {
            sentences: self.samples.len(),
            tokens,
            gold: counts.target(),
            predicted: counts.selected(),
            correct: counts.true_positives(),
            precision: counts.precision(),
            recall: counts.recall(),
            f1: counts.f_measure(),
        })
    }

    /// Require the same model bytes, corpus bytes, label subset and sample
    /// stream before checking exact spans/types and absolute probability error.
    /// Returns the largest observed probability error. Aggregate F1 agreement
    /// alone does not establish prediction parity.
    pub fn compare(&self, reference: &Self, tolerance: f64) -> Result<f64, String> {
        self.validate()?;
        reference.validate()?;
        if !tolerance.is_finite() || !(0.0..=1.0).contains(&tolerance) {
            return Err("tolerance must be finite and in [0,1]".into());
        }
        if self.model_sha256 != reference.model_sha256
            || self.corpus_sha256 != reference.corpus_sha256
            || self.types != reference.types
            || self.samples.len() != reference.samples.len()
        {
            return Err("model, corpus, types or sample count differ".into());
        }
        let mut max_error = 0.0f64;
        for (i, (s, r)) in self.samples.iter().zip(&reference.samples).enumerate() {
            if s.tokens != r.tokens || s.clear_adaptive_data != r.clear_adaptive_data || s.gold != r.gold {
                return Err(format!("sample {i}: reference stream differs"));
            }
            if s.predicted.len() != r.predicted.len() {
                return Err(format!("sample {i}: prediction count differs"));
            }
            for (p, q) in s.predicted.iter().zip(&r.predicted) {
                if p.mention != q.mention {
                    return Err(format!("sample {i}: span or type differs"));
                }
                let error = (p.probability - q.probability).abs();
                if error > tolerance {
                    return Err(format!("sample {i}: probability error {error} exceeds {tolerance}"));
                }
                max_error = max_error.max(error);
            }
        }
        Ok(max_error)
    }
}
