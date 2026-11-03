//! Native sentence evaluation against corpus labels and a deterministic baseline.
//! Callers supply predictions from rules, SaT, or an explicitly identified native
//! head. No reference model bytes, external runtime, or downloads are accepted.

use crate::{
    GateRun,
    card::HeldOutCard,
    error::{Error, Result},
    segmentation::HeldOut,
    spans,
};
use turbo_analysis::{Annotator, Span};
use turbo_eval::model_gate::{HeldOutSet, ModelGateRecord, ModelId, Scoring, Task};
use turbo_formats::{evaluate::SentenceDetectorEvaluator, sample::SentenceSample};
use turbo_zoo::Sha256;

pub const BASELINE_ID: &str = "sentence-newline-v1";

/// Versioned algorithm identity, not a model-file digest.
pub fn baseline() -> ModelId {
    let detector = turbo_sentdetect::NewlineDetector::new();
    let identity = format!(
        "{BASELINE_ID}:{}:{}:{:016x}",
        turbo_sentdetect::NewlineDetector::ID,
        turbo_sentdetect::NewlineDetector::VERSION,
        detector.config_hash()
    );
    ModelId { id: BASELINE_ID.into(), sha256: Sha256::of(identity.as_bytes()).to_string() }
}

fn validate(text: &str, values: &[Span]) -> Result<()> {
    let mut end = 0;
    for span in values {
        if span.start < end || span.start >= span.end || text.get(span.range()).is_none() {
            return Err(Error::Model(
                "sentence spans must be ordered, nonempty, nonoverlapping UTF-8 byte ranges".into(),
            ));
        }
        end = span.end;
    }
    Ok(())
}

/// Score aligned predictions and the newline baseline against the same gold
/// sentence spans. The record retains per-sample counts and the existing paired
/// gate limits. A passing comparison does not claim peer-model parity.
pub fn gate(
    candidate: ModelId,
    held_out: &HeldOut<'_>,
    predictions: &[Vec<Span>],
    set_name: &str,
    corpus_sha256: &str,
) -> Result<GateRun> {
    let HeldOut::Sentences(samples) = held_out else {
        return Err(Error::ComponentMismatch { recipe: "sentence gold", samples: "token" });
    };
    if samples.is_empty() {
        return Err(Error::EmptySplit { what: "held-out sentence" });
    }
    if samples.len() != predictions.len() {
        return Err(Error::Model("one prediction vector is required per gold sentence sample".into()));
    }
    if samples.iter().all(|sample| sample.sentences().is_empty()) {
        return Err(Error::EmptySplit { what: "held-out gold sentence labels" });
    }
    let mut candidate_counts = Vec::with_capacity(samples.len());
    let mut baseline_counts = Vec::with_capacity(samples.len());
    let mut newline = Vec::new();
    for (sample, predicted) in samples.iter().zip(predictions) {
        validate_text_len(sample.document().len())?;
        validate(sample.document(), sample.sentences())?;
        if sample.source().start > sample.source().end || sample.sources().iter().any(|s| s.start > s.end) {
            return Err(Error::Model("sentence source ranges must have ordered bounds".into()));
        }
        validate(sample.document(), predicted)?;
        newline.clear();
        turbo_sentdetect::newline_spans(sample.document(), &mut newline);
        candidate_counts.push(score(sample, predicted)?);
        baseline_counts.push(score(sample, &newline)?);
    }
    let set = HeldOutSet {
        name: set_name.into(),
        corpus_sha256: corpus_sha256.into(),
        samples_sha256: samples_sha256(held_out),
        skipped_sentences: 0,
        scoring: Scoring::Spans,
    };
    let record = ModelGateRecord::new(
        Task::SentenceDetector,
        set,
        candidate,
        baseline(),
        spans::pair(candidate_counts, baseline_counts),
    )
    .map_err(Error::Gate)?;
    record.check().map_err(Error::Gate)?;
    Ok(GateRun {
        record,
        held_out: HeldOutCard { samples: samples.len(), skipped_sentences: 0, tokens: None, names: None, chunks: None },
    })
}

fn score(sample: &SentenceSample<'_>, predicted: &[Span]) -> Result<turbo_eval::model_gate::Counts> {
    let mut evaluator = SentenceDetectorEvaluator::new();
    evaluator.evaluate(sample, predicted)?;
    Ok(spans::counts(evaluator.fmeasure()))
}

/// Hash the scored text/labels and source coordinates. Source ranges may
/// belong to different documents (MASC), so they are not globally ordered.
fn samples_sha256(held_out: &HeldOut<'_>) -> String {
    let mut bytes = b"native-sentence-gold-v1".to_vec();
    bytes.extend_from_slice(held_out.sha256().as_bytes());
    if let HeldOut::Sentences(samples) = held_out {
        for sample in samples {
            for span in std::iter::once(&sample.source()).chain(sample.sources()) {
                bytes.extend_from_slice(&span.start.to_le_bytes());
                bytes.extend_from_slice(&span.end.to_le_bytes());
            }
        }
    }
    Sha256::of(&bytes).to_string()
}

fn validate_text_len(len: usize) -> Result<()> {
    u32::try_from(len).map(|_| ()).map_err(|_| Error::Model("sentence sample exceeds the source offset limit".into()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn text_length_limit_returns_an_error_without_allocating_text() {
        assert!(super::validate_text_len(0).is_ok());
        assert!(super::validate_text_len(u32::MAX as usize).is_ok());
        if let Ok(too_long) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(super::validate_text_len(too_long).is_err());
        }
    }
}
