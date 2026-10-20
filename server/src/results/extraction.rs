//! ExtractionResult: the ranked answers of an extraction or
//! question-answering run, the turbo-analysis-embed `Answers` layer
//! (extractive QA, issue #745) plus the model identity. The schema-driven
//! structured records of issue #715 have no merged layer yet; this
//! message carries what exists today.

use turbo_analysis::{Layer, ModelRef, Span, ViewId};
use turbo_analysis_embed::Answers;

use super::{MAX_ANSWERS, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::ExtractionResult;

/// The extraction answers unbundled from the layer: one span per answer
/// in rank order, its score and normalized probability, the abstention
/// score and its probability, and the model's identity.
#[derive(Debug)]
pub struct Extraction {
    pub model: ModelRef,
    /// The text view the spans are offsets into; 0 is the source text.
    pub view: u16,
    pub spans: Vec<Span>,
    pub scores: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub null_score: f64,
    pub null_probability: f64,
}

impl Extraction {
    /// The answers layer the extraction came from.
    pub fn new(a: &Answers, model: ModelRef) -> Extraction {
        Extraction {
            model,
            view: a.spans.view().0,
            spans: (0..a.len()).map(|i| a.spans.get(turbo_analysis::Ix::new(i as u32))).collect(),
            scores: a.scores.clone(),
            probabilities: a.probabilities.clone(),
            null_score: a.null_score,
            null_probability: a.null_probability,
        }
    }

    /// The layer this extraction's columns make, checked the way
    /// `Answers` checks itself.
    pub fn answers(&self) -> Result<Answers, Failure> {
        let mut spans = turbo_analysis::Spans::new(ViewId(self.view));
        for &s in &self.spans {
            if s.start >= s.end {
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "an answer that ends at or before it starts"));
            }
            spans.push(s);
        }
        let out = Answers {
            spans,
            scores: self.scores.clone(),
            probabilities: self.probabilities.clone(),
            null_score: self.null_score,
            null_probability: self.null_probability,
        };
        out.check().map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("extraction answers: {e}")))?;
        Ok(out)
    }
}

impl Wire for Extraction {
    type Msg = ExtractionResult;

    fn to_msg(&self) -> ExtractionResult {
        ExtractionResult {
            model: Some(model_to_proto(&self.model)),
            view: self.view as u32,
            span_start: self.spans.iter().map(|s| s.start).collect(),
            span_end: self.spans.iter().map(|s| s.end).collect(),
            score: self.scores.clone(),
            probability: self.probabilities.clone(),
            null_score: self.null_score,
            null_probability: self.null_probability,
        }
    }

    fn from_msg(msg: &ExtractionResult) -> Result<Self, Failure> {
        if msg.span_start.len() as u64 > MAX_ANSWERS {
            return Err(over("max_answers", "answers", msg.span_start.len() as u64));
        }
        if msg.span_start.len() != msg.span_end.len() {
            return Err(mismatched("span_start", msg.span_start.len(), msg.span_end.len()));
        }
        if msg.view > u16::MAX as u32 {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("view {} is outside 0 to 65535", msg.view)));
        }
        let out = Extraction {
            model: msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
                id: String::new(),
                revision: String::new(),
                sha256: String::new(),
            }),
            view: msg.view as u16,
            spans: msg.span_start.iter().zip(&msg.span_end).map(|(&s, &e)| Span { start: s, end: e }).collect(),
            scores: msg.score.clone(),
            probabilities: msg.probability.clone(),
            null_score: msg.null_score,
            null_probability: msg.null_probability,
        };
        out.answers()?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TURBO_E_CAPACITY;

    fn model() -> ModelRef {
        ModelRef { id: "org/qa".into(), revision: "v3".into(), sha256: "cafe".into() }
    }

    /// Two ranked answers over the source view and a 0.2 abstention mass.
    fn sample() -> Answers {
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 4));
        spans.push(Span::new(16, 24));
        let out = Answers {
            spans,
            scores: vec![8.0, 4.5],
            probabilities: vec![0.5, 0.3],
            null_score: -2.0,
            null_probability: 0.2,
        };
        out.check().unwrap();
        out
    }

    #[test]
    fn the_wire_round_trip_is_unchanged() {
        let a = sample();
        let x = Extraction::new(&a, model());
        assert_eq!(x.view, 0);
        let back = Extraction::from_msg(&x.to_msg()).unwrap();
        assert_eq!(back.model, x.model);
        assert_eq!(back.spans, x.spans);
        assert_eq!(back.scores, x.scores);
        assert_eq!(back.probabilities, x.probabilities);
        assert_eq!((back.null_score, back.null_probability), (-2.0, 0.2));
        // The rebuilt layer holds what the original held.
        let ra = back.answers().unwrap();
        assert_eq!(ra.spans.columns(), a.spans.columns());
        assert_eq!((ra.scores, ra.probabilities), (a.scores.clone(), a.probabilities.clone()));
        assert_eq!(ra.null_probability, a.null_probability);
    }

    #[test]
    fn an_empty_result_is_a_calibrated_abstention() {
        let a = Answers {
            spans: turbo_analysis::Spans::new(ViewId::SOURCE),
            scores: Vec::new(),
            probabilities: Vec::new(),
            null_score: -1.0,
            null_probability: 1.0,
        };
        let back = Extraction::from_msg(&Extraction::new(&a, model()).to_msg()).unwrap();
        assert!(back.spans.is_empty() && back.null_probability == 1.0);
        back.answers().unwrap();
    }

    fn refused(mut f: impl FnMut(&mut ExtractionResult)) -> Failure {
        let mut msg = Extraction::new(&sample(), model()).to_msg();
        f(&mut msg);
        Extraction::from_msg(&msg).unwrap_err()
    }

    #[test]
    fn garbage_is_refused_without_panicking() {
        let e = refused(|m| {
            m.span_start.pop();
        });
        assert!(e.message.contains("span_start"), "{e:?}");
        let e = refused(|m| {
            m.span_end[0] = m.span_start[0];
        });
        assert!(e.message.contains("ends at or before"), "{e:?}");
        let e = refused(|m| {
            m.score[0] = f64::NAN;
        });
        assert!(e.message.contains("finite"), "{e:?}");
        let e = refused(|m| {
            m.probability[0] = 1.5;
        });
        assert!(e.message.contains("extraction answers"), "{e:?}");
        let e = refused(|m| {
            m.null_probability = 0.1;
        });
        assert!(e.message.contains("extraction answers"), "{e:?}");
        let e = refused(|m| {
            m.view = 70000;
        });
        assert!(e.message.contains("view"), "{e:?}");
        let e = Extraction::from_msg(&ExtractionResult::default()).unwrap_err();
        assert!(e.message.contains("extraction answers"), "{e:?}");
    }

    #[test]
    fn over_the_answer_limit_names_it() {
        let msg = ExtractionResult {
            null_probability: 1.0,
            span_start: vec![0; MAX_ANSWERS as usize + 1],
            span_end: vec![1; MAX_ANSWERS as usize + 1],
            ..Default::default()
        };
        let e = Extraction::from_msg(&msg).unwrap_err();
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_answers"), "{e:?}");
    }
}
