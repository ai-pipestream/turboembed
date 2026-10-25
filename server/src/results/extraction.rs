//! ExtractionResult: the ranked answers of an extraction or
//! question-answering run, the turbo-analysis-embed `Answers` layer
//! (extractive QA, issue #745) plus the model identity. Schema-driven
//! structured records travel as RecordsResult (`super::records`).

use turbo_analysis::{Layer, ModelRef, Span, ViewId};
use turbo_analysis_embed::Answers;

use super::{MAX_ANSWER_FRAGMENTS, MAX_ANSWERS, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::ExtractionResult;

/// The extraction answers unbundled from the layer: one bounding span
/// per answer in rank order, the fragments of answers made of separated
/// pieces (empty when every answer is one continuous piece), scores,
/// probabilities, the abstention mass, and the model's identity.
#[derive(Debug)]
pub struct Extraction {
    pub model: ModelRef,
    /// The text view the spans are offsets into; 0 is the source text.
    pub view: u16,
    /// The view text the spans are offsets into; empty when `offsets`
    /// is `OffsetMode::Utf8` and no reference text is carried.
    pub text: String,
    /// The wire unit `to_msg` emits.
    pub offsets: OffsetMode,
    pub spans: Vec<Span>,
    /// The fragments of discontinuous answers, flat in answer order.
    pub fragments: Vec<Span>,
    /// One past the last fragment of each answer; empty with fragments.
    pub frag_at: Vec<u32>,
    pub scores: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub null_score: f64,
    pub null_probability: f64,
}

impl Extraction {
    /// The answers layer the extraction came from.
    pub fn new(a: &Answers, model: ModelRef) -> Extraction {
        let (fragments, frag_at) = match a.spans.fragment_columns() {
            Some((start, end, at)) => {
                (start.iter().zip(end).map(|(&s, &e)| Span { start: s, end: e }).collect(), at.to_vec())
            }
            None => (Vec::new(), Vec::new()),
        };
        Extraction {
            model,
            view: a.spans.view().0,
            text: String::new(),
            offsets: OffsetMode::Utf8,
            spans: (0..a.len()).map(|i| a.spans.get(turbo_analysis::Ix::new(i as u32))).collect(),
            fragments,
            frag_at,
            scores: a.scores.clone(),
            probabilities: a.probabilities.clone(),
            null_score: a.null_score,
            null_probability: a.null_probability,
        }
    }

    /// Attaches reference view text and wire [`OffsetMode`].
    pub fn with_offsets(mut self, text: impl Into<String>, offsets: OffsetMode) -> Self {
        self.text = text.into();
        self.offsets = offsets;
        self
    }

    /// The layout checks that keep the rebuild below from slicing or
    /// pushing out of range, naming the column at fault.
    fn layout(&self) -> Result<(), Failure> {
        let n = self.spans.len();
        if self.scores.len() != n || self.probabilities.len() != n {
            return Err(mismatched("score", self.scores.len(), n));
        }
        for (i, &s) in self.spans.iter().enumerate() {
            if s.start >= s.end {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    format!("answer {i} ends at or before it starts"),
                ));
            }
        }
        let frags = self.fragments.len();
        if self.frag_at.is_empty() && frags == 0 {
            return Ok(());
        }
        // Fragment columns present: one per answer, laid over the
        // fragment column, each answer holding at least one fragment.
        if self.frag_at.len() != n {
            return Err(mismatched("frag_at", self.frag_at.len(), n));
        }
        if !self.frag_at.windows(2).all(|w| w[0] <= w[1]) || self.frag_at.last().is_some_and(|&e| e as usize != frags) {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                "frag_at: fragment ends in answer order, the last past the last fragment",
            ));
        }
        let mut k = 0;
        for (a, &span) in self.spans.iter().enumerate() {
            let lo = k;
            let hi = self.frag_at[a] as usize;
            if hi <= lo {
                return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("answer {a} has no fragments")));
            }
            let frags = &self.fragments[lo..hi];
            let multi = frags.len() > 1;
            for w in frags.windows(2) {
                if w[0].end >= w[1].start {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        format!("answer {a}: fragments separated in text order"),
                    ));
                }
            }
            for f in frags {
                if f.start > f.end || (multi && f.start == f.end) {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        format!("answer {a}: a fragment that ends at or before it starts"),
                    ));
                }
            }
            let (first, last) = (frags[0], frags[frags.len() - 1]);
            if span.start != first.start || span.end != last.end {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    format!("answer {a}: the envelope does not match its fragments"),
                ));
            }
            k = hi;
        }
        Ok(())
    }

    /// The layer this extraction's columns make, checked the way
    /// `Answers` checks itself.
    pub fn answers(&self) -> Result<Answers, Failure> {
        self.layout()?;
        let mut spans = turbo_analysis::Spans::new(ViewId(self.view));
        if self.frag_at.is_empty() {
            for &s in &self.spans {
                spans.push(s);
            }
        } else {
            let mut k = 0;
            for &hi in &self.frag_at {
                spans.push_fragments(&self.fragments[k..hi as usize]);
                k = hi as usize;
            }
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
        let (span_start, span_end, frag_start, frag_end) = if self.text.is_empty() && self.offsets == OffsetMode::Utf8 {
            (
                self.spans.iter().map(|s| s.start).collect(),
                self.spans.iter().map(|s| s.end).collect(),
                self.fragments.iter().map(|s| s.start).collect(),
                self.fragments.iter().map(|s| s.end).collect(),
            )
        } else {
            let offsetter = Offsetter::new(&self.text, self.offsets);
            let (ss, se) = offsetter.spans_to_wire(&self.text, &self.spans);
            let (fs, fe) = offsetter.spans_to_wire(&self.text, &self.fragments);
            (ss, se, fs, fe)
        };
        ExtractionResult {
            model: Some(model_to_proto(&self.model)),
            view: self.view as u32,
            span_start,
            span_end,
            score: self.scores.clone(),
            probability: self.probabilities.clone(),
            null_score: self.null_score,
            null_probability: self.null_probability,
            frag_start,
            frag_end,
            frag_at: self.frag_at.clone(),
            text: self.text.clone(),
            offsets: self.offsets.to_proto(),
        }
    }

    fn from_msg(msg: &ExtractionResult) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        if msg.span_start.len() as u64 > MAX_ANSWERS {
            return Err(over("max_answers", "answers", msg.span_start.len() as u64));
        }
        // The fragment columns refuse past the listed limit before
        // anything is allocated for them.
        if msg.frag_start.len() as u64 > MAX_ANSWER_FRAGMENTS {
            return Err(over("max_answer_fragments", "answer fragments", msg.frag_start.len() as u64));
        }
        if msg.span_start.len() != msg.span_end.len() {
            return Err(mismatched("span_start", msg.span_start.len(), msg.span_end.len()));
        }
        if msg.frag_start.len() != msg.frag_end.len() {
            return Err(mismatched("frag_start", msg.frag_start.len(), msg.frag_end.len()));
        }
        if msg.view > u16::MAX as u32 {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("view {} is outside 0 to 65535", msg.view)));
        }
        let (spans, fragments) = if msg.text.is_empty() && offsets == OffsetMode::Utf8 {
            (
                msg.span_start.iter().zip(&msg.span_end).map(|(&s, &e)| Span { start: s, end: e }).collect(),
                msg.frag_start.iter().zip(&msg.frag_end).map(|(&s, &e)| Span { start: s, end: e }).collect(),
            )
        } else {
            let offsetter = Offsetter::new(&msg.text, offsets);
            let spans = offsetter.spans_from_wire(&msg.text, "answer span", &msg.span_start, &msg.span_end)?;
            let fragments = offsetter.spans_from_wire(&msg.text, "answer fragment", &msg.frag_start, &msg.frag_end)?;
            (spans, fragments)
        };
        let out = Extraction {
            model: msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
                id: String::new(),
                revision: String::new(),
                sha256: String::new(),
            }),
            view: msg.view as u16,
            text: msg.text.clone(),
            offsets,
            spans,
            fragments,
            frag_at: msg.frag_at.clone(),
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

    /// One answer in two separated pieces: [0, 2) and [8, 10), envelope
    /// [0, 10). The gap is not answer text.
    fn discontinuous() -> Answers {
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        spans.push_fragments(&[Span::new(0, 2), Span::new(8, 10)]);
        let out =
            Answers { spans, scores: vec![6.0], probabilities: vec![0.8], null_score: -1.0, null_probability: 0.2 };
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
        assert!(back.fragments.is_empty() && back.frag_at.is_empty());
        // The rebuilt layer holds what the original held.
        let ra = back.answers().unwrap();
        assert_eq!(ra.spans.columns(), a.spans.columns());
        assert_eq!((ra.scores, ra.probabilities), (a.scores.clone(), a.probabilities.clone()));
        assert_eq!(ra.null_probability, a.null_probability);
    }

    #[test]
    fn discontinuous_answers_keep_their_fragments() {
        let a = discontinuous();
        assert!(a.spans.is_discontinuous(turbo_analysis::Ix::new(0)));
        let x = Extraction::new(&a, model());
        assert_eq!(x.spans, [Span::new(0, 10)]);
        assert_eq!(x.fragments, [Span::new(0, 2), Span::new(8, 10)]);
        assert_eq!(x.frag_at, [2]);
        // Through the conversions, and back to the same layer.
        let back = Extraction::from_msg(&x.to_msg()).unwrap();
        assert_eq!(back.fragments, x.fragments);
        assert_eq!(back.frag_at, x.frag_at);
        let ra = back.answers().unwrap();
        assert_eq!(ra.spans.fragment_columns(), a.spans.fragment_columns());
        assert_eq!(
            ra.spans.fragments(turbo_analysis::Ix::new(0)).collect::<Vec<_>>(),
            [Span::new(0, 2), Span::new(8, 10)]
        );
        // A continuous answer beside a discontinuous one keeps one
        // fragment entry each.
        let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
        spans.push(Span::new(0, 4));
        spans.push_fragments(&[Span::new(10, 12), Span::new(18, 20)]);
        let mixed = Answers {
            spans,
            scores: vec![5.0, 4.0],
            probabilities: vec![0.6, 0.3],
            null_score: -1.0,
            null_probability: 0.1,
        };
        mixed.check().unwrap();
        let back = Extraction::from_msg(&Extraction::new(&mixed, model()).to_msg()).unwrap();
        assert_eq!(back.fragments, [Span::new(0, 4), Span::new(10, 12), Span::new(18, 20)]);
        assert_eq!(back.frag_at, [1, 3]);
        back.answers().unwrap();
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
        assert!(back.spans.is_empty() && back.fragments.is_empty() && back.null_probability == 1.0);
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
        // Fragment corruptions refuse by name.
        let mut msg = Extraction::new(&discontinuous(), model()).to_msg();
        msg.frag_at[0] = 3;
        let e = Extraction::from_msg(&msg).unwrap_err();
        assert!(e.message.contains("frag_at"), "{e:?}");
        let mut msg = Extraction::new(&discontinuous(), model()).to_msg();
        msg.frag_end[0] = 0;
        let e = Extraction::from_msg(&msg).unwrap_err();
        assert!(e.message.contains("fragment"), "{e:?}");
        let mut msg = Extraction::new(&discontinuous(), model()).to_msg();
        msg.span_end[0] = 4;
        let e = Extraction::from_msg(&msg).unwrap_err();
        assert!(e.message.contains("envelope"), "{e:?}");
        let e = Extraction::from_msg(&ExtractionResult::default()).unwrap_err();
        assert!(e.message.contains("extraction answers"), "{e:?}");
    }

    #[test]
    fn the_fragments_refuse_before_allocating() {
        // One answer claiming a fragment column past the listed limit.
        let msg = ExtractionResult {
            model: Some(model_to_proto(&model())),
            view: 0,
            span_start: vec![0],
            span_end: vec![8],
            score: vec![1.0],
            probability: vec![1.0],
            null_score: 0.0,
            null_probability: 0.0,
            frag_start: vec![0; MAX_ANSWER_FRAGMENTS as usize + 1],
            frag_end: vec![1; MAX_ANSWER_FRAGMENTS as usize + 1],
            frag_at: vec![1],
            ..Default::default()
        };
        let e = Extraction::from_msg(&msg).unwrap_err();
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_answer_fragments"), "{e:?}");
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
