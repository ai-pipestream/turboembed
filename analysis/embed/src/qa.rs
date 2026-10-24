//! Question answering over a view with explicit source-span projection.
use std::{path::Path, sync::Mutex};
use turbo::qa::{Answer, DecodeOptions, ExtractiveQa};
use turbo_analysis::{
    Anchor, Annotator, Document, Error, Ix, Layer, ModelRef, Overlap, Refs, Span, Spans, TextSpans, View, ViewId,
};

pub struct Question(pub String);
impl Layer for Question {
    const ID: &'static str = "turbo:question";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        1
    }
}

/// Ranked answers in the input view. An empty layer means calibrated abstention.
pub struct Answers {
    pub spans: Spans<Self>,
    pub scores: Vec<f64>,
    pub probabilities: Vec<f64>,
    pub null_score: f64,
    pub null_probability: f64,
}
impl Layer for Answers {
    const ID: &'static str = "turbo:answers";
    const OVERLAP: Overlap = Overlap::Any;
    fn len(&self) -> usize {
        self.spans.len()
    }
    fn text_spans(&self) -> Option<TextSpans<'_>> {
        Some(self.spans.text_spans())
    }
    fn check(&self) -> Result<(), String> {
        self.spans.check(Self::OVERLAP)?;
        if self.spans.columns().0.iter().zip(self.spans.columns().1).any(|(a, b)| a >= b) {
            return Err("answer span must be nonempty".into());
        }
        if self.scores.len() != self.len()
            || self.probabilities.len() != self.len()
            || !self.null_score.is_finite()
            || !self.null_probability.is_finite()
            || !(0.0..=1.0).contains(&self.null_probability)
            || self.scores.iter().any(|v| !v.is_finite())
            || self.probabilities.iter().any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || (self.probabilities.iter().sum::<f64>() + self.null_probability - 1.0).abs() > 1e-6
        {
            return Err("answers require matching columns, finite scores and normalized probabilities".into());
        }
        Ok(())
    }
}
impl Answers {
    pub fn from_decoded(view: ViewId, decoded: Answer) -> Result<Self, String> {
        let mut out = Self {
            spans: Spans::new(view),
            scores: Vec::new(),
            probabilities: Vec::new(),
            null_score: 0.0,
            null_probability: 1.0,
        };
        match decoded {
            Answer::None { null_score } => out.null_score = null_score,
            Answer::Spans { answers, null_score, null_probability } => {
                out.null_score = null_score;
                out.null_probability = null_probability;
                for a in answers {
                    let start = u32::try_from(a.bytes.start).map_err(|_| "answer start exceeds u32")?;
                    let end = u32::try_from(a.bytes.end).map_err(|_| "answer end exceeds u32")?;
                    if start >= end {
                        return Err("answer span must be nonempty".into());
                    }
                    out.spans.push(Span { start, end });
                    out.scores.push(a.score);
                    out.probabilities.push(a.probability);
                }
            }
        }
        out.check()?;
        Ok(out)
    }
    /// Project each answer through the view LCA. Offsets are revalidated by
    /// Document::map_span before projection. Scores retain their original rank.
    pub fn project(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        self.check().map_err(failure)?;
        let mut spans = Spans::new(to);
        for i in 0..self.len() {
            spans.push(doc.map_span(
                self.spans.view(),
                to,
                self.spans.get(Ix::new(u32::try_from(i).map_err(|_| failure("too many answer rows"))?)),
            )?);
        }
        let out = Self {
            spans,
            scores: self.scores.clone(),
            probabilities: self.probabilities.clone(),
            null_score: self.null_score,
            null_probability: self.null_probability,
        };
        out.check().map_err(failure)?;
        Ok(out)
    }
}
pub struct ExtractiveQaAnnotator {
    session: Mutex<ExtractiveQa>,
    model: ModelRef,
    options: DecodeOptions,
}
fn failure(detail: impl Into<String>) -> Error {
    Error::Annotator { annotator: "turbo:extractive-qa", detail: detail.into() }
}
impl ExtractiveQaAnnotator {
    pub fn open(path: &Path, threads: u32, options: DecodeOptions) -> Result<Self, Error> {
        if options.top_k == 0 || options.max_answer_tokens == 0 || !options.null_threshold.is_finite() {
            return Err(failure("QA decoder options must have positive counts and a finite threshold"));
        }
        let session = ExtractiveQa::open(path, threads).map_err(|e| failure(e.message))?;
        let model = ModelRef {
            id: session.model_id.clone(),
            revision: session.revision.clone(),
            sha256: session.manifest_sha256.clone(),
        };
        Ok(Self { session: Mutex::new(session), model, options })
    }
}
impl Annotator for ExtractiveQaAnnotator {
    const ID: &'static str = "turbo:extractive-qa";
    const VERSION: &'static str = "1";
    type Reads = (View, Question);
    type Writes = (Answers,);
    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }
    fn config_hash(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        for b in (self.options.top_k as u64)
            .to_le_bytes()
            .into_iter()
            .chain((self.options.max_answer_tokens as u64).to_le_bytes())
            .chain(self.options.null_threshold.to_bits().to_le_bytes())
            .chain([u8::from(self.options.non_overlapping)])
        {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
        h
    }
    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let mut session = match self.session.lock() {
            Ok(s) => s,
            Err(_) => return inputs.iter().map(|_| Err(failure("QA session lock poisoned"))).collect(),
        };
        let pairs: Vec<_> = inputs.iter().map(|(view, q)| (q.0.as_str(), view.text())).collect();
        let logits = match session.logits_many(&pairs) {
            Ok(rows) => rows.into_iter().map(Ok).collect::<Vec<_>>(),
            // Tokenization can reject one pair. Isolate it without losing valid
            // neighbors; successful batches never take this recovery path.
            Err(_) => pairs.iter().map(|&(q, c)| session.logits(q, c)).collect(),
        };
        logits
            .into_iter()
            .zip(inputs)
            .map(|(rows, (view, _))| {
                let answer = turbo::qa::decode(view.text(), &rows.map_err(|e| failure(e.message))?, self.options)
                    .map_err(|e| failure(e.message))?;
                Ok((Answers::from_decoded(view.id(), answer).map_err(failure)?,))
            })
            .collect()
    }
}
