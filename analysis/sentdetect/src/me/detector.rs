//! The maxent sentence detector: `SentenceDetectorME.sentPosDetect`.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Confidence, Sentences};
use turbo_unicode::{WhitespaceMode, java};

use crate::context::{Lookup, SdContext};
use crate::eos::Position;
use crate::features::Features;
use crate::me::model::SentenceModel;
use crate::scan::{decode, find_whitespace, skip_whitespace, trim};
use crate::{check_len, mode_hash};

/// What became of a candidate the model was asked about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The model's best outcome is a split, and it was taken.
    Split,
    /// The model's best outcome is not a split.
    NoSplit,
    /// The model said split, and the abbreviation dictionary refused it.
    Vetoed,
}

/// One candidate the model was asked about, as the detector saw it.
#[derive(Copy, Clone, Debug)]
pub struct Candidate<'a> {
    /// The end character's position.
    pub at: Position,
    /// The context's features.
    pub features: &'a Features,
    /// The probability of each outcome, in the model's outcome order.
    pub probs: &'a [f64],
    pub decision: Decision,
}

/// Buffers one detection works in, kept between calls so that detecting
/// many texts allocates only while they grow.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    enders: Vec<Position>,
    features: Features,
    probs: Vec<f64>,
    lookup: Lookup,
    window: Vec<u16>,
    /// The accepted sentence starts, as byte offsets, and their
    /// probabilities.
    starts: Vec<u32>,
    start_probs: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// Splits a view into sentences with a maxent model, writing `Sentences`
/// and a probability per sentence.
#[derive(Clone, Debug)]
pub struct SentenceDetector {
    model: Arc<SentenceModel>,
    /// The whitespace definition of the context features. It must be the
    /// one the model was trained with; OpenNLP's models are trained in the
    /// mode of their OpenNLP version, `Unicode` for 3.x.
    pub whitespace: WhitespaceMode,
}

impl SentenceDetector {
    pub fn new(model: impl Into<Arc<SentenceModel>>, whitespace: WhitespaceMode) -> SentenceDetector {
        SentenceDetector { model: model.into(), whitespace }
    }

    /// The model it runs.
    pub fn sentence_model(&self) -> &SentenceModel {
        &self.model
    }

    /// Appends to `out` every sentence of `text` with its probability, in
    /// text order, as `sentPosDetect` and `probs` give them.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn detect(&self, text: &str, out: &mut Vec<(Span, f64)>) {
        self.detect_with(text, &mut Scratch::new(), out);
    }

    /// [`detect`](Self::detect) with buffers kept by the caller.
    pub fn detect_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<(Span, f64)>) {
        self.detect_traced(text, scratch, out, |_| {});
    }

    /// [`detect_with`](Self::detect_with), calling `trace` with every
    /// candidate the model is asked about: its features, the model's
    /// probabilities and the decision.
    pub fn detect_traced(
        &self,
        text: &str,
        scratch: &mut Scratch,
        out: &mut Vec<(Span, f64)>,
        mut trace: impl FnMut(Candidate<'_>),
    ) {
        let model = &*self.model;
        let maxent = model.maxent();
        let ctx = SdContext::new(model.eos(), model.abbreviations(), self.whitespace);
        let Scratch { enders, features, probs, lookup, window, starts, start_probs } = scratch;
        let len16 = model.eos().positions(text, enders);
        probs.resize(maxent.num_outcomes(), 0.0);
        starts.clear();
        start_probs.clear();
        let bytes = text.as_bytes();

        // The start of the current segment, just after the last split.
        let mut index = Position::default();
        for i in 0..enders.len() {
            let cint = enders[i];
            let after = cint.byte as usize + decode(bytes, cint.byte as usize).1;
            // A run of end characters with no whitespace inside (z.B.) is
            // one candidate, at its last character, unless the character
            // after this one is upper case. The sentence-start whitespace
            // is always the Unicode set, as in the Java.
            if let Some(next) = enders.get(i + 1) {
                let through = next.byte as usize + decode(bytes, next.byte as usize).1;
                if find_whitespace(text, after, through, WhitespaceMode::Unicode) == through {
                    let upper = after < bytes.len() && {
                        let cp = decode(bytes, after).0;
                        cp < 0x1_0000 && java::is_upper_case(cp)
                    };
                    if !upper {
                        continue;
                    }
                }
            }
            if starts.last().is_some_and(|&s| cint.byte < s) {
                continue;
            }

            ctx.context(text, len16, cint, features, lookup);
            maxent.eval_features(features.iter(), probs);
            let best = maxent.best_outcome(probs);
            let decision = if best != model.split_outcome() {
                Decision::NoSplit
            } else if model.abbreviation_index().is_some_and(|a| !a.allows_break(text, len16, index, cint, window)) {
                Decision::Vetoed
            } else {
                Decision::Split
            };
            trace(Candidate { at: cint, features, probs, decision });
            if decision != Decision::Split {
                continue;
            }
            if index.unit != cint.unit {
                let from = if model.use_token_end() {
                    find_whitespace(text, after, text.len(), WhitespaceMode::Unicode)
                } else {
                    after
                };
                starts.push(skip_whitespace(text, from, text.len(), WhitespaceMode::Unicode) as u32);
                start_probs.push(probs[best]);
            }
            index = Position { byte: after as u32, unit: cint.unit + 1 };
        }

        // mapPositionsToSpans: each sentence runs from the previous start
        // to the next, trimmed of White_Space; one that trims to nothing is
        // dropped with its probability. The text after the last start is a
        // last sentence with probability 1.
        let mut push = |start: usize, end: usize, p: f64| {
            if let Some((s, e)) = trim(text, start, end, WhitespaceMode::Unicode) {
                out.push((Span::new(s as u32, e as u32), p));
            }
        };
        let mut sentence_start = 0;
        for (&s, &p) in starts.iter().zip(start_probs.iter()) {
            push(sentence_start, s as usize, p);
            sentence_start = s as usize;
        }
        if starts.last().is_none_or(|&s| s as usize != text.len()) {
            push(sentence_start, text.len(), 1.0);
        }
    }
}

impl Annotator for SentenceDetector {
    const ID: &'static str = "turbo-sentdetect:maxent";
    type Reads = (View,);
    type Writes = (Sentences, Confidence<Sentences>);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences, Confidence<Sentences>), Error> {
        let text = view.text();
        check_len(Self::ID, text)?;
        let mut found = Vec::new();
        self.detect_with(text, &mut Scratch::new(), &mut found);
        let mut spans = Spans::with_capacity(view.id(), found.len());
        // The confidence layer stores f32; the probabilities are computed
        // in f64, as in Java, and narrowed only here.
        let confidence = found.iter().map(|&(s, p)| {
            spans.push(s);
            p as f32
        });
        let confidence = Confidence::new(confidence.collect());
        Ok((Sentences { spans }, confidence))
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        mode_hash(self.whitespace)
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::Document;
    use turbo_maxent::ModelFile;

    use super::*;

    fn detector(name: &str, mode: WhitespaceMode) -> SentenceDetector {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models").join(name);
        SentenceDetector::new(SentenceModel::open(ModelFile::read(path).unwrap()).unwrap(), mode)
    }

    fn detect(d: &SentenceDetector, text: &str) -> Vec<(u32, u32, f64)> {
        let mut out = Vec::new();
        d.detect(text, &mut out);
        out.iter().map(|&(s, p)| (s.start, s.end, p)).collect()
    }

    #[test]
    fn empty_and_blank_texts_have_no_sentences() {
        let d = detector("en-abbrev.bin", WhitespaceMode::Unicode);
        assert_eq!(detect(&d, ""), []);
        assert_eq!(detect(&d, " \u{85}\u{2028}\t"), []);
        // Text with no end character is one sentence of probability 1.
        assert_eq!(detect(&d, "  no end here "), [(2, 13, 1.0)]);
    }

    #[test]
    fn a_split_at_the_segment_start_adds_no_sentence() {
        // From the golden file lone-period: the split at index 0 is taken,
        // but `index != cint` is false, so no start is added and the text
        // is one sentence of probability 1.
        let d = detector("en-abbrev.bin", WhitespaceMode::Unicode);
        let mut decisions = Vec::new();
        let mut out = Vec::new();
        d.detect_traced(".", &mut Scratch::new(), &mut out, |c| decisions.push((c.at, c.decision)));
        assert_eq!(decisions, [(Position { byte: 0, unit: 0 }, Decision::Split)]);
        assert_eq!(out, [(Span::new(0, 1), 1.0)]);
    }

    #[test]
    fn scratch_reuse_gives_the_same_result() {
        let d = detector("en-plain.bin", WhitespaceMode::Unicode);
        let texts = ["Mr. Smith left. Then he came back.", "", "One. Two! Three?"];
        let mut scratch = Scratch::new();
        for text in texts.iter().chain(texts.iter()) {
            let (mut fresh, mut reused) = (Vec::new(), Vec::new());
            d.detect(text, &mut fresh);
            d.detect_with(text, &mut scratch, &mut reused);
            assert_eq!(fresh, reused, "{text:?}");
        }
    }

    #[test]
    fn annotator_writes_both_layers() {
        let d = detector("en-abbrev.bin", WhitespaceMode::Unicode);
        let doc = Document::new("Mr. Smith met Dr. Brown at 5 p.m. on Main St. today. He left early.");
        let view = doc.get::<View>().unwrap();
        let (sentences, confidence) = d.annotate((view,)).unwrap();
        assert_eq!(sentences.spans.view(), view.id());
        assert_eq!(sentences.spans.columns(), (&[0, 53][..], &[52, 67][..]));
        // The golden probabilities, narrowed to f32.
        assert_eq!(confidence.value, [0.9955300941022667f64 as f32, 0.9466056890177881f64 as f32]);
        assert_ne!(d.config_hash(), detector("en-abbrev.bin", WhitespaceMode::Legacy).config_hash());
        assert_eq!(d.model().unwrap().id, "opennlp:SentenceDetectorME:en");
    }
}
