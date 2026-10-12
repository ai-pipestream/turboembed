//! The maxent sentence detector: `SentenceDetectorME.sentPosDetect`.

use std::cell::RefCell;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Confidence, Sentences};
use turbo_maxent::best_outcome;
use turbo_unicode::{WhitespaceMode, java::code_point as java};

use crate::context::{Limits, Lookup, SdContext, Walks};
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
        self.run(text, scratch, out, self.model.limits(), |_| {});
    }

    /// [`detect_with`](Self::detect_with), calling `trace` with every
    /// candidate the model is asked about: its features, the model's
    /// probabilities and the decision.
    ///
    /// The features are all those the Java builds. `detect_with` leaves
    /// out a word feature longer than every predicate of the model, which
    /// cannot change the probabilities; this does not, so on a long text
    /// without whitespace, where the words around each candidate reach to
    /// the ends of the text, it takes time quadratic in the text length.
    pub fn detect_traced(
        &self,
        text: &str,
        scratch: &mut Scratch,
        out: &mut Vec<(Span, f64)>,
        trace: impl FnMut(Candidate<'_>),
    ) {
        self.run(text, scratch, out, Limits::NONE, trace);
    }

    fn run(
        &self,
        text: &str,
        scratch: &mut Scratch,
        out: &mut Vec<(Span, f64)>,
        limits: Limits,
        mut trace: impl FnMut(Candidate<'_>),
    ) {
        let model = &*self.model;
        let maxent = model.maxent();
        let ctx = SdContext::new(model.eos(), model.abbreviations(), self.whitespace, limits);
        let Scratch { enders, features, probs, lookup, window, starts, start_probs } = scratch;
        let scanned = model.eos().positions(text, enders);
        let mut walks = Walks::new();
        probs.resize(maxent.num_outcomes(), 0.0);
        starts.clear();
        start_probs.clear();
        let bytes = text.as_bytes();
        // The first whitespace at or after `ws_from` is at `ws_at` (the text
        // length when there is none), kept so that a text without
        // whitespace is not searched to its end at every split.
        let (mut ws_from, mut ws_at) = (usize::MAX, 0);

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
            // A candidate before the last sentence start is not asked
            // about: with `useTokenEnd`, a split puts the rest of its token
            // in the sentence it ends ("hard.Bekannt." after a split at
            // "hard."), and the end characters there are passed over.
            if starts.last().is_some_and(|&s| cint.byte < s) {
                continue;
            }

            ctx.context(scanned, cint, &mut walks, features, lookup);
            maxent.eval_features(features.iter(), probs);
            let best = best_outcome(probs);
            let decision = if best != model.split_outcome() {
                Decision::NoSplit
            } else if model.abbreviation_index().is_some_and(|a| !a.allows_break(scanned, index, cint, window)) {
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
                    if !(ws_from <= after && after <= ws_at) {
                        ws_from = after;
                        ws_at = find_whitespace(text, after, text.len(), WhitespaceMode::Unicode);
                    }
                    ws_at
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

thread_local! {
    /// The buffers of the annotator, one set per thread, so that a
    /// pipeline running many documents allocates only while they grow.
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new());
}

impl Annotator for SentenceDetector {
    const ID: &'static str = "turbo-sentdetect:maxent";
    type Reads = (View,);
    type Writes = (Sentences, Confidence<Sentences>);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences, Confidence<Sentences>), Error> {
        let text = view.text();
        check_len(Self::ID, text)?;
        let mut found = Vec::new();
        SCRATCH.with_borrow_mut(|scratch| self.detect_with(text, scratch, &mut found));
        let mut spans = Spans::with_capacity(view.id(), found.len());
        let confidence = found.iter().map(|&(s, p)| {
            spans.push(s);
            p
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
    fn leaving_out_long_words_changes_nothing() {
        // detect_with leaves out word features longer than every predicate
        // and skips lookups of words longer than every dictionary entry;
        // detect_traced builds every feature as the Java does. On texts
        // without whitespace every word reaches to an end of the text.
        let models = [
            "tests/fixtures/models/en-abbrev.bin",
            "tests/fixtures/models/en-plain.bin",
            "tests/models/opennlp-en-ud-ewt-sentence-1.3-2.5.4.bin",
            "tests/models/opennlp-pt-ud-gsd-sentence-1.3-2.5.4.bin",
        ];
        let texts = [
            "A.".repeat(1500),
            "x.Y".repeat(1000),
            "Mr.Smith.went.Home!Then?Dr.Who.et.al.".repeat(80),
            "He thought hard.Bekannt.Sie kam.".repeat(90),
            format!("Start. {} End. Mr. Brown left.", "\u{E9}t\u{E9}.\u{1F600}".repeat(400)),
        ];
        let mut splits = 0;
        for model in models {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(model);
            let model = Arc::new(SentenceModel::open(ModelFile::read(path).unwrap()).unwrap());
            assert!(model.limits().feature < 1000, "{}", model.limits().feature);
            for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
                let d = SentenceDetector::new(model.clone(), mode);
                for text in &texts {
                    let (mut fast, mut full) = (Vec::new(), Vec::new());
                    d.detect(text, &mut fast);
                    d.detect_traced(text, &mut Scratch::new(), &mut full, |c| {
                        splits += usize::from(c.decision == Decision::Split);
                    });
                    let bits = |v: &[(Span, f64)]| v.iter().map(|&(s, p)| (s, p.to_bits())).collect::<Vec<_>>();
                    assert_eq!(
                        bits(&fast),
                        bits(&full),
                        "{} {mode:?} on {:?}",
                        model.language(),
                        text.chars().take(30).collect::<String>()
                    );
                }
            }
        }
        assert!(splits > 1000, "only {splits} splits");
    }

    #[test]
    fn annotator_writes_both_layers() {
        let d = detector("en-abbrev.bin", WhitespaceMode::Unicode);
        let doc = Document::new("Mr. Smith met Dr. Brown at 5 p.m. on Main St. today. He left early.");
        let view = doc.get::<View>().unwrap();
        let (sentences, confidence) = d.annotate((view,)).unwrap();
        assert_eq!(sentences.spans.view(), view.id());
        assert_eq!(sentences.spans.columns(), (&[0, 53][..], &[52, 67][..]));
        // The golden probabilities.
        assert_eq!(confidence.value, [0.9955300941022667, 0.9466056890177881]);
        assert_ne!(d.config_hash(), detector("en-abbrev.bin", WhitespaceMode::Legacy).config_hash());
        assert_eq!(d.model().unwrap().id, "opennlp:SentenceDetectorME:en");
    }
}
