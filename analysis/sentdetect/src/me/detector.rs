//! The maxent sentence detector: `SentenceDetectorME.sentPosDetect`.

use std::cell::RefCell;
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_layers::{Confidence, Sentences};
use turbo_maxent::best_outcome;
use turbo_unicode::java::code_point as java;

use crate::context::{Limits, Lookup, SdContext, Walks};
use crate::eos::Position;
use crate::features::Features;
use crate::inventory::{AbbreviationInventories, AbbreviationInventory, Verdict};
use crate::me::model::SentenceModel;
use crate::scan::{decode, find_whitespace, skip_whitespace, trim};
use crate::{WHITESPACE_HASH, check_len, fold_hash};

/// What became of a candidate the model was asked about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    /// The model's best outcome is a split, and it was taken.
    Split,
    /// The model's best outcome is not a split.
    NoSplit,
    /// The model said split, and the abbreviation dictionary or the
    /// abbreviation inventory refused it.
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
    /// Consulted when the model says split; the bundled inventory of the
    /// model's language by default.
    inventory: AbbreviationInventory,
}

impl SentenceDetector {
    /// The detector of a model with the bundled abbreviation inventory of
    /// the model's language ([`AbbreviationInventories::for_language`]):
    /// every class for English, none for a language without one. Use
    /// [`with_jvm_parity`](Self::with_jvm_parity) to split exactly as
    /// `SentenceDetectorME` does.
    pub fn new(model: impl Into<Arc<SentenceModel>>) -> SentenceDetector {
        let model = model.into();
        let inventory = AbbreviationInventories::for_language(model.language());
        SentenceDetector { model, inventory }
    }

    /// The same detector with no abbreviation inventory: its spans,
    /// probabilities and configuration hash are those of OpenNLP's
    /// `SentenceDetectorME` with the same model, bit for bit. The JVM
    /// golden comparisons run in this mode.
    pub fn with_jvm_parity(self) -> SentenceDetector {
        self.with_abbreviations(AbbreviationInventory::EMPTY)
    }

    /// Whether it splits as `SentenceDetectorME` does: no inventory.
    pub fn is_jvm_parity(&self) -> bool {
        self.inventory.is_unset()
    }

    /// The same detector with an abbreviation inventory. Where the model
    /// says split, a [`Verdict::Suppress`] refuses the split, a
    /// [`Verdict::Allow`] takes it whatever the model's dictionary says,
    /// and [`Verdict::Neutral`] leaves the dictionary to decide. The
    /// features are the model's own, so the probabilities do not change.
    pub fn with_abbreviations(self, inventory: AbbreviationInventory) -> SentenceDetector {
        SentenceDetector { inventory, ..self }
    }

    /// The abbreviation inventory it consults.
    pub fn abbreviations(&self) -> &AbbreviationInventory {
        &self.inventory
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
        let ctx = SdContext::new(model.eos(), model.abbreviations(), limits);
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
                if find_whitespace(text, after, through) == through {
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
            } else {
                match self.inventory.judge(text, cint.byte as usize) {
                    Verdict::Suppress => Decision::Vetoed,
                    Verdict::Allow => Decision::Split,
                    Verdict::Neutral
                        if model
                            .abbreviation_index()
                            .is_some_and(|a| !a.allows_break(scanned, index, cint, window)) =>
                    {
                        Decision::Vetoed
                    }
                    Verdict::Neutral => Decision::Split,
                }
            };
            trace(Candidate { at: cint, features, probs, decision });
            if decision != Decision::Split {
                continue;
            }
            if index.unit != cint.unit {
                let from = if model.use_token_end() {
                    if !(ws_from <= after && after <= ws_at) {
                        ws_from = after;
                        ws_at = find_whitespace(text, after, text.len());
                    }
                    ws_at
                } else {
                    after
                };
                starts.push(skip_whitespace(text, from, text.len()) as u32);
                start_probs.push(probs[best]);
            }
            index = Position { byte: after as u32, unit: cint.unit + 1 };
        }

        // mapPositionsToSpans: each sentence runs from the previous start
        // to the next, trimmed of White_Space; one that trims to nothing is
        // dropped with its probability. The text after the last start is a
        // last sentence with probability 1.
        let mut push = |start: usize, end: usize, p: f64| {
            if let Some((s, e)) = trim(text, start, end) {
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

    /// The whitespace definition, and the abbreviation inventory when
    /// there is one; in JVM parity mode the hash is what it was before
    /// inventories.
    fn config_hash(&self) -> u64 {
        let h = WHITESPACE_HASH;
        if self.inventory.is_unset() { h } else { fold_hash(h, self.inventory.config_hash()) }
    }
}
