//! Recipes that train the sentence detector, tokenizer, part-of-speech
//! tagger and lemmatizer on the CC BY 4.0 treebanks of [`crate::treebank`]
//! (issue #415), and their gates against the OpenNLP UD 1.3 models.
//!
//! Each recipe trains on the treebank's UD `train` file and is gated on
//! its UD `test` file; the `dev` file is checked against its digest and
//! not read. Training is what OpenNLP 2.5.4's CoNLL-U trainer tools do,
//! with the parameters the published UD 1.3 models were trained with
//! ([`turbo_train_components::ud`]: GIS, 300 iterations, cutoff 5,
//! 4 threads, the factory defaults for the language, OpenNLP 2.5.4's
//! whitespace, 10 sentences per sentence detector sample, the universal
//! tags), so a trained model and its reference are trained with the same
//! settings. Training uses no randomness, so a recipe has no seed.
//!
//! OpenNLP trained the Catalan references on this treebank's `train` file.
//! The Catalan sentence detector and tokenizer recipes make OpenNLP's
//! events exactly (the same `Training-Eventhash`), and the models differ
//! only in the last bits of some parameters, so those two gates check
//! that the recipes reproduce OpenNLP's models rather than compare two
//! models. The Catalan tagger and lemmatizer recipes make OpenNLP's
//! events exactly when the empty nodes are kept (the event hash OpenNLP's
//! trainer gives), so the data their models and the references are
//! trained on differ only by the 4,999 empty nodes the recipes leave out
//! (below).
//! The other references were trained on another treebank of the language.
//!
//! The samples are those of OpenNLP's CoNLL-U streams, with one change:
//! the tagger's and the lemmatizer's samples leave out empty nodes (ids
//! like `6.1`), which `ConlluPOSSampleStream` and
//! `ConlluLemmaSampleStream` keep. An empty node is not a word of the
//! text (AnCora writes its form `_`, Porttinari `__NOFORM__`), and
//! scoring one would count a token no text has; a model trained with them
//! learns to tag `_`, which on a held-out set with them would favour a
//! model trained on the same treebank over one trained on another.
//!
//! A treebank has a recipe for each component it annotates. FTB and
//! MarkIT have no sentence detector and no tokenizer: their `text`
//! comments put a space between almost every two tokens, and their words
//! carry almost no `SpaceAfter=No` (none in FTB's files; in MarkIT's, 2
//! in `train`, on the `1` and `,` of "1,9", 7 in `dev` and none in
//! `test`), so they hold no tokenization or sentence ends of running
//! text.
//!
//! Each model is gated against the OpenNLP UD 1.3 model for its language
//! and component, both scored on the same `test` samples with OpenNLP's
//! evaluators: a model of the zoo's built-in index ([`Reference::Zoo`]),
//! or for Italian a model the recipe records ([`Reference::Recorded`]).
//! The gates are the MASC recipes' ([`crate::segmentation`],
//! [`crate::postag`]), and the records and cards are of the same versions:
//! a sentence detector or tokenizer is gated as a [`SegmentationRecipe`],
//! and a tagger as a [`PosRecipe`] whose tags, the trained model's and the
//! reference's, are both asked for in [`TAG_FORMAT`] and compared with the
//! held-out tag as written, with no mapping. A lemmatizer, given the
//! held-out tags, is scored on its lemmas as `LemmatizerEvaluator` counts
//! them, as a tagger on its tags ([`score_lemmas`]).
//!
//! Except for Catalan, the reference was trained on another treebank, so
//! the gate scores it against annotation conventions it was not trained
//! on, and that favours the candidate: for FTB, the Finnish TDT reference
//! tags 303 scored `VERB` tokens `AUX`. Only the Catalan gates compare
//! models trained on the same treebank.
//!
//! A reference trained on another treebank may also lack outcomes the
//! `test` file needs. Its tagger may not have every tag: FTB tags
//! determiners `DET` and particles `PART`, and the Finnish reference has
//! neither. Its lemmatizer may have no outcome that gives a token's
//! held-out lemma where the token's gold lemma class, the edit script from
//! the token to its lemma, gives it. A token with a tag the reference has
//! no outcome for, or such a lemma, is one the reference cannot get right
//! and the candidate can, so such tokens are left out of both models' scores
//! ([`crate::postag::ReferenceTags`], [`ReferenceClasses`]), and the card
//! lists how many there were, by tag; when they are more than
//! [`MAX_TOKENS_LEFT_OUT`] of the tokens, the recipe fails.
//!
//! The limit, 5 in 100, is one for every UD tagger and lemmatizer, and it
//! is a floor on how much of the `test` file the gate covers. It is not
//! the MASC tagger's 1 in 1,000 ([`crate::MAX_TOKENS_LEFT_OUT`]) because a
//! reference trained on another treebank can lack whole tags: FTB's
//! `test` file has 635 such tokens of 16,261 (3.9 in 100) for the Finnish
//! reference tagger, and 177 (1.1 in 100) for its lemmatizer. What leaving
//! tokens out can do to the gate follows from every left-out token being
//! one the reference gets wrong. With `n` tokens scored, `l` left out, the
//! candidate right on `x` of the left-out ones, and `d` the candidate's
//! lead on the scored tokens as a fraction, the lead over all the tokens
//! is `d` less `l / (n + l)` times `d - x / l`:
//!
//! - When the candidate trails (`d` below 0), leaving tokens out always
//!   lowers its lead on the point estimate. The same holds in each of the
//!   gate's resamples, which draw the same sentences either way, so when
//!   the resamples the interval's low end is read from have the candidate
//!   trailing, the low end is no higher with the tokens left out: leaving
//!   them out cannot turn a failing gate into a passing one.
//! - When the candidate leads, leaving tokens out raises its lead only
//!   when its accuracy on them, `x / l`, is below `d`, and then by at most
//!   `l / (n + l)` times `d`: under the limit, 5 in 100 of the lead.
//!
//! For FTB's tagger the candidate leads and gets 428 of the 635 left-out
//! tokens right, far more than its lead, so leaving them out lowers its
//! lead: +2.00 points with them left out, +4.55 with them counted.
//!
//! A sentence a recipe's reader skips (one with no word once the empty
//! nodes are left out) is counted, and the recipe fails when the training
//! or the held-out sentences skipped are more than
//! [`crate::MAX_SKIPPED_SENTENCES`] of those read. The binary checks each
//! recipe's data before it trains any ([`UdRecipe::check_data`]).
//!
//! OpenNLP's Italian UD models are trained on UD_Italian-VIT, which is
//! CC BY-NC-SA 3.0, so they are local-only (`docs/licences.md`,
//! "Non-commercial data") and the zoo's built-in index does not list them.
//! The Italian recipes record each one's file, size and SHA-256
//! ([`IT_UD_VIT_POS`], [`IT_UD_VIT_LEMMAS`]); a run reads it from its
//! references directory, checks it, and only scores it. It is never
//! shipped, bundled or trained on, and the trained model is trained on
//! MarkIT alone, so its class is permissive.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, PairedUnit, Scoring, TagScoring, Task};
use turbo_formats::FormatError;
use turbo_formats::conllu::{ConlluReader, ConlluSentenceReader, ConlluTokenReader, Sentence, Tagset, WordLine};
use turbo_formats::evaluate::LemmatizerEvaluator;
use turbo_formats::sample::{LemmaSample, PosSample};
use turbo_lemmatizer::{LemmatizerModel, MaxentLemmatizer, TokenTag, edit_script::decode_lemma};
use turbo_maxent::{JavaString, ModelFile};
use turbo_postag::PosTagFormat;
use turbo_train::TrainingConfiguration;
use turbo_train_components::ud::{self as published, Component};
use turbo_train_components::{
    LemmaEventStream, LemmatizerFactory, PosEventStream, PosTaggerFactory, SdEventStream, SentenceDetectorFactory,
    Stamp, TokSpanEventStream, TokenizerFactory, lemmatize, postag, sentdetect, tokenize,
};
use turbo_zoo::Sha256;

use crate::card::{HeldOutCard, Mismatch, TagCount, TokenCoverage};
use crate::error::{Error, LeftOut, Result};
use crate::postag::{HeldOutTags, PosRecipe, Scored, most_frequent};
use crate::segmentation::{self, HeldOut, SegmentationRecipe, TIMESTAMP_MS, WHITESPACE};
use crate::treebank::TreebankText;
use crate::treebank::{
    CATALAN_ANCORA, FINNISH_FTB, ITALIAN_MARKIT, PORTUGUESE_PORTTINARI, RELEASE, SPANISH_ANCORA, Treebank,
};
use crate::ud_card::UdModelCard;
use crate::{GateRun, Limit, RecordedReference, Reference, Trained, check_skipped};

/// Sentences per sentence detector sample.
pub const SENTENCES_PER_SAMPLE: usize = published::SENTENCES_PER_SAMPLE;
/// The tag column the tagger and the lemmatizer read: the universal tags.
pub const TAGSET: Tagset = published::TAGSET;
/// The tags both taggers are scored in: their outcomes as they are, the
/// universal tags with a multiword token's tags joined by `+`, which both
/// were trained on.
pub const TAG_FORMAT: PosTagFormat = PosTagFormat::Custom;

/// The most held-out tokens a UD tagger's or lemmatizer's gate may leave
/// out of both models' scores because the reference cannot give their tag
/// or the lemma their gold lemma class gives: 5 in 100, a floor on how much of
/// the `test` file the gate covers (the module documentation says why it
/// is not the MASC tagger's limit, and what leaving tokens out can do to
/// the gate). More and the recipe fails ([`Error::TagsOutsideReference`],
/// [`Error::ClassesOutsideReference`]).
pub const MAX_TOKENS_LEFT_OUT: Limit = Limit { count: 5, of: 100 };

/// OpenNLP's Italian UD 1.3 tagger, trained on UD_Italian-VIT (CC
/// BY-NC-SA 3.0), as published. Its SHA-512 is the one Apache publishes
/// beside it, and it is the file in Maven Central's
/// `org.apache.opennlp:opennlp-models-pos-it:1.3.0`.
pub const IT_UD_VIT_POS: RecordedReference = RecordedReference {
    id: "opennlp-it-ud-vit-pos-1.3-2.5.4",
    file: "opennlp-it-ud-vit-pos-1.3-2.5.4.bin",
    url: "https://archive.apache.org/dist/opennlp/models/ud-models-1.3/opennlp-it-ud-vit-pos-1.3-2.5.4.bin",
    size: 1_149_395,
    sha256: "d38e645dc81289c035f2f126f813092ea657ae704769bd65a26f6da89fdd18f1",
};

/// OpenNLP's Italian UD 1.3 lemmatizer, trained on UD_Italian-VIT, as
/// [`IT_UD_VIT_POS`] is (Maven Central's
/// `org.apache.opennlp:opennlp-models-lemmatizer-it:1.3.0`).
pub const IT_UD_VIT_LEMMAS: RecordedReference = RecordedReference {
    id: "opennlp-it-ud-vit-lemmas-1.3-2.5.4",
    file: "opennlp-it-ud-vit-lemmas-1.3-2.5.4.bin",
    url: "https://archive.apache.org/dist/opennlp/models/ud-models-1.3/opennlp-it-ud-vit-lemmas-1.3-2.5.4.bin",
    size: 704_996,
    sha256: "10b25bc4506af3a5f06e56b9e1058aecccb3021e5d08902a1fa1ee33606510c7",
};

/// One model trained on a treebank and its reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UdRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    pub treebank: Treebank,
    pub component: Component,
    pub reference: Reference,
}

const fn recipe(model_id: &'static str, treebank: Treebank, component: Component, reference: Reference) -> UdRecipe {
    UdRecipe { model_id, treebank, component, reference }
}

use Component::{Lemmas, Pos, Sentence as Sentences, Tokens};
use Reference::{Recorded, Zoo};

/// The recipes, by treebank, in the order the binary runs them.
pub const UD_RECIPES: [UdRecipe; 16] = [
    recipe(
        "turbo-es-ud-ancora-2.16-sentence-1",
        SPANISH_ANCORA,
        Sentences,
        Zoo("opennlp-es-ud-gsd-sentence-1.3-2.5.4"),
    ),
    recipe("turbo-es-ud-ancora-2.16-tokens-1", SPANISH_ANCORA, Tokens, Zoo("opennlp-es-ud-gsd-tokens-1.3-2.5.4")),
    recipe("turbo-es-ud-ancora-2.16-pos-1", SPANISH_ANCORA, Pos, Zoo("opennlp-es-ud-gsd-pos-1.3-2.5.4")),
    recipe("turbo-es-ud-ancora-2.16-lemmas-1", SPANISH_ANCORA, Lemmas, Zoo("opennlp-es-ud-gsd-lemmas-1.3-2.5.4")),
    recipe(
        "turbo-ca-ud-ancora-2.16-sentence-1",
        CATALAN_ANCORA,
        Sentences,
        Zoo("opennlp-ca-ud-ancora-sentence-1.3-2.5.4"),
    ),
    recipe("turbo-ca-ud-ancora-2.16-tokens-1", CATALAN_ANCORA, Tokens, Zoo("opennlp-ca-ud-ancora-tokens-1.3-2.5.4")),
    recipe("turbo-ca-ud-ancora-2.16-pos-1", CATALAN_ANCORA, Pos, Zoo("opennlp-ca-ud-ancora-pos-1.3-2.5.4")),
    recipe("turbo-ca-ud-ancora-2.16-lemmas-1", CATALAN_ANCORA, Lemmas, Zoo("opennlp-ca-ud-ancora-lemmas-1.3-2.5.4")),
    recipe(
        "turbo-pt-ud-porttinari-2.16-sentence-1",
        PORTUGUESE_PORTTINARI,
        Sentences,
        Zoo("opennlp-pt-ud-gsd-sentence-1.3-2.5.4"),
    ),
    recipe(
        "turbo-pt-ud-porttinari-2.16-tokens-1",
        PORTUGUESE_PORTTINARI,
        Tokens,
        Zoo("opennlp-pt-ud-gsd-tokens-1.3-2.5.4"),
    ),
    recipe("turbo-pt-ud-porttinari-2.16-pos-1", PORTUGUESE_PORTTINARI, Pos, Zoo("opennlp-pt-ud-gsd-pos-1.3-2.5.4")),
    recipe(
        "turbo-pt-ud-porttinari-2.16-lemmas-1",
        PORTUGUESE_PORTTINARI,
        Lemmas,
        Zoo("opennlp-pt-ud-gsd-lemmas-1.3-2.5.4"),
    ),
    recipe("turbo-fi-ud-ftb-2.16-pos-1", FINNISH_FTB, Pos, Zoo("opennlp-fi-ud-tdt-pos-1.3-2.5.4")),
    recipe("turbo-fi-ud-ftb-2.16-lemmas-1", FINNISH_FTB, Lemmas, Zoo("opennlp-fi-ud-tdt-lemmas-1.3-2.5.4")),
    recipe("turbo-it-ud-markit-2.16-pos-1", ITALIAN_MARKIT, Pos, Recorded(IT_UD_VIT_POS)),
    recipe("turbo-it-ud-markit-2.16-lemmas-1", ITALIAN_MARKIT, Lemmas, Recorded(IT_UD_VIT_LEMMAS)),
];

/// The words of a sentence the tagger and the lemmatizer read: its word
/// lines without the empty nodes.
fn words<'s, 'a>(sentence: &'s Sentence<'a>) -> impl Iterator<Item = &'s WordLine<'a>> {
    sentence.word_lines.iter().filter(|l| !l.id.as_str().contains('.'))
}

/// Reads the part-of-speech samples of a CoNLL-U text: one per sentence,
/// its words' forms and universal tags, empty nodes left out. A sentence
/// with no word is skipped and counted.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn pos_samples(conllu: &str) -> std::result::Result<(Vec<PosSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        samples.push(PosSample::new(tokens, tags, sentence.source)?);
    }
    Ok((samples, skipped))
}

/// Reads the lemma samples of a CoNLL-U text as [`pos_samples`] reads the
/// part-of-speech samples, with each word's lemma.
///
/// # Errors
///
/// A sentence the CoNLL-U reader refuses.
pub fn lemma_samples(conllu: &str) -> std::result::Result<(Vec<LemmaSample<'_>>, usize), FormatError> {
    let mut samples = Vec::new();
    let mut skipped = 0;
    for sentence in ConlluReader::new(conllu)? {
        let sentence = sentence?;
        let tokens: Vec<_> = words(&sentence).map(|l| l.form.clone()).collect();
        if tokens.is_empty() {
            skipped += 1;
            continue;
        }
        let tags = words(&sentence).map(|l| l.tag(TAGSET).clone()).collect();
        let lemmas = words(&sentence).map(|l| l.lemma.clone()).collect();
        samples.push(LemmaSample::new(tokens, tags, lemmas, sentence.source)?);
    }
    Ok((samples, skipped))
}

/// The samples of one component read from one CoNLL-U file, borrowed
/// from its text.
#[derive(Debug, Clone)]
pub enum UdSamples<'c> {
    /// Sentence or token samples.
    Segments(HeldOut<'c>),
    Tags(Vec<PosSample<'c>>),
    Lemmas(Vec<LemmaSample<'c>>),
}

impl UdSamples<'_> {
    pub fn len(&self) -> usize {
        match self {
            UdSamples::Segments(s) => s.len(),
            UdSamples::Tags(t) => t.len(),
            UdSamples::Lemmas(l) => l.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The kind of the samples, as [`Error::ComponentMismatch`] names it.
    fn kind(&self) -> &'static str {
        match self {
            UdSamples::Segments(HeldOut::Sentences(_)) => "sentence",
            UdSamples::Segments(HeldOut::Tokens { .. }) => "token",
            UdSamples::Tags(_) => "part-of-speech",
            UdSamples::Lemmas(_) => "lemma",
        }
    }
}

/// The lemma classes a reference lemmatizer has outcomes for, which decide
/// the held-out tokens a lemmatizer's gate scores. A token's gold class is
/// the class training gives the token and its held-out lemma
/// ([`lemmatize::lemma_class`], `LemmatizerME.encodeLemmas`). A token is
/// left out of both models' scores when its gold class gives its lemma
/// back, so a lemmatizer trained on the treebank can give it, and no
/// outcome of the reference gives that lemma for the token, so the
/// reference cannot get it right, as a token whose tag a reference tagger
/// cannot give ([`crate::postag::ReferenceTags`]). Lacking the gold class
/// is not enough: `decodeShortestEditScript` deletes a character without
/// reading the letter the class names for it, so classes that differ only
/// in those letters give the same lemma, and a reference without a token's
/// gold class may give its lemma through another class.
///
/// A token whose gold class gives another string is scored, as
/// `LemmatizerEvaluator` counts it: training teaches the candidate the
/// gold classes, so the rule leaves out only tokens it can be taught.
/// `decodeLemmas` applies a class to the lowercased token, and every class
/// is made from a lowercased lemma, so no model gives a lemma with a
/// capital letter (MarkIT writes `[PUNCT]` as the lemma of most
/// punctuation); and it reads a class's index as one digit, so a gold
/// class with an index of ten or more gives another string
/// (`turbo_lemmatizer`'s documentation). Making classes from the lemma as
/// written and applying them to the token as written would let a model
/// give capitalised lemmas, an improvement over OpenNLP that the recipes
/// leave out so that they train and score as OpenNLP does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceClasses {
    classes: HashSet<JavaString>,
}

impl ReferenceClasses {
    /// The classes `outcomes`.
    pub fn new(outcomes: impl IntoIterator<Item = JavaString>) -> ReferenceClasses {
        ReferenceClasses { classes: outcomes.into_iter().collect() }
    }

    /// The outcomes of the lemmatizer in `bytes`.
    ///
    /// # Errors
    ///
    /// A model file that is not a lemmatizer model.
    pub fn of_model(bytes: &[u8]) -> Result<ReferenceClasses> {
        Ok(ReferenceClasses::new(open_lemmatizer(bytes)?.maxent().outcomes().cloned()))
    }

    /// Whether a held-out token `token` of the lemma `lemma` is scored:
    /// unless its gold class gives the lemma and no class of the reference
    /// does.
    pub fn scores(&self, token: &str, lemma: &str) -> bool {
        let gives = |class: &JavaString| decode_lemma(token, class).is_ok_and(|given| given == lemma);
        let class = lemmatize::lemma_class(token, lemma);
        self.classes.contains(&class) || !gives(&class) || self.classes.iter().any(gives)
    }

    /// The held-out tokens left out of the scores, by their tag as
    /// written, sorted by tag.
    pub fn left_out(&self, samples: &[LemmaSample<'_>]) -> Vec<TagCount> {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for s in samples {
            for ((token, tag), lemma) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()) {
                if !self.scores(token.as_str(), lemma.as_str()) {
                    *counts.entry(tag.as_str()).or_default() += 1;
                }
            }
        }
        counts.into_iter().map(|(tag, tokens)| TagCount { tag: tag.to_owned(), tokens }).collect()
    }
}

/// The tokens of lemma samples.
fn lemma_tokens(samples: &[LemmaSample<'_>]) -> u64 {
    samples.iter().map(|s| s.len() as u64).sum()
}

/// The SHA-256 of lemma samples as scored with a reference of the classes
/// `reference`: for each sample, its number of tokens as a little endian
/// `u64`, then for each token its text, its tag and its lemma, each as its
/// length in bytes as a little endian `u64` and the bytes, and a byte, 1
/// when the token is scored ([`ReferenceClasses::scores`]) and 0 when it
/// is left out. Two gates whose references leave out other tokens have
/// other digests.
pub fn lemma_samples_sha256(samples: &[LemmaSample<'_>], reference: &ReferenceClasses) -> String {
    let mut bytes = Vec::new();
    let put = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    for s in samples {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        for ((token, tag), lemma) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()) {
            put(&mut bytes, token.as_str());
            put(&mut bytes, tag.as_str());
            put(&mut bytes, lemma.as_str());
            bytes.push(u8::from(reference.scores(token.as_str(), lemma.as_str())));
        }
    }
    Sha256::of(&bytes).to_string()
}

/// The samples a recipe read from one file, and how many sentences it
/// left out.
#[derive(Debug, Clone)]
pub struct ReadSamples<'c> {
    pub samples: UdSamples<'c>,
    /// Sentences with no word once the empty nodes are left out (0 for the
    /// sentence detector and the tokenizer).
    pub skipped: usize,
}

impl UdRecipe {
    /// The recipe of [`UD_RECIPES`] that trains the model `id`.
    pub fn trains(id: &str) -> Option<UdRecipe> {
        UD_RECIPES.into_iter().find(|r| r.model_id == id)
    }

    /// The gate's task.
    pub fn task(&self) -> Task {
        match self.component {
            Component::Sentence => Task::SentenceDetector,
            Component::Tokens => Task::Tokenizer,
            Component::Pos => Task::PartOfSpeechTagger,
            Component::Lemmas => Task::Lemmatizer,
        }
    }

    /// The component the zoo lists the trained model under.
    pub fn zoo_component(&self) -> turbo_zoo::Component {
        match self.component {
            Component::Sentence => turbo_zoo::Component::SentenceDetector,
            Component::Tokens => turbo_zoo::Component::Tokenizer,
            Component::Pos => turbo_zoo::Component::Pos,
            Component::Lemmas => turbo_zoo::Component::Lemmatizer,
        }
    }

    /// For a sentence detector or a tokenizer, the MASC recipes' gate for
    /// it: a [`SegmentationRecipe`] of the model and its reference.
    pub fn segmentation(&self) -> Option<SegmentationRecipe> {
        let component = match self.component {
            Component::Sentence => segmentation::Component::Sentence,
            Component::Tokens => segmentation::Component::Tokens,
            Component::Pos | Component::Lemmas => return None,
        };
        Some(SegmentationRecipe { component, model_id: self.model_id, reference_id: self.reference.id() })
    }

    /// For a tagger, the MASC tagger's gate for it: a [`PosRecipe`] of the
    /// model and its reference, the reference asked for [`TAG_FORMAT`], no
    /// mapping, and [`MAX_TOKENS_LEFT_OUT`].
    pub fn tagger(&self) -> Option<PosRecipe> {
        (self.component == Component::Pos).then_some(PosRecipe {
            model_id: self.model_id,
            reference: self.reference,
            reference_format: TAG_FORMAT,
            reference_mappings: &[],
            max_left_out: MAX_TOKENS_LEFT_OUT,
        })
    }

    /// The name of the held-out set in the records.
    pub fn held_out_name(&self) -> String {
        format!("{} {RELEASE} test ({})", self.treebank.repository, self.treebank.test.name)
    }

    /// Reads the recipe's samples from a CoNLL-U text, as the trainer
    /// tool reads them (with the change the module documentation gives).
    ///
    /// # Errors
    ///
    /// A sentence the reader refuses ([`Error::Conllu`], naming `file`),
    /// or no sample ([`Error::EmptySplit`]).
    pub fn samples<'c>(&self, conllu: &'c str, file: &'static str) -> Result<ReadSamples<'c>> {
        let conllu_error =
            |error: FormatError| Error::Conllu { treebank: self.treebank.repository, file, error: Box::new(error) };
        let (samples, skipped) = match self.component {
            Component::Sentence => (
                UdSamples::Segments(HeldOut::Sentences(
                    ConlluSentenceReader::new(conllu, SENTENCES_PER_SAMPLE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                )),
                0,
            ),
            Component::Tokens => (
                UdSamples::Segments(HeldOut::Tokens {
                    samples: ConlluTokenReader::new(conllu, WHITESPACE)
                        .map_err(conllu_error)?
                        .collect::<std::result::Result<_, _>>()
                        .map_err(conllu_error)?,
                    skipped: 0,
                }),
                0,
            ),
            Component::Pos => {
                let (s, skipped) = pos_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Tags(s), skipped)
            }
            Component::Lemmas => {
                let (s, skipped) = lemma_samples(conllu).map_err(conllu_error)?;
                (UdSamples::Lemmas(s), skipped)
            }
        };
        if samples.is_empty() {
            return Err(Error::EmptySplit { what: "sample of the treebank file" });
        }
        Ok(ReadSamples { samples, skipped })
    }

    /// The held-out samples of the `test` file, checked as [`run_ud`]
    /// checks them: at most [`crate::MAX_SKIPPED_SENTENCES`] skipped.
    ///
    /// # Errors
    ///
    /// What [`UdRecipe::samples`] fails on, or [`Error::TooManyLeftOut`].
    pub fn held_out<'c>(&self, text: &'c TreebankText) -> Result<ReadSamples<'c>> {
        let held_out = self.samples(&text.test, self.treebank.test.name)?;
        check_skipped(LeftOut::HeldOutSentences, held_out.skipped, held_out.samples.len())?;
        Ok(held_out)
    }

    /// Reads the training and held-out samples of `text` without training
    /// or scoring, and checks what [`run_ud`] checks of them: each file has
    /// a sample, the reader skips at most [`crate::MAX_SKIPPED_SENTENCES`]
    /// of the sentences of either, and for a tagger or a lemmatizer, given
    /// its reference's bytes, that the reference can give the tag, or the
    /// lemma the gold lemma class gives, of all but [`MAX_TOKENS_LEFT_OUT`]
    /// of the held-out tokens ([`ReferenceClasses::scores`]). The binary runs it for every recipe before it
    /// trains any, as it runs [`crate::Recipe::check_data`] for the MASC
    /// recipes.
    ///
    /// # Errors
    ///
    /// What reading the samples fails on, [`Error::TooManyLeftOut`],
    /// [`Error::TagsOutsideReference`], [`Error::ClassesOutsideReference`],
    /// or a reference that is not a model of the component.
    pub fn check_data(&self, text: &TreebankText, reference: &[u8]) -> Result<()> {
        let train = self.samples(&text.train, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, train.skipped, train.samples.len())?;
        let held_out = self.held_out(text)?;
        match (self.tagger(), held_out.samples) {
            (Some(tagger), UdSamples::Tags(samples)) => {
                let tags = tagger.reference_tags(reference)?;
                tagger.check_coverage(self.reference.id(), &tags, &HeldOutTags::new(samples, held_out.skipped))?;
            }
            (None, UdSamples::Lemmas(samples)) => {
                self.check_classes(&ReferenceClasses::of_model(reference)?, &samples)?;
            }
            _ => {}
        }
        Ok(())
    }

    /// The held-out tokens whose gold lemma class gives their lemma and no
    /// class of `classes`, a reference's, does, by tag
    /// ([`ReferenceClasses::left_out`]).
    ///
    /// # Errors
    ///
    /// More of them than [`MAX_TOKENS_LEFT_OUT`] allows
    /// ([`Error::ClassesOutsideReference`]).
    pub fn check_classes(&self, classes: &ReferenceClasses, samples: &[LemmaSample<'_>]) -> Result<Vec<TagCount>> {
        let left_out = classes.left_out(samples);
        let tokens = lemma_tokens(samples);
        if MAX_TOKENS_LEFT_OUT.allows(left_out.iter().map(|t| t.tokens).sum(), tokens) {
            Ok(left_out)
        } else {
            Err(Error::ClassesOutsideReference {
                reference: self.reference.id().to_owned(),
                left_out,
                of: tokens,
                limit: MAX_TOKENS_LEFT_OUT,
            })
        }
    }

    /// Trains the model from the treebank's `train` file.
    ///
    /// # Errors
    ///
    /// What reading the samples fails on, more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]),
    /// or what training fails on.
    pub fn train(&self, train: &str) -> Result<Trained> {
        let read = self.samples(train, self.treebank.train.name)?;
        check_skipped(LeftOut::TrainingSentences, read.skipped, read.samples.len())?;
        let language = self.treebank.language;
        let stamp = Stamp::at(TIMESTAMP_MS);
        let params = published::training_parameters();
        let mut config = TrainingConfiguration::default();
        let mut events = Vec::new();
        let (bytes, report) = match &read.samples {
            UdSamples::Segments(HeldOut::Sentences(samples)) => {
                let factory = SentenceDetectorFactory::new(language, true, None, None)?;
                let mut stream = SdEventStream::new(&factory, WHITESPACE);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = sentdetect::train_sentence_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Segments(HeldOut::Tokens { samples, .. }) => {
                let factory = TokenizerFactory::new(language, None, false, None)?;
                let mut stream = TokSpanEventStream::new(&factory, WHITESPACE);
                for s in samples {
                    stream.events(s, &mut events)?;
                }
                let t = tokenize::train_tokenizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Tags(samples) => {
                let factory = PosTaggerFactory::new(language, None, None)?;
                let mut stream = PosEventStream::new(&factory)?;
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = postag::train_pos_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
            UdSamples::Lemmas(samples) => {
                let factory = LemmatizerFactory::new(language);
                let mut stream = LemmaEventStream::new(&factory);
                samples.iter().for_each(|s| stream.events(s, &mut events));
                let t = lemmatize::train_lemmatizer_model_from_events(&events, &factory, &params, &mut config, stamp)?;
                (t.package.to_bytes(), t.report)
            }
        };
        Ok(Trained {
            bytes,
            samples: read.samples.len(),
            skipped_sentences: read.skipped,
            events: events.len(),
            event_hash: report.event_hash.expect("trained from events"),
        })
    }

    /// Each held-out sample's counts for the model in `bytes`, scored with
    /// OpenNLP's evaluator for the component: for a tagger, as the trained
    /// model is scored ([`Scored::Candidate`]) over the tokens whose tag the
    /// model in `bytes` can give, and for a lemmatizer, over the tokens it
    /// scores as the reference ([`ReferenceClasses::scores`]).
    ///
    /// # Errors
    ///
    /// A model file that is not a model of the component, samples of
    /// another component, or a prediction the evaluator throws on.
    pub fn score(&self, bytes: &[u8], held_out: &ReadSamples<'_>) -> Result<Vec<Counts>> {
        match (self.segmentation(), self.tagger(), &held_out.samples) {
            (Some(s), _, UdSamples::Segments(samples)) => s.score(bytes, samples),
            (_, Some(t), UdSamples::Tags(samples)) => {
                let tags = HeldOutTags::new(samples.clone(), held_out.skipped);
                t.score(bytes, Scored::Candidate, &tags, &t.reference_tags(bytes)?)
            }
            (None, None, UdSamples::Lemmas(samples)) => {
                score_lemmas(bytes, samples, &ReferenceClasses::of_model(bytes)?)
            }
            (_, _, samples) => Err(self.mismatch(samples)),
        }
    }

    fn mismatch(&self, samples: &UdSamples<'_>) -> Error {
        Error::ComponentMismatch { recipe: component_name(self.component), samples: samples.kind() }
    }

    /// Scores the trained model and the reference (its file's bytes in
    /// `reference`) on the held-out samples and runs the gate, as the
    /// MASC recipes' gates do (the module documentation says how), with
    /// the `test` file's name and SHA-256 naming the held-out set.
    ///
    /// # Errors
    ///
    /// A tagger's reference that cannot give the tags of more than
    /// [`MAX_TOKENS_LEFT_OUT`] of the held-out tokens
    /// ([`Error::TagsOutsideReference`]), a lemmatizer's reference that
    /// cannot give the lemmas of more than that
    /// ([`Error::ClassesOutsideReference`]), samples of another component,
    /// what scoring fails on, or the gate's errors.
    pub fn gate(&self, trained: &[u8], reference: &[u8], held_out: ReadSamples<'_>) -> Result<GateRun> {
        let reference_id = self.reference.id();
        let (name, corpus_sha256) = (self.held_out_name(), self.treebank.test.sha256);
        match (self.segmentation(), self.tagger(), held_out.samples) {
            (Some(s), _, UdSamples::Segments(samples)) => {
                s.gate(trained, reference_id, reference, &samples, &name, corpus_sha256)
            }
            (_, Some(t), UdSamples::Tags(samples)) => {
                let samples = HeldOutTags::new(samples, held_out.skipped);
                t.gate(trained, reference_id, reference, &samples, &name, corpus_sha256)
            }
            (None, None, UdSamples::Lemmas(samples)) => {
                let classes = ReferenceClasses::of_model(reference)?;
                let left_out = self.check_classes(&classes, &samples)?;
                let candidate = score_lemmas(trained, &samples, &classes)?;
                let baseline = lemma_scores(reference, &samples, &classes)?;
                let units = candidate
                    .into_iter()
                    .zip(baseline.units)
                    .map(|(candidate, reference)| PairedUnit { candidate, reference })
                    .collect();
                let tokens = lemma_tokens(&samples);
                let set = HeldOutSet {
                    name,
                    corpus_sha256: corpus_sha256.to_owned(),
                    samples_sha256: lemma_samples_sha256(&samples, &classes),
                    skipped_sentences: held_out.skipped as u64,
                    scoring: Scoring::Tags(TagScoring {
                        tokens,
                        left_out: left_out.iter().map(|t| t.tokens).sum(),
                        reference_mappings: Vec::new(),
                    }),
                };
                let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
                let record = ModelGateRecord::new(
                    self.task(),
                    set,
                    id(self.model_id, trained),
                    id(reference_id, reference),
                    units,
                )
                .map_err(Error::Gate)?;
                let held_out = HeldOutCard {
                    samples: samples.len(),
                    skipped_sentences: held_out.skipped,
                    tokens: Some(TokenCoverage {
                        tokens,
                        reference_mappings: Vec::new(),
                        left_out,
                        reference_mismatches: baseline.mismatches,
                    }),
                };
                Ok(GateRun { record, held_out })
            }
            (_, _, samples) => Err(self.mismatch(&samples)),
        }
    }
}

/// What one UD recipe made.
#[derive(Debug)]
pub struct UdOutcome {
    pub trained: Trained,
    pub gate: ModelGateRecord,
    pub card: UdModelCard,
}

/// Trains `recipe` on the treebank's `train` file, scores it and the
/// reference (the bytes of its file) on the `test` file, and makes the
/// gate record and the card.
///
/// # Errors
///
/// What training, reading the samples, scoring or the gate fail on, and
/// [`Error::TooManyLeftOut`] when the reader skips more than
/// [`crate::MAX_SKIPPED_SENTENCES`] of the training or of the held-out
/// sentences.
pub fn run_ud(recipe: &UdRecipe, text: &TreebankText, reference: &[u8]) -> Result<UdOutcome> {
    let trained = recipe.train(&text.train)?;
    let GateRun { record: gate, held_out } = recipe.gate(&trained.bytes, reference, recipe.held_out(text)?)?;
    let card = UdModelCard::new(recipe, &trained, held_out, &gate);
    Ok(UdOutcome { trained, gate, card })
}

fn component_name(component: Component) -> &'static str {
    match component {
        Component::Sentence => "sentence detector",
        Component::Tokens => "tokenizer",
        Component::Pos => "part-of-speech tagger",
        Component::Lemmas => "lemmatizer",
    }
}

/// The lemmatizer in `bytes`.
fn open_lemmatizer(bytes: &[u8]) -> Result<LemmatizerModel> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    LemmatizerModel::open(file).map_err(|e| Error::Model(e.to_string()))
}

/// Each sample's counts for the lemmatizer in `bytes`, given the
/// sample's tags: the tokens given the sample's lemma, of its tokens that
/// `classes` scores ([`ReferenceClasses::scores`]). The
/// lemmas are those `LemmatizerME.lemmatize` gives, which
/// `LemmatizerEvaluator` compares with the sample's: `decodeLemmas` applies
/// each class to the lowercased token, so a capitalised held-out lemma is
/// never given.
///
/// # Errors
///
/// A model file that is not a lemmatizer model, a sentence no class
/// sequence scores above the search's minimum for, a class the decoder
/// cannot read, or what the evaluator throws on.
pub fn score_lemmas(bytes: &[u8], samples: &[LemmaSample<'_>], classes: &ReferenceClasses) -> Result<Vec<Counts>> {
    Ok(lemma_scores(bytes, samples, classes)?.units)
}

/// A lemmatizer's counts on the held-out samples, and its most frequent
/// mismatches on the scored tokens: the held-out lemma and the lemma
/// given.
struct LemmaScores {
    units: Vec<Counts>,
    /// At most [`crate::postag::MISMATCHES_LISTED`], most frequent first.
    mismatches: Vec<Mismatch>,
}

/// [`score_lemmas`], with the mismatches.
fn lemma_scores<'s>(bytes: &[u8], samples: &'s [LemmaSample<'_>], classes: &ReferenceClasses) -> Result<LemmaScores> {
    let lemmatizer = MaxentLemmatizer::new(Arc::new(open_lemmatizer(bytes)?));
    let mut units = Vec::with_capacity(samples.len());
    let mut mismatches: BTreeMap<(&'s str, String), u64> = BTreeMap::new();
    let (mut predicted_classes, mut probs) = (Vec::new(), Vec::new());
    let mut words: Vec<TokenTag<'_>> = Vec::new();
    let mut lemmas: Vec<String> = Vec::new();
    for sample in samples {
        words.clear();
        words.extend(
            sample.tokens().iter().zip(sample.tags()).map(|(t, g)| TokenTag { token: t.as_str(), tag: g.as_str() }),
        );
        predicted_classes.clear();
        probs.clear();
        if !lemmatizer.classes(&words, &mut predicted_classes, &mut probs) {
            return Err(Error::Model(format!("no lemma class sequence scored above the minimum for: {sample:?}")));
        }
        lemmas.clear();
        for (w, class) in words.iter().zip(&predicted_classes) {
            lemmas.push(decode_lemma(w.token, class).map_err(|e| Error::Model(format!("{:?}: {e}", w.token)))?);
        }
        let predicted: Vec<&str> = lemmas.iter().map(String::as_str).collect();
        let mut evaluator = LemmatizerEvaluator::default();
        evaluator.evaluate(sample, &predicted)?;
        let (mut scored, mut right, mut right_as_written) = (0u64, 0u64, 0u64);
        for ((token, expected), &given) in sample.tokens().iter().zip(sample.lemmas()).zip(&predicted) {
            let expected: &'s str = expected.as_str();
            right_as_written += u64::from(expected == given);
            if classes.scores(token.as_str(), expected) {
                scored += 1;
                if expected == given {
                    right += 1;
                } else {
                    *mismatches.entry((expected, given.to_owned())).or_default() += 1;
                }
            }
        }
        // The lemmas counted over every token are what
        // `LemmatizerEvaluator` counts.
        debug_assert_eq!(evaluator.word_count(), sample.len() as u64);
        debug_assert_eq!(right_as_written as f64 / sample.len() as f64, evaluator.word_accuracy());
        units.push(Counts::accuracy(right, scored));
    }
    Ok(LemmaScores { units, mismatches: most_frequent(mismatches) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::treebank::TREEBANKS;

    /// A token is left out only when its gold class gives its lemma and no
    /// class of the reference does; a lemma its gold class does not give is
    /// scored.
    #[test]
    fn a_lemma_no_class_gives_is_scored_and_one_the_reference_lacks_is_left_out() {
        let none = ReferenceClasses::new([]);
        assert!(!none.scores("gatos", "gato") && !none.scores("fue", "ir"));
        // Lowercased before decoding, so a capital letter is never given.
        assert_eq!(decode_lemma("López", &lemmatize::lemma_class("López", "López")).unwrap(), "lópez");
        assert!(none.scores("López", "López") && none.scores(",", "[PUNCT]"));
        // An index of ten or more is read as one digit (AnCora's test file).
        let class = lemmatize::lemma_class("Ultimamente", "últimamente");
        assert_eq!(decode_lemma("Ultimamente", &class).unwrap(), "ultimamente");
        assert!(none.scores("Ultimamente", "últimamente"));
        let gato = ReferenceClasses::new([lemmatize::lemma_class("gatos", "gato")]);
        assert!(gato.scores("gatos", "gato") && gato.scores("perros", "perro") && !gato.scores("fue", "ir"));
        // A deletion's letter is not read, so a class that names another
        // letter gives the same lemma (AnCora's test file has "Burlonas").
        let burlones = lemmatize::lemma_class("burlones", "burlón");
        assert_ne!(burlones, lemmatize::lemma_class("Burlonas", "burlón"));
        assert_eq!(decode_lemma("Burlonas", &burlones).unwrap(), "burlón");
        assert!(ReferenceClasses::new([burlones]).scores("Burlonas", "burlón"));
    }

    #[test]
    fn the_recipes_name_their_treebank_component_and_reference() {
        for r in UD_RECIPES {
            let t = r.treebank;
            let id = format!("turbo-{}-ud-{}-2.16-{}-1", t.language, t.name, r.component.name());
            assert_eq!(r.model_id, id);
            let parsed: turbo_zoo::ArtifactId = r.model_id.parse().unwrap();
            assert_eq!(parsed.as_str(), r.model_id);
            match r.reference {
                Reference::Zoo(reference) => {
                    let index = turbo_zoo::Index::builtin();
                    let a = index.get(reference).unwrap();
                    assert_eq!(a.component, Some(r.zoo_component()), "{reference}");
                    assert_eq!(a.language.as_ref().unwrap().as_str(), t.language, "{reference}");
                    assert!(reference.ends_with(&format!("-{}-1.3-2.5.4", r.component.name())));
                }
                Reference::Recorded(reference) => {
                    // OpenNLP's Italian models are local-only and not in the zoo.
                    assert_eq!(t.language, "it");
                    assert_eq!(reference.file, format!("{}.bin", reference.id));
                    assert_eq!(reference.id, format!("opennlp-it-ud-vit-{}-1.3-2.5.4", r.component.name()));
                    assert!(reference.url.ends_with(reference.file));
                    assert_eq!(turbo_zoo::Index::builtin().find("it", r.zoo_component()).count(), 0);
                }
            }
            assert_eq!(UdRecipe::trains(r.model_id), Some(r));
            assert_eq!(r.segmentation().is_some(), matches!(r.component, Sentences | Tokens));
            let tagger = r.tagger();
            assert_eq!(tagger.is_some(), r.component == Pos);
            if let Some(pos) = tagger {
                assert_eq!((pos.model_id, pos.reference, pos.reference_format), (r.model_id, r.reference, TAG_FORMAT));
                assert!(pos.reference_mappings.is_empty());
                assert_eq!(pos.max_left_out, MAX_TOKENS_LEFT_OUT);
            }
        }
        // Every treebank has a tagger and a lemmatizer; the ones whose text
        // is running text have a sentence detector and a tokenizer.
        for t in TREEBANKS {
            let components: Vec<Component> =
                UD_RECIPES.iter().filter(|r| r.treebank == t).map(|r| r.component).collect();
            let running = t.name != "ftb" && t.name != "markit";
            let expected: &[Component] = if running { &[Sentences, Tokens, Pos, Lemmas] } else { &[Pos, Lemmas] };
            assert_eq!(components, expected, "{}", t.repository);
        }
        assert_eq!(UD_RECIPES[3].task(), Task::Lemmatizer);
        assert_eq!(UdRecipe::trains("turbo-en-ud-ewt-2.16-pos-1"), None);
    }
}
