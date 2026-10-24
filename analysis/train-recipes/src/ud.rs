//! Recipes that train the sentence detector, tokenizer, part-of-speech
//! tagger and lemmatizer on the CC BY 4.0 treebanks of [`crate::treebank`]
//! (issue #415), and their gates against the OpenNLP UD 1.3 models.
//!
//! Each recipe trains on the treebank's UD `train` file and is gated on
//! its UD `test` file; the `dev` file is checked against its digest and
//! not read. Training is what OpenNLP 2.5.4's CoNLL-U trainer tools do,
//! with the settings used for this project's `models-ud-2.16-1` release:
//! GIS, 300 iterations, cutoff 5, 4 partitions, the factory defaults for
//! the language, OpenNLP 2.5.4's whitespace, 10 sentences per sentence
//! detector sample and the universal tags. The separate OpenNLP UD 1.3
//! reproduction recipes use 12 partitions. Training uses no randomness,
//! so a recipe has no seed.
//!
//! OpenNLP trained the Catalan references on this treebank's `train` file.
//! The Catalan sentence detector and tokenizer recipes make OpenNLP's
//! events exactly (the same `Training-Eventhash`), but their four-partition
//! sums differ from the references in the last parameter bits. Those two
//! gates compare the released recipes with the references; exact parameter
//! reproduction uses the separate 12-partition recipes. The Catalan tagger
//! and lemmatizer recipes make OpenNLP's events exactly when the empty nodes are kept (the event hash OpenNLP's
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
//! a sentence detector or tokenizer is gated as a
//! [`SegmentationRecipe`](crate::segmentation::SegmentationRecipe), and a
//! tagger as a [`PosRecipe`](crate::postag::PosRecipe) whose tags, the
//! trained model's and the reference's, are both asked for in
//! [`TAG_FORMAT`] and compared with the held-out tag as written, with no
//! mapping. A lemmatizer, given the held-out tags, is scored on its lemmas
//! as `LemmatizerEvaluator` counts them, as a tagger on its tags
//! ([`score_lemmas`]).
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

mod lemma;
mod samples;

use turbo_eval::model_gate::ModelGateRecord;
use turbo_formats::conllu::Tagset;
use turbo_formats::sample::{LemmaSample, PosSample};
use turbo_postag::PosTagFormat;
use turbo_train_components::ud::{self as published, Component};

pub use lemma::{ReferenceClasses, lemma_samples_sha256, score_lemmas};
pub use samples::{lemma_samples, pos_samples};

use crate::error::Result;
use crate::segmentation::HeldOut;
use crate::treebank::{
    CATALAN_ANCORA, FINNISH_FTB, ITALIAN_MARKIT, PORTUGUESE_PORTTINARI, SPANISH_ANCORA, Treebank, TreebankText,
};
use crate::ud_card::UdModelCard;
use crate::{GateRun, Limit, RecordedReference, Reference, Trained};

/// GIS partitions used for the published `models-ud-2.16-1` models.
pub const THREADS: i32 = 4;

/// Release parameters, independent of OpenNLP UD 1.3 reproduction.
pub(crate) fn training_parameters() -> turbo_train::TrainingParameters {
    let mut params = published::training_parameters();
    params.put_str(turbo_train::Key::THREADS, &THREADS.to_string());
    params
}

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
/// the gate). More and the recipe fails ([`Error::TagsOutsideReference`](crate::error::Error::TagsOutsideReference),
/// [`Error::ClassesOutsideReference`](crate::error::Error::ClassesOutsideReference)).
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

    /// The kind of the samples, as [`Error::ComponentMismatch`](crate::error::Error::ComponentMismatch) names it.
    pub(super) fn kind(&self) -> &'static str {
        match self {
            UdSamples::Segments(HeldOut::Sentences(_)) => "sentence",
            UdSamples::Segments(HeldOut::Tokens { .. }) => "token",
            UdSamples::Tags(_) => "part-of-speech",
            UdSamples::Lemmas(_) => "lemma",
        }
    }
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
/// [`Error::TooManyLeftOut`](crate::error::Error::TooManyLeftOut) when the
/// reader skips more than [`crate::MAX_SKIPPED_SENTENCES`] of the training
/// or of the held-out sentences.
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

#[cfg(test)]
mod tests {
    #[test]
    fn ud_release_keeps_four_partitions_when_reproduction_uses_twelve() {
        use turbo_train::Key;
        let release = super::training_parameters();
        let reproduction = turbo_train_components::ud::training_parameters();
        assert_eq!(release.get_int(Key::THREADS).unwrap(), Some(4));
        assert_eq!(reproduction.get_int(Key::THREADS).unwrap(), Some(12));
        for key in [Key::ITERATIONS, Key::CUTOFF] {
            assert_eq!(release.get_int(key).unwrap(), reproduction.get_int(key).unwrap());
        }
        assert_eq!(turbo_train::GisParams::default().threads, 1);
    }

    use super::*;
    use crate::error::Error;
    use crate::treebank::TREEBANKS;
    use turbo_eval::model_gate::Task;
    use turbo_lemmatizer::edit_script::decode_lemma;
    use turbo_train_components::lemmatize;

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

    #[test]
    fn check_data_rejects_invalid_segmentation_reference_bytes() {
        let conllu = "# text = Hola.\n1\tHola\thola\tINTJ\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n";
        let text = TreebankText { train: conllu.to_owned(), test: conllu.to_owned() };
        assert!(matches!(UD_RECIPES[0].check_data(&text, b"not a model"), Err(Error::Model(_))));
        assert!(matches!(UD_RECIPES[1].check_data(&text, b"not a model"), Err(Error::Model(_))));
    }

    #[test]
    fn score_lemmas_handles_empty_sample_without_nan_panic() {
        let conllu = "# text = Gatos.\n1\tGatos\tgato\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n\
                      # text = Perros.\n1\tPerros\tperro\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n\
                      # text = Casas.\n1\tCasas\tcasa\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n\
                      # text = Mesas.\n1\tMesas\tmesa\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n\
                      # text = Libros.\n1\tLibros\tlibro\tNOUN\t_\t_\t0\troot\t_\tSpaceAfter=No\n2\t.\t.\tPUNCT\t_\t_\t1\tpunct\t_\t_\n\n";
        let trained = UD_RECIPES[3].train(conllu).unwrap();
        let classes = ReferenceClasses::of_model(&trained.bytes).unwrap();
        let empty = LemmaSample::new(Vec::new(), Vec::new(), Vec::new(), turbo_analysis::Span::new(0, 0)).unwrap();
        let counts = score_lemmas(&trained.bytes, &[empty], &classes).unwrap();
        assert_eq!(counts.len(), 1);
    }
}
