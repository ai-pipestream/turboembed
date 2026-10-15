//! The MASC recipes end to end on corpora written in MASC's layout: the
//! split, training, scoring both models on the held-out samples, the
//! gate, the card and the model's directory; and the part-of-speech
//! recipe's samples on the MASC corpora checked in for the goldens.

/// The bundles the MASC goldens' inputs are written in, read as the
/// `turbo-formats` tests read them.
#[path = "../../formats/tests/bundle/mod.rs"]
mod bundle;
mod common;

use std::path::PathBuf;

use common::{GATED_DOCUMENTS, SENTENCES, nest_token, rewrite_document, temp_dir, with_tags_rotated, write_corpus};
use turbo_eval::model_gate::{Measure, ModelGateRecord, Task};
use turbo_formats::masc::{MascCorpus, MascTokenReader};
use turbo_postag::PosTagFormat;
use turbo_train_components::{TokSpanEventStream, TokenizerFactory};
use turbo_train_recipes::card::{
    CARD_FILE, CARD_VERSION, GATE_FILE, HeldOutCard, ModelCard, SpellingCard, TagCount, TagFormat, TokenCoverage,
    write_model_dir,
};
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, SplitCorpus};
use turbo_train_recipes::postag::{self, CANDIDATE_FORMAT, POS, PosRecipe, ReferenceTags};
use turbo_train_recipes::segmentation::{SENTENCE, TOKENS, in_text_order, read_token_samples};
use turbo_train_recipes::{Error, LeftOut, RECIPES, Recipe, Reference, held_out_name, run};
use turbo_zoo::{Index, LicenceClass};

/// The pool without abbreviations, attached punctuation or contractions:
/// a model trained on it has not seen what the held-out text holds.
const PLAIN: &[&str] = &["Who called ~? Nobody did ~!", "Mr. Jones left early ~.", "It rained all day ~."];

/// Whether CI asks for the published models to be present.
fn models_required() -> bool {
    std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1")
}

fn split_of(root: &std::path::Path) -> (SplitCorpus, masc::CorpusDigest) {
    let digest = masc::check_corpus(root, RELEASE, true).unwrap();
    let corpus = MascCorpus::read(root).unwrap();
    (SPLIT.apply(corpus, root).unwrap(), digest)
}

#[test]
fn the_split_holds_out_every_tenth_document_in_read_order() {
    let root = temp_dir("split");
    write_corpus(&root, 25, SENTENCES);
    let all: Vec<PathBuf> = MascCorpus::read(&root).unwrap().documents().iter().map(|d| d.path().to_owned()).collect();
    let (split, digest) = split_of(&root);
    assert_eq!(digest.files, 25 * 5);
    assert_eq!(split.train.documents().len(), 23);
    let held: Vec<String> =
        [9, 19].iter().map(|&i| all[i].strip_prefix(&root).unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(split.held_out_paths, held);
    assert!(split.held_out_paths.iter().all(|p| !p.starts_with('/')), "{:?}", split.held_out_paths);
    assert!(split.skipped_paths.is_empty());
    // The read order is the reader's: each directory's entries by name,
    // a directory's subdirectories after its files, the last pushed first.
    assert!(split.held_out_paths[0].starts_with("written/news/"), "{:?}", split.held_out_paths);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn the_recipes_train_score_gate_and_write_their_cards() {
    let root = temp_dir("recipes");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, digest) = split_of(&root);

    // The candidate is trained on documents made of the same eight
    // sentences as the held-out documents, so it has seen every held-out
    // sentence and its score here is not a measure of anything. The
    // reference is the same recipe trained on a corpus without
    // abbreviations, contractions or attached punctuation; for the tagger,
    // which is scored only on tokens whose tag the reference can give, on
    // the same sentences with their tags moved one token along, so it can
    // give every tag but has learned the wrong one for nearly every word.
    // The check that tells the gate's rule apart from one that passes
    // everything is the swapped one below, where the weaker model is the
    // candidate.
    let plain_root = temp_dir("plain");
    write_corpus(&plain_root, 30, PLAIN);
    let (plain, _) = split_of(&plain_root);
    let rotated_root = temp_dir("rotated");
    let rotated_pool = with_tags_rotated(SENTENCES);
    write_corpus(&rotated_root, 30, &rotated_pool.iter().map(String::as_str).collect::<Vec<_>>());
    let (rotated, _) = split_of(&rotated_root);

    let out = temp_dir("out");
    for recipe in RECIPES {
        let weak = match recipe {
            Recipe::Pos(_) => recipe.train(&rotated.train).unwrap(),
            Recipe::Segmentation(_) => recipe.train(&plain.train).unwrap(),
        };
        println!(
            "{}: the weak reference was trained on {} samples, {} events",
            recipe.model_id(),
            weak.samples,
            weak.events
        );
        let outcome = run(&recipe, &split, &digest, false, "weak", &weak.bytes).unwrap();
        let gate = &outcome.gate;
        gate.check().unwrap();
        println!("{gate}");
        assert_eq!(gate.task, recipe.task());
        let measure = if recipe.task() == Task::PartOfSpeechTagger { Measure::Accuracy } else { Measure::FMeasure };
        assert_eq!(gate.measure, measure);
        assert_eq!(gate.set.name, held_out_name());
        assert_eq!(gate.set.corpus_sha256, digest.sha256);
        assert_eq!(gate.candidate.id, recipe.model_id());
        assert_eq!(gate.reference.id, "weak");
        // The model that has seen the held-out sentences beats the weak
        // one, and passes.
        assert!(gate.result.difference.mean > 0.0, "{gate}");
        assert!(gate.result.pass);
        assert!(outcome.trained.samples > 0 && outcome.trained.events > 0);
        assert_eq!(outcome.trained.skipped_sentences, 0);

        // The same models with their places swapped: the weak model as
        // the candidate loses far more than the margin, and fails. This is
        // the check that discriminates.
        let swapped = recipe
            .gate(
                &weak.bytes,
                recipe.model_id(),
                &outcome.trained.bytes,
                &split.held_out,
                &held_out_name(),
                &digest.sha256,
            )
            .unwrap()
            .record;
        assert!(swapped.result.difference.high < 0.0, "{swapped}");
        assert!(!swapped.result.pass);
        // A model against itself passes with no difference.
        let itself = recipe
            .gate(
                &outcome.trained.bytes,
                recipe.model_id(),
                &outcome.trained.bytes,
                &split.held_out,
                &held_out_name(),
                &digest.sha256,
            )
            .unwrap()
            .record;
        assert_eq!(itself.result.difference.mean, 0.0);
        assert!(itself.result.pass);

        // Training is deterministic: the same corpus gives the same file.
        assert_eq!(recipe.train(&split.train).unwrap().bytes, outcome.trained.bytes);

        let card = &outcome.card;
        assert_eq!(card.id, recipe.model_id());
        assert_eq!(card.component, recipe.component());
        assert_eq!(card.class, LicenceClass::Permissive);
        assert_eq!(card.trained_on, vec![masc::training_data()]);
        assert!(card.attribution.contains("CC") || card.attribution.contains("Creative Commons"));
        assert!(!card.corpus.release_recorded);
        assert_eq!(card.corpus.held_out_documents, split.held_out_paths);
        assert_eq!(card.training.seed, None);
        let tokens = match recipe {
            Recipe::Pos(p) => Some(TokenCoverage {
                tokens: p.held_out(&split.held_out).unwrap().tokens(),
                spellings: postag::REFERENCE_SPELLINGS
                    .iter()
                    .map(|s| SpellingCard { masc: s.masc.to_owned(), reference: s.reference.to_owned() })
                    .collect(),
                left_out: Vec::new(),
            }),
            Recipe::Segmentation(_) => None,
        };
        assert_eq!(card.held_out, HeldOutCard { samples: gate.units.len(), skipped_sentences: 0, tokens });
        assert_eq!(card.training.sentences_per_sample.is_some(), recipe.task() == Task::SentenceDetector);
        assert_eq!(card.gate.pass, gate.result.pass);
        match recipe {
            Recipe::Pos(_) => {
                assert_eq!(card.training.whitespace, None);
                assert_eq!(card.gate.candidate_tag_format, Some(TagFormat::Custom));
                assert_eq!(card.gate.reference_tag_format, Some(TagFormat::Penn));
            }
            Recipe::Segmentation(_) => {
                assert_eq!(card.training.whitespace.as_deref(), Some("legacy"));
                assert_eq!((card.gate.candidate_tag_format, card.gate.reference_tag_format), (None, None));
            }
        }

        write_model_dir(&out, card, &outcome.trained.bytes, gate).unwrap();
        let dir = out.join(recipe.model_id());
        assert_eq!(ModelCard::read(&dir.join(CARD_FILE)).unwrap(), *card);
        let other = dir.join("other-card.json");
        let current = format!("\"card_version\": {CARD_VERSION},");
        assert!(card.to_json().contains(&current));
        std::fs::write(&other, card.to_json().replace(&current, "\"card_version\": 1,")).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::CardVersion { found: 1, .. })));
        // A card of version 2, without the fields version 3 added, is of
        // another version, not a card missing fields.
        let mut old: serde_json::Value = serde_json::from_str(&card.to_json()).unwrap();
        old["card_version"] = 2.into();
        old.as_object_mut().unwrap().remove("held_out");
        old["gate"].as_object_mut().unwrap().remove("reference_sha256");
        std::fs::write(&other, serde_json::to_string_pretty(&old).unwrap()).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::CardVersion { found: 2, .. })));
        // A card of this version with a field missing is not a card.
        old["card_version"] = CARD_VERSION.into();
        std::fs::write(&other, serde_json::to_string_pretty(&old).unwrap()).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::Json { .. })));
        std::fs::write(&other, "{}").unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::Json { .. })));
        let bad_split = card.to_json().replace("\"every\": 10,", "\"every\": 0,");
        assert_ne!(bad_split, card.to_json());
        std::fs::write(&other, bad_split).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::Json { .. })), "a split of every 0");
        assert_eq!(ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap(), *gate);
        assert_eq!(std::fs::read(dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
        assert!(write_model_dir(&out, card, &outcome.trained.bytes, gate).is_err(), "nothing is replaced");

        // A card gated against another model than the recipe's reference,
        // as every card here is, makes no zoo entry, even with its gate
        // passed and the release recorded; nor does a card no recipe makes.
        let url = format!("https://example.org/models/{}", card.file.name);
        let refused = |card: &ModelCard| match card.artifact(&url) {
            Err(Error::NotShippable { reason, .. }) => reason,
            other => panic!("{other:?}"),
        };
        let mut weak = card.clone();
        weak.corpus.release_recorded = true;
        assert_eq!(refused(&weak), "it was not gated against its recipe's reference");
        let mut named_only = weak.clone();
        named_only.gate.reference = recipe.reference_id().to_owned();
        assert_eq!(refused(&named_only), "it was not gated against its recipe's reference");
        let mut released = named_only.clone();
        released.gate.reference_sha256 = recipe.reference().sha256().unwrap();
        let mut unknown = released.clone();
        unknown.id = "turbo-en-masc-3.0.0-chunks-1".to_owned();
        assert_eq!(refused(&unknown), "no recipe trains it");
        // Against the reference, one whose gate failed, or trained on a
        // corpus that is not the recorded release, makes none either.
        let mut failed = released.clone();
        failed.gate.pass = false;
        assert_eq!(refused(&failed), "it failed its gate");
        let mut unrecorded = released.clone();
        unrecorded.corpus.release_recorded = false;
        assert_eq!(refused(&unrecorded), "it was trained on a corpus that is not the recorded release");
        // With the reference's id and digest, the release recorded and the
        // gate passed, the entry is permissive, shippable and names MASC.
        let artifact = released.artifact(&url).unwrap();
        let index = Index::parse(Index::to_canonical_json(&[artifact]).as_bytes()).unwrap();
        let entry = index.get(recipe.model_id()).unwrap();
        assert_eq!(entry.component, Some(recipe.component()));
        assert_eq!(entry.computed_class(), LicenceClass::Permissive);
        assert_eq!(index.shippable().count(), 1);
        assert_eq!(entry.trained_on[0].name, masc::NAME);
        // A model also trained on non-commercial data is local-only: it
        // makes no zoo entry, whatever its gate says. Nor does a card
        // whose class is not the one its licences give.
        let mut local = released.clone();
        local.trained_on.push(turbo_zoo::TrainingData {
            name: "UD_English-GUM r2.16".to_owned(),
            license: turbo_zoo::Licence::CcByNcSa40,
            license_terms: "https://github.com/UniversalDependencies/UD_English-GUM/blob/r2.16/README.md"
                .parse()
                .unwrap(),
        });
        assert!(matches!(local.artifact(&url), Err(Error::InvalidCard { field: "class", .. })));
        local.class = LicenceClass::LocalOnly;
        assert!(matches!(local.artifact(&url), Err(Error::NotShippable { .. })));
        let mut mislabelled = released.clone();
        mislabelled.class = LicenceClass::ShareAlike;
        assert!(matches!(mislabelled.artifact(&url), Err(Error::InvalidCard { field: "class", .. })));
    }
    // A tagger trained on the plain corpus gives only `NN` and `.`: as the
    // reference it could never be right on most held-out tokens, and the
    // gate refuses it.
    let plain_tagger = POS.train(&plain.train).unwrap();
    let candidate = POS.train(&split.train).unwrap();
    let refused = Recipe::Pos(POS)
        .gate(&candidate.bytes, "plain", &plain_tagger.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .err();
    match refused {
        Some(Error::TagsOutsideReference { reference, left_out, .. }) => {
            assert_eq!(reference, "plain");
            let tags: Vec<&str> = left_out.iter().map(|t| t.tag.as_str()).collect();
            assert!(tags.contains(&"VBD") && tags.contains(&"POS") && !tags.contains(&"NN"), "{tags:?}");
        }
        other => panic!("{other:?}"),
    }
    for dir in [root, plain_root, rotated_root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// [`SENTENCES`] with two sentences the recipes' readers skip: one with a
/// token without a tag, which the tagger's reader skips, and one with no
/// Penn token, which both readers skip.
fn with_skipped_sentences() -> Vec<&'static str> {
    let mut pool = SENTENCES.to_vec();
    pool.extend(["Dogs/NNS ran/ home/NN ~./.", "^Not ^tokenized ~^."]);
    pool
}

/// The document's sentences: `first`, then the corpus's sentences.
fn sentences_after(first: &[usize]) -> Vec<usize> {
    first.iter().copied().chain((0..SENTENCES.len()).cycle()).take(common::SENTENCES_PER_DOCUMENT).collect()
}

fn too_many(err: Option<Error>, side: LeftOut, skipped: u64, of: u64) {
    match err {
        Some(Error::TooManyLeftOut { what, count, of: total, limit })
            if (what, count, total) == (side, skipped, of) =>
        {
            assert_eq!(limit, turbo_train_recipes::MAX_SKIPPED_SENTENCES);
        }
        other => panic!("{side:?}, {skipped} of {of}: {other:?}"),
    }
}

/// A recipe's reader skips held-out sentences as it skips training ones;
/// the gate run and the card record how many, and a recipe fails when
/// either side skips more than one sentence in a thousand.
#[test]
fn skipped_sentences_are_recorded_and_bounded() {
    let pool = with_skipped_sentences();
    let (untagged, no_tokens) = (SENTENCES.len(), SENTENCES.len() + 1);
    // 250 documents of 40 sentences: 1,000 held-out sentences.
    let root = temp_dir("skips");
    write_corpus(&root, 250, SENTENCES);
    let (split, _) = split_of(&root);
    rewrite_document(&root, &split.held_out_paths[0], &pool, &sentences_after(&[no_tokens]));
    let (split, digest) = split_of(&root);
    assert_eq!(split.held_out.documents().len(), 25);

    // One sentence in a thousand is within the limit.
    let tags = POS.held_out(&split.held_out).unwrap();
    assert_eq!((tags.len(), tags.skipped()), (999, 1));
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert_eq!((tokens.len(), tokens.skipped()), (999, 1));
    assert_eq!(SENTENCE.held_out(&split.held_out).unwrap().skipped(), 0);

    // The gate run reports the count, which `run` puts on the card.
    let small = temp_dir("skips-train");
    write_corpus(&small, GATED_DOCUMENTS, SENTENCES);
    let (small_split, _) = split_of(&small);
    for recipe in [Recipe::Segmentation(TOKENS), Recipe::Pos(POS)] {
        let trained = recipe.train(&small_split.train).unwrap();
        let gated = recipe
            .gate(&trained.bytes, "itself", &trained.bytes, &split.held_out, &held_out_name(), &digest.sha256)
            .unwrap();
        assert_eq!((gated.held_out.samples, gated.held_out.skipped_sentences), (999, 1), "{}", recipe.model_id());
        assert_eq!(gated.record.units.len(), 999);
    }

    // Two in a thousand are not: a sentence without a tag for the tagger,
    rewrite_document(&root, &split.held_out_paths[1], &pool, &sentences_after(&[untagged]));
    let (split, _) = split_of(&root);
    too_many(POS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);
    let message = POS.held_out(&split.held_out).unwrap_err().to_string();
    assert_eq!(message, "2 of 1000 held-out sentences skipped, more than the limit of 1 in 1000");
    assert_eq!(TOKENS.held_out(&split.held_out).unwrap().skipped(), 1);
    // and another sentence without tokens for both.
    rewrite_document(&root, &split.held_out_paths[1], &pool, &sentences_after(&[no_tokens]));
    let (split, _) = split_of(&root);
    too_many(POS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);
    too_many(TOKENS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 2, 1000);

    // Training sentences are bounded the same way, before any training.
    let first = split.train.documents()[0].path().strip_prefix(&root).unwrap().to_string_lossy().into_owned();
    rewrite_document(&root, &first, &pool, &[[untagged; 20], [no_tokens; 20]].concat());
    let (split, _) = split_of(&root);
    too_many(POS.train(&split.train).err(), LeftOut::TrainingSentences, 40, 9000);
    too_many(TOKENS.train(&split.train).err(), LeftOut::TrainingSentences, 20, 9000);
    for dir in [root, small] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// A sentence whose tokens are out of text order (a token inside the one
/// before it) is skipped alone by the token recipe's reader, on both
/// sides, and counted with the other skipped sentences, where
/// `TokSpanEventStream` would refuse it; the tagger's reader reads it.
#[test]
fn a_sentence_with_a_token_inside_another_is_skipped_by_the_token_recipe() {
    // 250 documents of 40 sentences: 1,000 held-out sentences.
    let root = temp_dir("nested");
    write_corpus(&root, 250, SENTENCES);
    let (split, _) = split_of(&root);
    nest_token(&root, &split.held_out_paths[0], 0);
    let (split, _) = split_of(&root);
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert_eq!((tokens.len(), tokens.skipped()), (999, 1));
    let tags = POS.held_out(&split.held_out).unwrap();
    assert_eq!((tags.len(), tags.skipped()), (1000, 0));

    let mut refused = 0;
    let skipped = read_token_samples(&split.held_out, |sample| {
        assert!(in_text_order(sample.tokens()));
        Ok(())
    })
    .unwrap();
    assert_eq!(skipped, 1);
    let factory = TokenizerFactory::new("en", None, false, None).unwrap();
    let mut stream = TokSpanEventStream::new(&factory, turbo_train_recipes::segmentation::WHITESPACE);
    for read in MascTokenReader::new(&split.held_out).unwrap() {
        let sample = read.unwrap().sample;
        if !in_text_order(sample.tokens()) {
            let err = stream.events(&sample, &mut Vec::new()).unwrap_err();
            assert!(matches!(err, turbo_train_components::Error::TokensOutOfOrder { token: 1 }), "{err}");
            refused += 1;
        }
    }
    assert_eq!(refused, 1);

    // A training sentence is skipped the same way, and training goes on.
    let small = temp_dir("nested-train");
    write_corpus(&small, GATED_DOCUMENTS, SENTENCES);
    let (split, _) = split_of(&small);
    let first = split.train.documents()[0].path().strip_prefix(&small).unwrap().to_string_lossy().into_owned();
    let all = split.train.documents().len() * common::SENTENCES_PER_DOCUMENT;
    nest_token(&small, &first, 0);
    let (split, _) = split_of(&small);
    let trained = TOKENS.train(&split.train).unwrap();
    assert_eq!((trained.samples, trained.skipped_sentences), (all - 1, 1));
    for dir in [root, small] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn a_held_out_set_of_another_component_is_refused() {
    let root = temp_dir("component");
    write_corpus(&root, 20, SENTENCES);
    let (split, _) = split_of(&root);
    let model = SENTENCE.train(&split.train).unwrap();
    let tokens = TOKENS.held_out(&split.held_out).unwrap();
    assert!(SENTENCE.score(&model.bytes, &tokens).is_err());
    // A sentence model is not a tokenizer model.
    let sentences = SENTENCE.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &sentences).is_err());
    let tokens_again = TOKENS.held_out(&split.held_out).unwrap();
    assert!(TOKENS.score(&model.bytes, &tokens_again).is_err());
    // A sentence model is not a part-of-speech model, in any format, and
    // a part-of-speech model is not a tokenizer model.
    let tags = POS.held_out(&split.held_out).unwrap();
    let penn = ReferenceTags::of(&turbo_analysis::TagSet::new("penn", &["NN", "."]));
    assert!(matches!(POS.score(&model.bytes, CANDIDATE_FORMAT, &tags, &penn), Err(Error::Model(_))));
    assert!(matches!(POS.score(&model.bytes, POS.reference_format, &tags, &penn), Err(Error::Model(_))));
    let tagger = POS.train(&split.train).unwrap();
    assert!(TOKENS.score(&tagger.bytes, &tokens_again).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// A sentence tagged as MASC tags brackets, a hyphen, an ellipsis and a
/// semicolon, and the same sentence as the reference tags them.
const MASC_SPELLED: &str = "Wait/VB .../... he/PRP said/VBD -/- no/DT ~;/; then/RB (/( yes/UH ~)/) ~./.";
const REFERENCE_SPELLED: &str = "Wait/VB .../: he/PRP said/VBD -/: no/DT ~;/: then/RB (/-LRB- yes/UH ~)/-RRB- ~./.";
/// A tag no Penn tagger gives, as MASC 3.0.0 has a few.
const ODD_TAG: &str = "Agh/UH` ~!/.";

/// The tagger is scored in the reference's spelling, over the held-out
/// tokens whose tag the reference can give: a candidate trained on MASC's
/// spellings and a reference trained on the same sentences in its own
/// both score every token, the token of a tag neither can give is left
/// out and counted, and more such tokens than one in a thousand are an
/// error that names their tags.
#[test]
fn the_tagger_is_scored_in_the_reference_s_spelling_over_the_tags_it_can_give() {
    let pool = |extra: &'static str| {
        let mut pool = SENTENCES.to_vec();
        pool.extend([extra, ODD_TAG]);
        pool
    };
    let (masc_pool, reference_pool) = (pool(MASC_SPELLED), pool(REFERENCE_SPELLED));
    let root = temp_dir("spellings");
    write_corpus(&root, GATED_DOCUMENTS, &masc_pool[..=SENTENCES.len()]);
    let reference_root = temp_dir("spellings-reference");
    write_corpus(&reference_root, GATED_DOCUMENTS, &reference_pool[..=SENTENCES.len()]);
    let (split, digest) = split_of(&root);
    let (reference_split, _) = split_of(&reference_root);
    let candidate = POS.train(&split.train).unwrap();
    let reference = POS.train(&reference_split.train).unwrap();

    // One held-out sentence with a tag neither model gives.
    let odd = SENTENCES.len() + 1;
    rewrite_document(&root, &split.held_out_paths[0], &masc_pool, &sentences_after(&[odd]));
    let (split, _) = split_of(&root);
    let tags = POS.held_out(&split.held_out).unwrap();
    let run = Recipe::Pos(POS)
        .gate(&candidate.bytes, "reference", &reference.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    let coverage = run.held_out.tokens.unwrap();
    assert_eq!(coverage.tokens, tags.tokens());
    assert_eq!(coverage.left_out, [TagCount { tag: "UH`".to_owned(), tokens: 1 }]);
    let scored: u64 = run.record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(scored, tags.tokens() - 1);
    // The reference gives `-LRB-` and `:` where MASC has `(` and `-`, and
    // is right on every scored token, as the candidate is.
    assert_eq!(
        (run.record.result.candidate.score, run.record.result.reference.score),
        (100.0, 100.0),
        "{}",
        run.record
    );

    // Five such tokens are more than one in a thousand.
    rewrite_document(&root, &split.held_out_paths[0], &masc_pool, &sentences_after(&[odd; 5]));
    let (split, _) = split_of(&root);
    let tokens = POS.held_out(&split.held_out).unwrap().tokens();
    assert!((1000..5000).contains(&tokens), "{tokens}");
    let err = Recipe::Pos(POS)
        .gate(&candidate.bytes, "reference", &reference.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap_err();
    assert!(
        matches!(&err, Error::TagsOutsideReference { left_out, of, .. } if *of == tokens && left_out == &[TagCount { tag: "UH`".to_owned(), tokens: 5 }]),
        "{err:?}"
    );
    assert_eq!(
        err.to_string(),
        format!(
            "the reference reference cannot give the tags of 5 of {tokens} held-out tokens, more than the limit of \
             1 in 1000: UH` (5)"
        )
    );
    for dir in [root, reference_root] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

/// What the part-of-speech recipe reads from the MASC corpora checked in
/// for the goldens (`analysis/goldens/data/inputs/formats/masc-*.txt`,
/// written for the tests in MASC's layout), against what OpenNLP's
/// `MascPOSSampleStream` read from them in the JVM (the `masc/pos`
/// goldens). The goldens' generator reads on after a read throws, so a
/// golden's first pass records every sample the stream read and every
/// exception it threw, each a token without a tag
/// (`IllegalArgumentException`): the recipe reads the same samples in the
/// same order and skips one sentence for each exception. A corpus the
/// stream could not open (`masc-aborts`) is refused when it is read, with
/// the JVM's exception. The recipe's other skip, a sentence of no tokens,
/// is in none of these corpora; `skipped_sentences_are_recorded_and_bounded`
/// reads one.
#[test]
fn the_tagger_recipe_reads_the_jvm_s_samples_from_the_checked_in_masc_corpora() {
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
    // A layer's rows of the first pass, without the pass column, or every
    // row of a layer without one.
    let jvm = |name: &str, id: &str| -> Vec<Vec<String>> {
        let text = std::fs::read_to_string(goldens.join(format!("masc/pos/formats/{name}.golden"))).unwrap();
        let start = text.find(&format!("layer\t{id}\t")).unwrap_or_else(|| panic!("{name}: no {id}"));
        let body = &text[start..];
        let end = body.find(&format!("\nend\t{id}")).unwrap();
        let rows = body[..end].lines().skip(1).map(|l| l.split('\t').map(str::to_owned).collect::<Vec<_>>());
        if id == "turbo:open-errors" {
            rows.collect()
        } else {
            rows.filter(|r| r[0] == "0").map(|r| r[1..].to_vec()).collect()
        }
    };
    let write = |name: &str| {
        let root = temp_dir(&format!("checked-in-{name}"));
        let bundle = std::fs::read_to_string(goldens.join(format!("inputs/formats/{name}.txt"))).unwrap();
        bundle::write_bundle(&bundle, &root);
        root
    };
    let names = [
        "masc-basic",
        "masc-names",
        "masc-opennlp",
        "masc-penn",
        "masc-sentences",
        "masc-skips",
        "masc-unicode",
        "masc-xml",
    ];
    let mut skipped_any = 0;
    for name in names {
        let expected: Vec<String> = jvm(name, "turbo:samples").into_iter().map(|r| r[0].clone()).collect();
        let thrown = jvm(name, "turbo:read-errors");
        assert!(jvm(name, "turbo:open-errors").is_empty(), "{name}");
        for row in &thrown {
            assert_eq!(row[1..], ["java.lang.IllegalArgumentException", "null elements are not allowed in tags!"]);
        }
        let root = write(name);
        let corpus = MascCorpus::read(&root).unwrap();
        let mut read = Vec::new();
        let skipped = postag::read_samples(&corpus, |s| read.push(s.to_string())).unwrap();
        assert_eq!(read, expected, "{name}");
        assert_eq!(skipped, thrown.len(), "{name}");
        println!(
            "{name}: {} samples read, {skipped} sentences skipped; the JVM threw {} times",
            read.len(),
            thrown.len()
        );
        skipped_any += skipped;
        std::fs::remove_dir_all(&root).unwrap();
    }
    // masc-penn and masc-xml have sentences with a token without a tag.
    assert_eq!(skipped_any, 3);

    let opened = jvm("masc-aborts", "turbo:open-errors");
    assert_eq!(opened.len(), 1);
    let root = write("masc-aborts");
    let err = MascCorpus::read(&root).unwrap_err();
    assert_eq!(err.java_class(), Some(opened[0][0].as_str()), "{err}");
    assert_eq!(err.to_string(), opened[0][1]);
    std::fs::remove_dir_all(&root).unwrap();
}

/// Where a recipe's reference is, or `None` when it is not there and the
/// test may skip. The OpenNLP UD models come from `TURBO_GOLDENS_MODELS`
/// (`cargo run -p turbo-goldens-tool -- fetch-models <dir>` writes them),
/// and are required when `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, as CI sets
/// it. The OpenNLP 1.5 tagger comes from `TURBO_TEST_OPENNLP_MODELS` and
/// is local-only, as in turbo-sequence's tests (its training data is not
/// documented, so CI does not fetch it): without that variable it is
/// skipped even when the models are required, and with it a missing file
/// fails.
fn reference_path(recipe: &Recipe) -> Option<PathBuf> {
    let file = recipe.reference_file().unwrap();
    match recipe.reference() {
        Reference::Zoo(_) => {
            let Some(dir) = std::env::var_os("TURBO_GOLDENS_MODELS").map(PathBuf::from) else {
                assert!(!models_required(), "TURBO_GOLDENS_MODELS is not set and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1");
                eprintln!("TURBO_GOLDENS_MODELS is not set: skipping {file}");
                return None;
            };
            let path = dir.join(&file);
            if path.is_file() {
                return Some(path);
            }
            assert!(!models_required(), "{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
            eprintln!("{} not found: skipping", path.display());
            None
        }
        Reference::Recorded(_) => {
            let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS").map(PathBuf::from) else {
                eprintln!("TURBO_TEST_OPENNLP_MODELS is not set: skipping {file}");
                return None;
            };
            let path = dir.join(&file);
            assert!(path.is_file(), "TURBO_TEST_OPENNLP_MODELS is set and {} is not there", path.display());
            Some(path)
        }
    }
}

/// The English UD tagger asked for Penn tags returns its UD outcomes
/// converted, one Penn tag per UD tag: it never gives an inflected verb, a
/// plural noun or a comma, which is why it is not the tagger recipe's
/// reference (`postag`'s documentation).
#[test]
fn the_ud_tagger_asked_for_penn_tags_gives_one_penn_tag_per_ud_tag() {
    let ewt = Recipe::Pos(PosRecipe { reference: Reference::Zoo("opennlp-en-ud-ewt-pos-1.3-2.5.4"), ..POS });
    let Some(path) = reference_path(&ewt) else { return };
    let bytes = std::fs::read(&path).unwrap();
    ewt.check_reference(&bytes).unwrap();
    let file = turbo_maxent::ModelFile::open(bytes).unwrap();
    let model = turbo_postag::PosModel::open_with_format(file, PosTagFormat::Penn).unwrap();
    assert!(model.converts_tags());
    assert_eq!(
        labels(model.tagset()),
        [".", "?", "CC", "CD", "DT", "FW", "IN", "JJ", "MD", "NN", "NNP", "PRP", "RB", "RP", "SYM", "UH", "VB"]
    );
}

/// The tagger recipe's reference, asked for Penn tags, returns its own
/// outcomes: Penn tags, unconverted, among them the inflected verbs,
/// plural nouns and punctuation the UD tagger cannot give. It writes the
/// tags of `postag::REFERENCE_SPELLINGS` as that list says: brackets
/// `-LRB-` and `-RRB-`, hyphens, dashes, ellipses and semicolons `:`.
#[test]
fn the_penn_reference_returns_its_own_outcomes() {
    let Some(path) = reference_path(&Recipe::Pos(POS)) else { return };
    let bytes = std::fs::read(&path).unwrap();
    POS.reference.check(&bytes).unwrap();
    let file = turbo_maxent::ModelFile::open(bytes).unwrap();
    let model = turbo_postag::PosModel::open_with_format(file, POS.reference_format).unwrap();
    assert!(!model.converts_tags());
    assert_eq!(model.tag_format(), PosTagFormat::Penn);
    let tags = labels(model.tagset());
    let outcomes: Vec<&str> =
        (0..model.maxent().num_outcomes()).map(|o| model.maxent().outcome_str(o).unwrap()).collect();
    assert_eq!(tags.len(), outcomes.len());
    assert_eq!(
        tags,
        [
            "#", "$", "''", ",", "-LRB-", "-RRB-", ".", ":", "CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS",
            "LS", "MD", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO",
            "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "WDT", "WP", "WP$", "WRB", "``"
        ]
    );
    for spelling in postag::REFERENCE_SPELLINGS {
        assert!(!tags.contains(&spelling.masc) && tags.contains(&spelling.reference), "{spelling:?}");
    }
    let tagger = turbo_postag::PosTagger::new(model);
    let words = ["Wait", "...", "he", "said", "-", "no", ";", "then", "(", "yes", ")", "."];
    let (mut found, mut probs) = (Vec::new(), Vec::new());
    assert!(tagger.tag(&words, &mut found, &mut probs).unwrap());
    let given: Vec<&str> = found.iter().map(|&t| tagger.pos_model().tagset().label(t)).collect();
    for (word, masc) in [("...", "..."), ("-", "-"), (";", ";"), ("(", "("), (")", ")")] {
        let at = words.iter().position(|w| *w == word).unwrap();
        assert_eq!(given[at], postag::in_reference_spelling(masc), "{word} in {given:?}");
    }
}

fn labels(tagset: &turbo_analysis::TagSet) -> Vec<&str> {
    let mut labels: Vec<&str> = (0..tagset.len()).map(|i| tagset.label(turbo_analysis::Tag(i as u16))).collect();
    labels.sort_unstable();
    labels
}

/// The gate against the published OpenNLP models, those present (see
/// [`reference_path`]). The corpus is the test corpus, not MASC, and the
/// candidate has seen its held-out sentences, so the scores say nothing
/// about the recipes; the test checks that the published models load,
/// match their recorded digests and are scored on the same samples, and
/// that each scores what it scored when the test was written, so a change
/// in how a reference is read or scored shows.
#[test]
fn the_published_references_are_checked_and_scored_when_present() {
    let root = temp_dir("published");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, digest) = split_of(&root);
    for recipe in RECIPES {
        let Some(path) = reference_path(&recipe) else { continue };
        let reference = std::fs::read(&path).unwrap();
        recipe.check_reference(&reference).unwrap();
        let outcome = run(&recipe, &split, &digest, false, recipe.reference_id(), &reference).unwrap();
        outcome.gate.check().unwrap();
        assert_eq!(outcome.gate.reference.id, recipe.reference_id());
        println!("{}", outcome.gate);
        let expected = match recipe.task() {
            Task::SentenceDetector => "63.16",
            Task::Tokenizer => "94.34",
            Task::PartOfSpeechTagger => "92.41",
        };
        let scores = (outcome.gate.result.candidate.score, outcome.gate.result.reference.score);
        assert_eq!(format!("{:.2} {:.2}", scores.0, scores.1), format!("100.00 {expected}"), "{}", outcome.gate);
    }
    std::fs::remove_dir_all(&root).unwrap();
}

/// The binary reads and checks every reference a run needs, and checks
/// that none of its model directories exists, before it trains anything:
/// a run that fails on one writes nothing, and says what every problem
/// is, in the order the recipes run, `RECIPES`'s, whatever the order of
/// `--recipe`.
#[test]
fn the_binary_checks_every_reference_and_model_directory_before_training() {
    let root = temp_dir("binary-checks");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let out = temp_dir("binary-checks-out");
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    let models = out.join("models");
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let run = |extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("masc").arg("--corpus").arg(&root).arg("--references").arg(&refs).arg("--out").arg(&models);
        c.arg("--unrecorded-corpus").args(extra);
        let output = c.output().unwrap();
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        String::from_utf8_lossy(&output.stderr).into_owned()
    };
    let (sentence, tokens, pos) = (RECIPES[0], RECIPES[1], RECIPES[2]);
    let file = |r: Recipe| r.reference_file().unwrap();

    // No references: the two named are missing, in the recipes' order.
    let stderr = run(&["--recipe", pos.model_id(), "--recipe", sentence.model_id()]);
    let (at_sentence, at_pos) = (stderr.find(&file(sentence)).unwrap(), stderr.find(&file(pos)).unwrap());
    assert!(at_sentence < at_pos, "{stderr}");
    assert!(!stderr.contains(&file(tokens)), "{stderr}");
    assert!(!models.exists(), "nothing is written");

    // References that are not the recorded files, and a model directory
    // already there.
    for r in RECIPES {
        std::fs::write(refs.join(file(r)), b"not the model").unwrap();
    }
    std::fs::create_dir_all(models.join(tokens.model_id())).unwrap();
    let stderr = run(&[]);
    assert_eq!(stderr.lines().count(), 4, "{stderr}");
    for r in RECIPES {
        assert!(stderr.contains(&format!("reference model {}: SHA-256", r.reference_id())), "{stderr}");
    }
    assert!(stderr.contains(&format!("{}: the model directory exists", tokens.model_id())), "{stderr}");
    let written: Vec<_> = std::fs::read_dir(&models).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(written, [tokens.model_id()], "nothing else is written");
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn the_binary_refuses_an_unrecorded_corpus_and_writes_each_model_s_directory() {
    let root = temp_dir("binary");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let out = temp_dir("binary-out");
    // The references present, copied into one directory under the file
    // names the binary looks for.
    let refs = out.join("references");
    std::fs::create_dir_all(&refs).unwrap();
    let mut present = Vec::new();
    for r in RECIPES {
        if let Some(path) = reference_path(&r) {
            std::fs::copy(&path, refs.join(r.reference_file().unwrap())).unwrap();
            present.push(r);
        }
    }
    let bin = env!("CARGO_BIN_EXE_turbo-train-recipes");
    let args = |extra: &[&str]| {
        let mut c = std::process::Command::new(bin);
        c.arg("masc").arg("--corpus").arg(&root).arg("--references").arg(&refs).arg("--out").arg(out.join("models"));
        c.args(extra);
        c.output().unwrap()
    };
    let refused = args(&[]);
    assert_eq!(refused.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("not the recorded release's"), "{refused:?}");
    assert!(!out.join("models").exists());
    let unknown = args(&["--unrecorded-corpus", "--recipe", "turbo-en-masc-3.0.0-chunks-1"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("no recipe trains"), "{unknown:?}");

    if present.len() < RECIPES.len() {
        // Without every reference, a run of every recipe fails before it
        // trains or writes anything, and names every one missing.
        let all = args(&["--unrecorded-corpus"]);
        assert_eq!(all.status.code(), Some(2), "{all:?}");
        let stderr = String::from_utf8_lossy(&all.stderr);
        for r in RECIPES.iter().filter(|r| !present.contains(r)) {
            assert!(stderr.contains(&r.reference_file().unwrap()), "{stderr}");
        }
        assert!(!out.join("models").exists(), "a run without every reference wrote models");
    }
    if present.is_empty() {
        eprintln!("no published model is present: only the refusals were checked");
    } else {
        // The models trained on the test corpus have seen its held-out
        // sentences and beat the published models on it, so the gates
        // pass; that says nothing about the recipes on MASC.
        let mut extra = vec!["--unrecorded-corpus"];
        for r in &present {
            extra.extend(["--recipe", r.model_id()]);
        }
        let ran = args(&extra);
        assert_eq!(ran.status.code(), Some(0), "{ran:?}");
        for r in RECIPES {
            let dir = out.join("models").join(r.model_id());
            if !present.contains(&r) {
                assert!(!dir.exists(), "{} ran without --recipe", r.model_id());
                continue;
            }
            let card = ModelCard::read(&dir.join(CARD_FILE)).unwrap();
            assert!(!card.corpus.release_recorded);
            assert_eq!(card.gate.reference, r.reference_id());
            assert_eq!(card.gate.reference_sha256, r.reference().sha256().unwrap());
            ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap();
        }
    }
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}
