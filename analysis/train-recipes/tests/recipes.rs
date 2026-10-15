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

use common::{
    GATED_DOCUMENTS, SENTENCES, add_chunk, nest_token, rewrite_document, set_text_class, temp_dir, with_chunks_rotated,
    with_tags_rotated, write_corpus,
};
use turbo_eval::model_gate::{Measure, ModelGateRecord, Scoring, Task};
use turbo_formats::masc::{MascCorpus, MascTokenReader};
use turbo_postag::PosTagFormat;
use turbo_train_components::{TokSpanEventStream, TokenizerFactory};
use turbo_train_recipes::card::{
    CARD_FILE, CARD_VERSION, ChunksCard, GATE_FILE, HeldOutCard, ModelCard, TagCount, TagFormat, TokenCoverage,
    write_model_dir,
};
use turbo_train_recipes::chunk::{self, CHUNKS, SPAN_TYPES, Skipped};
use turbo_train_recipes::masc::{self, ExcludedDocument, Exclusion, RELEASE, SPLIT, SplitCorpus};
use turbo_train_recipes::postag::{self, POS, PosRecipe, ReferenceTags, Scored};
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

/// The split of the corpus at `root`, read with its chunk files, as the
/// binary reads it for the chunker, and its digest.
fn split_of(root: &std::path::Path) -> (SplitCorpus, masc::CorpusDigest) {
    let digest = masc::check_corpus(root, RELEASE, true).unwrap();
    let corpus = MascCorpus::read_with_chunks(root).unwrap();
    (SPLIT.apply(corpus, root).unwrap(), digest)
}

#[test]
fn the_split_holds_out_every_tenth_document_in_read_order() {
    let root = temp_dir("split");
    write_corpus(&root, 25, SENTENCES);
    let all: Vec<PathBuf> = MascCorpus::read(&root).unwrap().documents().iter().map(|d| d.path().to_owned()).collect();
    let (split, digest) = split_of(&root);
    // A header, the text, and the quark, sentence, Penn, noun chunk and
    // verb chunk files.
    assert_eq!(digest.files, 25 * 7);
    assert_eq!(split.train.documents().len(), 23);
    let held: Vec<String> =
        [9, 19].iter().map(|&i| all[i].strip_prefix(&root).unwrap().to_string_lossy().into_owned()).collect();
    assert_eq!(split.held_out_paths, held);
    assert!(split.held_out_paths.iter().all(|p| !p.starts_with('/')), "{:?}", split.held_out_paths);
    // The read order is the reader's: each directory's entries by name,
    // a directory's subdirectories after its files, the last pushed first.
    assert!(split.held_out_paths[0].starts_with("written/news/"), "{:?}", split.held_out_paths);

    // A document the read skips (here, its header names no sentence
    // file) would move the documents after it: the split refuses the
    // corpus.
    let header = root.join(&split.held_out_paths[1]);
    let xml = std::fs::read_to_string(&header).unwrap();
    let without = xml.lines().filter(|l| !l.contains("f.id=\"f.s\"")).collect::<Vec<_>>().join("\n");
    assert_ne!(without, xml);
    std::fs::write(&header, without).unwrap();
    let corpus = MascCorpus::read(&root).unwrap();
    assert_eq!(corpus.skipped().len(), 1);
    let err = SPLIT.apply(corpus, &root).unwrap_err();
    assert!(matches!(&err, Error::SkippedDocuments { documents } if documents.len() == 1), "{err:?}");
    assert!(err.to_string().contains(&split.held_out_paths[1]), "{err}");
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
    // give every tag but has learned the wrong one for nearly every word;
    // for the chunker, on the same sentences with their chunk tags moved
    // one token along, so its chunks start and end a token late.
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
    let shifted_root = temp_dir("shifted");
    let shifted_pool = with_chunks_rotated(SENTENCES);
    write_corpus(&shifted_root, 30, &shifted_pool.iter().map(String::as_str).collect::<Vec<_>>());
    let (shifted, _) = split_of(&shifted_root);

    let out = temp_dir("out");
    for recipe in RECIPES {
        let weak = match recipe {
            Recipe::Pos(_) => recipe.train(&rotated.train).unwrap(),
            Recipe::Segmentation(_) => recipe.train(&plain.train).unwrap(),
            Recipe::Chunk(_) => recipe.train(&shifted.train).unwrap(),
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
            Recipe::Pos(p) => {
                // The weak reference's mismatches, most frequent first.
                let mismatches = card.held_out.tokens.as_ref().unwrap().reference_mismatches.clone();
                assert!(!mismatches.is_empty() && mismatches.len() <= postag::MISMATCHES_LISTED);
                assert!(mismatches.windows(2).all(|w| w[0].tokens >= w[1].tokens), "{mismatches:?}");
                Some(TokenCoverage {
                    tokens: p.held_out(&split.held_out).unwrap().tokens(),
                    reference_mappings: postag::tag_mappings(),
                    left_out: Vec::new(),
                    reference_mismatches: mismatches,
                })
            }
            Recipe::Segmentation(_) | Recipe::Chunk(_) => None,
        };
        // The gate record holds the skipped sentences and the scoring too.
        assert_eq!(gate.set.skipped_sentences, 0);
        match (&gate.set.scoring, &tokens) {
            (Scoring::Tags(t), Some(c)) => {
                assert_eq!((t.tokens, t.left_out, &t.reference_mappings), (c.tokens, 0, &c.reference_mappings));
            }
            (Scoring::Spans, None) => assert!(matches!(recipe, Recipe::Segmentation(_))),
            (Scoring::TypedSpans(t), None) => {
                let held_out = CHUNKS.held_out(&split.held_out).unwrap();
                assert_eq!(t.types, ["NP", "VP"]);
                assert!(t.renamed.is_empty());
                assert_eq!((t.spans, t.left_out), (held_out.spans(), 0));
                assert!(t.spans > held_out.len() as u64, "{t:?}");
                // The reference is given `:` for MASC's `-`, `...` and `;`,
                // and the record says so.
                let inputs: Vec<(&str, &str)> =
                    t.reference_inputs.iter().map(|m| (m.held_out.as_str(), m.reference.as_str())).collect();
                assert_eq!(inputs, [("-", ":"), ("...", ":"), (";", ":")]);
                assert_eq!(t.reference_inputs, chunk::reference_inputs());
            }
            other => panic!("{other:?}"),
        }
        // The chunker's card counts the chunks in no sentence its reader
        // reads, on both sides: here every chunk is in a sentence.
        let (training_chunks, held_out_chunks) = match recipe {
            Recipe::Chunk(_) => {
                let (training, held_out) = (chunk::chunk_counts(&split.train), chunk::chunk_counts(&split.held_out));
                assert!(training.chunks > 0 && held_out.chunks > 0);
                assert_eq!(ChunksCard { chunks: 0, ..training.clone() }, ChunksCard::default());
                assert_eq!(ChunksCard { chunks: 0, ..held_out.clone() }, ChunksCard::default());
                (Some(training), Some(held_out))
            }
            _ => (None, None),
        };
        assert_eq!(card.training.chunks, training_chunks);
        assert_eq!(outcome.trained.chunks, training_chunks);
        let held_out =
            HeldOutCard { samples: gate.units.len(), skipped_sentences: 0, tokens, chunks: held_out_chunks.clone() };
        assert_eq!(card.held_out, held_out);
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
            Recipe::Chunk(_) => {
                assert_eq!(card.training.whitespace, None);
                assert_eq!((card.gate.candidate_tag_format, card.gate.reference_tag_format), (None, None));
            }
        }
        assert_eq!(card.training.parameters, recipe.parameters());

        // What the binary prints of the card, with the counts of a run.
        let mut counted = card.clone();
        (counted.training.samples, counted.training.skipped_sentences, counted.training.events) = (27149, 1, 1495034);
        (counted.held_out.samples, counted.held_out.skipped_sentences) = (6384, 2);
        let trained = format!(
            "{}: trained on 27149 samples, 1 sentences skipped, 1495034 events; held out 6384 samples, 2 sentences \
             skipped",
            recipe.model_id()
        );
        if let (Some(training), Some(held_out)) = (&mut counted.training.chunks, &mut counted.held_out.chunks) {
            (training.chunks, training.outside_sentences, training.across_sentences, training.without_tokens) =
                (183681, 13509, 83, 2);
            (held_out.chunks, held_out.outside_sentences, held_out.across_sentences, held_out.without_tokens) =
                (38250, 3088, 12, 0);
            let id = recipe.model_id();
            let lines = format!(
                "{trained}\n{id}: of 183681 training chunks, 13509 outside every kept sentence, 83 across \
                 sentences, 2 without tokens\n{id}: of 38250 held-out chunks, 3088 outside every kept sentence, 12 \
                 across sentences, 0 without tokens"
            );
            assert_eq!(counted.summary().to_string(), lines);
        } else if counted.held_out.tokens.is_none() {
            assert_eq!(counted.summary().to_string(), trained);
        } else {
            let id = recipe.model_id();
            fn coverage(card: &mut ModelCard) -> &mut TokenCoverage {
                card.held_out.tokens.as_mut().unwrap()
            }
            coverage(&mut counted).tokens = 85292;
            let none = format!("{id}: 0 of 85292 held-out tokens left out, their tags not the reference's: none");
            assert_eq!(counted.summary().to_string(), format!("{trained}\n{none}"));
            coverage(&mut counted).left_out =
                vec![TagCount { tag: "\"".to_owned(), tokens: 2 }, TagCount { tag: "UH`".to_owned(), tokens: 3 }];
            let some =
                format!("{id}: 5 of 85292 held-out tokens left out, their tags not the reference's: \" (2), UH` (3)");
            assert_eq!(counted.summary().to_string(), format!("{trained}\n{some}"));
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
        // Nor is a card of version 3, without the chunk counts.
        let mut three: serde_json::Value = serde_json::from_str(&card.to_json()).unwrap();
        three["card_version"] = 3.into();
        three["training"].as_object_mut().unwrap().remove("chunks");
        three["held_out"].as_object_mut().unwrap().remove("chunks");
        std::fs::write(&other, serde_json::to_string_pretty(&three).unwrap()).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::CardVersion { found: 3, .. })));
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
        unknown.id = "turbo-en-masc-3.0.0-unknown-1".to_owned();
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
    for dir in [root, plain_root, rotated_root, shifted_root, out] {
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
        assert_eq!(gated.record.set.skipped_sentences, 1);
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

/// The Penn number of the first token of each sentence of a document
/// written of `sentences` of `pool`, as `write_document` numbers them.
fn sentence_starts(pool: &[&str], sentences: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(sentences.len());
    let mut q = 0;
    for &i in sentences {
        starts.push(q);
        q += common::tokens(pool[i]).len();
    }
    starts
}

/// The chunker's reader skips a sentence with chunks that chunk tags
/// cannot express: one with part of a chunk that crosses into another sentence,
/// or with two chunks over a token. The same chunk given twice is one
/// chunk. The gate run records the skipped sentences and the chunks in no
/// sentence, and a recipe fails when either side skips more than one in
/// ten, before any training, or on the recorded release skips other
/// counts than the recorded ones.
#[test]
fn sentences_whose_chunks_cross_or_overlap_are_skipped_by_the_chunker_and_bounded() {
    let root = temp_dir("chunk-skips");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, _) = split_of(&root);
    let held = split.held_out_paths[0].clone();
    let sentences = sentences_after(&[0, 1]);
    rewrite_document(&root, &held, SENTENCES, &sentences);
    let starts = sentence_starts(SENTENCES, &sentences);
    assert_eq!(&starts[..3], [0, 15, 20]);
    // `Dr. Smith` again as a noun chunk
    add_chunk(&root, &held, "nc", 0, &[0, 1]);
    let (split, _) = split_of(&root);
    let chunks = CHUNKS.held_out(&split.held_out).unwrap();
    assert_eq!((chunks.len(), chunks.skipped().total()), (320, 0));
    let read = chunks.chunks().chunks;
    assert_eq!(chunks.chunks(), &ChunksCard { chunks: read, ..ChunksCard::default() });

    // a noun chunk from the first sentence's last token into the second,
    // and a verb chunk over `Smith said` in the third
    add_chunk(&root, &held, "nc", 1, &[14, 15]);
    add_chunk(&root, &held, "vc", 2, &[21, 22]);
    // a noun chunk of a Penn token no sentence has, and a verb chunk of
    // no token: in no sentence, so no sentence is skipped for them
    add_chunk(&root, &held, "nc", 3, &[99_999]);
    add_chunk(&root, &held, "vc", 4, &[]);
    let (split, digest) = split_of(&root);
    let chunks = CHUNKS.held_out(&split.held_out).unwrap();
    assert_eq!(chunks.skipped(), Skipped { partial_chunk: 2, overlapping_chunks: 1, ..Skipped::default() });
    assert_eq!(chunks.len(), 317);
    let left_out = ChunksCard { chunks: read + 4, outside_sentences: 1, across_sentences: 1, without_tokens: 1 };
    assert_eq!(chunks.chunks(), &left_out);
    let trained = CHUNKS.train(&split.train).unwrap();
    assert_eq!(trained.chunks, Some(chunk::chunk_counts(&split.train)));
    let gated = Recipe::Chunk(CHUNKS)
        .gate(&trained.bytes, "itself", &trained.bytes, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    assert_eq!((gated.held_out.samples, gated.held_out.skipped_sentences), (317, 3));
    assert_eq!((gated.record.set.skipped_sentences, gated.record.units.len()), (3, 317));
    assert_eq!(gated.held_out.chunks, Some(left_out));

    // On the recorded release the sentences skipped must be the counts
    // the recipe records (1,872 training, 349 held out), checked before
    // training and after it; on another corpus only the bound applies.
    let e = Recipe::Chunk(CHUNKS).check_data(&split, true, "reference", None).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::TrainingSentences, count: 0, recorded: 1872 }),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "0 training sentences skipped, not the recorded release's 1872");
    Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).unwrap();
    let e = run(&Recipe::Chunk(CHUNKS), &split, &digest, true, "itself", &trained.bytes).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::TrainingSentences, count: 0, recorded: 1872 }),
        "{e:?}"
    );
    chunk::check_recorded_skips(LeftOut::TrainingSentences, 1872).unwrap();
    chunk::check_recorded_skips(LeftOut::HeldOutSentences, 349).unwrap();
    let e = chunk::check_recorded_skips(LeftOut::HeldOutSentences, 350).unwrap_err();
    assert!(
        matches!(e, Error::SkippedNotRecorded { what: LeftOut::HeldOutSentences, count: 350, recorded: 349 }),
        "{e:?}"
    );
    assert_eq!(e.to_string(), "350 held-out sentences skipped, not the recorded release's 349");
    assert!(chunk::check_recorded_skips(LeftOut::TrainingSentences, 1871).is_err());
    Recipe::Chunk(CHUNKS).check_recorded_skips(false, LeftOut::TrainingSentences, 0).unwrap();
    for recipe in RECIPES.into_iter().filter(|r| !matches!(r, Recipe::Chunk(_))) {
        recipe.check_recorded_skips(true, LeftOut::TrainingSentences, 0).unwrap();
    }

    // A verb chunk over the first two tokens of each sentence of another
    // held-out document: 43 of 320 is more than one in ten.
    let other = split.held_out_paths[1].clone();
    let doc = split.held_out.documents()[1].path().to_owned();
    assert!(doc.ends_with(&other));
    let written: Vec<usize> = (0..common::SENTENCES_PER_DOCUMENT).map(|k| k % SENTENCES.len()).collect();
    rewrite_document(&root, &other, SENTENCES, &written);
    for (k, &q) in sentence_starts(SENTENCES, &written).iter().enumerate() {
        add_chunk(&root, &other, "vc", 10 + k, &[q, q + 1]);
    }
    let (split, _) = split_of(&root);
    let limited = |err: Option<Error>, side: LeftOut, count: u64, of: u64| match err {
        Some(Error::TooManyLeftOut { what, count: c, of: o, limit }) => {
            assert_eq!((what, c, o, limit), (side, count, of, chunk::MAX_SKIPPED_SENTENCES));
        }
        other => panic!("{other:?}"),
    };
    limited(CHUNKS.held_out(&split.held_out).err(), LeftOut::HeldOutSentences, 43, 320);
    let message = CHUNKS.held_out(&split.held_out).unwrap_err().to_string();
    assert_eq!(message, "43 of 320 held-out sentences skipped, more than the limit of 1 in 10");
    // The check made before training finds it too.
    let checked = Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).unwrap_err();
    assert_eq!(checked.to_string(), message);

    // Training sentences are bounded the same way, before any training:
    // every sentence of eight of the 72 training documents is skipped.
    let training: Vec<String> = split.train.documents()[..8]
        .iter()
        .map(|d| d.path().strip_prefix(&root).unwrap().to_string_lossy().into_owned())
        .collect();
    for doc in &training {
        rewrite_document(&root, doc, SENTENCES, &written);
        for (k, &q) in sentence_starts(SENTENCES, &written).iter().enumerate() {
            add_chunk(&root, doc, "vc", 10 + k, &[q, q + 1]);
        }
    }
    let (split, _) = split_of(&root);
    assert_eq!(split.train.documents().len() * common::SENTENCES_PER_DOCUMENT, 2880);
    limited(CHUNKS.train(&split.train).err(), LeftOut::TrainingSentences, 320, 2880);
    limited(CHUNKS.check_training(&split.train).err(), LeftOut::TrainingSentences, 320, 2880);
    std::fs::remove_dir_all(&root).unwrap();
}

/// The chunker's reference gives chunk types MASC does not annotate; its
/// tags of those types are made `O` before it is scored, so its `NP` and
/// `VP` chunks are scored as they are and the others count neither for
/// nor against it. Here the reference is the recipe trained on the same
/// samples with every `IN` token a `PP` chunk, as CoNLL-2000 has them:
/// scored, it is right on every chunk, as the candidate is, where its
/// `PP` chunks would count as wrong predictions if they were kept.
#[test]
fn the_reference_s_chunks_of_types_the_gate_does_not_score_are_left_out() {
    use turbo_formats::evaluate::ChunkerEvaluator;
    use turbo_formats::sample::ChunkSample;
    use turbo_formats::source::Field;

    let root = temp_dir("chunk-types");
    write_corpus(&root, GATED_DOCUMENTS, SENTENCES);
    let (split, digest) = split_of(&root);
    let candidate = CHUNKS.train(&split.train).unwrap();
    let mut samples = Vec::new();
    chunk::read_samples(&split.train, |s| samples.push(s)).unwrap();
    let with_pp: Vec<ChunkSample<'_>> = samples
        .iter()
        .map(|s| {
            let chunks = s
                .tags()
                .iter()
                .zip(s.chunks())
                .map(|(tag, c)| if tag.as_str() == "IN" { Field::made("B-PP".to_owned(), c.span) } else { c.clone() })
                .collect();
            ChunkSample::new(s.tokens().to_vec(), s.tags().to_vec(), chunks, s.source()).unwrap()
        })
        .collect();
    let factory = turbo_train_components::ChunkerFactory::new("en");
    let mut config = turbo_train::TrainingConfiguration::default();
    let params = chunk::training_parameters();
    let stamp = turbo_train_components::Stamp::at(turbo_train_recipes::segmentation::TIMESTAMP_MS);
    let reference = turbo_train_components::chunk::train_chunker_model(&with_pp, &factory, &params, &mut config, stamp)
        .unwrap()
        .package
        .to_bytes();

    let run = Recipe::Chunk(CHUNKS)
        .gate(&candidate.bytes, "with-pp", &reference, &split.held_out, &held_out_name(), &digest.sha256)
        .unwrap();
    assert_eq!((run.record.result.candidate.score, run.record.result.reference.score), (100.0, 100.0));
    let selected = |counts: &[turbo_eval::model_gate::Counts]| counts.iter().map(|c| c.selected).sum::<u64>();
    let candidate_selected = selected(&run.record.units.iter().map(|u| u.candidate).collect::<Vec<_>>());
    let reference_selected = selected(&run.record.units.iter().map(|u| u.reference).collect::<Vec<_>>());
    assert_eq!(candidate_selected, reference_selected);

    // Kept, its `PP` chunks are predictions no held-out chunk matches.
    let held_out = CHUNKS.held_out(&split.held_out).unwrap();
    let chunker = turbo_phrase_chunker::PhraseChunker::new(
        turbo_phrase_chunker::ChunkerModel::open(turbo_maxent::ModelFile::open(reference).unwrap()).unwrap(),
    );
    let mut evaluator = ChunkerEvaluator::default();
    let mut pp = 0;
    for sample in held_out.samples() {
        let words: Vec<turbo_sequence::chunker::TokenTag<'_>> = sample
            .tokens()
            .iter()
            .zip(sample.tags())
            .map(|(t, tag)| turbo_sequence::chunker::TokenTag { token: t.as_str(), tag: tag.as_str() })
            .collect();
        let (mut outcomes, mut probs) = (Vec::new(), Vec::new());
        assert!(chunker.outcomes(&words, &mut outcomes, &mut probs));
        pp += outcomes.iter().filter(|o| **o == "B-PP").count() as u64;
        evaluator.evaluate(sample, &outcomes).unwrap();
    }
    assert!(pp > 0);
    assert_eq!(evaluator.fmeasure().selected(), reference_selected + pp);
    assert!(evaluator.fmeasure().f_measure() < 1.0);
    std::fs::remove_dir_all(&root).unwrap();
}

/// A corpus read without its chunk files has no chunk sample: the
/// chunker's reader refuses it, and so does the check made before
/// training.
#[test]
fn the_chunker_refuses_a_corpus_read_without_its_chunk_files() {
    let root = temp_dir("chunk-files");
    write_corpus(&root, 20, SENTENCES);
    let split = SPLIT.apply(MascCorpus::read(&root).unwrap(), &root).unwrap();
    let err = CHUNKS.held_out(&split.held_out).unwrap_err();
    assert!(matches!(err, Error::Masc(turbo_formats::masc::MascError::NoChunks)), "{err:?}");
    assert!(Recipe::Chunk(CHUNKS).check_data(&split, false, "reference", None).is_err());
    assert!(CHUNKS.train(&split.train).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// The documents of MASC's Twitter genre, those whose header's
/// `textClass` names the category `TW`, are left out of both sides
/// wherever the split puts them; every other document keeps its side,
/// and the card and the gate record say what was left out.
#[test]
fn the_twitter_genre_is_left_out_of_both_sides_and_moves_no_other_document() {
    let root = temp_dir("twitter");
    // 90 documents: 9 held out, 8 once one is left out, 32 sentence
    // detector samples.
    write_corpus(&root, 90, SENTENCES);
    let corpus = || MascCorpus::read(&root).unwrap();
    let rel = |p: &std::path::Path| p.strip_prefix(&root).unwrap().to_string_lossy().into_owned();
    let all: Vec<String> = corpus().documents().iter().map(|d| rel(d.path())).collect();
    let (tweets_trained, tweets_held_out) = (all[0].clone(), all[19].clone());
    set_text_class(&root, &tweets_trained, "WR TW");
    set_text_class(&root, &tweets_held_out, "WR TW");
    set_text_class(&root, &all[1], "WR BL");
    set_text_class(&root, &all[9], "WR TWX");

    let kept = SPLIT.apply_excluding(corpus(), &root, &[]).unwrap();
    assert!(kept.excluded.is_empty());
    let (split, digest) = split_of(&root);
    let excluded = |path: &str| ExcludedDocument { path: path.to_owned(), exclusion: Exclusion::TwitterGenre };
    assert_eq!(split.excluded, [excluded(&tweets_trained), excluded(&tweets_held_out)]);
    let paths = |c: &MascCorpus| c.documents().iter().map(|d| rel(d.path())).collect::<Vec<_>>();
    let without = |paths: Vec<String>, left_out: &str| paths.into_iter().filter(|p| p != left_out).collect::<Vec<_>>();
    assert_eq!(paths(&split.train), without(paths(&kept.train), &tweets_trained));
    assert_eq!(split.held_out_paths, without(kept.held_out_paths.clone(), &tweets_held_out));
    assert_eq!((split.train.documents().len(), split.held_out.documents().len()), (80, 8));

    let trained = SENTENCE.train(&split.train).unwrap();
    let outcome = run(&Recipe::Segmentation(SENTENCE), &split, &digest, false, "itself", &trained.bytes).unwrap();
    assert_eq!(outcome.card.corpus.excluded_documents, split.excluded);
    assert_eq!(outcome.card.corpus.held_out_documents, split.held_out_paths);
    assert_eq!(
        outcome.gate.set.name,
        "MASC 3.0.0 held out (every 10th document from the 10th, the Twitter genre left out)"
    );
    let json = outcome.card.to_json();
    assert!(
        json.contains(&format!("\"path\": \"{tweets_held_out}\",\n        \"exclusion\": \"twitter_genre\"")),
        "{json}"
    );
    std::fs::remove_dir_all(&root).unwrap();
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
    assert!(matches!(POS.score(&model.bytes, Scored::Candidate, &tags, &penn), Err(Error::Model(_))));
    assert!(matches!(POS.score(&model.bytes, Scored::Reference, &tags, &penn), Err(Error::Model(_))));
    let tagger = POS.train(&split.train).unwrap();
    assert!(TOKENS.score(&tagger.bytes, &tokens_again).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// A sentence tagged as MASC tags brackets, a hyphen, an ellipsis, a
/// semicolon and an opening quotation mark, and the same sentence as the
/// reference tags them.
const MASC_SPELLED: &str =
    "Wait/VB .../... he/PRP said/VBD -/- no/DT ~;/; then/RB (/( yes/UH ~)/) and/CC \"/'' ~fine/JJ ~\"/'' ~./.";
const REFERENCE_SPELLED: &str =
    "Wait/VB .../: he/PRP said/VBD -/: no/DT ~;/: then/RB (/-LRB- yes/UH ~)/-RRB- and/CC \"/`` ~fine/JJ ~\"/'' ~./.";
/// A tag no Penn tagger gives, as MASC 3.0.0 has a few.
const ODD_TAG: &str = "Agh/UH` ~!/.";

/// The tagger's gate scores the held-out tokens whose tag the reference
/// can give, as written or through the mapping of its tags: a candidate
/// trained on MASC's tags is compared with the held-out tags as written,
/// and a reference trained on the same sentences in its own conventions
/// through the mapping, so both are right on every token; the reference's
/// tags are not counted right for the candidate. The token of a tag
/// neither can give is left out and counted, and more such tokens than
/// one in a thousand are an error that names their tags.
#[test]
fn the_tagger_is_scored_over_the_tags_the_reference_can_give_mapping_only_the_reference_s() {
    let pool = |extra: &'static str| {
        let mut pool = SENTENCES.to_vec();
        pool.extend([extra, ODD_TAG]);
        pool
    };
    let (masc_pool, reference_pool) = (pool(MASC_SPELLED), pool(REFERENCE_SPELLED));
    let root = temp_dir("mappings");
    write_corpus(&root, GATED_DOCUMENTS, &masc_pool[..=SENTENCES.len()]);
    let reference_root = temp_dir("mappings-reference");
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
    assert_eq!(coverage.reference_mismatches, []);
    let scored: u64 = run.record.units.iter().map(|u| u.reference.target).sum();
    assert_eq!(scored, tags.tokens() - 1);
    // The reference gives `-LRB-`, `:` and ``` `` ``` where MASC has `(`, `-`
    // and `''`, and is right on every scored token, as the candidate is.
    assert_eq!(
        (run.record.result.candidate.score, run.record.result.reference.score),
        (100.0, 100.0),
        "{}",
        run.record
    );
    // Scored as the candidate, the reference's tags are compared as
    // written, and its `-LRB-`, `:` and ``` `` ``` are wrong.
    let reference_tags = POS.reference_tags(&reference.bytes).unwrap();
    let as_candidate: u64 = POS
        .score(&reference.bytes, Scored::Candidate, &tags, &reference_tags)
        .unwrap()
        .iter()
        .map(|c| c.true_positives)
        .sum();
    let mapped: u64 = POS
        .score(&reference.bytes, Scored::Reference, &tags, &reference_tags)
        .unwrap()
        .iter()
        .map(|c| c.true_positives)
        .sum();
    // Each held-out copy of the sentence has six such tokens: the
    // brackets, the hyphen, the ellipsis, the semicolon and the opening
    // quotation mark (the closing one is `''` in both).
    let copies = tags.samples().iter().filter(|s| s.tokens()[0].as_str() == "Wait").count() as u64;
    assert!(copies > 0);
    assert_eq!(mapped, scored);
    assert_eq!(mapped - as_candidate, 6 * copies);

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
    // The check made before training finds the same, given the
    // reference; without it, it checks only the samples.
    let checked = Recipe::Pos(POS).check_data(&split, false, "reference", Some(&reference.bytes)).unwrap_err();
    assert_eq!(checked.to_string(), err.to_string());
    Recipe::Pos(POS).check_data(&split, false, "reference", None).unwrap();
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
/// it. The OpenNLP 1.5 tagger and chunker come from
/// `TURBO_TEST_OPENNLP_MODELS` and are local-only, as in turbo-sequence's
/// tests (their training data is not published, so CI does not fetch
/// them): without that variable they are skipped even when the models are
/// required, and with it a missing file fails.
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
/// plural nouns and punctuation the UD tagger cannot give. It tags the
/// tokens of `postag::REFERENCE_MAPPINGS` as that list says: brackets
/// `-LRB-` and `-RRB-`, hyphens, ellipses and semicolons `:`, and an
/// opening quotation mark ``` `` ```.
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
    for mapping in postag::REFERENCE_MAPPINGS {
        // The held-out tag is one the reference does not give, but for
        // `''`, which it gives a closing quotation mark.
        assert_eq!(tags.contains(&mapping.held_out), mapping.held_out == "''", "{mapping:?}");
        assert!(tags.contains(&mapping.reference), "{mapping:?}");
    }
    let tagger = turbo_postag::PosTagger::new(model);
    let words = ["Wait", "...", "he", "said", "-", "no", ";", "then", "(", "yes", ")", "and", "\"", "fine", "\"", "."];
    let (mut found, mut probs) = (Vec::new(), Vec::new());
    assert!(tagger.tag(&words, &mut found, &mut probs).unwrap());
    let given: Vec<&str> = found.iter().map(|&t| tagger.pos_model().tagset().label(t)).collect();
    for (at, held_out) in [(1, "..."), (4, "-"), (6, ";"), (8, "("), (10, ")"), (12, "''")] {
        let mapping = postag::REFERENCE_MAPPINGS.iter().find(|m| m.held_out == held_out).unwrap();
        assert_eq!(given[at], mapping.reference, "{} in {given:?}", words[at]);
    }
    assert_eq!(given[14], "''", "{given:?}");
}

/// The chunker recipe's reference gives CoNLL-2000's chunk types, of
/// which the gate scores `NP` and `VP`, and its features name the tags of
/// its training data: brackets `(` and `)`, as MASC writes them, and `:`,
/// and none of the three tags `chunk::REFERENCE_TAGS` maps to `:`.
#[test]
fn the_chunk_reference_gives_conll_types_and_knows_the_tags_it_is_given() {
    let Some(path) = reference_path(&Recipe::Chunk(CHUNKS)) else { return };
    let bytes = std::fs::read(&path).unwrap();
    CHUNKS.reference.check(&bytes).unwrap();
    let model = turbo_phrase_chunker::ChunkerModel::open(turbo_maxent::ModelFile::open(bytes).unwrap()).unwrap();
    let maxent = model.maxent();
    let mut outcomes: Vec<&str> = (0..maxent.num_outcomes()).map(|o| maxent.outcome_str(o).unwrap()).collect();
    outcomes.sort_unstable();
    let mut types: Vec<&str> = outcomes.iter().filter_map(|o| o.strip_prefix("B-")).collect();
    types.dedup();
    assert_eq!(types, ["ADJP", "ADVP", "CONJP", "INTJ", "LST", "NP", "PP", "PRT", "SBAR", "UCP", "VP"]);
    assert!(outcomes.contains(&"O"));
    for t in SPAN_TYPES.types {
        assert!(types.contains(t), "{t}");
    }
    let knows = |tag: &str| maxent.predicate_index(format!("t0={tag}").as_bytes()).is_some();
    for tag in ["(", ")", ":", "''", "``", "NN", "VBZ"] {
        assert!(knows(tag), "{tag}");
    }
    for m in chunk::REFERENCE_TAGS {
        assert!(!knows(m.held_out) && knows(m.reference), "{m:?}");
    }
    assert!(!knows("-LRB-"));
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
            Task::Chunker => "93.33",
            task @ Task::NameFinder => panic!("no recipe of the task {task:?}"),
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
fn the_binary_checks_every_reference_model_directory_and_recipe_s_samples_before_training() {
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
    assert_eq!(stderr.lines().count(), RECIPES.len() + 1, "{stderr}");
    for r in RECIPES {
        assert!(stderr.contains(&format!("reference model {}: SHA-256", r.reference_id())), "{stderr}");
    }
    assert!(stderr.contains(&format!("{}: the model directory exists", tokens.model_id())), "{stderr}");
    let written: Vec<_> = std::fs::read_dir(&models).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(written, [tokens.model_id()], "nothing else is written");

    // A held-out sentence of no tokens, which the token, tagger and
    // chunker recipes skip: one of 320 is more than one in a thousand,
    // the token and tagger recipes are named with the other problems,
    // before anything is trained, and the chunker, whose limit is one in
    // ten, is not.
    let (split, _) = split_of(&root);
    let no_tokens = SENTENCES.len() + 1;
    rewrite_document(&root, &split.held_out_paths[0], &with_skipped_sentences(), &sentences_after(&[no_tokens]));
    let stderr = run(&[]);
    assert_eq!(stderr.lines().count(), RECIPES.len() + 3, "{stderr}");
    for r in [tokens, pos] {
        let line = format!("{}: 1 of 320 held-out sentences skipped, more than the limit of 1 in 1000", r.model_id());
        assert!(stderr.contains(&line), "{stderr}");
    }
    for r in [sentence, RECIPES[3]] {
        assert!(!stderr.contains(&format!("{}: 1 of", r.model_id())), "{stderr}");
    }
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
    let unknown = args(&["--unrecorded-corpus", "--recipe", "turbo-en-masc-3.0.0-unknown-1"]);
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
            let stdout = String::from_utf8_lossy(&ran.stdout);
            let gate = ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap();
            assert!(stdout.contains(&format!("{}\n{gate}\n", card.summary())), "{stdout}");
            assert!(!card.corpus.release_recorded);
            assert_eq!(card.gate.reference, r.reference_id());
            assert_eq!(card.gate.reference_sha256, r.reference().sha256().unwrap());
        }
    }
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}
