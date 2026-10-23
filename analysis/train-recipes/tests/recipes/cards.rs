//! The recipes trained, scored, gated and written to their cards.

use crate::common::{
    GATED_DOCUMENTS, SENTENCES, temp_dir, with_chunks_rotated, with_names_shifted, with_tags_rotated, write_corpus,
};
use crate::split_of;
use crate::zoo::{Released, released_models_make_the_release_s_zoo_entries};
use turbo_eval::model_gate::{Measure, ModelGateRecord, ModelId, Scoring, Task};
use turbo_train_recipes::card::{
    CARD_FILE, CARD_VERSION, ChunksCard, GATE_FILE, HeldOutCard, ModelCard, NamesCard, TagCount, TagFormat,
    TokenCoverage, write_model_dir,
};
use turbo_train_recipes::chunk::{self, CHUNKS};
use turbo_train_recipes::masc;
use turbo_train_recipes::postag::{self, POS};
use turbo_train_recipes::{Error, RECIPES, Recipe, held_out_name, run};
use turbo_zoo::{Index, LicenceClass};

/// The pool without abbreviations, attached punctuation or contractions:
/// a model trained on it has not seen what the held-out text holds.
const PLAIN: &[&str] = &["Who called ~? Nobody did ~!", "Mr. Jones left early ~.", "It rained all day ~."];

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
    // one token along, so its chunks start and end a token late; for a
    // name finder, on the same sentences with each name moved one token
    // along.
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
    let shifted_pool = with_names_shifted(&shifted_pool.iter().map(String::as_str).collect::<Vec<_>>());
    write_corpus(&shifted_root, 30, &shifted_pool.iter().map(String::as_str).collect::<Vec<_>>());
    let (shifted, _) = split_of(&shifted_root);

    let out = temp_dir("out");
    let mut released_models = Vec::new();
    for recipe in RECIPES {
        let weak = match recipe {
            Recipe::Pos(_) => recipe.train(&rotated.train).unwrap(),
            Recipe::Segmentation(_) => recipe.train(&plain.train).unwrap(),
            Recipe::Chunk(_) | Recipe::Name(_) => recipe.train(&shifted.train).unwrap(),
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
                    reference_mappings: postag::tag_mappings(&postag::REFERENCE_MAPPINGS),
                    left_out: Vec::new(),
                    reference_mismatches: mismatches,
                })
            }
            Recipe::Segmentation(_) | Recipe::Chunk(_) | Recipe::Name(_) => None,
        };
        let names = match recipe {
            Recipe::Name(n) => Some(n.held_out(&split.held_out).unwrap().counts().clone()),
            Recipe::Segmentation(_) | Recipe::Pos(_) | Recipe::Chunk(_) => None,
        };
        // The gate record holds the skipped sentences and the scoring too.
        assert_eq!(gate.set.skipped_sentences, 0);
        match (&gate.set.scoring, &tokens, &names, recipe) {
            (Scoring::Tags(t), Some(c), None, Recipe::Pos(_)) => {
                assert_eq!((t.tokens, t.left_out, &t.reference_mappings), (c.tokens, 0, &c.reference_mappings));
            }
            (Scoring::Spans, None, None, Recipe::Segmentation(_)) => {}
            (Scoring::TypedSpans(t), None, None, Recipe::Chunk(_)) => {
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
            (Scoring::TypedSpans(t), None, Some(n), Recipe::Name(r)) => {
                assert_eq!(t.types, [r.kind()]);
                assert_eq!((t.spans, t.left_out), (n.names, n.other_types));
                assert!(n.names > 0 && n.other_types > 0, "{n:?}");
                assert_eq!(n.documents, split.held_out.documents().len());
                assert_eq!(card.training.names.as_ref().unwrap().documents, split.train.documents().len());
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
        let held_out = HeldOutCard {
            samples: gate.units.len(),
            skipped_sentences: 0,
            tokens,
            names,
            chunks: held_out_chunks.clone(),
        };
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
            Recipe::Chunk(_) | Recipe::Name(_) => {
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
        if let (Some(training), Some(held_out)) = (&mut counted.training.names, &mut counted.held_out.names) {
            *training = NamesCard {
                documents: 351,
                names: 2934,
                gapped: 1,
                overlapping: 36,
                other_types: 11622,
                outside_sentences: 192,
                in_dropped_sentences: 143,
            };
            *held_out = NamesCard {
                documents: 39,
                names: 321,
                gapped: 0,
                overlapping: 0,
                other_types: 2157,
                outside_sentences: 18,
                in_dropped_sentences: 25,
            };
            let id = recipe.model_id();
            let names = format!(
                "{id}: 2934 names in the training samples of 351 documents, 1 with a gap; 36 overlapping and 11622 of \
                 other types left out; 192 outside every sentence region and 143 in sentence regions the reader \
                 drops, in no sample\n{id}: 321 names in the held-out samples of 39 documents, 0 with a gap; 0 \
                 overlapping and 2157 of other types left out; 18 outside every sentence region and 25 in sentence \
                 regions the reader drops, in no sample"
            );
            assert_eq!(counted.summary().to_string(), format!("{trained}\n{names}"));
        } else if let (Some(training), Some(held_out)) = (&mut counted.training.chunks, &mut counted.held_out.chunks) {
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
        // Nor is a card of version 4, with the chunk counts or the name
        // counts but not both.
        for kept in ["chunks", "names"] {
            let mut four: serde_json::Value = serde_json::from_str(&card.to_json()).unwrap();
            four["card_version"] = 4.into();
            let dropped = if kept == "chunks" { "names" } else { "chunks" };
            four["training"].as_object_mut().unwrap().remove(dropped);
            four["held_out"].as_object_mut().unwrap().remove(dropped);
            std::fs::write(&other, serde_json::to_string_pretty(&four).unwrap()).unwrap();
            assert!(matches!(ModelCard::read(&other), Err(Error::CardVersion { found: 4, .. })));
        }
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
        let mut released_gate = outcome.gate.clone();
        released_gate.reference =
            ModelId { id: recipe.reference_id().to_owned(), sha256: recipe.reference().sha256().unwrap() };
        released_models.push(Released {
            card: released.clone(),
            gate: released_gate,
            model: outcome.trained.bytes.clone(),
        });
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
    released_models_make_the_release_s_zoo_entries(&out.join("release"), &released_models);
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
