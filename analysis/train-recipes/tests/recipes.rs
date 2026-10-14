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

use common::{GATED_DOCUMENTS, SENTENCES, temp_dir, write_corpus};
use turbo_eval::model_gate::{Measure, ModelGateRecord, Task};
use turbo_formats::masc::MascCorpus;
use turbo_postag::PosTagFormat;
use turbo_train_recipes::card::{CARD_FILE, GATE_FILE, ModelCard, TagFormat, write_model_dir};
use turbo_train_recipes::masc::{self, RELEASE, SPLIT, SplitCorpus};
use turbo_train_recipes::postag::{self, CANDIDATE_FORMAT, POS, PosRecipe};
use turbo_train_recipes::segmentation::{SENTENCE, TOKENS};
use turbo_train_recipes::{Error, RECIPES, Recipe, Reference, held_out_name, run};
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
    // abbreviations, contractions or attached punctuation. The check that
    // tells the gate's rule apart from one that passes everything is the
    // swapped one below, where the weaker model is the candidate.
    let plain_root = temp_dir("plain");
    write_corpus(&plain_root, 30, PLAIN);
    let (plain, _) = split_of(&plain_root);

    let out = temp_dir("out");
    for recipe in RECIPES {
        let weak = recipe.train(&plain.train).unwrap();
        println!("{}: the plain corpus gave {} samples, {} events", recipe.model_id(), weak.samples, weak.events);
        let outcome = run(&recipe, &split, &digest, false, "plain", &weak.bytes).unwrap();
        let gate = &outcome.gate;
        gate.check().unwrap();
        println!("{gate}");
        assert_eq!(gate.task, recipe.task());
        let measure = if recipe.task() == Task::PartOfSpeechTagger { Measure::Accuracy } else { Measure::FMeasure };
        assert_eq!(gate.measure, measure);
        assert_eq!(gate.set.name, held_out_name());
        assert_eq!(gate.set.corpus_sha256, digest.sha256);
        assert_eq!(gate.candidate.id, recipe.model_id());
        assert_eq!(gate.reference.id, "plain");
        // The model that has seen the held-out sentences beats the one
        // that never saw their abbreviations and contractions, and passes.
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
            .unwrap();
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
            .unwrap();
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
        std::fs::write(&other, card.to_json().replace("\"card_version\": 2,", "\"card_version\": 1,")).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::CardVersion { found: 1, .. })));
        std::fs::write(&other, "{}").unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::Json { .. })));
        let bad_split = card.to_json().replace("\"every\": 10,", "\"every\": 0,");
        assert_ne!(bad_split, card.to_json());
        std::fs::write(&other, bad_split).unwrap();
        assert!(matches!(ModelCard::read(&other), Err(Error::Json { .. })), "a split of every 0");
        assert_eq!(ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap(), *gate);
        assert_eq!(std::fs::read(dir.join(&card.file.name)).unwrap(), outcome.trained.bytes);
        assert!(write_model_dir(&out, card, &outcome.trained.bytes, gate).is_err(), "nothing is replaced");

        // A card trained on a corpus that is not the recorded release, as
        // every card here is, or whose gate failed, makes no zoo entry.
        let url = format!("https://example.org/models/{}", card.file.name);
        assert!(matches!(card.artifact(&url), Err(Error::NotShippable { .. })));
        let mut failed = card.clone();
        failed.corpus.release_recorded = true;
        failed.gate.pass = false;
        assert!(matches!(failed.artifact(&url), Err(Error::NotShippable { .. })));
        // With the release recorded and the gate passed, the entry is
        // permissive, shippable and names MASC.
        let mut released = card.clone();
        released.corpus.release_recorded = true;
        let artifact = released.artifact(&url).unwrap();
        let index = Index::parse(Index::to_canonical_json(&[artifact]).as_bytes()).unwrap();
        let entry = index.get(recipe.model_id()).unwrap();
        assert_eq!(entry.component, Some(recipe.component()));
        assert_eq!(entry.computed_class(), LicenceClass::Permissive);
        assert_eq!(index.shippable().count(), 1);
        assert_eq!(entry.trained_on[0].name, masc::NAME);
    }
    for dir in [root, plain_root, out] {
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
    assert!(matches!(POS.score(&model.bytes, CANDIDATE_FORMAT, &tags), Err(Error::Model(_))));
    assert!(matches!(POS.score(&model.bytes, POS.reference_format, &tags), Err(Error::Model(_))));
    let tagger = POS.train(&split.train).unwrap();
    assert!(TOKENS.score(&tagger.bytes, &tokens_again).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}

/// What the part-of-speech recipe reads from the MASC corpora checked in
/// for the goldens (`analysis/goldens/data/inputs/formats/masc-*.txt`,
/// written for the tests in MASC's layout): the samples OpenNLP's
/// `MascPOSSampleStream` read from them in the JVM, as the `masc/pos`
/// goldens record them, up to the first sentence the stream threw on; and
/// where it threw, the recipe skips that sentence alone and reads on.
#[test]
fn the_tagger_recipe_reads_the_jvm_s_samples_from_the_checked_in_masc_corpora() {
    let goldens = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../goldens/data");
    // The samples of the first pass of each golden, and whether the
    // stream threw in that pass.
    let jvm = |name: &str| -> (Vec<String>, bool) {
        let text = std::fs::read_to_string(goldens.join(format!("masc/pos/formats/{name}.golden"))).unwrap();
        let layer = |id: &str| -> Vec<Vec<String>> {
            let start = text.find(&format!("layer\t{id}\t")).unwrap_or_else(|| panic!("{name}: no {id}"));
            let body = &text[start..];
            let end = body.find(&format!("\nend\t{id}")).unwrap();
            body[..end].lines().skip(1).map(|l| l.split('\t').map(str::to_owned).collect()).collect()
        };
        let samples = layer("turbo:samples").into_iter().filter(|r| r[0] == "0").map(|r| r[1].clone()).collect();
        let threw = layer("turbo:read-errors").iter().any(|r| r[0] == "0");
        (samples, threw)
    };
    let mut checked = 0;
    for name in ["masc-basic", "masc-names", "masc-opennlp", "masc-penn", "masc-sentences", "masc-skips", "masc-xml"] {
        let (expected, threw) = jvm(name);
        let root = temp_dir(&format!("checked-in-{name}"));
        let bundle = std::fs::read_to_string(goldens.join(format!("inputs/formats/{name}.txt"))).unwrap();
        bundle::write_bundle(&bundle, &root);
        let corpus = MascCorpus::read(&root).unwrap();
        let mut read = Vec::new();
        let skipped = postag::read_samples(&corpus, |s| read.push(s.to_string())).unwrap();
        if threw {
            // The JVM read the samples before the sentence it threw on;
            // the recipe reads those, skips that sentence and goes on.
            assert!(skipped > 0, "{name}: nothing skipped");
            assert_eq!(read[..expected.len()], expected[..], "{name}");
        } else {
            assert_eq!(skipped, 0, "{name}");
            assert_eq!(read, expected, "{name}");
        }
        println!("{name}: {} samples read, {} sentences skipped; the JVM read {}", read.len(), skipped, expected.len());
        checked += 1;
        std::fs::remove_dir_all(&root).unwrap();
    }
    assert_eq!(checked, 7);
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
/// plural nouns and punctuation the UD tagger cannot give.
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
    let outcomes: Vec<&str> = (0..model.maxent().num_outcomes()).map(|o| model.maxent().outcome(o)).collect();
    assert_eq!(tags.len(), outcomes.len());
    for tag in ["VBD", "VBZ", "NNS", ",", "TO", "POS", "PRP$"] {
        assert!(tags.contains(&tag), "{tag} is not an outcome: {tags:?}");
    }
    println!("{} Penn outcomes: {tags:?}", tags.len());
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
/// match their recorded digests and are scored on the same samples.
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
    }
    std::fs::remove_dir_all(&root).unwrap();
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
    assert!(String::from_utf8_lossy(&refused.stderr).contains("no recorded digest"), "{refused:?}");
    assert!(!out.join("models").exists());
    let unknown = args(&["--unrecorded-corpus", "--recipe", "turbo-en-masc-3.0.0-chunks-1"]);
    assert_eq!(unknown.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&unknown.stderr).contains("no recipe trains"), "{unknown:?}");

    if present.len() < RECIPES.len() {
        // Without every reference, a run of every recipe fails on the
        // first one missing, before it writes anything for that recipe.
        let all = args(&["--unrecorded-corpus"]);
        assert_eq!(all.status.code(), Some(2), "{all:?}");
        std::fs::remove_dir_all(out.join("models")).ok();
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
            ModelGateRecord::read(&dir.join(GATE_FILE)).unwrap();
        }
    }
    for dir in [root, out] {
        std::fs::remove_dir_all(dir).unwrap();
    }
}
