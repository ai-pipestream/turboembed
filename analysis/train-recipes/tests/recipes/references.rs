//! The published reference models, when present.

use crate::common::{GATED_DOCUMENTS, SENTENCES, temp_dir, write_corpus};
use crate::{reference_path, split_of};
use turbo_eval::model_gate::Task;
use turbo_postag::PosTagFormat;
use turbo_train_recipes::chunk::{self, CHUNKS, SPAN_TYPES};
use turbo_train_recipes::postag::{self, POS, PosRecipe};
use turbo_train_recipes::{RECIPES, Recipe, Reference, run};

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

/// Each name recipe's reference finds names of the recipe's type, and
/// only those: its outcomes are `other` and that type's start and
/// continuation, so the gate scores both models on the same type.
#[test]
fn each_name_reference_finds_the_type_of_its_recipe() {
    for recipe in &RECIPES[4..] {
        let Recipe::Name(r) = recipe else { panic!("{} is not a name recipe", recipe.model_id()) };
        let Some(path) = reference_path(recipe) else { continue };
        let bytes = std::fs::read(&path).unwrap();
        recipe.check_reference(&bytes).unwrap();
        let file = turbo_maxent::ModelFile::open(bytes).unwrap();
        let model = file.maxent(turbo_sequence::namefind::MODEL_ENTRY_NAME).unwrap();
        let kind = r.kind();
        assert_eq!(
            model.outcome_texts().unwrap(),
            ["other".to_owned(), format!("{kind}-start"), format!("{kind}-cont")]
        );
        // A 1.5 model: no descriptor, so it runs with the feature
        // generators OpenNLP gives a model without one.
        assert!(file.entry("generator.featuregen").is_err());
        assert_eq!(file.manifest().get("OpenNLP-Version"), Some("1.5.0"));
    }
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
            // The 1.5 name finders find none of the test corpus's persons,
            // locations and organizations; the date finder finds `Monday`,
            // `May` and `June` but not `that week` or `10 a.m.`.
            Task::NameFinder => match recipe {
                Recipe::Name(n) if n.kind() == "date" => "75.00",
                _ => "0.00",
            },
            task @ (Task::Lemmatizer | Task::LanguageDetector | Task::DependencyParser) => {
                panic!("no recipe of the task {task:?}")
            }
        };
        let scores = (outcome.gate.result.candidate.score, outcome.gate.result.reference.score);
        assert_eq!(format!("{:.2} {:.2}", scores.0, scores.1), format!("100.00 {expected}"), "{}", outcome.gate);
    }
    std::fs::remove_dir_all(&root).unwrap();
}
