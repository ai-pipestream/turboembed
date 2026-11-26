#[path = "wordnet_tests/morphy.rs"]
mod morphy;
#[path = "wordnet_tests/similarity.rs"]
mod similarity;
#[path = "wordnet_tests/wn_lmf.rs"]
mod wn_lmf;
#[path = "wordnet_tests/wn_lmf_extension.rs"]
mod wn_lmf_extension;
#[path = "wordnet_tests/wn_lmf_relations.rs"]
mod wn_lmf_relations;
#[path = "wordnet_tests/wndb.rs"]
mod wndb;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use turbo_wordnet::{
    Expansion, ExpansionKind, HypernymTyper, InMemoryWordNetLexicon, LemmaKey, LexicalExpander, LexicalKnowledgeBase,
    MAX_EXTENSION_DEPTH, MorphyExceptions, MorphyLemmatizer, QName, Synset, SynsetSimilarity, WnLmfDependency,
    WnLmfLexicon, WnLmfReader, WnLmfResolver, WnLmfSource, WndbReader, WordNetError, WordNetPos, WordNetRelation, fold,
    split_on_spaces,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn mini_wndb_dir() -> PathBuf {
    fixtures_dir().join("mini-wndb")
}

fn wndb_fixture() -> InMemoryWordNetLexicon {
    WndbReader::read(mini_wndb_dir()).expect("read mini-wndb")
}

fn lmf_fixture() -> InMemoryWordNetLexicon {
    let path = fixtures_dir().join("mini-wn-lmf.xml");
    WnLmfReader::read(path).expect("read mini-wn-lmf.xml")
}

fn morphy_exceptions_fixture() -> MorphyExceptions {
    MorphyExceptions::load(mini_wndb_dir()).expect("load mini-wndb exceptions")
}

fn fixture_string(name: &str) -> String {
    std::fs::read_to_string(fixtures_dir().join(name)).expect("read fixture")
}

fn parse_lmf(document: &str) -> Result<InMemoryWordNetLexicon, WordNetError> {
    WnLmfReader::read_named(Cursor::new(document.as_bytes()), "inline.xml")
}

fn wrap_lmf(body: &str) -> String {
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<LexicalResource>\n\
         <Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\n\
         {body}\n</Lexicon>\n</LexicalResource>\n"
    )
}

// ============================================================================
// LemmaFoldingTest
// ============================================================================

#[test]
fn test_fold_uses_locale_independent_lowercase_and_spaces_for_underscores() {
    assert_eq!("mice", fold("MICE"));
    assert_eq!("domestic dog", fold("Domestic_Dog"));
    assert_eq!("attorney general", fold("attorney_general"));
    assert_eq!("dog", fold("dog"));
    assert_eq!("", fold(""));
}

#[test]
fn test_split_on_spaces_collapses_runs_and_ignores_edges() {
    assert_eq!(vec!["a", "b", "c"], split_on_spaces("a b c"));
    assert_eq!(vec!["a", "b"], split_on_spaces("a   b"));
    assert_eq!(vec!["a"], split_on_spaces("a"));
    assert_eq!(vec!["a"], split_on_spaces(" a "));
    assert!(split_on_spaces("").is_empty());
    assert!(split_on_spaces("   ").is_empty());
}

#[test]
fn test_lemma_key_and_exception_lookup_agree_on_the_fold() {
    assert_eq!(LemmaKey::of("Domestic_Dog", WordNetPos::Noun), LemmaKey::of(&fold("DOMESTIC_DOG"), WordNetPos::Noun));
}

// ============================================================================
// InMemoryWordNetLexiconTest
// ============================================================================

fn dummy_synset(id: &str, relations: BTreeMap<WordNetRelation, Vec<String>>) -> Synset {
    Synset::new(id, WordNetPos::Noun, vec!["lemma".to_string()], "a gloss", relations).expect("valid synset")
}

#[test]
fn test_in_memory_lexicon_accepts_consistent_maps() {
    let mut a_rel = BTreeMap::new();
    a_rel.insert(WordNetRelation::Hypernym, vec!["b".to_string()]);
    let a = dummy_synset("a", a_rel);
    let b = dummy_synset("b", BTreeMap::new());
    let mut synsets = HashMap::new();
    synsets.insert("a".to_string(), a.clone());
    synsets.insert("b".to_string(), b.clone());
    let mut sense_order = HashMap::new();
    sense_order.insert(LemmaKey::of("lemma", WordNetPos::Noun), vec!["a".to_string(), "b".to_string()]);
    let lexicon = InMemoryWordNetLexicon::new(synsets, sense_order).expect("valid lexicon");
    assert_eq!(2, lexicon.size());
    assert_eq!(vec![&a, &b], lexicon.lookup("lemma", WordNetPos::Noun));
}

#[test]
fn test_in_memory_lexicon_rejects_key_that_does_not_match_synset_id() {
    let mut synsets = HashMap::new();
    synsets.insert("wrong-key".to_string(), dummy_synset("real-id", BTreeMap::new()));
    let err = InMemoryWordNetLexicon::new(synsets, HashMap::new()).unwrap_err();
    assert!(err.to_string().contains("wrong-key"), "{err}");
}

#[test]
fn test_in_memory_lexicon_rejects_dangling_relation_target() {
    let mut rel = BTreeMap::new();
    rel.insert(WordNetRelation::Hypernym, vec!["nope".to_string()]);
    let mut synsets = HashMap::new();
    synsets.insert("a".to_string(), dummy_synset("a", rel));
    let err = InMemoryWordNetLexicon::new(synsets, HashMap::new()).unwrap_err();
    assert!(err.to_string().contains("nope"), "{err}");
    assert!(err.to_string().contains("HYPERNYM"), "{err}");
}

#[test]
fn test_in_memory_lexicon_rejects_sense_order_entry_with_unknown_synset() {
    let mut synsets = HashMap::new();
    synsets.insert("a".to_string(), dummy_synset("a", BTreeMap::new()));
    let mut sense_order = HashMap::new();
    sense_order.insert(LemmaKey::of("lemma", WordNetPos::Noun), vec!["missing".to_string()]);
    let err = InMemoryWordNetLexicon::new(synsets, sense_order).unwrap_err();
    assert!(err.to_string().contains("missing"), "{err}");
    assert!(err.to_string().contains("lemma"), "{err}");
}

#[test]
fn test_in_memory_lexicon_rejects_sense_order_entry_that_does_not_describe_synset() {
    for (lemma, pos) in [("other", WordNetPos::Noun), ("lemma", WordNetPos::Verb)] {
        let mut synsets = HashMap::new();
        synsets.insert("a".to_string(), dummy_synset("a", BTreeMap::new()));
        let mut sense_order = HashMap::new();
        sense_order.insert(LemmaKey::of(lemma, pos), vec!["a".to_string()]);
        let err = InMemoryWordNetLexicon::new(synsets, sense_order).unwrap_err();
        assert!(err.to_string().contains("a"), "{err}");
        assert!(err.to_string().contains(lemma), "{err}");
    }
}

#[test]
fn test_in_memory_lexicon_rejects_duplicate_sense_order_entry() {
    let mut synsets = HashMap::new();
    synsets.insert("a".to_string(), dummy_synset("a", BTreeMap::new()));
    let mut sense_order = HashMap::new();
    sense_order.insert(LemmaKey::of("lemma", WordNetPos::Noun), vec!["a".to_string(), "a".to_string()]);
    let err = InMemoryWordNetLexicon::new(synsets, sense_order).unwrap_err();
    assert!(err.to_string().contains("duplicate synset a"), "{err}");
}

// ============================================================================
// ReaderEquivalenceTest
// ============================================================================

#[test]
fn test_both_readers_produce_equivalent_views() {
    let lmf = lmf_fixture();
    let wndb = wndb_fixture();
    assert_eq!(lmf.size(), wndb.size());

    let by_gloss = |lex: &InMemoryWordNetLexicon| -> HashMap<String, Synset> {
        let mut map = HashMap::new();
        for s in lex.synsets() {
            assert!(map.insert(s.gloss().to_string(), s.clone()).is_none(), "duplicate gloss: {}", s.gloss());
        }
        map
    };

    let lmf_by_gloss = by_gloss(&lmf);
    let wndb_by_gloss = by_gloss(&wndb);
    assert_eq!(lmf_by_gloss.keys().collect::<HashSet<_>>(), wndb_by_gloss.keys().collect::<HashSet<_>>());

    let relations_by_gloss =
        |synset: &Synset, lex: &InMemoryWordNetLexicon| -> BTreeMap<WordNetRelation, HashSet<String>> {
            let mut result = BTreeMap::new();
            for (&rel, targets) in synset.relations() {
                let glosses: HashSet<String> =
                    targets.iter().map(|tid| lex.synset(tid).unwrap().gloss().to_string()).collect();
                result.insert(rel, glosses);
            }
            result
        };

    for expected in lmf.synsets() {
        let actual = wndb_by_gloss.get(expected.gloss()).unwrap();
        assert_eq!(expected.pos(), actual.pos());
        assert_eq!(expected.lemmas(), actual.lemmas());
        assert_eq!(relations_by_gloss(expected, &lmf), relations_by_gloss(actual, &wndb));
    }

    for synset in wndb.synsets() {
        for lemma in synset.lemmas() {
            let lmf_glosses: Vec<&str> = lmf.lookup(lemma, synset.pos()).iter().map(|s| s.gloss()).collect();
            let wndb_glosses: Vec<&str> = wndb.lookup(lemma, synset.pos()).iter().map(|s| s.gloss()).collect();
            assert_eq!(lmf_glosses, wndb_glosses);
        }
    }
}
