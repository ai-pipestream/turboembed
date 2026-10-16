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
// WndbReaderTest
// ============================================================================

const DOG_ID: &str = "wndb-00001075-n";
const CANID_ID: &str = "wndb-00001160-n";

#[test]
fn test_wndb_lookup_returns_synset_with_all_components() {
    let lexicon = wndb_fixture();
    let senses = lexicon.lookup("dog", WordNetPos::Noun);
    assert_eq!(1, senses.len());
    let dog = senses[0];
    assert_eq!(DOG_ID, dog.id());
    assert_eq!(WordNetPos::Noun, dog.pos());
    assert_eq!(&["dog", "domestic dog"], dog.lemmas());
    assert_eq!("a domesticated canid", dog.gloss());
    assert_eq!(&[CANID_ID], dog.related(WordNetRelation::Hypernym));
}

#[test]
fn test_wndb_lookup_folds_case_and_underscore() {
    let lexicon = wndb_fixture();
    assert_eq!(DOG_ID, lexicon.lookup("Domestic_Dog", WordNetPos::Noun)[0].id());
    assert_eq!(DOG_ID, lexicon.lookup("DOG", WordNetPos::Noun)[0].id());
}

#[test]
fn test_wndb_lookup_keeps_index_sense_order() {
    let lexicon = wndb_fixture();
    let ids: Vec<&str> = lexicon.lookup("run", WordNetPos::Noun).iter().map(|s| s.id()).collect();
    assert_eq!(vec!["wndb-00001427-n", "wndb-00001669-n"], ids);
}

#[test]
fn test_wndb_relation_navigation() {
    let lexicon = wndb_fixture();
    assert_eq!(&[DOG_ID], lexicon.related(CANID_ID, WordNetRelation::Hyponym));
    assert_eq!(&["wndb-00001075-v", "wndb-00001171-v"], lexicon.related("wndb-00001324-v", WordNetRelation::Hyponym));
    assert_eq!(&["wndb-00001075-v"], lexicon.related("wndb-00001427-n", WordNetRelation::DerivationallyRelated));
}

#[test]
fn test_wndb_lexical_pointers_surface_at_synset_level() {
    let lexicon = wndb_fixture();
    assert_eq!(&["wndb-00001141-a"], lexicon.related("wndb-00001075-a", WordNetRelation::Antonym));
    assert_eq!(&["wndb-00001075-a"], lexicon.related("wndb-00001141-a", WordNetRelation::Antonym));
}

#[test]
fn test_wndb_satellite_normalizes_to_adjective_and_marker_is_stripped() {
    let lexicon = wndb_fixture();
    let large = lexicon.lookup("large", WordNetPos::Adjective)[0];
    assert_eq!(WordNetPos::Adjective, large.pos());
    assert_eq!(&["wndb-00001211-a"], large.related(WordNetRelation::SimilarTo));
    assert_eq!(&["short"], lexicon.lookup("short", WordNetPos::Adjective)[0].lemmas());
}

fn write_empty_db(dir: &Path, suffixes: &[&str]) {
    for suffix in suffixes {
        std::fs::write(dir.join(format!("data.{suffix}")), "").unwrap();
        std::fs::write(dir.join(format!("index.{suffix}")), "").unwrap();
    }
}

fn copy_wndb_fixture(target: &Path) {
    for entry in std::fs::read_dir(mini_wndb_dir()).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), target.join(entry.file_name())).unwrap();
    }
}

fn mutate_file(dir: &Path, file_name: &str, edit: impl Fn(&str) -> String) {
    let path = dir.join(file_name);
    let content = std::fs::read_to_string(&path).unwrap();
    let mut out = String::new();
    for line in content.lines() {
        out.push_str(&edit(line));
        out.push('\n');
    }
    std::fs::write(&path, out).unwrap();
}

#[test]
fn test_wndb_verb_group_pointer_maps_to_verb_group() {
    let temp = tempfile::tempdir().unwrap();
    write_empty_db(temp.path(), &["noun", "adj", "adv"]);
    let template = "00000000 29 v 01 sing 0 001 $ XXXXXXXX v 0000 00 | produce musical tones";
    let off2 = format!("{:08}", template.len() + 1);
    let line1 = template.replace("XXXXXXXX", &off2);
    let line2 = format!("{off2} 29 v 01 chant 0 001 $ 00000000 v 0000 00 | sing monotonously");
    std::fs::write(temp.path().join("data.verb"), format!("{line1}\n{line2}\n")).unwrap();
    std::fs::write(temp.path().join("index.verb"), format!("chant v 1 1 $ 1 0 {off2}\nsing v 1 1 $ 1 0 00000000\n"))
        .unwrap();
    let lexicon = WndbReader::read(temp.path()).unwrap();
    assert_eq!(&[format!("wndb-{off2}-v")], lexicon.related("wndb-00000000-v", WordNetRelation::VerbGroup));
    assert_eq!(
        &["wndb-00000000-v".to_string()],
        lexicon.related(&format!("wndb-{off2}-v"), WordNetRelation::VerbGroup)
    );
}

#[test]
fn test_wndb_unknown_lemma_or_synset_is_empty() {
    let lexicon = wndb_fixture();
    assert!(lexicon.lookup("zebra", WordNetPos::Noun).is_empty());
    assert!(lexicon.synset("wndb-99999999-n").is_none());
}

#[test]
fn test_wndb_rejects_missing_directory_and_files() {
    let temp = tempfile::tempdir().unwrap();
    assert!(WndbReader::read(temp.path().join("absent")).is_err());

    copy_wndb_fixture(temp.path());
    std::fs::remove_file(temp.path().join("data.verb")).unwrap();
    let err = WndbReader::read(temp.path()).unwrap_err();
    assert!(err.to_string().contains("data.verb"), "{err}");
}

#[test]
fn test_wndb_rejects_malformed_fields() {
    let cases = [
        ("data.noun", "00001564 03 n", "0000156x 03 n", "Synset offset must be 8 digits"),
        ("data.noun", "00001564 03 n", "00001564 xx n", "lex_filenum must be a 2-digit base-10 integer"),
        ("data.noun", "00001564 03 n", "00001564 99 n", "lex_filenum must be between 00 and 44"),
        ("data.noun", "00001075 03 n", "00001075 03 v", "Synset type v does not belong in"),
        ("data.noun", "n 01 box", "n 0z box", "w_cnt must be a 2-digit base-16 integer"),
        ("data.noun", "n 01 man", "n 00 man", "Word count must be at least 1"),
        ("data.noun", "n 01 box 0 000 |", "n 01 box 0 -01 |", "Pointer count must not be negative"),
        ("data.noun", "00001160 n 0000", "00001160 q 0000", "Pointer pos must be one of"),
        ("data.noun", "00001160 n 0000", "00001160 n zzzz", "pointer source/target must be a 4-digit base-16 integer"),
        ("data.noun", "00001160 n 0000", "00001160 n 0300", "Pointer source word 3 exceeds word count 2"),
        ("data.noun", "00001160 n 0000", "00001160 n 0102", "Pointer target word 2 exceeds target word count 1"),
        ("data.noun", "00001160 n 0000", "00001160 n 0100", "must both be zero or nonzero"),
        ("data.verb", "n 0101 01 + 02", "n 0101 -1 + 02", "Verb frame count must not be negative"),
        ("data.verb", "01 + 02 00 |", "01 ? 02 00 |", "Expected + before a verb frame"),
        ("data.verb", "01 + 02 00 |", "01 + xx 00 |", "f_num must be a 2-digit base-10 integer"),
        ("data.verb", "01 + 02 00 |", "01 + 36 00 |", "f_num must be between 01 and 35"),
        ("data.verb", "01 + 02 00 |", "01 + 02 xx |", "w_num must be a 2-digit base-16 integer"),
        ("data.verb", "01 + 02 00 |", "01 + 02 02 |", "Verb frame word 2 exceeds word count 1"),
        ("data.noun", "| a domesticated canid", "! a domesticated canid", "Expected the | gloss separator"),
        ("data.adj", "short(p)", "short(x)", "Unknown syntactic marker on word: short(x)"),
        ("data.noun", "dog 0", "dog(p) 0", "Syntactic marker (p) is only valid in data.adj"),
        ("index.noun", "berry n 1", "berry v 1", "Index pos v does not belong in"),
        ("index.noun", "berry n 1 0 1 0", "berry n +1 0 1 0", "synset count is not an unsigned decimal integer: +1"),
        ("index.noun", "berry n 1 0 1 0", "berry n 1 -1 1 0", "Pointer count must not be negative"),
        ("index.noun", "dog n 1 1 @ 1 0", "dog n 1 1 ? 1 0", "Undeclared pointer symbol: ?"),
        ("index.noun", "dog n 1 1 @ 1 0", "dog n 1 2 @ @ 1 0", "Duplicate pointer symbol: @"),
        ("index.noun", "berry n 1 0 1 0", "berry n 1 0 x 0", "sense count is not an unsigned decimal integer"),
        ("index.noun", "berry n 1 0 1 0", "berry n 1 0 1 -1", "Tagged-sense count must not be negative"),
        ("index.noun", "berry n 1 0 1 0", "berry n 1 0 2 0", "Sense count 2 does not match synset count 1"),
        ("index.noun", "berry n 1 0 1 0", "berry n 1 0 1 2", "Tagged-sense count 2 exceeds sense count 1"),
        (
            "index.noun",
            "berry n 1 0 1 0",
            "berry n 2147483647 0 2147483647 0",
            "Synset count 2147483647 exceeds the available offset fields",
        ),
        ("index.noun", "00001427 00001669", "00001427 00001427", "Duplicate synset offset 00001427"),
        ("index.noun", "00001564  ", "00001564 extra ", "Unexpected field after synset offsets: extra"),
    ];

    for (file_name, find_str, replacement, expected) in cases {
        let temp = tempfile::tempdir().unwrap();
        copy_wndb_fixture(temp.path());
        mutate_file(temp.path(), file_name, |line| line.replace(find_str, replacement));
        let err = WndbReader::read(temp.path()).unwrap_err();
        assert!(err.to_string().contains(expected), "expected '{expected}' in '{err}' for {file_name}");
    }
}

#[test]
fn test_wndb_dangling_pointer_error_names_pointer_line() {
    let temp = tempfile::tempdir().unwrap();
    write_empty_db(temp.path(), &["noun", "adj", "adv"]);
    std::fs::write(
        temp.path().join("data.verb"),
        "00000000 29 v 01 sing 0 001 $ 00009999 v 0000 00 | produce musical tones\n",
    )
    .unwrap();
    std::fs::write(temp.path().join("index.verb"), "sing v 1 1 $ 1 0 00000000\n").unwrap();
    let err = WndbReader::read(temp.path()).unwrap_err();
    assert!(err.to_string().contains("wndb-00009999-v"), "{err}");
    assert!(err.to_string().contains("line 1"), "{err}");
}

// ============================================================================
// WnLmfReaderTest
// ============================================================================

#[test]
fn test_lmf_lookup_returns_synset_with_all_components() {
    let lexicon = lmf_fixture();
    let senses = lexicon.lookup("dog", WordNetPos::Noun);
    assert_eq!(1, senses.len());
    let dog = senses[0];
    assert_eq!("mini-n1", dog.id());
    assert_eq!(WordNetPos::Noun, dog.pos());
    assert_eq!(&["dog", "domestic dog"], dog.lemmas());
    assert_eq!("a domesticated canid", dog.gloss());
    assert_eq!(&["mini-n2"], dog.related(WordNetRelation::Hypernym));
}

#[test]
fn test_lmf_lookup_folds_case_and_underscore() {
    let lexicon = lmf_fixture();
    assert_eq!("mini-n1", lexicon.lookup("Domestic_Dog", WordNetPos::Noun)[0].id());
    assert_eq!("mini-n1", lexicon.lookup("DOG", WordNetPos::Noun)[0].id());
}

#[test]
fn test_lmf_lookup_keeps_sense_order() {
    let lexicon = lmf_fixture();
    let ids: Vec<&str> = lexicon.lookup("run", WordNetPos::Noun).iter().map(|s| s.id()).collect();
    assert_eq!(vec!["mini-n5", "mini-n9"], ids);
}

#[test]
fn test_lmf_preserves_multiple_definitions() {
    let lexicon = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\">\
         <Definition>a feline</Definition>\
         <Definition>a domesticated cat</Definition></Synset>",
    ))
    .unwrap();
    assert_eq!("a feline; a domesticated cat", lexicon.lookup("cat", WordNetPos::Noun)[0].gloss());
}

#[test]
fn test_lmf_lookup_is_pos_scoped() {
    let lexicon = lmf_fixture();
    assert_eq!(1, lexicon.lookup("run", WordNetPos::Verb).len());
    assert!(lexicon.lookup("dog", WordNetPos::Verb).is_empty());
    assert!(!lexicon.contains("walk", WordNetPos::Noun));
    assert!(lexicon.contains("walk", WordNetPos::Verb));
}

#[test]
fn test_lmf_relation_navigation() {
    let lexicon = lmf_fixture();
    assert_eq!(&["mini-n1"], lexicon.related("mini-n2", WordNetRelation::Hyponym));
    assert_eq!(&["mini-v1", "mini-v2"], lexicon.related("mini-v4", WordNetRelation::Hyponym));
    assert_eq!(&["mini-v4"], lexicon.related("mini-v1", WordNetRelation::Hypernym));
}

#[test]
fn test_lmf_sense_relations_are_represented_at_synset_level() {
    let lexicon = lmf_fixture();
    assert_eq!(&["mini-a2"], lexicon.related("mini-a1", WordNetRelation::Antonym));
    assert_eq!(&["mini-a1"], lexicon.related("mini-a2", WordNetRelation::Antonym));
    assert_eq!(&["mini-v1"], lexicon.related("mini-n5", WordNetRelation::DerivationallyRelated));
    assert_eq!(&["mini-n5"], lexicon.related("mini-v1", WordNetRelation::DerivationallyRelated));
}

#[test]
fn test_lmf_satellite_normalizes_to_adjective() {
    let lexicon = lmf_fixture();
    let senses = lexicon.lookup("large", WordNetPos::Adjective);
    assert_eq!(1, senses.len());
    assert_eq!(WordNetPos::Adjective, senses[0].pos());
    assert_eq!(&["mini-a4"], lexicon.related("mini-a3", WordNetRelation::SimilarTo));
    assert_eq!(&["mini-a3"], lexicon.related("mini-a4", WordNetRelation::SimilarTo));
}

#[test]
fn test_lmf_similar_on_verb_synset_maps_to_verb_group() {
    let lexicon = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-sing-v\"><Lemma writtenForm=\"sing\" partOfSpeech=\"v\"/>\
         <Sense id=\"t-sing-v-1\" synset=\"t-v1\"/></LexicalEntry>\
         <LexicalEntry id=\"t-chant-v\"><Lemma writtenForm=\"chant\" partOfSpeech=\"v\"/>\
         <Sense id=\"t-chant-v-1\" synset=\"t-v2\"/></LexicalEntry>\
         <Synset id=\"t-v1\" partOfSpeech=\"v\">\
         <Definition>produce musical tones</Definition>\
         <SynsetRelation relType=\"similar\" target=\"t-v2\"/></Synset>\
         <Synset id=\"t-v2\" partOfSpeech=\"v\">\
         <Definition>sing monotonously</Definition></Synset>",
    ))
    .unwrap();
    assert_eq!(&["t-v2"], lexicon.related("t-v1", WordNetRelation::VerbGroup));
    assert!(lexicon.related("t-v1", WordNetRelation::SimilarTo).is_empty());
}

#[test]
fn test_lmf_skips_doctype_and_rejects_xxe() {
    let doc = "<?xml version=\"1.0\"?>\n\
        <!DOCTYPE LexicalResource SYSTEM \"http://example.invalid/WN-LMF-1.3.dtd\">\n\
        <LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\
        <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
        <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
        <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
        </Lexicon></LexicalResource>";
    let lexicon = parse_lmf(doc).unwrap();
    assert_eq!("a feline", lexicon.lookup("cat", WordNetPos::Noun)[0].gloss());

    let xxe_doc = "<?xml version=\"1.0\"?>\n\
        <!DOCTYPE LexicalResource [<!ENTITY xxe SYSTEM \"file:///etc/passwd\">]>\n\
        <LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\
        <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"&xxe;\" partOfSpeech=\"n\"/>\
        <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
        <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
        </Lexicon></LexicalResource>";
    assert!(parse_lmf(xxe_doc).is_err());
}

#[test]
fn test_lmf_rejects_structurally_invalid_documents() {
    let cases: &[(&str, &[&str])] = &[
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\"/></LexicalEntry>",
            &["synset"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-9\"/></LexicalEntry>",
            &["t-9"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
             <SynsetRelation relType=\"hypernym\" target=\"t-9\"/></Synset>",
            &["t-9"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
             <SynsetRelation relType=\"quasi_synonym\" target=\"t-1\"/></Synset>",
            &["quasi_synonym"],
        ),
        (
            "<LexicalEntry id=\"t-cat-x\"><Lemma writtenForm=\"cat\" partOfSpeech=\"x\"/>\
             <Sense id=\"t-cat-x-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
            &["x"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a repeat</Definition></Synset>",
            &["Duplicate synset id t-1"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-dog-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
            &["Duplicate lexical entry id t-cat-n"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-2\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>\
             <Synset id=\"t-2\" partOfSpeech=\"n\"/>",
            &["Duplicate sense id t-cat-n-1"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"v\"><Definition>a feline</Definition></Synset>",
            &["t-cat-n", "Verb", "Noun"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
             <LexicalEntry id=\"t-dog-n\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-dog-n-1\" synset=\"t-2\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\" members=\"t-dog-n-1\"/>\
             <Synset id=\"t-2\" partOfSpeech=\"n\" members=\"t-dog-n-1\"/>",
            &["t-dog-n-1", "t-1", "t-2"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
             <SenseRelation relType=\"antonym\" target=\"t-ghost-1\"/></Sense>\
             </LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["t-ghost-1"],
        ),
        ("<Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>", &["Lemma outside a LexicalEntry"]),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/></LexicalEntry>",
            &["Duplicate Lemma in LexicalEntry t-cat-n"],
        ),
        (
            "<LexicalEntry id=\"t-outer-n\"><Lemma writtenForm=\"outer\" partOfSpeech=\"n\"/>\
             <LexicalEntry id=\"t-inner-n\"><Lemma writtenForm=\"inner\" partOfSpeech=\"n\"/>\
             </LexicalEntry></LexicalEntry>",
            &["Nested LexicalEntry inside t-outer-n"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["Sense before its entry's Lemma"],
        ),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
             <Sense id=\"t-cat-n-2\" synset=\"t-1\"/></Sense></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["Nested Sense inside t-cat-n-1"],
        ),
        (
            "<Synset id=\"t-1\" partOfSpeech=\"n\">\
             <Synset id=\"t-2\" partOfSpeech=\"n\"/></Synset>",
            &["Nested Synset inside t-1"],
        ),
        ("<Definition>orphaned definition</Definition>", &["Definition outside a Synset"]),
        (
            "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-cat-n-1\" synset=\"t-1\"/>\
             <SenseRelation relType=\"antonym\" target=\"t-cat-n-1\"/></LexicalEntry>\
             <Synset id=\"t-1\" partOfSpeech=\"n\"/>",
            &["SenseRelation outside a Sense"],
        ),
        ("<SynsetRelation relType=\"hypernym\" target=\"t-1\"/>", &["SynsetRelation outside a Synset"]),
    ];

    for (body, fragments) in cases {
        let err = parse_lmf(&wrap_lmf(body)).unwrap_err();
        let msg = err.to_string();
        for fragment in *fragments {
            assert!(msg.contains(fragment), "expected '{fragment}' in '{msg}' for body: {body}");
        }
    }
}

#[test]
fn test_lmf_skips_other_relation_type_on_sense_and_synset_relations() {
    let lex1 = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\">\
         <SenseRelation relType=\"other\" target=\"t-cat-n-1\"/></Sense></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>",
    ))
    .unwrap();
    assert!(lex1.synset("t-1").unwrap().relations().is_empty());

    let lex2 = parse_lmf(&wrap_lmf(
        "<LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
         <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
         <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition>\
         <SynsetRelation relType=\"other\" target=\"t-1\"/></Synset>",
    ))
    .unwrap();
    assert!(lex2.synset("t-1").unwrap().relations().is_empty());
}

// ============================================================================
// WnLmfResourceTest
// ============================================================================

#[test]
fn test_lmf_reads_multilingual_resource_without_merging_lexicons() {
    let path = fixtures_dir().join("omw-multilingual.xml");
    let resource = WnLmfReader::read_resource(&path).unwrap();
    let ids: Vec<&str> = resource.lexicons().iter().map(WnLmfLexicon::id).collect();
    assert_eq!(vec!["omw-it", "omw-es", "omw-sv"], ids);

    let italian = resource.lexicon("omw-it").unwrap();
    assert_eq!("it", italian.language());
    assert_eq!("2.0", italian.version());
    assert_eq!(
        Some(&"https://creativecommons.org/licenses/by/3.0/".to_string()),
        italian.metadata().get(&QName::local("license"))
    );
    assert_eq!(
        Some(&"Open Multilingual Wordnet".to_string()),
        italian.metadata().get(&QName::new("https://globalwordnet.github.io/schemas/dc/", "publisher"))
    );
    assert_eq!(&[WnLmfDependency::new("omw-en", "2.0", None).unwrap()], italian.dependencies());
    assert_eq!("omw-it-02084071-n", italian.knowledge_base().lookup("cane", WordNetPos::Noun)[0].id());
    assert_eq!(
        "omw-es-02084071-n",
        resource.lexicon("omw-es").unwrap().knowledge_base().lookup("perro", WordNetPos::Noun)[0].id()
    );
    assert_eq!(
        "omw-sv-02084071-n",
        resource.lexicon("omw-sv").unwrap().knowledge_base().lookup("hund", WordNetPos::Noun)[0].id()
    );
    assert!(italian.knowledge_base().lookup("perro", WordNetPos::Noun).is_empty());
    assert!(resource.lexicon("missing").is_none());

    // Single-lexicon convenience rejects multi-lexicon resource
    let err = WnLmfReader::read(&path).unwrap_err();
    assert!(err.to_string().contains("contains 3 lexicons"), "{err}");
}

// ============================================================================
// WnLmfRelationCoverageTest
// ============================================================================

#[test]
fn test_every_legal_synset_and_sense_relation() {
    let common = [
        ("agent", WordNetRelation::Agent),
        ("also", WordNetRelation::AlsoSee),
        ("anto_converse", WordNetRelation::AntoConverse),
        ("anto_gradable", WordNetRelation::AntoGradable),
        ("anto_simple", WordNetRelation::AntoSimple),
        ("antonym", WordNetRelation::Antonym),
        ("augmentative", WordNetRelation::Augmentative),
        ("diminutive", WordNetRelation::Diminutive),
        ("domain_region", WordNetRelation::DomainRegion),
        ("domain_topic", WordNetRelation::DomainTopic),
        ("exemplifies", WordNetRelation::DomainUsage),
        ("feminine", WordNetRelation::Feminine),
        ("has_augmentative", WordNetRelation::HasAugmentative),
        ("has_diminutive", WordNetRelation::HasDiminutive),
        ("has_domain_region", WordNetRelation::MemberOfDomainRegion),
        ("has_domain_topic", WordNetRelation::MemberOfDomainTopic),
        ("has_feminine", WordNetRelation::HasFeminine),
        ("has_masculine", WordNetRelation::HasMasculine),
        ("has_young", WordNetRelation::HasYoung),
        ("instrument", WordNetRelation::Instrument),
        ("is_exemplified_by", WordNetRelation::MemberOfDomainUsage),
        ("location", WordNetRelation::Location),
        ("masculine", WordNetRelation::Masculine),
        ("result", WordNetRelation::Result),
        ("young", WordNetRelation::Young),
    ];

    let synset_only = [
        ("attribute", WordNetRelation::Attribute),
        ("be_in_state", WordNetRelation::BeInState),
        ("causes", WordNetRelation::Cause),
        ("classified_by", WordNetRelation::ClassifiedBy),
        ("classifies", WordNetRelation::Classifies),
        ("co_agent_instrument", WordNetRelation::CoAgentInstrument),
        ("co_agent_patient", WordNetRelation::CoAgentPatient),
        ("co_agent_result", WordNetRelation::CoAgentResult),
        ("co_instrument_agent", WordNetRelation::CoInstrumentAgent),
        ("co_instrument_patient", WordNetRelation::CoInstrumentPatient),
        ("co_instrument_result", WordNetRelation::CoInstrumentResult),
        ("co_patient_agent", WordNetRelation::CoPatientAgent),
        ("co_patient_instrument", WordNetRelation::CoPatientInstrument),
        ("co_result_agent", WordNetRelation::CoResultAgent),
        ("co_result_instrument", WordNetRelation::CoResultInstrument),
        ("co_role", WordNetRelation::CoRole),
        ("direction", WordNetRelation::Direction),
        ("entails", WordNetRelation::Entailment),
        ("eq_synonym", WordNetRelation::EqSynonym),
        ("holo_location", WordNetRelation::LocationHolonym),
        ("holo_member", WordNetRelation::MemberHolonym),
        ("holo_part", WordNetRelation::PartHolonym),
        ("holo_portion", WordNetRelation::PortionHolonym),
        ("holo_substance", WordNetRelation::SubstanceHolonym),
        ("holonym", WordNetRelation::Holonym),
        ("hypernym", WordNetRelation::Hypernym),
        ("hyponym", WordNetRelation::Hyponym),
        ("in_manner", WordNetRelation::InManner),
        ("instance_hypernym", WordNetRelation::InstanceHypernym),
        ("instance_hyponym", WordNetRelation::InstanceHyponym),
        ("involved", WordNetRelation::Involved),
        ("involved_agent", WordNetRelation::InvolvedAgent),
        ("involved_direction", WordNetRelation::InvolvedDirection),
        ("involved_instrument", WordNetRelation::InvolvedInstrument),
        ("involved_location", WordNetRelation::InvolvedLocation),
        ("involved_patient", WordNetRelation::InvolvedPatient),
        ("involved_result", WordNetRelation::InvolvedResult),
        ("involved_source_direction", WordNetRelation::InvolvedSourceDirection),
        ("involved_target_direction", WordNetRelation::InvolvedTargetDirection),
        ("ir_synonym", WordNetRelation::IrSynonym),
        ("is_caused_by", WordNetRelation::CausedBy),
        ("is_entailed_by", WordNetRelation::EntailedBy),
        ("is_subevent_of", WordNetRelation::IsSubeventOf),
        ("manner_of", WordNetRelation::MannerOf),
        ("mero_location", WordNetRelation::LocationMeronym),
        ("mero_member", WordNetRelation::MemberMeronym),
        ("mero_part", WordNetRelation::PartMeronym),
        ("mero_portion", WordNetRelation::PortionMeronym),
        ("mero_substance", WordNetRelation::SubstanceMeronym),
        ("meronym", WordNetRelation::Meronym),
        ("patient", WordNetRelation::Patient),
        ("restricted_by", WordNetRelation::RestrictedBy),
        ("restricts", WordNetRelation::Restricts),
        ("role", WordNetRelation::Role),
        ("similar", WordNetRelation::SimilarTo),
        ("source_direction", WordNetRelation::SourceDirection),
        ("state_of", WordNetRelation::StateOf),
        ("subevent", WordNetRelation::Subevent),
        ("target_direction", WordNetRelation::TargetDirection),
    ];

    let sense_only = [
        ("body_part", WordNetRelation::BodyPart),
        ("by_means_of", WordNetRelation::ByMeansOf),
        ("derivation", WordNetRelation::DerivationallyRelated),
        ("destination", WordNetRelation::Destination),
        ("event", WordNetRelation::Event),
        ("has_metaphor", WordNetRelation::HasMetaphor),
        ("has_metonym", WordNetRelation::HasMetonym),
        ("material", WordNetRelation::Material),
        ("metaphor", WordNetRelation::Metaphor),
        ("metonym", WordNetRelation::Metonym),
        ("participle", WordNetRelation::Participle),
        ("pertainym", WordNetRelation::Pertainym),
        ("property", WordNetRelation::Property),
        ("secondary_aspect_ip", WordNetRelation::SecondaryAspectIp),
        ("secondary_aspect_pi", WordNetRelation::SecondaryAspectPi),
        ("similar", WordNetRelation::SimilarTo),
        ("simple_aspect_ip", WordNetRelation::SimpleAspectIp),
        ("simple_aspect_pi", WordNetRelation::SimpleAspectPi),
        ("state", WordNetRelation::State),
        ("undergoer", WordNetRelation::Undergoer),
        ("uses", WordNetRelation::Uses),
        ("vehicle", WordNetRelation::Vehicle),
    ];

    assert_eq!(84, common.len() + synset_only.len());
    assert_eq!(47, common.len() + sense_only.len());

    for (rel_type, expected) in common.iter().chain(synset_only.iter()) {
        let doc = wrap_lmf(&format!(
            "<LexicalEntry id=\"t-source-entry-n\">\
             <Lemma writtenForm=\"source\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-source-sense-n\" synset=\"t-source-n\"/></LexicalEntry>\
             <LexicalEntry id=\"t-target-entry-n\">\
             <Lemma writtenForm=\"target\" partOfSpeech=\"n\"/>\
             <Sense id=\"t-target-sense-n\" synset=\"t-target-n\"/></LexicalEntry>\
             <Synset id=\"t-source-n\" partOfSpeech=\"n\">\
             <SynsetRelation relType=\"{rel_type}\" target=\"t-target-n\"/></Synset>\
             <Synset id=\"t-target-n\" partOfSpeech=\"n\"/>"
        ));
        let lex = parse_lmf(&doc).unwrap();
        assert_eq!(&["t-target-n"], lex.related("t-source-n", *expected));
    }

    for (rel_type, expected) in common.iter().chain(sense_only.iter()) {
        let doc = wrap_lmf(&format!(
            "<LexicalEntry id=\"t-source-entry-v\">\
             <Lemma writtenForm=\"source\" partOfSpeech=\"v\"/>\
             <Sense id=\"t-source-sense-v\" synset=\"t-source-v\">\
             <SenseRelation relType=\"{rel_type}\" target=\"t-target-sense-v\"/></Sense></LexicalEntry>\
             <LexicalEntry id=\"t-target-entry-v\">\
             <Lemma writtenForm=\"target\" partOfSpeech=\"v\"/>\
             <Sense id=\"t-target-sense-v\" synset=\"t-target-v\"/></LexicalEntry>\
             <Synset id=\"t-source-v\" partOfSpeech=\"v\"/>\
             <Synset id=\"t-target-v\" partOfSpeech=\"v\"/>"
        ));
        let lex = parse_lmf(&doc).unwrap();
        assert_eq!(&["t-target-v"], lex.related("t-source-v", *expected));
    }
}

// ============================================================================
// WnLmfExtensionTest, WnLmfCompositionValidationTest, WnLmfResolutionContractTest
// ============================================================================

struct CatalogResolver {
    catalog: Mutex<HashMap<String, String>>,
    calls: AtomicUsize,
}

impl CatalogResolver {
    fn new() -> Self {
        Self { catalog: Mutex::new(HashMap::new()), calls: AtomicUsize::new(0) }
    }

    fn with(self, r: &str, version: &str, doc: impl Into<String>) -> Self {
        self.catalog.lock().unwrap().insert(format!("{r}@{version}"), doc.into());
        self
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl WnLmfResolver for CatalogResolver {
    fn resolve(&self, reference: &WnLmfDependency) -> Result<Option<WnLmfSource>, WordNetError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let key = format!("{}@{}", reference.id(), reference.version());
        let Some(doc) = self.catalog.lock().unwrap().get(&key).cloned() else {
            return Ok(None);
        };
        Ok(Some(WnLmfSource::named_bytes(format!("{}-{}.xml", reference.id(), reference.version()), doc.into_bytes())?))
    }
}

fn base_resolver() -> CatalogResolver {
    CatalogResolver::new().with("ewn", "2020", fixture_string("extension-base.xml"))
}

#[test]
fn test_computer_science_extension_example() {
    let ext_xml = fixture_string("extension-computer-science.xml");
    let resolver = base_resolver();
    let resource = WnLmfReader::read_resource_named_with_resolver(
        Cursor::new(ext_xml.as_bytes()),
        "extension-computer-science.xml",
        &resolver,
    )
    .unwrap();
    assert_eq!(1, resource.lexicons().len());
    let composed = &resource.lexicons()[0];
    assert_eq!("ewn-cs-example", composed.id());
    assert_eq!("English WordNet Computer Science Terms (example)", composed.label());
    assert_eq!("en", composed.language());
    assert_eq!("1.0", composed.version());
    assert_eq!(Some(&WnLmfDependency::new("ewn", "2020", None).unwrap()), composed.extension_of());
    assert_eq!(
        &[WnLmfDependency::new("omw-fr", "1.4", Some("https://example.org/omw-fr-1.4.xml".to_string())).unwrap()],
        composed.dependencies()
    );

    let kb = composed.knowledge_base();
    let process_ids: Vec<&str> = kb.lookup("process", WordNetPos::Noun).iter().map(|s| s.id()).collect();
    assert_eq!(vec!["ewn-08056231-n", "ewn-05984756-n", "ewn-20000123-n"], process_ids);

    let computer_process = kb.synset("ewn-20000123-n").unwrap();
    assert_eq!(&["process"], computer_process.lemmas());
    assert_eq!("a running instance of a computer program", computer_process.gloss());
    assert_eq!(&["ewn-06581154-n"], computer_process.related(WordNetRelation::Hypernym));
    assert_eq!(6, kb.size());
}

#[test]
fn test_extension_definition_fills_empty_base_gloss_but_never_replaces_non_empty() {
    let doc = "<LexicalResource><LexiconExtension id=\"ext\" label=\"ext\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <ExternalSynset id=\"ewn-90000001-n\">\
        <Definition>a placeholder definition</Definition></ExternalSynset>\
        <ExternalSynset id=\"ewn-08056231-n\">\
        <Definition>must not replace the base definition</Definition></ExternalSynset>\
        </LexiconExtension></LexicalResource>";
    let kb = WnLmfReader::read_named_with_resolver(Cursor::new(doc.as_bytes()), "ext.xml", &base_resolver()).unwrap();
    assert_eq!("a placeholder definition", kb.synset("ewn-90000001-n").unwrap().gloss());
    assert_eq!("a phenomenon that unfolds over time", kb.synset("ewn-08056231-n").unwrap().gloss());
}

#[test]
fn test_sibling_extensions_compose_independently_and_share_one_resolver_call() {
    let doc = "<LexicalResource>\
        <LexiconExtension id=\"ext-a\" label=\"ext-a\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <LexicalEntry id=\"ext-a-alpha-n\"><Lemma writtenForm=\"alpha\" partOfSpeech=\"n\"/>\
        <Sense id=\"ext-a-alpha-n-1\" synset=\"ext-a-1-n\"/></LexicalEntry>\
        <Synset id=\"ext-a-1-n\" partOfSpeech=\"n\"/></LexiconExtension>\
        <LexiconExtension id=\"ext-b\" label=\"ext-b\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"ewn\" version=\"2020\"/>\
        <LexicalEntry id=\"ext-b-beta-n\"><Lemma writtenForm=\"beta\" partOfSpeech=\"n\"/>\
        <Sense id=\"ext-b-beta-n-1\" synset=\"ext-b-1-n\"/></LexicalEntry>\
        <Synset id=\"ext-b-1-n\" partOfSpeech=\"n\"/></LexiconExtension>\
        </LexicalResource>";
    let resolver = base_resolver();
    let resource =
        WnLmfReader::read_resource_named_with_resolver(Cursor::new(doc.as_bytes()), "siblings.xml", &resolver).unwrap();
    assert_eq!(1, resolver.calls());
    let first = resource.lexicon("ext-a").unwrap().knowledge_base();
    let second = resource.lexicon("ext-b").unwrap().knowledge_base();
    assert_eq!(1, first.lookup("alpha", WordNetPos::Noun).len());
    assert!(first.lookup("beta", WordNetPos::Noun).is_empty());
    assert_eq!(1, second.lookup("beta", WordNetPos::Noun).len());
    assert!(second.lookup("alpha", WordNetPos::Noun).is_empty());
}

#[test]
fn test_extension_cycles_and_depth_limit_are_enforced() {
    let self_cycle = "<LexicalResource><LexiconExtension id=\"loop\" label=\"loop\" language=\"en\" version=\"1.0\">\
        <Extends ref=\"loop\" version=\"1.0\"/></LexiconExtension></LexicalResource>";
    let resolver = CatalogResolver::new().with("loop", "1.0", self_cycle);
    let err =
        WnLmfReader::read_named_with_resolver(Cursor::new(self_cycle.as_bytes()), "loop.xml", &resolver).unwrap_err();
    assert!(err.to_string().contains("cycle"), "{err}");

    let make_chain = |depth: usize| {
        let mut r = CatalogResolver::new().with(
            "chain-0",
            "1.0",
            "<LexicalResource><Lexicon id=\"chain-0\" label=\"chain-0\" language=\"en\" version=\"1.0\">\
             <LexicalEntry id=\"c0-e\"><Lemma writtenForm=\"process\" partOfSpeech=\"n\"/>\
             <Sense id=\"c0-s\" synset=\"c0-syn\"/></LexicalEntry>\
             <Synset id=\"c0-syn\" partOfSpeech=\"n\"/></Lexicon></LexicalResource>",
        );
        for i in 1..depth {
            let prev = i - 1;
            r = r.with(
                &format!("chain-{i}"),
                "1.0",
                format!(
                    "<LexicalResource><LexiconExtension id=\"chain-{i}\" label=\"chain-{i}\" \
                     language=\"en\" version=\"1.0\"><Extends ref=\"chain-{prev}\" version=\"1.0\"/>\
                     </LexiconExtension></LexicalResource>"
                ),
            );
        }
        let prev = depth - 1;
        let top = format!(
            "<LexicalResource><LexiconExtension id=\"chain-{depth}\" label=\"chain-{depth}\" \
             language=\"en\" version=\"1.0\"><Extends ref=\"chain-{prev}\" version=\"1.0\"/>\
             </LexiconExtension></LexicalResource>"
        );
        (top, r)
    };

    let (top_ok, res_ok) = make_chain(MAX_EXTENSION_DEPTH);
    let kb = WnLmfReader::read_named_with_resolver(Cursor::new(top_ok.as_bytes()), "ok.xml", &res_ok).unwrap();
    assert_eq!(1, kb.lookup("process", WordNetPos::Noun).len());

    let (top_deep, res_deep) = make_chain(MAX_EXTENSION_DEPTH + 1);
    let err =
        WnLmfReader::read_named_with_resolver(Cursor::new(top_deep.as_bytes()), "deep.xml", &res_deep).unwrap_err();
    assert!(err.to_string().contains(&MAX_EXTENSION_DEPTH.to_string()), "{err}");
}

#[test]
fn test_normalized_and_rejected_member_separators() {
    let base_entry = "<LexicalEntry id=\"base-entry\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
        <Sense id=\"base-sense\" synset=\"base-synset\"/></LexicalEntry>\
        <LexicalEntry id=\"dog-entry\"><Lemma writtenForm=\"dog\" partOfSpeech=\"n\"/>\
        <Sense id=\"dog-sense\" synset=\"base-synset\"/></LexicalEntry>";

    for sep in [" ", "\t", "\n", "\r", "\r\n", "\t \r\n", "&#32;", "&#x20;"] {
        let doc = wrap_lmf(&format!(
            "{base_entry}<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"{sep}dog-sense{sep}{sep}base-sense{sep}\"/>"
        ));
        let kb = parse_lmf(&doc).unwrap();
        assert_eq!(&["dog", "cat"], kb.synset("base-synset").unwrap().lemmas());
    }

    for sep in ["&#9;", "&#x9;", "&#10;", "&#xA;", "&#13;", "&#xD;", "&#xA0;", "&#x2003;"] {
        let doc = wrap_lmf(&format!(
            "{base_entry}<Synset id=\"base-synset\" partOfSpeech=\"n\" members=\"dog-sense{sep}base-sense\"/>"
        ));
        let err = parse_lmf(&doc).unwrap_err();
        assert!(err.to_string().contains("dog-sense"), "{err}");
    }
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

// ============================================================================
// MorphyExceptionsTest & MorphyLemmatizerTest
// ============================================================================

#[test]
fn test_morphy_exceptions_lookup_and_validation() {
    let exc = morphy_exceptions_fixture();
    assert_eq!(&["mouse"], exc.lookup("mice", WordNetPos::Noun));
    assert_eq!(&["mouse"], exc.lookup("Mice", WordNetPos::Noun));
    assert_eq!(&["mouse"], exc.lookup("MICE", WordNetPos::Noun));
    assert_eq!(&["go"], exc.lookup("went", WordNetPos::Verb));
    assert_eq!(&["good"], exc.lookup("better", WordNetPos::Adjective));
    assert_eq!(&["well"], exc.lookup("best", WordNetPos::Adverb));
    assert!(exc.lookup("went", WordNetPos::Noun).is_empty());
    assert!(exc.lookup("dog", WordNetPos::Noun).is_empty());
    assert_eq!(&["axis", "ax"], exc.lookup("axes", WordNetPos::Noun));
}

#[test]
fn test_morphy_lemmatizer() {
    let morphy = MorphyLemmatizer::new(wndb_fixture(), morphy_exceptions_fixture());
    let one = |token: &str, tag: &str| morphy.lemmatize(&[token], &[tag]).unwrap()[0].clone();

    let cases = [
        ("mice", "NN", "mouse"),
        ("Mice", "NNS", "mouse"),
        ("men", "NNS", "man"),
        ("ran", "VBD", "run"),
        ("running", "VBG", "run"),
        ("went", "VBD", "go"),
        ("gone", "VBN", "go"),
        ("best", "RBS", "well"),
        ("dogs", "NNS", "dog"),
        ("boxes", "NNS", "box"),
        ("berries", "NNS", "berry"),
        ("runs", "NNS", "run"),
        ("runs", "VBZ", "run"),
        ("walked", "VBD", "walk"),
        ("walking", "VBG", "walk"),
        ("walks", "VBZ", "walk"),
        ("moved", "VBD", "move"),
        ("taller", "JJR", "tall"),
        ("tallest", "JJS", "tall"),
        ("larger", "JJR", "large"),
        ("dog", "NN", "dog"),
        ("quickly", "RB", "quickly"),
        ("dogs", "n", "dog"),
        ("walked", "v", "walk"),
        ("taller", "a", "tall"),
        ("best", "r", "well"),
        ("oxen", "NNS", "ox"),
        ("better", "JJR", "good"),
    ];
    for (token, tag, expected) in cases {
        assert_eq!(expected, one(token, tag), "{token}/{tag}");
    }

    let unknowns = [
        ("dogged", "VBD"),
        ("boxes", "VBZ"),
        ("glarbs", "NNS"),
        ("walk", "NN"),
        ("dog", "DT"),
        ("dog", "XYZ"),
        ("dogs", ""),
        ("was", "AUX"),
        ("taller", "AUX"),
    ];
    for (token, tag) in unknowns {
        assert_eq!("O", one(token, tag), "{token}/{tag}");
    }

    assert_eq!(
        vec!["O", "mouse", "run", "quickly"],
        morphy.lemmatize(&["The", "mice", "ran", "quickly"], &["DT", "NNS", "VBD", "RB"]).unwrap()
    );

    let candidates = morphy.lemmatize_candidates(&["glarbs", "berries", "axes"], &["NNS", "NNS", "NNS"]).unwrap();
    assert_eq!(vec![vec!["O"], vec!["berry"], vec!["axis", "ax"]], candidates);
    assert_eq!("axis", one("axes", "NNS"));

    assert_eq!(Some(WordNetPos::Noun), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("NNP"));
    assert_eq!(Some(WordNetPos::Verb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("VBZ"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("JJ"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("RBR"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("a"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("s"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("ADJ"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("ADV"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("r"));
    for tag in ["DT", "", "AUX", "ADP", "SCONJ", "SYM", "ß", "ſ"] {
        assert_eq!(None, MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag(tag), "{tag}");
    }
}

// ============================================================================
// LexicalExpanderTest & LexicalExpanderLexiconTest
// ============================================================================

fn find_expansion<'a>(expansions: &'a [Expansion], term: &str) -> Option<&'a Expansion> {
    expansions.iter().find(|e| e.term() == term)
}

#[test]
fn test_lexical_expander_over_fixtures_and_morphy() {
    let lmf = lmf_fixture();
    let wndb = wndb_fixture();

    let exp_lmf = LexicalExpander::builder(&lmf).build().expand("dog", WordNetPos::Noun).unwrap();
    let dom = find_expansion(&exp_lmf, "domestic dog").unwrap();
    assert_eq!(ExpansionKind::Synonym, dom.kind());
    assert_eq!(1.0, dom.weight());
    assert_eq!(0, dom.sense_rank());

    let canid = find_expansion(&exp_lmf, "canid").unwrap();
    assert_eq!(ExpansionKind::Hypernym, canid.kind());
    assert_eq!(1, canid.depth());
    assert_eq!(0.5, canid.weight());

    let exp_multi = LexicalExpander::builder(&wndb).build().expand("domestic_dog", WordNetPos::Noun).unwrap();
    assert_eq!(
        vec![
            Expansion::new("dog", ExpansionKind::Synonym, 0, 0, 1.0).unwrap(),
            Expansion::new("canid", ExpansionKind::Hypernym, 1, 0, 0.5).unwrap(),
        ],
        exp_multi
    );

    let morphy = MorphyLemmatizer::new(lmf.clone(), morphy_exceptions_fixture());
    let expander = LexicalExpander::builder(&lmf).lemmatizer(morphy).build();

    let dogs = expander.expand("dogs", WordNetPos::Noun).unwrap();
    assert_eq!(ExpansionKind::Synonym, find_expansion(&dogs, "dog").unwrap().kind());
    assert!(find_expansion(&dogs, "canid").is_some());

    let mice = expander.expand("mice", WordNetPos::Noun).unwrap();
    assert_eq!(ExpansionKind::Synonym, find_expansion(&mice, "mouse").unwrap().kind());
    assert!(find_expansion(&mice, "rodent").is_some());
}

// ============================================================================
// SynsetSimilarityTest & HypernymTyperTest
// ============================================================================

struct FixtureKb {
    by_id: HashMap<String, Synset>,
    by_lemma: HashMap<String, Vec<String>>,
}

impl FixtureKb {
    fn new() -> Self {
        Self { by_id: HashMap::new(), by_lemma: HashMap::new() }
    }

    fn add(&mut self, id: &str, lemma: &str, rel: WordNetRelation, parents: &[&str]) {
        let mut relations = BTreeMap::new();
        if !parents.is_empty() {
            relations.insert(rel, parents.iter().map(|s| (*s).to_string()).collect());
        }
        let synset = Synset::new(id, WordNetPos::Noun, vec![lemma.to_string()], "fixture", relations).unwrap();
        self.by_id.insert(id.to_string(), synset);
        self.by_lemma.entry(lemma.to_string()).or_default().push(id.to_string());
    }
}

impl LexicalKnowledgeBase for FixtureKb {
    fn lookup(&self, lemma: &str, _pos: WordNetPos) -> Vec<&Synset> {
        let Some(ids) = self.by_lemma.get(lemma) else {
            return Vec::new();
        };
        ids.iter().filter_map(|id| self.by_id.get(id)).collect()
    }

    fn synset(&self, synset_id: &str) -> Option<&Synset> {
        self.by_id.get(synset_id)
    }
}

fn similarity_taxonomy() -> FixtureKb {
    let mut kb = FixtureKb::new();
    kb.add("n1", "entity", WordNetRelation::Hypernym, &[]);
    kb.add("n2", "physical", WordNetRelation::Hypernym, &["n1"]);
    kb.add("n3", "organism", WordNetRelation::Hypernym, &["n2"]);
    kb.add("n4", "person", WordNetRelation::Hypernym, &["n3"]);
    kb.add("n5", "scientist", WordNetRelation::Hypernym, &["n4"]);
    kb.add("n6", "chemist", WordNetRelation::Hypernym, &["n5"]);
    kb.add("n7", "location", WordNetRelation::Hypernym, &["n2"]);
    kb.add("n8", "city", WordNetRelation::Hypernym, &["n7"]);
    kb.add("n9", "organization", WordNetRelation::Hypernym, &["n1"]);
    kb.add("n10", "company", WordNetRelation::Hypernym, &["n9"]);
    kb.add("n11", "paris", WordNetRelation::InstanceHypernym, &["n8"]);
    kb.add("n12", "abstract", WordNetRelation::Hypernym, &[]);
    kb
}

#[test]
fn test_synset_similarity_measures() {
    let sim = SynsetSimilarity::new(similarity_taxonomy());
    assert!((sim.path("n5", "n5") - 1.0).abs() < 1e-9);
    assert!((sim.path("n6", "n5") - 0.5).abs() < 1e-9);
    assert!((sim.path("n6", "n8") - 1.0 / 7.0).abs() < 1e-9);
    assert!((sim.path("n6", "n12") - 0.0).abs() < 1e-9);
    assert!((sim.path("n11", "n8") - 0.5).abs() < 1e-9);

    assert!((sim.wu_palmer("n5", "n6") - 10.0 / 11.0).abs() < 1e-9);
    let sibling = sim.wu_palmer("n6", "n8");
    assert!(sibling < sim.wu_palmer("n5", "n6") && sibling > 0.0);
    assert!((sim.wu_palmer("n6", "n12") - 0.0).abs() < 1e-9);
    assert!((sim.wu_palmer("n1", "n1") - 1.0).abs() < 1e-9);
    assert!(sim.wu_palmer("n8", "n10") > 0.0);

    assert!((sim.leacock_chodorow("n5", "n6", 10).unwrap() - 10.0_f64.ln()).abs() < 1e-9);
    assert!((sim.leacock_chodorow("n6", "n12", 10).unwrap() - 0.0).abs() < 1e-9);
    assert!((sim.leacock_chodorow("n6", "n8", 3).unwrap() - (-(7.0_f64 / 6.0).ln())).abs() < 1e-9);
    assert!(sim.leacock_chodorow("n5", "n6", 0).is_err());

    assert_eq!(None, sim.shortest_distance("n5", "missing"));
    assert_eq!(None, sim.shortest_distance("missing", "missing"));
    assert_eq!(0.0, sim.path("missing", "missing"));
    assert_eq!(0.0, sim.wu_palmer("missing", "missing"));

    let mut dag1 = FixtureKb::new();
    dag1.add("root", "root", WordNetRelation::Hypernym, &[]);
    dag1.add("middle-1", "middle-1", WordNetRelation::Hypernym, &["root"]);
    dag1.add("middle-2", "middle-2", WordNetRelation::Hypernym, &["middle-1"]);
    dag1.add("common", "common", WordNetRelation::Hypernym, &["root", "middle-2"]);
    dag1.add("left", "left", WordNetRelation::Hypernym, &["common"]);
    dag1.add("right", "right", WordNetRelation::Hypernym, &["common"]);
    assert!((SynsetSimilarity::new(dag1).wu_palmer("left", "right") - 0.8).abs() < 1e-9);

    let mut dag2 = FixtureKb::new();
    dag2.add("root", "root", WordNetRelation::Hypernym, &[]);
    dag2.add("common", "common", WordNetRelation::Hypernym, &["root"]);
    dag2.add("right", "right", WordNetRelation::Hypernym, &["common"]);
    dag2.add("branch-1", "branch-1", WordNetRelation::Hypernym, &["root"]);
    dag2.add("branch-2", "branch-2", WordNetRelation::Hypernym, &["branch-1"]);
    dag2.add("branch-3", "branch-3", WordNetRelation::Hypernym, &["branch-2"]);
    dag2.add("left", "left", WordNetRelation::Hypernym, &["common", "branch-3"]);
    assert!((SynsetSimilarity::new(dag2).wu_palmer("left", "right") - 0.5).abs() < 1e-9);

    let mut dangling = FixtureKb::new();
    dangling.add("left", "left", WordNetRelation::Hypernym, &["missing-parent"]);
    dangling.add("right", "right", WordNetRelation::Hypernym, &["missing-parent"]);
    let dsim = SynsetSimilarity::new(dangling);
    assert_eq!(None, dsim.shortest_distance("left", "right"));
    assert_eq!(0.0, dsim.path("left", "right"));
    assert_eq!(0.0, dsim.wu_palmer("left", "right"));
}

#[test]
fn test_hypernym_typer() {
    let typer = HypernymTyper::new(similarity_taxonomy(), [("person", "person"), ("location", "location")]).unwrap();
    assert_eq!(Some("person"), typer.type_lemma("chemist").unwrap());
    assert_eq!(Some("location"), typer.type_lemma("paris").unwrap());
    assert_eq!(Some("person"), typer.type_lemma("person").unwrap());
    assert_eq!(Some("location"), typer.type_synset("n8"));
    assert_eq!(None, typer.type_lemma("abstract").unwrap());
    assert_eq!(None, typer.type_lemma("unknownword").unwrap());
    assert_eq!(None, typer.type_synset("n12"));
    assert_eq!(None, typer.type_synset("missing"));

    let nested = HypernymTyper::new(similarity_taxonomy(), [("person", "person"), ("scientist", "scientist")]).unwrap();
    assert_eq!(Some("scientist"), nested.type_lemma("chemist").unwrap());
    assert_eq!(Some("scientist"), nested.type_lemma("scientist").unwrap());
    assert_eq!(None, nested.type_lemma("organism").unwrap());

    // Conflicting labels for same synset rejected
    let lmf = lmf_fixture();
    assert!(HypernymTyper::new(&lmf, [("dog", "animal"), ("domestic dog", "pet")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), Vec::<(&str, &str)>::new()).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [(" ", "person")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [("person", "\u{00A0}")]).is_err());
    assert!(HypernymTyper::new(similarity_taxonomy(), [("notaword", "label")]).is_err());
    assert!(typer.type_lemma(" ").is_err());
}

#[test]
fn test_navigate_semantic_role_relations_and_report_footprint() {
    let path = fixtures_dir().join("relation-usage-wn-lmf.xml");
    let start = Instant::now();
    let lexicon = WnLmfReader::read(&path).unwrap();
    let load_elapsed = start.elapsed();

    let purchase = lexicon.lookup("purchase", WordNetPos::Verb)[0];
    let agent_id = &purchase.related(WordNetRelation::InvolvedAgent)[0];
    let instrument_id = &purchase.related(WordNetRelation::InvolvedInstrument)[0];
    assert_eq!(&["buyer"], lexicon.synset(agent_id).unwrap().lemmas());
    assert_eq!(&["payment card"], lexicon.synset(instrument_id).unwrap().lemmas());

    // Verify load time and memory footprint for miniature WNDB + WN-LMF fixtures (#201)
    let wndb_start = Instant::now();
    let wndb = wndb_fixture();
    let wndb_elapsed = wndb_start.elapsed();
    let wndb_bytes = wndb.memory_footprint_bytes();
    let lmf_bytes = lexicon.memory_footprint_bytes();
    assert!(load_elapsed.as_secs() < 5);
    assert!(wndb_elapsed.as_secs() < 5);
    assert!(wndb_bytes > 0);
    assert!(lmf_bytes > 0);
}
