//! The WNDB reader.

use super::*;

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
fn test_wndb_lexical_pointer_targets_raw_word_count_after_lemma_dedup() {
    let temp = tempfile::tempdir().unwrap();
    write_empty_db(temp.path(), &["noun", "verb", "adv"]);
    // Line 1 has w_cnt=02 with `galore(p)` and `galore(a)` which deduplicate to a single
    // cleaned lemma `"galore"`. Line 2 has a lexical pointer (`0102`) targeting word 2 of line 1.
    let template = "00000000 00 s 02 galore(p) 0 galore(a) 0 001 & XXXXXXXX a 0201 | in abundance";
    let off2 = format!("{:08}", template.len() + 1);
    let line1 = template.replace("XXXXXXXX", &off2);
    let line2 = format!("{off2} 00 a 01 abundant 0 001 & 00000000 a 0102 | present in great quantity");
    std::fs::write(temp.path().join("data.adj"), format!("{line1}\n{line2}\n")).unwrap();
    std::fs::write(
        temp.path().join("index.adj"),
        format!("abundant a 1 1 & 1 0 {off2}\ngalore a 1 1 & 1 0 00000000\n"),
    )
    .unwrap();

    let lexicon = WndbReader::read(temp.path()).unwrap();
    let syn1 = lexicon.synset("wndb-00000000-a").unwrap();
    assert_eq!(syn1.lemmas(), &["galore".to_string()]);
    assert_eq!(lexicon.related(&format!("wndb-{off2}-a"), WordNetRelation::SimilarTo), &["wndb-00000000-a"]);
}

#[test]
fn test_wndb_reads_crlf_data_and_index_files() {
    let temp = tempfile::tempdir().unwrap();
    for entry in std::fs::read_dir(mini_wndb_dir()).unwrap() {
        let entry = entry.unwrap();
        let content = std::fs::read_to_string(entry.path()).unwrap();
        let crlf = content.replace('\n', "\r\n");
        std::fs::write(temp.path().join(entry.file_name()), crlf).unwrap();
    }
    let lf_lexicon = wndb_fixture();
    let crlf_lexicon = WndbReader::read(temp.path()).unwrap();
    assert_eq!(crlf_lexicon.size(), lf_lexicon.size());
    for synset in lf_lexicon.synsets() {
        let got = crlf_lexicon.synset(synset.id()).expect("synset present in CRLF lexicon");
        assert_eq!(got.gloss(), synset.gloss());
        assert_eq!(got.lemmas(), synset.lemmas());
    }
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
