//! The part-of-speech tag dictionary and tag format mapper against the
//! JVM (`golden/src/PosGolden.java`): the words and tags of each
//! dictionary, `getTags` for every probe, the format guessed from a
//! model's outcomes, `convertTag` of every probe, and the tags
//! `POSTaggerME` returns for each requested format.
//!
//! The dictionary files under `golden/resources` and the fixture model
//! are always checked. The public English models are not checked in: set
//! TURBO_TEST_OPENNLP_MODELS to a directory holding them (see the crate
//! README) to run those tests; without it they say they were skipped and
//! pass, unless TURBO_TEST_REQUIRE_OPENNLP_MODELS=1.

mod common;

use std::path::{Path, PathBuf};

use common::{body, escape, field, unescape, unescape_list};
use turbo_maxent::{ModelFile, sha256};
use turbo_sequence::postag::{self, PosTagFormat, TagDictionary, TagFormatMapper};

/// The dictionary's lines `word TAB count TAB tag TAB ...`, escaped, sorted by word
/// in UTF-16 order, as the golden file hashes and lists them.
fn dump(dict: &TagDictionary) -> Vec<String> {
    let mut words: Vec<(&str, &[String])> = dict.iter().collect();
    words.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
    words
        .into_iter()
        .map(|(word, tags)| {
            let mut line = escape(word);
            line.push('\t');
            line.push_str(&tags_field(tags));
            line
        })
        .collect()
}

/// The tag count and the escaped tags, joined by tabs.
fn tags_field(tags: &[String]) -> String {
    let mut out = tags.len().to_string();
    for t in tags {
        out.push('\t');
        out.push_str(&escape(t));
    }
    out
}

fn check_dictionary(dict: Option<&TagDictionary>, name: &str) {
    let text = common::read(&format!("expected/tagdict-{name}.txt"));
    let mut lines = body(&text, "POSDictionary").peekable();
    let present = field(lines.next().unwrap(), "present") == "true";
    assert_eq!(dict.is_some(), present, "{name}: present");
    let Some(dict) = dict else {
        assert_eq!(lines.next(), None);
        return;
    };
    assert_eq!(field(lines.next().unwrap(), "case_sensitive"), dict.is_case_sensitive().to_string(), "{name}");
    assert_eq!(field(lines.next().unwrap(), "words").parse::<usize>().unwrap(), dict.len(), "{name}: words");
    let ours = dump(dict);
    let mut all = String::new();
    for line in &ours {
        all.push_str(line);
        all.push('\n');
    }
    assert_eq!(sha256(all.as_bytes()).to_hex(), field(lines.next().unwrap(), "sha256"), "{name}: words and tags");
    let listed: Vec<&str> =
        std::iter::from_fn(|| lines.next_if(|l| l.starts_with("entry:"))).map(|l| field(l, "entry")).collect();
    if !listed.is_empty() {
        assert_eq!(ours, listed, "{name}: entries");
    }
    let mut lookups = 0;
    for line in lines {
        let (probe, tags) = field(line, "lookup").split_once('\t').unwrap();
        let probe = unescape(probe);
        let got = dict.tags(&probe).map_or_else(|| "null".to_owned(), tags_field);
        assert_eq!(got, tags, "{name}: getTags({probe:?})");
        lookups += 1;
    }
    assert!(lookups > 0, "{name}: no lookups");
}

fn check_file(name: &str) {
    let xml = std::fs::read(common::golden().join(format!("resources/{name}.tagdict"))).unwrap();
    check_dictionary(Some(&TagDictionary::read(&xml).unwrap()), name);
}

fn check_model(dir: &Path, name: &str, file_name: &str) {
    let file = ModelFile::read(dir.join(file_name)).unwrap();
    check_dictionary(TagDictionary::for_model(&file).unwrap().as_ref(), name);

    let text = common::read(&format!("expected/tagformat-{name}.txt"));
    let mut lines = body(&text, "POSTagFormatMapper");
    assert_eq!(field(lines.next().unwrap(), "model"), file.sha256().to_hex(), "{name}: model file");
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let outcomes: Vec<&str> = model.outcome_texts().unwrap();
    assert_eq!(outcomes, unescape_list(field(lines.next().unwrap(), "outcomes")), "{name}: outcomes");
    let mapper = TagFormatMapper::guess(outcomes.iter().copied());
    assert_eq!(mapper.guessed_format().as_str(), field(lines.next().unwrap(), "guessed"), "{name}");

    let mut lines = lines.peekable();
    while let Some(line) = lines.next_if(|l| l.starts_with("convert:")) {
        let (tag, want) = field(line, "convert").split_once('\t').unwrap();
        let tag = unescape(tag);
        assert_eq!(mapper.convert_tag(&tag), unescape(want), "{name}: convertTag({tag:?})");
    }

    let mut requested = None;
    let mut sentences = 0;
    while let Some(line) = lines.next() {
        if let Some(format) = line.strip_prefix("requested: ") {
            let format = PosTagFormat::parse(format).unwrap();
            requested = Some((format, TagFormatMapper::for_tagger(format, outcomes.iter().copied())));
            continue;
        }
        let (_, tagger) = requested.unwrap();
        let raw = unescape_list(field(line, "raw"));
        let want = unescape_list(field(lines.next().unwrap(), "tags"));
        let got: Vec<&str> = match &tagger {
            Some(m) => raw.iter().map(|t| m.convert_tag(t)).collect(),
            None => raw.iter().map(String::as_str).collect(),
        };
        assert_eq!(got, want, "{name}: {requested:?}");
        sentences += 1;
    }
    assert_eq!(sentences, 4 * common::read("inputs/sentences.txt").lines().count(), "{name}");
}

#[test]
fn fixture_dictionary() {
    check_file("tags");
}

/// Case-insensitive, with several words lowercasing to the same one,
/// where Java's hash order, not the file order, picks the tags.
#[test]
fn fixture_dictionary_with_case_collisions() {
    check_file("tags-cases");
}

/// The same entries, case sensitive (the root has no `case_sensitive`).
#[test]
fn fixture_dictionary_case_sensitive() {
    check_file("tags-exact");
}

#[test]
fn pos_fixture() {
    check_model(&common::golden().join("models"), "pos-fixture", "pos-fixture.bin");
}

#[test]
fn guesses() {
    let text = common::read("expected/tagformat-guess.txt");
    let mut lines = body(&text, "POSTagFormatMapper");
    let mut n = 0;
    while let Some(line) = lines.next() {
        let outcomes = unescape_list(field(line, "outcomes"));
        let want = field(lines.next().unwrap(), "guessed");
        let got = TagFormatMapper::guess(outcomes.iter().map(String::as_str)).guessed_format();
        assert_eq!(got.as_str(), want, "{outcomes:?}");
        n += 1;
    }
    assert!(n > 0);
}

/// The directory of the public English models, or None after saying the
/// test is skipped.
fn real_models(test: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS") else {
        if std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1") {
            panic!("{test}: TURBO_TEST_OPENNLP_MODELS is not set");
        }
        println!("{test}: skipped: TURBO_TEST_OPENNLP_MODELS is not set");
        return None;
    };
    Some(PathBuf::from(dir))
}

/// No tag dictionary; UD outcomes, among them joined ones like
/// `AUX+PART` that convert to `?`.
#[test]
fn en_ud_ewt_pos() {
    if let Some(dir) = real_models("en_ud_ewt_pos") {
        check_model(&dir, "en-ud-ewt-pos", "opennlp-en-ud-ewt-pos-1.3-2.5.4.bin");
    }
}

#[test]
fn en_pos_maxent() {
    if let Some(dir) = real_models("en_pos_maxent") {
        check_model(&dir, "en-pos-maxent", "en-pos-maxent.bin");
    }
}

#[test]
fn en_pos_perceptron() {
    if let Some(dir) = real_models("en_pos_perceptron") {
        check_model(&dir, "en-pos-perceptron", "en-pos-perceptron.bin");
    }
}
