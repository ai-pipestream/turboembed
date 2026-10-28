//! The part-of-speech tag dictionary and tag format mapper against the
//! JVM (the goldens of `TagDictionaryGolden.java` and
//! `TagFormatGolden.java` in `analysis/goldens-tool`): the words and tags
//! of each dictionary, `getTags` for every probe, the format guessed from a
//! model's outcomes, `convertTag` of every probe, and the tags
//! `POSTaggerME` returns for each requested format.
//!
//! The dictionary files and the fixture model in
//! `analysis/goldens/data/fixtures` are always checked. The published
//! English UD model comes from `turbo-goldens fetch-models`
//! (`turbo_goldens::published_model`). The OpenNLP 1.5 models are
//! local-only: set `TURBO_TEST_OPENNLP_MODELS` to a directory holding them
//! (`turbo-goldens fetch-models <dir> --local` fetches them) to run those
//! tests.

mod common;

use std::path::{Path, PathBuf};

use common::{escape, grouped, layer, strs, u32s};
use turbo_goldens::Golden;
use turbo_maxent::{ModelFile, sha256};
use turbo_sequence::postag::{self, PosTagFormat, TagDictionary, TagFormatMapper};

/// The dictionary's lines `word TAB count TAB tag TAB ...`, escaped, sorted
/// by word in UTF-16 order, each ending with a line feed, as the golden
/// hashes them.
fn dump(dict: &TagDictionary) -> String {
    let mut words: Vec<(&str, &[String])> = dict.iter().collect();
    words.sort_by(|a, b| a.0.encode_utf16().cmp(b.0.encode_utf16()));
    let mut out = String::new();
    for (word, tags) in words {
        out.push_str(&escape(word));
        out.push('\t');
        out.push_str(&tags.len().to_string());
        for t in tags {
            out.push('\t');
            out.push_str(&escape(t));
        }
        out.push('\n');
    }
    out
}

/// The values of `value` for each group of rows numbered by `key` in a
/// layer whose groups the layer `of` numbers.
fn group_values(g: &Golden, of: &str, l: &str, key: &str, value: &str) -> Vec<Vec<String>> {
    let n = layer(g, of).len();
    let l = layer(g, l);
    let values = strs(l, value);
    grouped(l, key, n).into_iter().map(|rows| rows.into_iter().map(|r| values[r].clone()).collect()).collect()
}

/// Checks the model file against the golden's first model record.
fn check_model_file(g: &Golden, path: &Path) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let model = &g.header.models[0];
    assert!(path.ends_with(&model.file), "{} is not {}", path.display(), model.file);
    assert_eq!(turbo_goldens::sha512(&bytes), model.sha512, "{}: not the file the golden was made with", model.file);
}

fn check_dictionary(dict: Option<&TagDictionary>, run: &str) {
    let g = common::read("tag-dictionary", run, "sequence", "sentences");
    let summary = layer(&g, "turbo:tagdict");
    assert_eq!(dict.is_some(), u32s(summary, "present")[0] == 1, "{run}: present");
    let Some(dict) = dict else {
        assert_eq!(g.layers.len(), 1, "{run}");
        return;
    };
    assert_eq!(dict.is_case_sensitive(), u32s(summary, "case-sensitive")[0] == 1, "{run}: case sensitive");
    assert_eq!(u32s(summary, "words")[0] as usize, dict.len(), "{run}: words");
    assert_eq!(sha256(dump(dict).as_bytes()).to_hex(), strs(summary, "sha256")[0], "{run}: words and tags");

    if let Some(entries) = g.layer("turbo:tagdict-entries") {
        let tags = group_values(&g, "turbo:tagdict-entries", "turbo:tagdict-entry-tags", "entry", "tag");
        let count = u32s(entries, "tags");
        for (e, word) in strs(entries, "word").iter().enumerate() {
            assert_eq!(tags[e].len(), count[e] as usize, "{run}: {word}");
            assert_eq!(dict.tags(word), Some(&tags[e][..]), "{run}: entry {word:?}");
        }
        assert_eq!(entries.len(), dict.len(), "{run}: entries");
    }

    let lookups = layer(&g, "turbo:tagdict-lookups");
    let tags = group_values(&g, "turbo:tagdict-lookups", "turbo:tagdict-lookup-tags", "lookup", "tag");
    let (found, count) = (u32s(lookups, "found"), u32s(lookups, "tags"));
    for (l, word) in strs(lookups, "word").iter().enumerate() {
        assert_eq!(tags[l].len(), count[l] as usize, "{run}: {word}");
        let want = (found[l] == 1).then_some(&tags[l][..]);
        assert_eq!(dict.tags(word), want, "{run}: getTags({word:?})");
    }
    assert!(!lookups.is_empty(), "{run}: no lookups");
}

fn check_file(name: &str) {
    let xml = std::fs::read(common::fixtures().join(format!("{name}.tagdict"))).unwrap();
    check_dictionary(Some(&TagDictionary::read(&xml).unwrap()), name);
}

fn check_model(path: &Path) {
    let run = path.file_stem().unwrap().to_str().unwrap();
    let file = ModelFile::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    check_dictionary(TagDictionary::for_model(&file).unwrap().as_ref(), run);

    let g = common::read("tag-format", run, "sequence", "sentences");
    check_model_file(&g, path);
    let model = file.maxent(postag::MODEL_ENTRY_NAME).unwrap();
    let outcomes: Vec<&str> = model.outcome_texts().unwrap();
    assert_eq!(outcomes, strs(layer(&g, "turbo:outcomes"), "outcome"), "{run}: outcomes");
    let mapper = TagFormatMapper::guess(outcomes.iter().copied());
    assert_eq!(mapper.guessed_format().as_str(), strs(layer(&g, "turbo:guessed-format"), "format")[0], "{run}");

    let converted = layer(&g, "turbo:converted-tags");
    for (tag, want) in strs(converted, "tag").iter().zip(strs(converted, "converted")) {
        assert_eq!(mapper.convert_tag(tag), want, "{run}: convertTag({tag:?})");
    }

    let requested = layer(&g, "turbo:requested-tags");
    let (format, sentence) = (strs(requested, "format"), u32s(requested, "sentence"));
    let (raw, tag) = (strs(requested, "raw"), strs(requested, "tag"));
    let mut sentences = std::collections::BTreeSet::new();
    for r in 0..requested.len() {
        let f = PosTagFormat::parse(&format[r]).unwrap();
        let got = match TagFormatMapper::for_tagger(f, outcomes.iter().copied()) {
            Some(m) => m.convert_tag(&raw[r]).to_string(),
            None => raw[r].clone(),
        };
        assert_eq!(got, tag[r], "{run}: {f:?} sentence {}", sentence[r]);
        sentences.insert((format[r].clone(), sentence[r]));
    }
    let lines = g.input_text(turbo_goldens::data_dir()).unwrap().lines().count();
    assert_eq!(sentences.len(), 4 * lines, "{run}");
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
    check_model(&common::fixtures().join("pos-fixture.bin"));
}

#[test]
fn guesses() {
    let g = common::read("tag-format", "guesses", "tag-format", "guesses");
    let input = g.input_text(turbo_goldens::data_dir()).unwrap();
    let lines: Vec<&str> = input.lines().collect();
    let l = layer(&g, "turbo:guessed-formats");
    let (line, format) = (u32s(l, "line"), strs(l, "format"));
    for r in 0..l.len() {
        let text = lines[line[r] as usize];
        let outcomes: Vec<&str> = if text.is_empty() { Vec::new() } else { text.split(' ').collect() };
        let got = TagFormatMapper::guess(outcomes.iter().copied()).guessed_format();
        assert_eq!(got.as_str(), format[r], "{outcomes:?}");
    }
    assert_eq!(l.len(), lines.len());
}

/// An OpenNLP 1.5 model in `TURBO_TEST_OPENNLP_MODELS` (local-only: the
/// 1.5 training corpora are LDC corpora or undocumented, so CI does not
/// fetch them), or `None` after saying the test is skipped.
fn model_15(file_name: &str, test: &str) -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("TURBO_TEST_OPENNLP_MODELS") else {
        println!("{test}: skipped: TURBO_TEST_OPENNLP_MODELS is not set");
        return None;
    };
    Some(PathBuf::from(dir).join(file_name))
}

/// No tag dictionary; UD outcomes, among them joined ones like
/// `AUX+PART` that convert to `?`.
#[test]
fn en_ud_ewt_pos() {
    if let Some(path) = turbo_goldens::published_model("opennlp-en-ud-ewt-pos-1.3-2.5.4.bin") {
        check_model(&path);
    }
}

#[test]
fn en_pos_maxent() {
    if let Some(path) = model_15("en-pos-maxent.bin", "en_pos_maxent") {
        check_model(&path);
    }
}

#[test]
fn en_pos_perceptron() {
    if let Some(path) = model_15("en-pos-perceptron.bin", "en_pos_perceptron") {
        check_model(&path);
    }
}
