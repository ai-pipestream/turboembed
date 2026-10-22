//! The readings layer of the lattice analyzer against the feature columns
//! the JVM printed for the same morphemes (analysis/goldens/data/lattice):
//! for every token of every golden, the reading, pronunciation and lemma
//! are the columns the format names, split out of the JVM's feature
//! string, and absent where that column is missing, empty or `*`.
//!
//! The Korean fixture is in mecab-ko-dic's layout and always runs. IPADIC
//! and mecab-ko-dic are read as `lattice_golden.rs` reads them: ignored by
//! default, run in CI with `--ignored` after `turbo-goldens fetch-models`.
//! No UniDic is fetched; its layouts are tested on entries written in them
//! (`analysis/tokenize/src/lattice/format.rs`).

mod common;

use std::path::{Path, PathBuf};

use common::{fixtures_dir, run};
use turbo_analysis::{Analyzer, Annotator, Document, Error, Key, Layer, Refs, Span, Spans, View};
use turbo_analysis_layers::{PosTags, Sentences, Strs, Tokens};
use turbo_tokenize::lattice::{DictionaryFormat, FormatError, UnidicLayout};
use turbo_tokenize::{LatticeAnalyzer, LatticeTokenizer, MecabDictionary, Readings, WhitespaceMode};

fn dictionaries_dir() -> PathBuf {
    match std::env::var_os("TURBO_MECAB_DICTIONARIES") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    }
}

fn analyzer(dir: &Path, format: DictionaryFormat) -> LatticeAnalyzer {
    let dictionary =
        MecabDictionary::load(dir, WhitespaceMode::Unicode).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    LatticeAnalyzer::new(LatticeTokenizer::new(dictionary, WhitespaceMode::Unicode), format).unwrap()
}

/// How many tokens were compared, and how many had each column.
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    tokens: usize,
    reading: usize,
    pronunciation: usize,
    lemma: usize,
}

/// Item `i` of a column as an option: `None` for an empty string.
fn item(column: &Option<Strs>, i: usize) -> Option<Option<&str>> {
    column.as_ref().map(|c| Some(c.get(i)).filter(|v| !v.is_empty()))
}

/// Runs the analyzer on every golden of `run_name` and compares each
/// token's readings with the JVM's feature columns. A column map naming a
/// column that is always `*` would match the JVM everywhere; the counts
/// the tests assert are what catch it.
fn matches(run_name: &str, analyzer: LatticeAnalyzer) -> Counts {
    let map = analyzer.format().map();
    let pipeline = Analyzer::builder().add(analyzer).build().unwrap();
    let mut counts = Counts::default();
    for case in run("lattice", run_name) {
        if case.golden.header.error.is_some() {
            continue;
        }
        let mut docs = vec![Document::new(case.text.as_str())];
        pipeline.run(&mut docs)[0].clone().unwrap_or_else(|e| panic!("{}: {e}", case.path.display()));
        let doc = &docs[0];
        let tokens = doc.get::<Tokens>().unwrap();
        let readings = doc.get::<Readings>().unwrap();
        assert_eq!(doc.get::<PosTags>().unwrap().tags.len(), tokens.len());
        assert_eq!(readings.len(), tokens.len(), "{}", case.path.display());
        let got: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        let want: Vec<(u32, u32)> = case.jvm_spans().iter().map(|s| s.utf8).collect();
        assert_eq!(got, want, "{}", case.path.display());

        let features = case.golden.layer("turbo:morphemes").expect("a morphemes layer").strs("features").unwrap();
        for (i, &(a, b)) in got.iter().enumerate() {
            // The published dictionaries' features hold no comma, so the
            // JVM's joined string splits back into its columns.
            let columns: Vec<&str> = features.get(i).split(',').collect();
            let expected = |c: Option<usize>| c.map(|c| columns.get(c).copied().filter(|v| !v.is_empty() && *v != "*"));
            let what = || format!("{} token {i} {:?}", case.path.display(), &case.text[a as usize..b as usize]);
            assert_eq!(item(&readings.reading, i), expected(map.reading), "reading, {}", what());
            assert_eq!(item(&readings.pronunciation, i), expected(map.pronunciation), "pronunciation, {}", what());
            assert_eq!(item(&readings.lemma, i), expected(map.lemma), "lemma, {}", what());
            counts.reading += usize::from(matches!(item(&readings.reading, i), Some(Some(_))));
            counts.pronunciation += usize::from(matches!(item(&readings.pronunciation, i), Some(Some(_))));
            counts.lemma += usize::from(matches!(item(&readings.lemma, i), Some(Some(_))));
        }
        counts.tokens += tokens.len();
    }
    counts
}

fn counts(tokens: usize, reading: usize, pronunciation: usize, lemma: usize) -> Counts {
    Counts { tokens, reading, pronunciation, lemma }
}

#[test]
fn korean_fixture() {
    let a = analyzer(&fixtures_dir().join("mecab-fixture-ko"), DictionaryFormat::MecabKoDic);
    let got = matches("mecab-fixture-ko", a);
    assert_eq!(got, counts(1310, 46, 0, 0));
}

/// The Japanese fixture has four feature columns, which is no published
/// format.
#[test]
fn the_japanese_fixture_is_no_published_format() {
    let dir = fixtures_dir().join("mecab-fixture-ja");
    let d = MecabDictionary::load(&dir, WhitespaceMode::Unicode).unwrap();
    let t = LatticeTokenizer::new(d, WhitespaceMode::Unicode);
    let formats = [
        DictionaryFormat::Ipadic,
        DictionaryFormat::MecabKoDic,
        DictionaryFormat::Unidic(UnidicLayout::Source17),
        DictionaryFormat::Unidic(UnidicLayout::Full26),
        DictionaryFormat::Unidic(UnidicLayout::Full29),
    ];
    for format in formats {
        let err: FormatError = LatticeAnalyzer::new(t.clone(), format).unwrap_err();
        assert_eq!(err.found, 4, "{format:?}");
    }
}

/// A toy sentence splitter: a sentence ends after "。".
struct Sentencer;

impl Annotator for Sentencer {
    const ID: &'static str = "test:sentences";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Sentences,), Error> {
        let text = view.text();
        let mut spans = Spans::new(view.id());
        let mut start = 0;
        for (i, m) in text.match_indices('。') {
            spans.push(Span::new(start as u32, (i + m.len()) as u32));
            start = i + m.len();
        }
        if start < text.len() {
            spans.push(Span::new(start as u32, text.len() as u32));
        }
        Ok((Sentences { spans },))
    }
}

/// With sentences, each is analyzed on its own; the readings stay aligned
/// to the tokens, the provenance names the dictionary, and the format is
/// part of the configuration.
#[test]
fn readings_follow_the_tokens_through_a_pipeline() {
    let a = analyzer(&fixtures_dir().join("mecab-fixture-ko"), DictionaryFormat::MecabKoDic);
    let text = "학교에 갔다。학생이 왔다";
    // What each sentence gives alone, the second at its offset in the text,
    // and each token's 읽기.
    let second = "학교에 갔다。".len();
    let mut want: Vec<((u32, u32), String)> = Vec::new();
    for (base, sentence) in [(0, &text[..second]), (second, &text[second..])] {
        let mut out = Vec::new();
        a.tokenizer().analyze(sentence, &mut out).unwrap();
        let d = a.tokenizer().dictionary();
        for m in &out {
            let reading = a.format().map().reading(d, m.entry).unwrap_or("").to_string();
            want.push(((m.span.start + base as u32, m.span.end + base as u32), reading));
        }
    }
    let config = a.config_hash();
    assert_ne!(config, a.tokenizer().config_hash());

    let pipeline = Analyzer::builder().add(a).add(Sentencer).build().unwrap();
    let mut docs = vec![Document::new(text)];
    pipeline.run(&mut docs)[0].clone().unwrap();
    let doc = &docs[0];
    let tokens = doc.get::<Tokens>().unwrap();
    let readings = doc.get::<Readings>().unwrap();
    assert!(tokens.sentence.is_some());
    assert!(readings.pronunciation.is_none() && readings.lemma.is_none());
    let reading = readings.reading.as_ref().unwrap();
    let got: Vec<((u32, u32), String)> =
        tokens.spans.iter().enumerate().map(|(i, (_, s))| ((s.start, s.end), reading.get(i).to_string())).collect();
    assert_eq!(got, want);
    assert!(want.iter().any(|(s, r)| s.0 as usize >= second && !r.is_empty()));
    let p = doc.provenance(Key::<Readings>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "turbo-tokenize:lattice-readings");
    assert_eq!(p.model.as_ref().unwrap().id, "opennlp:MecabDictionary");
    assert_eq!(p.config, config);
}

/// IPADIC, from its UTF-8 copy: every lexicon entry has 読み, 発音 and
/// 原形; the unknown words have none.
#[test]
#[ignore = "needs IPADIC from turbo-goldens fetch-models"]
fn ipadic() {
    let a = analyzer(&dictionaries_dir().join("mecab-ipadic-2.7.0-20070801.utf8"), DictionaryFormat::Ipadic);
    // Spot checks against the dictionary's own lines.
    let d = a.tokenizer().dictionary();
    let map = a.format().map();
    let mut out = Vec::new();
    a.tokenizer().analyze("東京へ行った", &mut out).unwrap();
    let seen: Vec<_> =
        out.iter().map(|m| (map.reading(d, m.entry), map.pronunciation(d, m.entry), map.lemma(d, m.entry))).collect();
    assert_eq!(
        seen,
        [
            (Some("トウキョウ"), Some("トーキョー"), Some("東京")),
            (Some("ヘ"), Some("エ"), Some("へ")),
            (Some("イッ"), Some("イッ"), Some("行く")),
            (Some("タ"), Some("タ"), Some("た")),
        ]
    );
    let got = matches("mecab-ipadic-2.7.0-20070801", a);
    assert_eq!(got, counts(1512, 444, 444, 444));
}

/// mecab-ko-dic: 읽기 only, `*` for symbols and unknown words.
#[test]
#[ignore = "needs mecab-ko-dic from turbo-goldens fetch-models"]
fn mecab_ko_dic() {
    let a = analyzer(&dictionaries_dir().join("mecab-ko-dic-2.1.1-20180720"), DictionaryFormat::MecabKoDic);
    // A Hanja entry is read in Hangul.
    let d = a.tokenizer().dictionary();
    let mut out = Vec::new();
    a.tokenizer().analyze("學校", &mut out).unwrap();
    let readings: Vec<_> = out.iter().map(|m| a.format().map().reading(d, m.entry)).collect();
    assert_eq!(readings, [Some("학교")]);
    // IPADIC's column count refuses it.
    let err = DictionaryFormat::Ipadic.check(d).unwrap_err();
    assert_eq!(err.found, 8);
    let got = matches("mecab-ko-dic-2.1.1-20180720", a);
    assert_eq!(got, counts(1360, 211, 0, 0));
}

/// UniDic 2.1.2 (17-column source layout): `pron` and `orthBase` (`lemma`),
/// no `kana` (`reading`) column.
#[test]
#[ignore = "needs UniDic from turbo-goldens fetch-models"]
fn unidic() {
    let a = analyzer(&dictionaries_dir().join("unidic-mecab-2.1.2"), DictionaryFormat::Unidic(UnidicLayout::Source17));
    let d = a.tokenizer().dictionary();
    assert_eq!(UnidicLayout::of(d), Some(UnidicLayout::Source17));
    let map = a.format().map();
    let mut out = Vec::new();
    a.tokenizer().analyze("東京へ行った", &mut out).unwrap();
    let seen: Vec<_> =
        out.iter().map(|m| (map.reading(d, m.entry), map.pronunciation(d, m.entry), map.lemma(d, m.entry))).collect();
    assert_eq!(
        seen,
        [
            (None, Some("トーキョー"), Some("東京")),
            (None, Some("エ"), Some("へ")),
            (None, Some("イッ"), Some("行く")),
            (None, Some("タ"), Some("た")),
        ]
    );
    let got = matches("unidic-mecab-2.1.2", a);
    assert_eq!(got, counts(2016, 0, 1219, 1496));
}
