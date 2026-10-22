//! The lattice tokenizer against what OpenNLP's `LatticeTokenizer`
//! returned on the JVM for the same MeCab dictionaries and inputs
//! (analysis/goldens/data/lattice): the morpheme spans in UTF-8 offsets,
//! each morpheme's feature columns and whether unknown word handling made
//! it, and, where the JVM threw, the position it named.
//!
//! The fixture dictionaries are checked in and always run. IPADIC and
//! mecab-ko-dic are fetched by `turbo-goldens fetch-models` (IPADIC with
//! its UTF-8 copy); those tests are ignored by default and read the
//! directory in `TURBO_MECAB_DICTIONARIES`, or target/turbo-goldens/models.
//! CI runs them with `--ignored` after `fetch-models`, so there a missing
//! dictionary fails them.

mod common;

use std::path::{Path, PathBuf};

use common::{Case, compare_layers, fixtures_dir, run};
use turbo_analysis::{Annotator, Document, View};
use turbo_goldens::sha512;
use turbo_tokenize::lattice::Scratch;
use turbo_tokenize::{LatticeTokenizer, MecabDictionary, WhitespaceMode};
use turbo_unicode::utf16::byte_to_utf16;

const CLASS: &str = "opennlp.tools.tokenize.lattice.LatticeTokenizer";

/// The directory of the fetched dictionaries.
fn dictionaries_dir() -> PathBuf {
    match std::env::var_os("TURBO_MECAB_DICTIONARIES") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The SHA-512 of the manifest the generator writes for a dictionary
/// directory: each file read, in the order read, as its name, a tab, the
/// SHA-512 of its bytes in hex and a line feed.
fn manifest_sha512(dir: &Path) -> [u8; 64] {
    let mut csv: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|n| n.ends_with(".csv"))
        .collect();
    csv.sort();
    let names = std::iter::once("matrix.def".to_string()).chain(csv).chain(["char.def".into(), "unk.def".into()]);
    let mut m = String::new();
    for name in names {
        let bytes = std::fs::read(dir.join(&name)).unwrap();
        m.push_str(&format!("{name}\t{}\n", hex(&sha512(&bytes))));
    }
    sha512(m.as_bytes())
}

/// Asserts the golden's model records: each directory's name and the
/// SHA-512 of its manifest.
#[track_caller]
fn assert_dictionaries(case: &Case, dirs: &[(&str, [u8; 64])]) {
    let models = &case.golden.header.models;
    assert_eq!(models.len(), dirs.len(), "{}", case.path.display());
    for (m, (name, hash)) in models.iter().zip(dirs) {
        assert_eq!((m.file.as_str(), &m.sha512), (*name, hash), "{}", case.path.display());
    }
}

/// The UTF-16 position of the JVM's "no candidate" error.
fn error_position(message: &str) -> u32 {
    let rest = message.strip_prefix("dictionary provides no candidate at position ").expect(message);
    rest.split(';').next().unwrap().parse().unwrap()
}

/// What a run compared.
#[derive(Debug, Default, PartialEq, Eq)]
struct Counts {
    goldens: usize,
    morphemes: usize,
    unknown: usize,
    errors: usize,
}

/// Analyzes every input of `run_name` with the dictionary at `dir` (whose
/// golden model records are `models`) and compares with the JVM.
fn matches(run_name: &str, dir: &Path, models: &[&Path], mode: WhitespaceMode) -> Counts {
    let dictionary = MecabDictionary::load(dir, mode).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let tokenizer = LatticeTokenizer::new(dictionary, mode);
    let mut counts = Counts::default();
    let records: Vec<(&str, [u8; 64])> =
        models.iter().map(|d| (d.file_name().unwrap().to_str().unwrap(), manifest_sha512(d))).collect();
    let mut scratch = Scratch::new();
    let mut out = Vec::new();
    for case in run("lattice", run_name) {
        case.assert_annotator(CLASS, "analyze");
        assert_dictionaries(&case, &records);
        assert_eq!(case.whitespace(), mode, "{}", case.path.display());
        assert!(!case.has_cut(), "{}", case.path.display());
        let text = &case.text;
        out.clear();
        let result = tokenizer.analyze_with(text, &mut scratch, &mut out);
        let doc = Document::new(text.as_str());
        let annotated = tokenizer.annotate((doc.get::<View>().unwrap(), None));
        counts.goldens += 1;
        if let Some(error) = &case.golden.header.error {
            assert_eq!(error.class, "java.lang.IllegalStateException", "{}", case.path.display());
            let offset = result.expect_err(&case.name()).offset;
            let unit = byte_to_utf16(text, offset).unwrap() as u32;
            assert_eq!(unit, error_position(&error.message), "{}", case.path.display());
            assert!(annotated.is_err(), "{}", case.path.display());
            counts.errors += 1;
            continue;
        }
        result.unwrap_or_else(|e| panic!("{}: {e}", case.path.display()));
        let got: Vec<(u32, u32)> = out.iter().map(|m| (m.span.start, m.span.end)).collect();
        let want: Vec<(u32, u32)> = case.jvm_spans().iter().map(|s| s.utf8).collect();
        assert_eq!(got, want, "{}", case.path.display());
        assert!(compare_layers(&case, &got, None));

        let layer = case.golden.layer("turbo:morphemes").expect("a morphemes layer");
        let (unknown, features) = (layer.u32s("unknown").unwrap(), layer.strs("features").unwrap());
        let d = tokenizer.dictionary();
        for (i, m) in out.iter().enumerate() {
            let joined = d.features(m.entry).collect::<Vec<_>>().join(",");
            assert_eq!(
                (m.unknown, joined.as_str()),
                (unknown[i] == 1, features.get(i)),
                "{} morpheme {i} {:?}",
                case.path.display(),
                &text[m.span.range()]
            );
            counts.unknown += usize::from(m.unknown);
        }
        counts.morphemes += out.len();

        // The annotator writes the same tokens, each tagged with its
        // entry's first feature column.
        let (tokens, pos) = annotated.unwrap();
        let spans: Vec<(u32, u32)> = tokens.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(spans, got, "{}", case.path.display());
        for (m, &tag) in out.iter().zip(&pos.tags) {
            assert_eq!(pos.tagset.label(tag), d.features(m.entry).next().unwrap_or(""), "{}", case.path.display());
        }
    }
    counts
}

fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn counts(goldens: usize, morphemes: usize, unknown: usize, errors: usize) -> Counts {
    Counts { goldens, morphemes, unknown, errors }
}

#[test]
fn japanese_fixture() {
    let dir = fixture("mecab-fixture-ja");
    let got = matches("mecab-fixture-ja", &dir, &[&dir], WhitespaceMode::Unicode);
    assert_eq!(got, counts(65, 1497, 1244, 0));
}

#[test]
fn japanese_fixture_legacy() {
    let dir = fixture("mecab-fixture-ja");
    let got = matches("mecab-fixture-ja+legacy", &dir, &[&dir], WhitespaceMode::Legacy);
    assert_eq!(got, counts(65, 1494, 1241, 0));
}

#[test]
fn korean_fixture() {
    let dir = fixture("mecab-fixture-ko");
    let got = matches("mecab-fixture-ko", &dir, &[&dir], WhitespaceMode::Unicode);
    assert_eq!(got, counts(55, 1310, 1155, 0));
}

/// A dictionary whose unk.def has no DEFAULT template: the JVM throws on
/// a character no other template covers, and the port fails at the same
/// character.
#[test]
fn a_dictionary_without_a_default_template() {
    let dir = fixture("mecab-fixture-partial");
    let got = matches("mecab-fixture-partial", &dir, &[&dir], WhitespaceMode::Unicode);
    assert_eq!(got, counts(48, 17, 11, 44));
}

/// IPADIC, which is EUC-JP: the JVM loaded the original and the port
/// reads the UTF-8 copy, which the generator checked is the same text.
#[test]
#[ignore = "needs IPADIC from turbo-goldens fetch-models"]
fn ipadic() {
    let models = dictionaries_dir();
    let original = models.join("mecab-ipadic-2.7.0-20070801");
    let copy = models.join("mecab-ipadic-2.7.0-20070801.utf8");
    let got = matches("mecab-ipadic-2.7.0-20070801", &copy, &[&original, &copy], WhitespaceMode::Unicode);
    assert_eq!(got, counts(65, 1512, 1068, 0));
}

#[test]
#[ignore = "needs mecab-ko-dic from turbo-goldens fetch-models"]
fn mecab_ko_dic() {
    let dir = dictionaries_dir().join("mecab-ko-dic-2.1.1-20180720");
    let got = matches("mecab-ko-dic-2.1.1-20180720", &dir, &[&dir], WhitespaceMode::Unicode);
    assert_eq!(got, counts(61, 1360, 971, 0));
}

#[test]
#[ignore = "needs UniDic from turbo-goldens fetch-models"]
fn unidic() {
    let dir = dictionaries_dir().join("unidic-mecab-2.1.2");
    let got = matches("unidic-mecab-2.1.2", &dir, &[&dir], WhitespaceMode::Unicode);
    assert_eq!(got, counts(65, 2016, 519, 0));
}

fn file_shapes(text: &str) -> Vec<turbo_goldens::Shape> {
    use turbo_goldens::{ColumnSpec, ColumnType, GoldenAnchor, Shape, segmentation_shapes};
    let mut shapes = segmentation_shapes();
    for line in text.lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first() != Some(&"layer") || fields.len() < 5 {
            continue;
        }
        let id = fields[1];
        if shapes.iter().any(|s| s.id == id) {
            continue;
        }
        let anchor = match fields[2] {
            "text" => GoldenAnchor::Text,
            "document" => GoldenAnchor::Document,
            a => GoldenAnchor::Aligned(a.strip_prefix("aligned:").unwrap().into()),
        };
        let cols: Vec<ColumnSpec> = fields[4..]
            .iter()
            .map(|f| {
                let (name, ty) = f.split_once(':').unwrap();
                let name: &'static str = Box::leak(name.to_string().into_boxed_str());
                let ty = match ty {
                    "u32" => ColumnType::U32,
                    "f64" => ColumnType::F64,
                    "str" => ColumnType::Str,
                    other => panic!("unexpected column type {other}"),
                };
                ColumnSpec::new(name, ty)
            })
            .collect();
        let columns: &'static [ColumnSpec] = Box::leak(cols.into_boxed_slice());
        shapes.push(Shape { id: id.into(), anchor, columns });
    }
    shapes
}

fn collect_goldens(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_goldens(&path, out);
        } else if path.extension().is_some_and(|e| e == "golden") {
            out.push(path);
        }
    }
}

#[test]
fn unidic_goldens_round_trip() {
    let cases = run("lattice", "unidic-mecab-2.1.2");
    assert_eq!(cases.len(), 65);
    for case in cases {
        let raw = std::fs::read_to_string(&case.path).unwrap();
        assert!(!raw.contains('\u{2014}'), "{}", case.path.display());
        assert_eq!(case.golden.to_text().unwrap(), raw, "{}", case.path.display());
    }

    let mut all = Vec::new();
    collect_goldens(&turbo_goldens::data_dir(), &mut all);
    all.sort();
    assert!(all.len() > 3000, "{} checked-in goldens", all.len());
    for path in all {
        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains('\u{2014}'), "{}", path.display());
        let shapes = file_shapes(&raw);
        match turbo_goldens::Golden::parse_with(&raw, &shapes) {
            Ok(golden) => assert_eq!(golden.to_text().unwrap(), raw, "{}", path.display()),
            Err(turbo_goldens::GoldenError::Escape { error, .. })
                if matches!(error.kind, turbo_goldens::escape::EscapeErrorKind::LoneSurrogate(_)) => {}
            Err(e) => panic!("{}: {e}", path.display()),
        }
    }
}
