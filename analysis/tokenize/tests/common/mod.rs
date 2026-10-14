//! Reading the goldens and dumps `turbo-goldens generate` wrote for the
//! tokenizers and the detokenizer (analysis/goldens/data, format in
//! docs/analysis.md), through `turbo_goldens`, and the rules that turn
//! what the JVM returned into what the Rust port is expected to return.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_analysis::{Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Confidence, Tokens};
use turbo_goldens::{Dump, Golden, Tolerance, Unit, assert_matches_golden, data_dir, segmentation_shapes, sha512};
use turbo_tokenize::me::{Features, TokenizerModel, UnitFacts, token_context};
use turbo_tokenize::{TokenizerOptions, WhitespaceMode, whitespace_spans};
use turbo_unicode::Utf16Cursor;
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

/// The checked in fixture models and dictionaries of the generator.
pub fn fixtures_dir() -> PathBuf {
    data_dir().join("fixtures")
}

/// A published tokenizer model, where `turbo-goldens fetch-models` puts
/// it: the directory in `TURBO_GOLDENS_MODELS`, or
/// `target/turbo-goldens/models` under the repository root. The models
/// are trained on CC BY-SA treebanks and are not checked in, so a test
/// that needs one says so and passes without it when it is missing,
/// unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1`, which makes a missing
/// model fail it, as CI sets it.
pub fn published_model(file: &str) -> Option<PathBuf> {
    let dir = match std::env::var_os("TURBO_GOLDENS_MODELS") {
        Some(d) => PathBuf::from(d),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/turbo-goldens/models"),
    };
    let path = dir.join(file);
    if path.is_file() {
        Some(path)
    } else if std::env::var("TURBO_TEST_REQUIRE_OPENNLP_MODELS").is_ok_and(|v| v == "1") {
        panic!("{} not found and TURBO_TEST_REQUIRE_OPENNLP_MODELS=1", path.display());
    } else {
        eprintln!("{} not found: skipping the test that needs it", path.display());
        None
    }
}

/// One golden of a run and its input text, checked against the header.
pub struct Case {
    pub path: PathBuf,
    pub golden: Golden,
    pub text: String,
}

impl Case {
    /// `<set>/<id>`.
    pub fn name(&self) -> String {
        let i = &self.golden.header.input;
        format!("{}/{}", i.set, i.id)
    }

    pub fn set(&self) -> &str {
        &self.golden.header.input.set
    }

    /// The value of a `config` record.
    pub fn config(&self, key: &str) -> Option<&str> {
        self.golden.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
    }

    /// The whitespace mode the run used: `config whitespace legacy`, or
    /// the default.
    pub fn whitespace(&self) -> WhitespaceMode {
        match self.config("whitespace") {
            None => WhitespaceMode::Unicode,
            Some("legacy") => WhitespaceMode::Legacy,
            Some(other) => panic!("{}: whitespace mode {other}", self.path.display()),
        }
    }

    /// Asserts the annotator that ran.
    #[track_caller]
    pub fn assert_annotator(&self, class: &str, method: &str) {
        let a = &self.golden.header.annotator;
        assert_eq!((a.class.as_str(), a.method.as_str()), (class, method), "{}", self.path.display());
    }

    /// Asserts the number of model records, and the file name and SHA-512
    /// of each against the file at `paths[i]`.
    #[track_caller]
    pub fn assert_models(&self, paths: &[PathBuf]) {
        let models = &self.golden.header.models;
        assert_eq!(models.len(), paths.len(), "{}", self.path.display());
        for (m, path) in models.iter().zip(paths) {
            assert_eq!(Some(m.file.as_str()), path.file_name().and_then(|n| n.to_str()), "{}", self.path.display());
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            assert_eq!(sha512(&bytes), m.sha512, "{}: {}", self.path.display(), path.display());
        }
    }

    /// The spans the JVM returned, with their UTF-16 offsets, the
    /// confidence and word type of each when the golden has them.
    pub fn jvm_spans(&self) -> Vec<JvmSpan> {
        let g = &self.golden;
        let tokens = g.layer("opennlp:tokens").unwrap_or_else(|| panic!("{}: no tokens", self.path.display()));
        let (starts, ends) = (tokens.u32s("start").unwrap(), tokens.u32s("end").unwrap());
        let units = g.layer("turbo:utf16<opennlp:tokens>");
        let prob = g.layer("turbo:confidence<opennlp:tokens>").map(|l| l.f64s("value").unwrap());
        let kind = g.layer("turbo:word-types").map(|l| l.strs("kind").unwrap());
        let unit = |b: u32| byte_to_utf16(&self.text, b as usize).expect("a span on a character boundary") as u32;
        (0..tokens.len())
            .map(|i| {
                let (utf16, cut) = match units {
                    Some(u) => {
                        ((u.u32s("start16").unwrap()[i], u.u32s("end16").unwrap()[i]), u.u32s("cut").unwrap()[i] == 1)
                    }
                    None => ((unit(starts[i]), unit(ends[i])), false),
                };
                JvmSpan {
                    utf8: (starts[i], ends[i]),
                    utf16,
                    cut,
                    prob: prob.map(|p| p[i]),
                    kind: kind.map(|k| k.get(i).to_string()),
                }
            })
            .collect()
    }

    /// The decision point dump beside the golden.
    pub fn dump(&self) -> Dump {
        Dump::read(self.path.with_extension("dump")).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Whether the golden has a layer of UTF-16 spans, so a row cuts a
    /// character.
    pub fn has_cut(&self) -> bool {
        self.golden.layer("turbo:utf16<opennlp:tokens>").is_some()
    }
}

/// One span as the JVM returned it.
#[derive(Clone, Debug, PartialEq)]
pub struct JvmSpan {
    /// The UTF-8 span, widened to whole characters where it cuts one.
    pub utf8: (u32, u32),
    /// The JVM's exact UTF-16 span.
    pub utf16: (u32, u32),
    /// Whether it starts or ends between the two halves of a pair.
    pub cut: bool,
    pub prob: Option<f64>,
    pub kind: Option<String>,
}

/// Every golden of `<annotator>/<run>`, over all its input sets, sorted
/// by set and id.
pub fn run(annotator: &str, run: &str) -> Vec<Case> {
    let dir = data_dir().join(annotator).join(run);
    let mut paths = Vec::new();
    for set in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let set = set.unwrap().path();
        for f in std::fs::read_dir(&set).unwrap() {
            let p = f.unwrap().path();
            if p.extension().is_some_and(|x| x == "golden") {
                paths.push(p);
            }
        }
    }
    paths.sort();
    assert!(!paths.is_empty(), "no goldens in {}", dir.display());
    let shapes = segmentation_shapes();
    paths
        .into_iter()
        .map(|path| {
            let golden = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
            let text = golden.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
            Case { path, golden, text }
        })
        .collect()
}

/// The tokenizer model at `path`.
pub fn read_model(path: &Path) -> TokenizerModel {
    TokenizerModel::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A fixture model of the generator, by file name.
pub fn fixture_model(name: &str) -> TokenizerModel {
    read_model(&fixtures_dir().join(name))
}

pub const LANGUAGES: [&str; 5] = ["en", "de", "es", "fr", "pt"];

/// The file name of the UD tokenizer model for `language`, one of
/// `LANGUAGES`.
pub fn ud_model_name(language: &str) -> &'static str {
    match language {
        "en" => "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin",
        "de" => "opennlp-de-ud-gsd-tokens-1.3-2.5.4.bin",
        "es" => "opennlp-es-ud-gsd-tokens-1.3-2.5.4.bin",
        "fr" => "opennlp-fr-ud-gsd-tokens-1.3-2.5.4.bin",
        "pt" => "opennlp-pt-ud-gsd-tokens-1.3-2.5.4.bin",
        _ => panic!("no UD model for {language}"),
    }
}

/// The path of each language's UD model, in the order of `LANGUAGES`, or
/// `None` when any of them is not downloaded (`published_model` says
/// which).
pub fn ud_models() -> Option<Vec<(&'static str, PathBuf)>> {
    let found: Vec<_> = LANGUAGES.iter().map(|&l| published_model(ud_model_name(l)).map(|p| (l, p))).collect();
    found.into_iter().collect()
}

/// The run directory of a model file: its name without `.bin`.
pub fn run_name(model_file: &str) -> &str {
    model_file.strip_suffix(".bin").expect("a .bin model file")
}

/// Asserts two probabilities agree within 1e-9.
#[track_caller]
pub fn assert_close(got: f64, want: f64, what: &dyn std::fmt::Display) {
    assert!((got - want).abs() <= 1e-9, "{what}: probability {got}, the JVM gave {want}");
}

/// What the Rust maxent tokenizer produces for the spans the JVM wrote:
/// a span whose end falls between the two halves of a pair is joined with
/// the span after it, whose start is the same point, and their
/// probabilities multiply, since the port does not take that split and
/// goes on multiplying the token's decisions. Gives (UTF-8 start, UTF-8
/// end, probability).
pub fn expected_me_spans(text: &str, spans: &[JvmSpan]) -> Vec<(u32, u32, f64)> {
    let byte = |unit: u32| utf16_to_byte(text, unit as usize).map(|b| b as u32);
    let mut out = Vec::new();
    let mut open: Option<(u32, f64)> = None;
    for s in spans {
        let p = s.prob.expect("a confidence per span");
        let joined = open.is_some();
        let (start, prob) = match open.take() {
            Some((start, prob)) => {
                assert!(byte(s.utf16.0).is_none(), "a joined span continues where the last one stopped");
                (start, prob * p)
            }
            None => (byte(s.utf16.0).expect("a span start between two halves without a split before it"), p),
        };
        match byte(s.utf16.1) {
            Some(end) => {
                if !s.cut && !joined {
                    assert_eq!(s.utf8, (start, end), "the UTF-8 offsets agree with the UTF-16 ones");
                }
                out.push((start, end, prob));
            }
            None => open = Some((start, prob)),
        }
    }
    assert!(open.is_none(), "no span ends between two halves at the end of the text");
    out
}

/// A document over `text` with `spans` as its tokens and, when given,
/// their confidences.
pub fn token_document(text: &str, spans: &[(u32, u32)], probs: Option<Vec<f64>>) -> Document {
    let mut doc = Document::new(text);
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: s, sentence: None }, Provenance::by("test")).unwrap();
    if let Some(p) = probs {
        doc.add(&Key::DEFAULT, Confidence::<Tokens>::new(p), Provenance::by("test")).unwrap();
    }
    doc
}

/// Where no row of the golden cuts a character, the Rust spans and
/// probabilities, as layers on a document, match the golden's layers
/// through `assert_matches_golden`. Returns whether it compared.
pub fn compare_layers(case: &Case, spans: &[(u32, u32)], probs: Option<Vec<f64>>) -> bool {
    if case.has_cut() {
        return false;
    }
    let has_probs = probs.is_some();
    let doc = token_document(&case.text, spans, probs);
    assert_matches_golden::<Tokens>(&doc, &case.golden, Tolerance::exact());
    if has_probs {
        assert_matches_golden::<Confidence<Tokens>>(&doc, &case.golden, Tolerance::abs(1e-9));
    }
    true
}

/// Code units as Java's modified UTF-8, the bytes a feature is compared as.
pub fn modified_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for &u in units {
        turbo_unicode::modified_utf8::push_modified_utf8(&mut out, u32::from(u));
    }
    out
}

/// Checks the dump of a `TokenizerME` golden against `model`, decision by
/// decision. The dump's points must be the positions the tokenizer's loop
/// visits given the model's decisions there: every UTF-16 unit from the
/// second of each whitespace token the model is asked about, skipping the
/// dots after a split in a token the abbreviation dictionary lists. At
/// each, the features the context generator builds and the probabilities
/// the model gives must be the JVM's. Returns the number of points.
pub fn check_dump(case: &Case, model: &TokenizerModel) -> usize {
    let dump = case.dump();
    assert_eq!(dump.header, case.golden.header, "{}", case.path.display());
    let maxent = model.maxent();
    let mut features = Features::new();
    let mut probs = vec![0.0; maxent.num_outcomes()];
    let mut points = dump.points.iter();
    let mut n = 0;
    let text = case.text.as_str();
    let mut units = Vec::new();
    whitespace_spans(text, TokenizerOptions { whitespace: WhitespaceMode::Unicode, keep_newlines: false }, &mut units);
    for unit in &units {
        let (a, b) = (unit.start as usize, unit.end as usize);
        let token = &text[a..b];
        let len16 = token.encode_utf16().count();
        if len16 < 2 || model.alphanumeric().is_some_and(|al| al.test(token)) {
            continue;
        }
        let abbreviation = model.abbreviations().is_some_and(|d| d.contains(token));
        let facts = UnitFacts::of(token, abbreviation);
        let dots = token.bytes().filter(|&b| b == b'.').count();
        let base16 = byte_to_utf16(text, a).unwrap();
        let mut index = 1;
        while index < len16 {
            let what = format!("{} token {token:?} at UTF-16 unit {index}", case.path.display());
            let p = points.next().unwrap_or_else(|| panic!("{what}: the dump has no point here"));
            let at16 = base16 + index;
            let expected = match utf16_to_byte(text, at16) {
                Some(byte) => (byte as u32, Unit::Start),
                None => (utf16_to_byte(text, at16 - 1).unwrap() as u32, Unit::Inside),
            };
            assert_eq!((p.position, p.unit), expected, "{what}");
            let mut at = Utf16Cursor::new(token);
            for _ in 0..index {
                at.step_forward();
            }
            token_context(&at, facts, WhitespaceMode::Unicode, &mut features);
            assert_eq!(features.len(), p.contexts.len(), "{what}");
            for (i, want) in p.contexts.iter().enumerate() {
                assert_eq!(features.get(i), modified_utf8(want), "{what}, feature {i}");
            }
            maxent.eval_features(features.iter(), &mut probs);
            for o in &p.outcomes {
                let k = maxent.outcome_index(&o.label).unwrap();
                assert_close(probs[k], o.prob, &what);
            }
            if abbreviation && p.best().label == "T" {
                index += dots;
            }
            index += 1;
            n += 1;
        }
    }
    assert!(points.next().is_none(), "{}: points the tokenizer never reached", case.path.display());
    n
}
