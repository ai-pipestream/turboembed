//! The golden reader, writer and comparison against a hand written golden
//! file, `tests/fixtures/zurich.golden`, for the input
//! `tests/fixtures/inputs/test/zurich.txt`. The file was written by hand
//! in the format a generator writes: its f64 fields are what OpenJDK 21's
//! `Double.toString` printed for the values, and its commit and model
//! digests are stand-ins, not those of a real build or model.

use std::path::PathBuf;

use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{Confidence, Entities, Language, Lemmas, PosTags, Sentences, Strs, TagSet, Tokens};
use turbo_goldens::*;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn fixture_path() -> PathBuf {
    fixtures().join("zurich.golden")
}

fn fixture_text() -> String {
    std::fs::read_to_string(fixture_path()).unwrap()
}

fn golden() -> Golden {
    Golden::read(fixture_path()).unwrap()
}

const TEXT: &str = "Dr. Smith went to Zürich. It rained!";

// ---------------------------------------------------------------------
// Reading

#[test]
fn the_fixture_reads_into_typed_fields() {
    let g = golden();
    let h = &g.header;
    assert_eq!(h.generator, Generator { name: "turbo-goldens-java".into(), version: 1 });
    assert_eq!(hex::to_hex(&h.opennlp.commit), "0123456789abcdef0123456789abcdef01234567");
    assert_eq!(h.opennlp.commit[0], 0x01);
    assert_eq!(h.opennlp.version, "3.0.0-PRE-SNAPSHOT");
    assert_eq!(
        h.jdk,
        Jdk { version: "21.0.5".into(), vm_version: "21.0.5+11-LTS".into(), vendor: "Eclipse Adoptium".into() }
    );
    assert_eq!(h.unicode, UnicodeVersion { major: 15, minor: 0 });
    assert_eq!(h.input.set, "test");
    assert_eq!(h.input.id, "zurich");
    assert_eq!(h.input.utf8_len, 37);
    assert_eq!(h.input.utf16_len, 36);
    assert_eq!(h.input.sha256, sha256(TEXT.as_bytes()));
    assert_eq!(h.models.len(), 2);
    assert_eq!(h.models[0].file, "fixture-sentence.bin");
    assert_eq!(h.models[0].sha512, sha512(b"fixture model"));
    assert_eq!(h.models[1].file, "fixture-tokens.bin");
    assert_eq!(h.models[1].sha512, sha512(b""));
    assert_eq!(
        h.annotator,
        AnnotatorRef { class: "opennlp.tools.tokenize.TokenizerME".into(), method: "tokenizePos".into() }
    );

    let ids: Vec<&str> = g.layers.iter().map(|l| l.id.as_str()).collect();
    assert_eq!(
        ids,
        [
            "opennlp:sentences",
            "turbo:confidence<opennlp:sentences>",
            "opennlp:tokens",
            "turbo:confidence<opennlp:tokens>",
            "opennlp:pos",
            "opennlp:lemmas",
            "opennlp:entities",
            "opennlp:language"
        ]
    );
    let s = g.layer("opennlp:sentences").unwrap();
    assert_eq!(s.anchor, GoldenAnchor::Text);
    assert_eq!(s.u32s("start").unwrap(), [0, 27]);
    assert_eq!(s.u32s("end").unwrap(), [26, 37]);
    let c = g.layer("turbo:confidence<opennlp:sentences>").unwrap();
    assert_eq!(c.anchor, GoldenAnchor::Aligned("opennlp:sentences".into()));
    assert_eq!(c.f64s("value").unwrap(), [0.9876543210987654, 0.95]);
    let t = g.layer("opennlp:tokens").unwrap();
    assert_eq!(t.len(), 9);
    assert_eq!(t.column_names(), ["start:u32", "end:u32"]);
    let pos = g.layer("opennlp:pos").unwrap();
    assert_eq!(pos.strs("tag").unwrap().iter().collect::<Vec<_>>()[..3], ["PROPN", "PROPN", "VERB"]);
    assert_eq!(g.layer("opennlp:lemmas").unwrap().strs("lemma").unwrap().get(4), "Zürich");
    let e = g.layer("opennlp:entities").unwrap();
    assert_eq!(e.u32s("token_start").unwrap(), [1, 4]);
    assert_eq!(e.strs("kind").unwrap().get(1), "location");
    let lang = g.layer("opennlp:language").unwrap();
    assert_eq!(lang.anchor, GoldenAnchor::Document);
    assert_eq!(lang.len(), 4);
    assert_eq!(lang.f64s("confidence").unwrap()[3], 1.0e-5);
}

#[test]
fn the_input_text_is_read_and_checked() {
    let g = golden();
    assert_eq!(g.input_text(fixtures()).unwrap(), TEXT);

    // A copy of the input with one byte changed has the right length and
    // the wrong digest.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("goldens-input");
    std::fs::create_dir_all(dir.join("inputs/test")).unwrap();
    std::fs::write(dir.join("inputs/test/zurich.txt"), TEXT.replace("Dr.", "Dr!")).unwrap();
    let e = g.input_text(&dir).unwrap_err();
    assert!(matches!(e, GoldenError::Input { what: "SHA-256", .. }), "{e}");
    std::fs::write(dir.join("inputs/test/zurich.txt"), format!("{TEXT}\n")).unwrap();
    let e = g.input_text(&dir).unwrap_err().to_string();
    assert!(e.contains("length in UTF-8 bytes is 38, expected 37"), "{e}");
    std::fs::remove_file(dir.join("inputs/test/zurich.txt")).unwrap();
    assert!(matches!(g.input_text(&dir).unwrap_err(), GoldenError::Io { .. }));
}

#[test]
fn a_missing_file_names_its_path() {
    let e = Golden::read(fixtures().join("no-such.golden")).unwrap_err();
    assert!(matches!(e, GoldenError::Io { .. }));
    assert!(e.to_string().contains("no-such.golden"), "{e}");
}

// ---------------------------------------------------------------------
// Writing

#[test]
fn writing_the_fixture_gives_its_bytes_and_reads_back_equal() {
    let g = golden();
    let mut out = Vec::new();
    g.write(&mut out).unwrap();
    assert_eq!(String::from_utf8(out.clone()).unwrap(), fixture_text());
    assert_eq!(Golden::parse_bytes(&out).unwrap(), g);
}

#[test]
fn escapes_and_special_values_round_trip() {
    let mut g = golden();
    // Lemmas with every kind of escape, and values Java prints in each form.
    let mut lemma = Strs::default();
    for s in ["tab\there", "line\nend", "cr\rx", "back\\slash", "nul\u{0}bell\u{7}", "😀", "", "x", "y"] {
        lemma.push(s);
    }
    let lemmas = g.layers.iter_mut().find(|l| l.id == "opennlp:lemmas").unwrap();
    lemmas.columns[0].data = ColumnData::Str(lemma);
    let conf = g.layers.iter_mut().find(|l| l.id == "turbo:confidence<opennlp:tokens>").unwrap();
    conf.columns[0].data = ColumnData::F64(vec![
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        1e7,
        1e-3,
        9.99e-4,
        0.1 + 0.2,
    ]);
    let text = g.to_text().unwrap();
    assert!(text.contains("\ntab\\there\nline\\nend\ncr\\rx\nback\\\\slash\nnul\\u0000bell\\u0007\n😀\n\nx\ny\n"));
    assert!(text.contains("\n8000000000000000 -0.0\n7ff8000000000000 NaN\n7ff0000000000000 Infinity\n"));
    assert!(text.contains("\n0000000000000001 4.9E-324\n416312d000000000 1.0E7\n3f50624dd2f1a9fc 0.001\n"));
    let back = Golden::parse(&text).unwrap();
    assert_eq!(back, g);
    assert_eq!(back.to_text().unwrap(), text);
}

#[test]
fn the_writer_refuses_what_the_format_cannot_hold() {
    let mut g = golden();
    g.header.jdk.vendor = "a\tb".into();
    assert_eq!(g.to_text().unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    let mut g = golden();
    g.layers[0].columns[1].data = ColumnData::U32(vec![26]);
    let e = g.to_text().unwrap_err().to_string();
    assert!(e.contains("opennlp:sentences has 2 rows in its first column and 1 in end"), "{e}");
}

// ---------------------------------------------------------------------
// Comparing: a document built to match the fixture, and changes to it.

/// The fixture's layers as plain values, so a test can change one.
struct Parts {
    sentences: Vec<(u32, u32)>,
    sentence_conf: Vec<f64>,
    tokens: Vec<(u32, u32)>,
    token_conf: Vec<f64>,
    tags: Vec<&'static str>,
    prob: Option<Vec<f64>>,
    lemmas: Vec<&'static str>,
    entities: Vec<(u32, u32, u32, u32, &'static str, f64)>,
    language: (&'static str, f64),
}

impl Parts {
    fn matching() -> Parts {
        Parts {
            sentences: vec![(0, 26), (27, 37)],
            sentence_conf: vec![0.9876543210987654, 0.95],
            tokens: vec![(0, 3), (4, 9), (10, 14), (15, 17), (18, 25), (25, 26), (27, 29), (30, 36), (36, 37)],
            token_conf: vec![0.9991, 0.99842, 0.9999, 0.98, 0.97321, 2.0 / 3.0, 0.999, 0.9935, 1.0],
            tags: vec!["PROPN", "PROPN", "VERB", "ADP", "PROPN", "PUNCT", "PRON", "VERB", "PUNCT"],
            prob: Some(vec![0.9712, 0.9554, 0.987, 0.9933, 0.8123, 0.99, 0.991, 0.9601, 0.99]),
            lemmas: vec!["Dr.", "Smith", "go", "to", "Zürich", ".", "it", "rain", "!"],
            entities: vec![(4, 9, 1, 2, "person", 0.8765432), (18, 25, 4, 5, "location", 0.7123)],
            language: ("eng", 0.9312345678901234),
        }
    }

    fn doc(&self) -> Document {
        let mut doc = Document::new(TEXT);
        let by = || Provenance::by("test");
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(s, e) in &self.sentences {
            spans.push(Span::new(s, e));
        }
        doc.add(&Key::DEFAULT, Sentences { spans }, by()).unwrap();
        doc.add(&Key::DEFAULT, Confidence::<Sentences>::new(self.sentence_conf.clone()), by()).unwrap();
        doc.add(&Key::DEFAULT, tokens(ViewId::SOURCE, &self.tokens), by()).unwrap();
        doc.add(&Key::DEFAULT, Confidence::<Tokens>::new(self.token_conf.clone()), by()).unwrap();
        let tagset = TagSet::new("ud", &["PROPN", "VERB", "ADP", "PUNCT", "PRON", "NOUN"]);
        let tags = self.tags.iter().map(|t| tagset.tag(t).unwrap()).collect();
        doc.add(&Key::DEFAULT, PosTags { tagset, tags, prob: self.prob.clone() }, by()).unwrap();
        let mut lemma = Strs::default();
        for l in &self.lemmas {
            lemma.push(l);
        }
        doc.add(&Key::DEFAULT, Lemmas { lemma }, by()).unwrap();
        let kinds = TagSet::new("kinds", &["person", "location", "organization"]);
        let mut spans = Spans::new(ViewId::SOURCE);
        let (mut toks, mut kind, mut score) = (Vec::new(), Vec::new(), Vec::new());
        for &(s, e, ts, te, k, p) in &self.entities {
            spans.push(Span::new(s, e));
            toks.push(IxRange::new(ts, te));
            kind.push(kinds.tag(k).unwrap());
            score.push(p);
        }
        doc.add(&Key::DEFAULT, Entities { spans, tokens: toks, kind, kinds, score }, by()).unwrap();
        let (code, score) = self.language;
        doc.add(&Key::DEFAULT, Language { code: code.into(), score }, by()).unwrap();
        doc
    }
}

fn tokens(view: ViewId, spans: &[(u32, u32)]) -> Tokens {
    let mut s = Spans::new(view);
    for &(a, b) in spans {
        s.push(Span::new(a, b));
    }
    Tokens { spans: s, sentence: None }
}

#[test]
fn a_matching_document_passes_every_layer_exactly() {
    let g = golden();
    let doc = Parts::matching().doc();
    assert_matches_golden::<Sentences>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Tokens>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Confidence<Sentences>>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Confidence<Tokens>>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<PosTags>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Lemmas>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Entities>(&doc, &g, Tolerance::exact());
    assert_matches_golden::<Language>(&doc, &g, Tolerance::exact());
}

fn mismatch<L: GoldenCompare>(p: &Parts, tol: Tolerance) -> (Mismatch, String) {
    let m = matches_golden::<L>(&p.doc(), &golden(), tol).unwrap_err();
    let shown = m.to_string();
    (m, shown)
}

fn has(shown: &str, parts: &[&str]) {
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in the report:\n{shown}");
    }
}

#[test]
fn a_span_shifted_by_one_byte() {
    let mut p = Parts::matching();
    p.tokens[1] = (4, 10);
    let (m, shown) = mismatch::<Tokens>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "span", .. }), "{m:?}");
    has(
        &shown,
        &["opennlp:tokens item 1: the span differs", "rust:   bytes 4..10 \"Smith \"", "golden: bytes 4..9 \"Smith\""],
    );
}

#[test]
fn a_changed_tag() {
    let mut p = Parts::matching();
    p.tags[2] = "NOUN";
    let (m, shown) = mismatch::<PosTags>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 2, what: "tag", .. }), "{m:?}");
    // The report shows the token the tag is for, on both sides.
    has(
        &shown,
        &[
            "opennlp:pos item 2: the tag differs",
            "rust:   bytes 10..14 \"went\", tag \"NOUN\", prob 3fef95810624dd2f (0.987)",
            "golden: bytes 10..14 \"went\", tag \"VERB\", prob 3fef95810624dd2f (0.987)",
        ],
    );
}

#[test]
fn a_value_just_outside_the_tolerance() {
    let mut p = Parts::matching();
    let golden_value: f64 = 0.98;
    // 2e-9 above the golden value, or the nearest double below that whose
    // distance from it, as f64 subtraction gives it, is at most 2e-9.
    let mut v = golden_value + 2e-9;
    while v - golden_value > 2e-9 {
        v = f64::from_bits(v.to_bits() - 1);
    }
    p.token_conf[3] = v;
    let diff = (p.token_conf[3] - golden_value).abs();
    assert!(diff > 1e-9 + 1e-10 && diff <= 2e-9, "the test's own distance: {diff:e}");
    let (m, shown) = mismatch::<Confidence<Tokens>>(&p, Tolerance::abs(1e-9));
    assert!(matches!(m, Mismatch::Item { index: 3, what: "value", .. }), "{m:?}");
    let rust_bits = format!("{:016x}", p.token_conf[3].to_bits());
    has(
        &shown,
        &[
            "turbo:confidence<opennlp:tokens> item 3: the value differs",
            "tolerance 1e-9",
            &format!("rust:   bytes 15..17 \"to\", value {rust_bits} ({:?})", p.token_conf[3]),
            "golden: bytes 15..17 \"to\", value 3fef5c28f5c28f5c (0.98)",
        ],
    );
    assert_eq!(matches_golden::<Confidence<Tokens>>(&p.doc(), &golden(), Tolerance::abs(2e-9)), Ok(()));
    // At exact bits a value one unit in the last place off differs.
    p.token_conf[3] = f64::from_bits(golden_value.to_bits() + 1);
    let (m, _) = mismatch::<Confidence<Tokens>>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 3, what: "value", .. }), "{m:?}");
}

#[test]
fn a_missing_last_row() {
    let mut p = Parts::matching();
    p.sentences.pop();
    p.sentence_conf.pop();
    let (m, shown) = mismatch::<Sentences>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 1, golden: 2, .. }), "{m:?}");
    has(
        &shown,
        &["opennlp:sentences: 1 items in Rust, 2 in the golden", "golden item 1", "bytes 27..37 \"It rained!\""],
    );
    let (m, shown) = mismatch::<Confidence<Sentences>>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Count { rust: 1, golden: 2, .. }), "{m:?}");
    has(&shown, &["golden item 1", "bytes 27..37 \"It rained!\", value 3fee666666666666 (0.95)"]);
}

#[test]
fn a_golden_layer_with_the_wrong_columns() {
    let mut g = golden();
    let conf = g.layers.iter_mut().find(|l| l.id == "turbo:confidence<opennlp:tokens>").unwrap();
    conf.columns[0].name = "prob".into();
    let m = matches_golden::<Confidence<Tokens>>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Columns { .. }), "{m:?}");
    has(&m.to_string(), &["turbo:confidence<opennlp:tokens>", "the columns [prob:f64], expected [value:f64]"]);

    let mut g = golden();
    let pos = g.layers.iter_mut().find(|l| l.id == "opennlp:pos").unwrap();
    pos.columns[1].data = ColumnData::U32(vec![0; 9]);
    let m = matches_golden::<PosTags>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    has(&m.to_string(), &["the columns [tag:str, prob:u32], expected [tag:str, prob:f64]"]);
}

#[test]
fn a_language_code_difference() {
    let mut p = Parts::matching();
    p.language.0 = "deu";
    let (m, shown) = mismatch::<Language>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "code", .. }), "{m:?}");
    has(
        &shown,
        &[
            "opennlp:language item 0: the code differs",
            "rust:   code \"deu\", confidence 3fedccac6fbfc4c9",
            "golden: code \"eng\", confidence 3fedccac6fbfc4c9",
        ],
    );
    // The best language is row 0 alone: the confidence is compared next.
    let mut p = Parts::matching();
    p.language.1 = 0.0412;
    let (m, _) = mismatch::<Language>(&p, Tolerance::abs(1e-9));
    assert!(matches!(m, Mismatch::Item { index: 0, what: "confidence", .. }), "{m:?}");
}

#[test]
fn tags_without_probabilities_do_not_pass() {
    let mut p = Parts::matching();
    p.prob = None;
    let (m, shown) = mismatch::<PosTags>(&p, Tolerance::exact());
    assert_eq!(m, Mismatch::NoValues { layer: "opennlp:pos".into(), column: "prob" });
    has(&shown, &["opennlp:pos: the Rust layer has no probabilities", "prob column"]);
}

#[test]
fn entity_tokens_kinds_and_lemmas_are_compared() {
    let mut p = Parts::matching();
    p.entities[1].3 = 6;
    let (m, shown) = mismatch::<Entities>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 1, what: "tokens", .. }), "{m:?}");
    has(
        &shown,
        &[
            "rust:   bytes 18..25 \"Zürich\", tokens 4..6, kind \"location\"",
            "golden: bytes 18..25 \"Zürich\", tokens 4..5",
        ],
    );

    let mut p = Parts::matching();
    p.entities[0].4 = "organization";
    let (m, shown) = mismatch::<Entities>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 0, what: "kind", .. }), "{m:?}");
    has(&shown, &["kind \"organization\"", "kind \"person\"", "bytes 4..9 \"Smith\""]);

    let mut p = Parts::matching();
    p.lemmas[7] = "rained";
    let (m, shown) = mismatch::<Lemmas>(&p, Tolerance::exact());
    assert!(matches!(m, Mismatch::Item { index: 7, what: "lemma", .. }), "{m:?}");
    has(&shown, &["bytes 30..36 \"rained\", lemma \"rained\"", "lemma \"rain\""]);
}

#[test]
fn a_document_with_another_text_is_refused() {
    let g = golden();
    let doc = Document::new("Dr. Smith went to Zurich. It rained!");
    let m = matches_golden::<Sentences>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Input { expected_len: 37, found_len: 36, .. }), "{m:?}");
}

#[test]
fn a_missing_or_skipped_layer_is_reported() {
    let g = golden();
    let mut doc = Document::new(TEXT);
    let m = matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert_eq!(m.to_string(), "opennlp:tokens: the document has no such layer");
    doc.skip(Key::<Tokens>::DEFAULT.id(), "no model".into(), Provenance::by("test")).unwrap();
    let m = matches_golden::<Tokens>(&doc, &g, Tolerance::exact()).unwrap_err();
    assert_eq!(m.to_string(), "opennlp:tokens: the document's layer is skipped: no model");
}

#[test]
fn a_golden_without_the_layer_lists_what_it_has() {
    let mut g = golden();
    g.layers.retain(|l| l.id != "opennlp:lemmas");
    let m = matches_golden::<Lemmas>(&Parts::matching().doc(), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(&m, Mismatch::MissingGolden { available, .. } if available.len() == 7), "{m:?}");
}

/// A named instance is compared when its key is given, and tokens in a
/// derived view are compared in source bytes, through the view's
/// alignment.
#[test]
fn a_named_instance_on_a_derived_view() {
    let g = golden();
    let mut doc = Document::new(TEXT);
    // "Zürich" becomes "Zurich": every token from there on is a byte earlier
    // in the view, and maps back to the source's offsets.
    let mut b = ViewBuilder::new(TEXT);
    b.keep(19);
    b.replace(21, "u");
    let (folded, alignment) = b.finish();
    assert_eq!(folded, "Dr. Smith went to Zurich. It rained!");
    let view = View::derived(doc.get::<View>().unwrap(), folded, alignment);
    doc.add(&Key::named("folded"), view, Provenance::by("test")).unwrap();
    let id = doc.get_key(&Key::<View>::named("folded")).unwrap().id();
    let spans = [(0, 3), (4, 9), (10, 14), (15, 17), (18, 24), (24, 25), (26, 28), (29, 35), (35, 36)];
    doc.add(&Key::named("gold"), tokens(id, &spans), Provenance::by("test")).unwrap();
    assert_matches_golden_key(&doc, &Key::<Tokens>::named("gold"), &g, Tolerance::exact());

    let mut shifted = spans;
    shifted[4] = (18, 23);
    let mut doc2 = Document::new(TEXT);
    let mut b = ViewBuilder::new(TEXT);
    b.keep(19);
    b.replace(21, "u");
    let (folded, alignment) = b.finish();
    let view = View::derived(doc2.get::<View>().unwrap(), folded, alignment);
    doc2.add(&Key::named("folded"), view, Provenance::by("test")).unwrap();
    let id2 = doc2.get_key(&Key::<View>::named("folded")).unwrap().id();
    doc2.add(&Key::named("gold"), tokens(id2, &shifted), Provenance::by("test")).unwrap();
    let m = matches_golden_key(&doc2, &Key::<Tokens>::named("gold"), &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 4, what: "span", .. }), "{m:?}");
    has(
        &m.to_string(),
        &["opennlp:tokens@gold item 4", "rust:   bytes 18..24 \"Züric\"", "golden: bytes 18..25 \"Zürich\""],
    );
}

// ---------------------------------------------------------------------
// Malformed files: each refused with its line.

/// The fixture with line `n` (from 1) replaced by `with`, or removed when
/// `with` is None.
fn edit(n: usize, with: Option<&str>) -> String {
    let text = fixture_text();
    let mut lines: Vec<&str> = text.lines().collect();
    match with {
        Some(w) => lines[n - 1] = w,
        None => {
            lines.remove(n - 1);
        }
    }
    lines.join("\n") + "\n"
}

/// Parses `text`, which must fail on `line`, and returns the error.
fn refused(text: &str, line: usize) -> GoldenError {
    let e = Golden::parse(text).unwrap_err();
    let shown = e.to_string();
    assert!(shown.starts_with(&format!("line {line}: ")), "expected an error on line {line}, got: {shown}");
    e
}

#[test]
fn malformed_headers() {
    use GoldenError::*;
    assert!(matches!(refused(&edit(1, Some("golden 2")), 1), Version { .. }));
    assert!(matches!(refused(&edit(1, Some("golden\t1")), 1), Version { .. }));
    assert!(matches!(refused(&edit(2, Some("generator\tx\t01")), 2), Field { .. }));
    let e = refused(&edit(3, Some("opennlp\t0123456789ABCDEF0123456789abcdef01234567\tv")), 3);
    assert!(e.to_string().contains("expected 40 lowercase hex digits"), "{e}");
    let e = refused(&edit(4, Some("jdk\t21\tEclipse Adoptium")), 4);
    assert!(matches!(e, Fields { found: 3, .. }), "{e:?}");
    assert!(e.to_string().contains("jdk <java.version> <java.vm.version> <java.vendor>"), "{e}");
    assert!(matches!(refused(&edit(4, Some("unicode\t15.0")), 4), Record { .. }));
    assert!(matches!(refused(&edit(5, Some("unicode\t15")), 5), Field { .. }));
    assert!(matches!(refused(&edit(6, Some("input\tzurich\t37\t36\tf0")), 6), Field { .. }));
    let sha = "f03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f";
    assert!(matches!(refused(&edit(6, Some(&format!("input\ttest/zurich\t037\t36\t{sha}"))), 6), Field { .. }));
    assert!(matches!(refused(&edit(6, Some(&format!("input\ttest/zurich\t37\t-1\t{sha}"))), 6), Field { .. }));
    assert!(matches!(refused(&edit(6, Some(&format!("input\ttest/zurich\t4294967296\t36\t{sha}"))), 6), Field { .. }));
    assert!(matches!(refused(&edit(7, Some("model\tx.bin\tabc")), 7), Field { .. }));
    assert!(matches!(refused(&edit(9, Some("annotation\tx\ty")), 9), Record { .. }));
    // The header ends and no layer follows.
    let header: String = fixture_text().lines().take(9).map(|l| format!("{l}\n")).collect();
    assert!(matches!(refused(&header, 10), NoLayers { .. }));
    let short: String = fixture_text().lines().take(4).map(|l| format!("{l}\n")).collect();
    assert!(matches!(refused(&short, 5), Eof { .. }));
}

#[test]
fn malformed_bytes_and_line_ends() {
    use GoldenError::*;
    let text = fixture_text();
    assert!(matches!(refused(text.trim_end_matches('\n'), 71), NoFinalLineEnd { .. }));
    let crlf = text.replacen("unicode\t15.0\n", "unicode\t15.0\r\n", 1);
    assert!(matches!(refused(&crlf, 5), CarriageReturn { .. }));
    let mut bytes = text.into_bytes();
    let at = bytes.windows(6).position(|w| w == b"Z\xc3\xbcric").unwrap();
    bytes[at + 1] = 0xff;
    let e = Golden::parse_bytes(&bytes).unwrap_err();
    assert!(matches!(e, NotUtf8 { line: 56 }), "{e:?}");
}

#[test]
fn malformed_layers() {
    use GoldenError::*;
    assert!(matches!(
        refused(&edit(10, Some("layer\topennlp:sentence\ttext\t2\tstart:u32\tend:u32")), 10),
        UnknownLayer { .. }
    ));
    let e = refused(&edit(10, Some("layer\topennlp:sentences\tdocument\t2\tstart:u32\tend:u32")), 10);
    assert!(matches!(e, Anchor { .. }), "{e:?}");
    assert!(e.to_string().contains("anchored to document, expected text"), "{e}");
    let e = refused(&edit(10, Some("layer\topennlp:sentences\ttext\t2\tend:u32\tstart:u32")), 10);
    assert!(matches!(e, Columns { .. }), "{e:?}");
    assert!(e.to_string().contains("has the columns [end:u32, start:u32], expected [start:u32, end:u32]"), "{e}");
    assert!(matches!(
        refused(&edit(10, Some("layer\topennlp:sentences\ttext\t2\tstart:i32\tend:u32")), 10),
        Field { .. }
    ));
    assert!(matches!(refused(&edit(10, Some("layer\topennlp:sentences\ttext\t2")), 10), Fields { .. }));
    assert!(matches!(
        refused(&edit(14, Some("layer\tturbo:confidence<opennlp:sentences>\taligned:\t2\tvalue:f64")), 14),
        Field { .. }
    ));
    let e = refused(
        &edit(14, Some("layer\tturbo:confidence<opennlp:sentences>\taligned:opennlp:sentences\t3\tvalue:f64")),
        14,
    );
    assert!(matches!(e, AlignedRows { expected: 2, found: 3, .. }), "{e:?}");
    // A second sentences block where the confidence block was.
    let text = fixture_text().replacen(
        "layer\tturbo:confidence<opennlp:sentences>\taligned:opennlp:sentences\t2\tvalue:f64\n\
         3fef9add3c1be9b8 0.9876543210987654\n3fee666666666666 0.95\nend\tturbo:confidence<opennlp:sentences>",
        "layer\topennlp:sentences\ttext\t2\tstart:u32\tend:u32\n0\t26\n27\t37\nend\topennlp:sentences",
        1,
    );
    assert!(matches!(refused(&text, 14), Duplicate { .. }));
    // The tokens' confidence before the tokens.
    let text = fixture_text();
    let lines: Vec<&str> = text.lines().collect();
    let mut moved: Vec<&str> = lines[..17].to_vec();
    moved.extend(&lines[28..39]);
    moved.extend(&lines[17..28]);
    moved.extend(&lines[39..]);
    let moved = moved.join("\n") + "\n";
    assert!(matches!(refused(&moved, 18), AnchorLayer { .. }));
}

#[test]
fn malformed_rows() {
    use GoldenError::*;
    let e = refused(&edit(12, Some("27\t38")), 12);
    assert!(e.to_string().contains("the span 27..38 ends past the input's 37 bytes"), "{e}");
    let e = refused(&edit(12, Some("27\t20")), 12);
    assert!(e.to_string().contains("ends before it starts"), "{e}");
    let e = refused(&edit(21, Some("0\t14")), 21);
    assert!(matches!(e, Span { .. }), "{e:?}");
    let e = refused(&edit(12, Some("27")), 12);
    assert!(matches!(e, Fields { found: 1, .. }), "{e:?}");
    assert!(e.to_string().contains("<start:u32> <end:u32>"), "{e}");
    assert!(matches!(refused(&edit(12, Some("27\t37\t")), 12), Fields { found: 3, .. }));
    assert!(matches!(refused(&edit(12, Some("27\t 37")), 12), Field { .. }));
    let e = refused(&edit(15, Some("3FEF9ADD3C1BE9B8 0.9876543210987654")), 15);
    assert!(e.to_string().contains("value: expected 16 lowercase hex digits"), "{e}");
    assert!(matches!(refused(&edit(15, Some("3fef9add3c1be9b8")), 15), Field { .. }));
    let e = refused(&edit(43, Some("VE\\qRB\t3fef95810624dd2f 0.987")), 43);
    assert!(matches!(e, Escape { .. }), "{e:?}");
    assert!(e.to_string().contains("tag: at byte 2 of the field"), "{e}");
    // A row left out: the end line comes where row 1 was expected.
    let e = refused(&edit(12, None), 12);
    assert!(matches!(e, Rows { declared: 2, found: 1, .. }), "{e:?}");
    assert!(e.to_string().contains("opennlp:sentences ends after 1 rows, expected the 2 it declares"), "{e}");
    let e = refused(&edit(13, Some("end\topennlp:tokens")), 13);
    assert!(matches!(e, Record { .. }), "{e:?}");
    let cut: String = fixture_text().lines().take(20).map(|l| format!("{l}\n")).collect();
    assert!(matches!(refused(&cut, 21), Eof { .. }));
}

#[test]
fn data_dir_is_beside_the_crate() {
    assert!(data_dir().ends_with("analysis/goldens/data"));
}
