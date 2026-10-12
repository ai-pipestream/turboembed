//! Golden layers loaded as test inputs: a document built from the hand
//! written fixture `tests/fixtures/zurich.golden`, whose layers (sentences,
//! tokens, POS tags, lemmas, entities, language and the confidences) are
//! hand made values in the format a generator writes, not the output of a
//! real OpenNLP run. A stub tagger runs on the golden tokens alone and its
//! tags are compared with the golden's own POS layer.

use std::path::PathBuf;
use std::sync::Arc;

use turbo_analysis::{Analyzer, Annotator, Bound, Document, Error, Ix, IxRange, Key, Layer, Refs, View};
use turbo_analysis_layers::{Confidence, Entities, Language, Lemmas, PosTags, Sentences, Tag, TagSet, Tokens};
use turbo_goldens::*;

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn golden() -> Golden {
    Golden::read(fixtures().join("zurich.golden")).unwrap()
}

fn text(g: &Golden) -> String {
    g.input_text(fixtures()).unwrap()
}

/// A tagger that looks each token's text up in a table: enough to stand
/// in for a POS tagger port that reads tokens and writes tags.
struct TableTagger {
    table: Vec<(&'static str, &'static str, f64)>,
    tagset: Arc<TagSet>,
}

impl TableTagger {
    /// The table the fixture's POS layer was written from.
    fn matching() -> TableTagger {
        let table = vec![
            ("Dr.", "PROPN", 0.9712),
            ("Smith", "PROPN", 0.9554),
            ("went", "VERB", 0.987),
            ("to", "ADP", 0.9933),
            ("Zürich", "PROPN", 0.8123),
            (".", "PUNCT", 0.99),
            ("It", "PRON", 0.991),
            ("rained", "VERB", 0.9601),
            ("!", "PUNCT", 0.99),
        ];
        TableTagger { table, tagset: TagSet::new("ud", &["PROPN", "VERB", "ADP", "PUNCT", "PRON", "NOUN"]) }
    }
}

impl Annotator for TableTagger {
    const ID: &'static str = "test:table-tagger";
    type Reads = (View, Tokens);
    type Writes = (PosTags,);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(PosTags,), Error> {
        let mut tags = Vec::new();
        let mut prob = Vec::new();
        for (_, span) in tokens.spans.iter() {
            let word = &view.text()[span.range()];
            let Some(&(_, tag, p)) = self.table.iter().find(|(w, _, _)| *w == word) else {
                return Err(Error::Annotator { annotator: Self::ID, detail: format!("no tag for {word:?}") });
            };
            tags.push(self.tagset.tag(tag).unwrap());
            prob.push(p);
        }
        Ok((PosTags { tagset: self.tagset.clone(), tags, prob: Some(prob) },))
    }
}

/// Runs the tagger on the golden tokens of `doc`, writing the default
/// instance of the tags.
fn tag(doc: &mut Document, tagger: TableTagger) {
    let analyzer =
        Analyzer::builder().given(golden_key::<Tokens>()).step(Bound::new(tagger).read(1, GOLDEN_KEY)).build().unwrap();
    let status = analyzer.run(std::slice::from_mut(doc));
    assert_eq!(status, [Ok(())]);
}

#[test]
fn a_tagger_runs_on_the_golden_tokens_and_matches_the_golden_tags() {
    let g = golden();
    let mut doc = load_document(&g, &text(&g)).unwrap();
    // The Rust tokenizer's instance is free: only the golden one is here.
    assert!(doc.get::<Tokens>().is_none());
    assert_eq!(doc.get_key(&golden_key::<Tokens>()).unwrap().len(), 9);

    tag(&mut doc, TableTagger::matching());
    let tags = Key::<PosTags>::DEFAULT;
    assert_eq!(doc.made_on(tags.id()), [golden_key::<Tokens>().id()]);
    assert_eq!(matches_golden_key(&doc, &tags, &g, Tolerance::exact()), Ok(()));

    // One wrong entry in the table is found, at the token it is for.
    let mut doc = load_document(&g, &text(&g)).unwrap();
    let mut wrong = TableTagger::matching();
    wrong.table[2].1 = "NOUN";
    tag(&mut doc, wrong);
    let m = matches_golden_key(&doc, &tags, &g, Tolerance::exact()).unwrap_err();
    assert!(matches!(m, Mismatch::Item { index: 2, what: "tag", .. }), "{m:?}");
    assert!(m.to_string().contains("golden: bytes 10..14 \"went\", tag \"VERB\""), "{m}");
}

#[test]
fn every_loaded_layer_matches_its_golden() {
    let g = golden();
    let doc = load_document(&g, &text(&g)).unwrap();
    let exact = Tolerance::exact();
    assert_eq!(matches_golden_key(&doc, &golden_key::<Sentences>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Confidence<Sentences>>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Tokens>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Confidence<Tokens>>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<PosTags>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Lemmas>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Entities>(), &g, exact), Ok(()));
    assert_eq!(matches_golden_key(&doc, &golden_key::<Language>(), &g, exact), Ok(()));

    // The view, then the eight layers in file order.
    let keys: Vec<String> = doc.layers().map(|k| k.to_string()).collect();
    assert_eq!(
        keys,
        [
            "turbo:view",
            "opennlp:sentences@golden",
            "turbo:confidence<opennlp:sentences>@golden",
            "opennlp:tokens@golden",
            "turbo:confidence<opennlp:tokens>@golden",
            "opennlp:pos@golden",
            "opennlp:lemmas@golden",
            "opennlp:entities@golden",
            "opennlp:language@golden"
        ]
    );
    let language = doc.get_key(&golden_key::<Language>()).unwrap();
    assert_eq!((language.code.as_str(), language.score), ("eng", 0.9312345678901234));
    let entities = doc.get_key(&golden_key::<Entities>()).unwrap();
    assert_eq!(entities.tokens[1].start.get()..entities.tokens[1].end.get(), 4..5);
    assert_eq!(entities.kinds.label(entities.kind[1]), "location");
    assert_eq!(entities.score, [0.8765432, 0.7123]);
    let pos = doc.get_key(&golden_key::<PosTags>()).unwrap();
    assert_eq!(pos.tags[..3], [Tag(0), Tag(0), Tag(1)]);
    assert_eq!(pos.tagset.label(Tag(1)), "VERB");
    assert_eq!(pos.tagset.len(), 5);
}

#[test]
fn provenance_names_the_golden_annotator_and_what_a_layer_is_made_on() {
    let g = golden();
    let doc = load_document(&g, &text(&g)).unwrap();
    let p = doc.provenance(golden_key::<PosTags>().id()).unwrap();
    assert_eq!(p.annotator, "golden:opennlp.tools.tokenize.TokenizerME");
    assert_eq!(p.version, "3.0.0-PRE-SNAPSHOT");
    assert_eq!(p.model, None);
    assert_eq!(p.reads, [golden_key::<Tokens>().id()]);
    assert_eq!(doc.made_on(golden_key::<Entities>().id()), [golden_key::<Tokens>().id()]);
    assert_eq!(doc.made_on(golden_key::<Confidence<Sentences>>().id()), [golden_key::<Sentences>().id()]);
    assert_eq!(doc.provenance(golden_key::<Tokens>().id()).unwrap().reads, [golden_key::<Sentences>().id()]);
    assert_eq!(doc.made_on(golden_key::<Tokens>().id()), [golden_key::<Sentences>().id()]);
    assert!(doc.provenance(golden_key::<Sentences>().id()).unwrap().reads.is_empty());

    // Every layer of one golden has one configuration hash; another model
    // or another option gives another.
    let config = p.config;
    assert_eq!(doc.provenance(golden_key::<Tokens>().id()).unwrap().config, config);
    let mut other = g.clone();
    other.header.models[1].sha512[0] ^= 1;
    let doc = load_document(&other, &text(&g)).unwrap();
    assert_ne!(doc.provenance(golden_key::<Tokens>().id()).unwrap().config, config);
    let mut other = g.clone();
    other.header.config.push(ConfigEntry { key: "useAlphaNumericOptimization".into(), value: "true".into() });
    let doc = load_document(&other, &text(&g)).unwrap();
    assert_ne!(doc.provenance(golden_key::<Tokens>().id()).unwrap().config, config);
}

#[test]
fn loading_under_two_names_gives_two_instances() {
    let g = golden();
    let mut doc = Document::new(text(&g));
    const A: Key<Tokens> = Key::named("a");
    const B: Key<Tokens> = Key::named("b");
    load_layer(&mut doc, &g, &A).unwrap();
    load_layer(&mut doc, &g, &B).unwrap();
    load_layer(&mut doc, &g, &Key::<PosTags>::named("a")).unwrap();
    load_layer(&mut doc, &g, &Key::<PosTags>::named("b")).unwrap();

    let (a, b) = (doc.get_key(&A).unwrap(), doc.get_key(&B).unwrap());
    assert!(!std::ptr::eq(a, b));
    assert_eq!(a.spans.columns(), b.spans.columns());
    assert_eq!(doc.made_on(Key::<PosTags>::named("a").id()), [A.id()]);
    assert_eq!(doc.made_on(Key::<PosTags>::named("b").id()), [B.id()]);
    assert!(doc.get::<Tokens>().is_none() && doc.get_key(&golden_key::<Tokens>()).is_none());

    // A name is loaded once.
    let e = load_layer(&mut doc, &g, &A).unwrap_err();
    assert_eq!(e, LoadError::Document(Error::Duplicate(A.id())));
}

#[test]
fn an_aligned_layer_without_its_anchor_is_refused() {
    let g = golden();
    let mut doc = Document::new(text(&g));
    let e = load_layer(&mut doc, &g, &golden_key::<PosTags>()).unwrap_err();
    assert_eq!(
        e,
        LoadError::MissingAnchor { layer: golden_key::<PosTags>().id(), anchor: golden_key::<Tokens>().id() }
    );
    assert_eq!(
        e.to_string(),
        "opennlp:pos@golden: the document has no opennlp:tokens@golden layer to load it onto; load that first"
    );
    assert!(!doc.has(golden_key::<PosTags>().id()));

    // Tokens under another name are not its anchor.
    load_layer(&mut doc, &g, &Key::<Tokens>::named("other")).unwrap();
    let e = load_layer(&mut doc, &g, &golden_key::<Entities>()).unwrap_err();
    assert!(matches!(e, LoadError::MissingAnchor { .. }), "{e:?}");

    // A golden built in code without the tokens block: the first layer
    // aligned to them is refused.
    let mut no_tokens = g.clone();
    no_tokens.layers.retain(|l| l.id != "opennlp:tokens");
    let e = load_document(&no_tokens, &text(&g)).err().unwrap();
    assert_eq!(
        e,
        LoadError::MissingAnchor {
            layer: golden_key::<Confidence<Tokens>>().id(),
            anchor: golden_key::<Tokens>().id()
        }
    );
}

#[test]
fn a_text_that_is_not_the_input_is_refused() {
    let g = golden();
    let other = text(&g).replace("Dr.", "Dr!");
    let e = load_document(&g, &other).err().unwrap();
    assert!(matches!(e, LoadError::Input { expected_len: 37, found_len: 37, .. }), "{e:?}");
    let shown = e.to_string();
    assert!(shown.starts_with("the text is not the golden's input: 37 bytes with SHA-256 "), "{shown}");
    assert!(shown.ends_with("f03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f"), "{shown}");

    let mut doc = Document::new("Dr. Smith");
    let e = load_layer(&mut doc, &g, &golden_key::<Tokens>()).unwrap_err();
    assert!(matches!(e, LoadError::Input { expected_len: 37, found_len: 9, .. }), "{e:?}");
}

#[test]
fn bad_blocks_are_refused_before_loading() {
    let g = golden();
    let source = text(&g);

    // An entity whose token range runs past the tokens.
    let mut bad = g.clone();
    let entities = bad.layers.iter_mut().find(|l| l.id == "opennlp:entities").unwrap();
    entities.columns[3].data = ColumnData::U32(vec![2, 10]);
    let e = load_document(&bad, &source).err().unwrap();
    assert_eq!(e, LoadError::TokenRange { layer: golden_key::<Entities>().id(), row: 1, start: 4, end: 10, tokens: 9 });
    assert!(e.to_string().contains("row 1: the token range 4..10 is not inside the 9 tokens"), "{e}");

    // A block whose columns are not its type's.
    let mut bad = g.clone();
    bad.layers.iter_mut().find(|l| l.id == "opennlp:lemmas").unwrap().columns[0].name = "form".into();
    let e = load_document(&bad, &source).err().unwrap();
    assert!(matches!(&e, LoadError::Shape(Mismatch::Columns { .. })), "{e:?}");
    assert!(e.to_string().contains("opennlp:lemmas@golden: the golden layer has the columns [form:str]"), "{e}");

    // A block no layer type loads.
    let mut bad = g.clone();
    bad.layers.iter_mut().find(|l| l.id == "opennlp:lemmas").unwrap().id = "opennlp:dependencies".into();
    let e = load_document(&bad, &source).err().unwrap();
    assert_eq!(e, LoadError::NoLoader { layer: "opennlp:dependencies".into() });

    // A language block with no row.
    let mut bad = g.clone();
    let lang = bad.layers.iter_mut().find(|l| l.id == "opennlp:language").unwrap();
    for c in &mut lang.columns {
        c.data = ColumnData::empty(c.data.ty());
    }
    let e = load_document(&bad, &source).err().unwrap();
    assert_eq!(e, LoadError::Empty { layer: golden_key::<Language>().id() });

    // A span past the text is the document's to refuse.
    let mut bad = g.clone();
    bad.layers.iter_mut().find(|l| l.id == "opennlp:sentences").unwrap().columns[1].data =
        ColumnData::U32(vec![26, 38]);
    let e = load_document(&bad, &source).err().unwrap();
    assert!(matches!(e, LoadError::Document(Error::Bounds { .. })), "{e:?}");

    // A golden without the block asked for.
    let mut no_lemmas = g.clone();
    no_lemmas.layers.retain(|l| l.id != "opennlp:lemmas");
    let mut doc = load_document(&no_lemmas, &source).unwrap();
    let e = load_layer(&mut doc, &no_lemmas, &Key::<Lemmas>::named("again")).unwrap_err();
    assert!(matches!(e, LoadError::MissingGolden { .. }), "{e:?}");
}

#[test]
fn a_view_golden_gives_the_source_alone() {
    let g = Golden::read(fixtures().join("ligature.golden")).unwrap();
    let doc = load_document(&g, &text(&g)).unwrap();
    assert_eq!(doc.layers().count(), 1);
    assert_eq!(doc.source(), "The \u{fb01}nal  report \u{ff21} \u{ff22}.");
}

#[test]
fn golden_tokens_get_the_sentence_they_are_in() {
    let g = golden();
    let doc = load_document(&g, &text(&g)).unwrap();
    let tokens = doc.get_key(&golden_key::<Tokens>()).unwrap();
    let sentence: Vec<usize> = tokens.sentence.as_ref().unwrap().iter().map(|s| s.get()).collect();
    assert_eq!(sentence, [0, 0, 0, 0, 0, 0, 1, 1, 1]);
    assert_eq!(tokens.of(Ix::new(0)), IxRange::new(0, 6));
    assert_eq!(tokens.of(Ix::new(1)), IxRange::new(6, 9));

    // load_layer uses the sentences under the same name when they are
    // there, and leaves the column empty when they are not.
    let mut doc = Document::new(text(&g));
    load_layer(&mut doc, &g, &Key::<Tokens>::named("alone")).unwrap();
    assert!(doc.get_key(&Key::<Tokens>::named("alone")).unwrap().sentence.is_none());
    load_layer(&mut doc, &g, &Key::<Sentences>::named("split")).unwrap();
    load_layer(&mut doc, &g, &Key::<Tokens>::named("split")).unwrap();
    let split = doc.get_key(&Key::<Tokens>::named("split")).unwrap();
    assert_eq!(split.of(Ix::new(1)), IxRange::new(6, 9));
    assert_eq!(doc.made_on(Key::<Tokens>::named("split").id()), [Key::<Sentences>::named("split").id()]);
}

#[test]
fn a_token_outside_the_sentences_is_refused() {
    // Hand made: the sentences split the text at byte 22, inside the
    // token "Zürich" (bytes 18..25), which then straddles two sentences.
    let g = golden();
    let mut bad = g.clone();
    let sentences = bad.layers.iter_mut().find(|l| l.id == "opennlp:sentences").unwrap();
    sentences.columns[0].data = ColumnData::U32(vec![0, 22]);
    sentences.columns[1].data = ColumnData::U32(vec![22, 37]);
    let e = load_document(&bad, &text(&g)).err().unwrap();
    assert_eq!(e, LoadError::OutsideSentences { layer: golden_key::<Tokens>().id(), token: 4, start: 18, end: 25 });
    assert!(e.to_string().starts_with("opennlp:tokens@golden token 4: bytes 18..25 are not inside a sentence"), "{e}");

    // A token after the last sentence.
    let mut bad = g.clone();
    let sentences = bad.layers.iter_mut().find(|l| l.id == "opennlp:sentences").unwrap();
    sentences.columns[0].data = ColumnData::U32(vec![0, 27]);
    sentences.columns[1].data = ColumnData::U32(vec![26, 36]);
    let e = load_document(&bad, &text(&g)).err().unwrap();
    assert_eq!(e, LoadError::OutsideSentences { layer: golden_key::<Tokens>().id(), token: 8, start: 36, end: 37 });
}
