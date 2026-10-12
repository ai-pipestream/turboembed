//! The annotator in pipelines: built in code ahead of the chunker, whose
//! chunks then stay inside the sections it found, and built from
//! configuration through the provider.

use turbo_analysis::registry::{Registry, Spec};
use turbo_analysis::{Analyzer, Document, Error, Key, ModelRef, Provenance, Span, Spans, ViewId, annotate::Step};
use turbo_analysis_layers::{Block, Boundaries, Chunks, Cut, Sentences, Structure, Subwords};
use turbo_chunker::{Budget, Chunker};
use turbo_structure::{StructureAnnotator, StructureProvider, Syntax};

const TEXT: &str = "# Cats\n\nCats purr. Cats nap.\n\n# Rust\n\nRust compiles. Rust ships.\n";

/// The document with what the chunker reads besides the structure: a
/// sentence per heading line and per period, a subword per word, and a
/// boundary cost of 1 after every sentence.
fn chunker_inputs() -> Document {
    let mut doc = Document::new(TEXT);
    let mut sentences = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for s in ["# Cats", "Cats purr.", "Cats nap.", "# Rust", "Rust compiles.", "Rust ships."] {
        let start = at + TEXT[at..].find(s).unwrap();
        at = start + s.len();
        sentences.push(Span::new(start as u32, at as u32));
    }
    let mut pieces = Spans::new(ViewId::SOURCE);
    for w in TEXT.split_ascii_whitespace() {
        let start = w.as_ptr() as usize - TEXT.as_ptr() as usize;
        pieces.push(Span::new(start as u32, (start + w.len()) as u32));
    }
    let n = sentences.len();
    let model = ModelRef { id: "toy".into(), revision: "1".into(), sha256: "0".into() };
    let ids = vec![0; pieces.len()];
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: pieces, ids, model, token: None }, Provenance::by("t")).unwrap();
    let bounds = Boundaries { cost: vec![Boundaries::UNIT; n], reason: vec![Cut::Sentence; n] };
    doc.add(&Key::DEFAULT, bounds, Provenance::by("t")).unwrap();
    doc
}

fn chunk_texts(doc: &Document) -> Vec<(u8, &str)> {
    let c = doc.get::<Chunks>().unwrap();
    c.spans.iter().map(|(i, s)| (c.level[i.get()], &doc.source()[s.range()])).collect()
}

#[test]
fn the_chunker_keeps_to_the_sections_found() {
    let budget = Budget { max: 100, target: 100, size_weight: 1, overhead: 0 };
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(Chunker { budget })
        .add(StructureAnnotator::new(Syntax::Markdown))
        .build()
        .unwrap();
    assert_eq!(analyzer.plan(), [vec!["turbo-structure:blocks"], vec!["turbo:chunker"]]);
    let mut docs = vec![chunker_inputs()];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));

    let s = docs[0].get::<Structure>().unwrap();
    assert_eq!(s.kind.iter().filter(|&&k| k == Block::Section).count(), 2);
    assert_eq!(
        chunk_texts(&docs[0]),
        [
            (1, "# Cats\n\nCats purr. Cats nap."),
            (0, "# Cats\n\nCats purr. Cats nap."),
            (1, "# Rust\n\nRust compiles. Rust ships."),
            (0, "# Rust\n\nRust compiles. Rust ships."),
        ]
    );

    // The same budget without the structure makes one chunk of it all.
    let without = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .add(Chunker { budget })
        .build()
        .unwrap();
    let mut docs = vec![chunker_inputs()];
    assert!(without.run(&mut docs).iter().all(Result::is_ok));
    assert_eq!(chunk_texts(&docs[0]), [(0, TEXT.trim_end())]);
}

fn registry() -> Registry<dyn Step> {
    let mut r = Registry::default();
    r.register(StructureProvider);
    r
}

#[test]
fn the_provider_builds_either_syntax() {
    let r = registry();
    let text = "SUMMARY\n\n    indented text\n";
    let mut kinds = Vec::new();
    let mut configs = Vec::new();
    for spec in [Spec::default(), Spec::default().with("syntax", "markdown"), Spec::default().with("syntax", "plain")] {
        let analyzer = r.pipeline(&[("structure", spec)], |b| b).unwrap();
        let mut docs = vec![Document::new(text)];
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
        kinds.push(docs[0].get::<Structure>().unwrap().kind.clone());
        configs.push(docs[0].provenance(Key::<Structure>::DEFAULT.id()).unwrap().config);
    }
    assert_eq!(kinds[0], [Block::Paragraph, Block::Code]);
    assert_eq!(kinds[1], kinds[0]);
    assert_eq!(kinds[2], [Block::Section, Block::Heading, Block::Paragraph]);
    assert_eq!(configs[0], configs[1]);
    assert_ne!(configs[0], configs[2]);
}

#[test]
fn the_provider_refuses_what_it_does_not_take() {
    let r = registry();
    let refused = |spec: Spec| match r.create("structure", &spec) {
        Err(Error::Pipeline(why)) => why,
        Err(e) => panic!("{e}"),
        Ok(_) => panic!("{spec:?} was taken"),
    };
    assert!(refused(Spec::default().with("syntax", "html")).contains("no available provider"));
    assert!(refused(Spec::default().with("tables", "off")).contains("no available provider"));
    let located = Spec { location: Some("model.bin".into()), options: Vec::new() };
    assert!(refused(located).contains("no available provider"));
}

#[test]
fn a_whitespace_document_gets_an_empty_layer_not_a_skip() {
    let analyzer = Analyzer::builder().add(StructureAnnotator::default()).build().unwrap();
    let mut docs = vec![Document::new(" \r\n\t\n"), Document::new("")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    for d in &docs {
        assert_eq!(d.get::<Structure>().unwrap().spans.len(), 0);
    }
}
