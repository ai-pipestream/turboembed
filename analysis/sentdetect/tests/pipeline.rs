//! The detectors in pipelines, each followed by a step that reads what
//! they wrote: the sentences inside each paragraph, and the maxent
//! detector's sentences with their probabilities.

use turbo_analysis::{Analyzer, Anchor, Annotator, Document, Error, Key, Layer, Refs, Status, View};
use turbo_analysis_layers::{Block, Confidence, Sentences, Structure};
use turbo_sentdetect::{NewlineDetector, ParagraphDetector, SentenceDetector, SentenceModel, WhitespaceMode};

/// How many sentences each paragraph holds, aligned to the structure.
struct PerParagraph {
    sentences: Vec<u32>,
}

impl Layer for PerParagraph {
    const ID: &'static str = "test:per-paragraph";
    fn anchor() -> Anchor {
        Anchor::aligned::<Structure>()
    }
    fn len(&self) -> usize {
        self.sentences.len()
    }
}

struct Count;

impl Annotator for Count {
    const ID: &'static str = "test:count";
    type Reads = (View, Sentences, Structure);
    type Writes = (PerParagraph,);

    fn annotate(&self, (view, sentences, structure): Refs<'_, Self>) -> Result<(PerParagraph,), Error> {
        assert_eq!(sentences.spans.view(), view.id());
        assert_eq!(structure.spans.view(), view.id());
        let sentences = structure.spans.iter().map(|(_, p)| sentences.spans.within(p).len() as u32).collect();
        Ok((PerParagraph { sentences },))
    }
}

/// The text of each paragraph, all of them `Block::Paragraph`.
fn paragraphs(d: &Document) -> Vec<&str> {
    let s = d.get::<Structure>().unwrap();
    assert!(s.kind.iter().all(|&k| k == Block::Paragraph));
    s.spans.iter().map(|(_, p)| &d.source()[p.range()]).collect()
}

#[test]
fn newline_then_paragraph_then_reader() {
    let analyzer = Analyzer::builder()
        .add(Count)
        .add(ParagraphDetector::new(WhitespaceMode::Unicode))
        .add(NewlineDetector::new(WhitespaceMode::Unicode))
        .build()
        .unwrap();
    assert_eq!(analyzer.plan(), [vec!["turbo-sentdetect:paragraph", "turbo-sentdetect:newline"], vec!["test:count"]]);

    let mut docs = vec![
        Document::new("Title line\n\nFirst para, line one.\nLine two.\n\n\nLast para."),
        Document::new("\u{1F600} one\r\ntwo\u{2028}still two\r\n\u{A0}\r\nthree"),
        Document::new(" \t\r\n\u{3000}\n\u{85}"),
    ];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let sentences = |d: &Document| d.get::<Sentences>().unwrap().spans.columns().0.len();
    let counts = |d: &Document| d.get::<PerParagraph>().unwrap().sentences.clone();

    let d = &docs[0];
    assert_eq!(sentences(d), 4);
    assert_eq!(paragraphs(d), ["Title line", "First para, line one.\nLine two.", "Last para."]);
    assert_eq!(counts(d), [1, 2, 1]);

    // CR LF ends a line once; U+2028 does not end one; a line holding only
    // U+00A0 is blank.
    let d = &docs[1];
    assert_eq!(sentences(d), 3);
    assert_eq!(paragraphs(d), ["\u{1F600} one\r\ntwo\u{2028}still two", "three"]);
    assert_eq!(counts(d), [2, 1]);

    // Whitespace only: both layers are written, complete and empty, not
    // skipped, and the reader runs on them.
    let d = &docs[2];
    for key in [Key::<Sentences>::DEFAULT.id(), Key::<Structure>::DEFAULT.id(), Key::<PerParagraph>::DEFAULT.id()] {
        assert_eq!(d.status(key), Some(&Status::Complete), "{key}");
    }
    assert_eq!(sentences(d), 0);
    assert!(paragraphs(d).is_empty());
    assert!(counts(d).is_empty());

    let p = d.provenance(Key::<Sentences>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "turbo-sentdetect:newline");
    assert_eq!(p.config, NewlineDetector::new(WhitespaceMode::Unicode).config_hash());
}

/// The probability of each sentence, read back from the confidence layer,
/// with the sentence text: what a step after the maxent detector sees.
struct Scored {
    text: Vec<String>,
    confidence: Vec<f64>,
}

impl Layer for Scored {
    const ID: &'static str = "test:scored";
    fn anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }
    fn len(&self) -> usize {
        self.text.len()
    }
}

struct ReadScores;

impl Annotator for ReadScores {
    const ID: &'static str = "test:read-scores";
    type Reads = (View, Sentences, Confidence<Sentences>);
    type Writes = (Scored,);

    fn annotate(&self, (view, sentences, confidence): Refs<'_, Self>) -> Result<(Scored,), Error> {
        assert_eq!(sentences.spans.len(), confidence.value.len());
        let text = sentences.spans.iter().map(|(_, s)| view.text()[s.range()].to_string()).collect();
        Ok((Scored { text, confidence: confidence.value.clone() },))
    }
}

#[test]
fn maxent_then_reader() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/models/en-abbrev.bin");
    let model = SentenceModel::open(turbo_maxent::ModelFile::read(&path).unwrap()).unwrap();
    let detector = SentenceDetector::new(model, WhitespaceMode::Unicode);
    let model_ref = detector.sentence_model().model_ref().clone();
    // The detector aligns its confidence to the sentences it writes in the
    // same step, so it is added as a plain annotator.
    let analyzer = Analyzer::builder().add(ReadScores).add(detector).build().unwrap();
    assert_eq!(analyzer.plan(), [vec!["turbo-sentdetect:maxent"], vec!["test:read-scores"]]);

    // The first two texts are fixture inputs, so the values are those of
    // the golden files.
    let mut docs = vec![
        Document::new("Mr. Smith met Dr. Brown at 5 p.m. on Main St. today. He left early."),
        Document::new("It ended.\u{1F600} Next one starts.\u{10400}Deseret here. Last.\u{1F600}"),
        Document::new(" \t\n  \u{3000} "),
    ];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let scored = |d: &Document| {
        let s = d.get::<Scored>().unwrap();
        (s.text.clone(), s.confidence.clone())
    };
    let (text, confidence) = scored(&docs[0]);
    assert_eq!(text, ["Mr. Smith met Dr. Brown at 5 p.m. on Main St. today.", "He left early."]);
    assert_eq!(confidence, [0.9955300941022667, 0.9466056890177881]);
    let (text, confidence) = scored(&docs[1]);
    assert_eq!(text, ["It ended.\u{1F600}", "Next one starts.\u{10400}Deseret", "here.", "Last.\u{1F600}"]);
    assert_eq!(confidence[3], 0.50730226664988);

    // Whitespace only: no sentences, and both layers complete, not skipped.
    let d = &docs[2];
    for key in [Key::<Sentences>::DEFAULT.id(), Key::<Confidence<Sentences>>::DEFAULT.id(), Key::<Scored>::DEFAULT.id()]
    {
        assert_eq!(d.status(key), Some(&Status::Complete), "{key}");
    }
    assert_eq!(scored(d), (vec![], vec![]));

    let p = docs[0].provenance(Key::<Sentences>::DEFAULT.id()).unwrap();
    assert_eq!(p.annotator, "turbo-sentdetect:maxent");
    assert_eq!(p.model.as_ref(), Some(&model_ref));
}
