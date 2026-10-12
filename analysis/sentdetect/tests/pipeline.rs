//! Both detectors in one pipeline, followed by a step that reads what they
//! wrote: the sentences inside each paragraph.

use turbo_analysis::{Analyzer, Anchor, Annotator, Document, Error, Key, Layer, Refs, Status, View};
use turbo_analysis_layers::{Block, Sentences, Structure};
use turbo_sentdetect::{NewlineDetector, ParagraphDetector, WhitespaceMode};

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
