//! The detectors in pipelines, each followed by a step that reads what
//! they wrote: sentences inside each paragraph and rule-derived Unicode
//! source spans, including UTF-16 and code-point boundaries.

use turbo_analysis::{Analyzer, Anchor, Annotator, Document, Error, Key, Layer, Refs, Status, View};
use turbo_analysis_layers::{Block, Sentences, Structure};
use turbo_sentdetect::{NewlineDetector, ParagraphDetector, RuleSentenceDetector};

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
    let analyzer =
        Analyzer::builder().add(Count).add(ParagraphDetector::new()).add(NewlineDetector::new()).build().unwrap();
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
    assert_eq!(p.config, NewlineDetector::new().config_hash());
}

/// Source text and offsets consumed by a downstream step.
struct ReadBack {
    text: Vec<String>,
    offsets: Vec<(usize, usize, usize, usize, usize, usize)>,
}

impl Layer for ReadBack {
    const ID: &'static str = "test:read-back";
    fn anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }
    fn len(&self) -> usize {
        self.text.len()
    }
}

struct ReadSentences;

impl Annotator for ReadSentences {
    const ID: &'static str = "test:read-sentences";
    type Reads = (View, Sentences);
    type Writes = (ReadBack,);

    fn annotate(&self, (view, sentences): Refs<'_, Self>) -> Result<(ReadBack,), Error> {
        use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};
        assert_eq!(sentences.spans.view(), view.id());
        let source = view.text();
        let mut text = Vec::new();
        let mut offsets = Vec::new();
        for (_, span) in sentences.spans.iter() {
            let (start, end) = (span.start as usize, span.end as usize);
            let (u_start, u_end) = (byte_to_utf16(source, start).unwrap(), byte_to_utf16(source, end).unwrap());
            assert_eq!(utf16_to_byte(source, u_start), Some(start));
            assert_eq!(utf16_to_byte(source, u_end), Some(end));
            let (c_start, c_end) = (source[..start].chars().count(), source[..end].chars().count());
            for (byte, point) in [(start, c_start), (end, c_end)] {
                assert_eq!(source.char_indices().map(|(b, _)| b).chain([source.len()]).nth(point), Some(byte));
            }
            text.push(source[span.range()].to_owned());
            offsets.push((start, end, u_start, u_end, c_start, c_end));
        }
        Ok((ReadBack { text, offsets },))
    }
}

#[test]
fn rules_then_unicode_source_reader() {
    let detector = RuleSentenceDetector::new("en");
    let hash = detector.config_hash();
    let analyzer = Analyzer::builder().add(ReadSentences).add(detector).build().unwrap();
    assert_eq!(analyzer.plan(), [vec!["turbo-sentdetect:rules"], vec!["test:read-sentences"]]);
    let mut docs = vec![
        Document::new("Mr. Smith met Dr. Brown at 5 p.m. on Main St. today. He left early."),
        Document::new("Été. 𐐀 Start. 😀 End."),
        Document::new(" \t\n  \u{3000} "),
    ];
    let results = analyzer.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let first = docs[0].get::<ReadBack>().unwrap();
    assert_eq!(first.text, ["Mr. Smith met Dr. Brown at 5 p.m. on Main St. today.", "He left early."]);
    assert_eq!(first.offsets, [(0, 52, 0, 52, 0, 52), (53, 67, 53, 67, 53, 67)]);
    let unicode = docs[1].get::<ReadBack>().unwrap();
    assert_eq!(unicode.text, ["Été.", "𐐀 Start.", "😀 End."]);
    assert_eq!(unicode.offsets, [(0, 6, 0, 4, 0, 4), (7, 18, 5, 14, 5, 13), (19, 28, 15, 22, 14, 20)]);
    let empty = &docs[2];
    for key in [Key::<Sentences>::DEFAULT.id(), Key::<ReadBack>::DEFAULT.id()] {
        assert_eq!(empty.status(key), Some(&Status::Complete), "{key}");
    }
    assert!(empty.get::<ReadBack>().unwrap().text.is_empty());
    assert!(empty.get::<ReadBack>().unwrap().offsets.is_empty());
    let provenance = docs[0].provenance(Key::<Sentences>::DEFAULT.id()).unwrap();
    assert_eq!(provenance.annotator, "turbo-sentdetect:rules");
    assert_eq!(provenance.config, hash);
    assert!(provenance.model.is_none());
}
