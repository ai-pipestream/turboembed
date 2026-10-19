//! Term counts with boilerplate left out, beside the plain counts.

use turbo_analysis::*;
use turbo_analysis_layers::{Block, Boilerplate, BoilerplateKind, Structure, Tokens};
use turbo_chunker::terms::{ContentTermCounter, TermCounter, TermIndexer, TermStats, Vocabulary};

const TEXT: &str = "Cats nap here. Cats purr too. Page one footer.";

/// Tokens at spaces with each closing period its own token, a paragraph
/// per sentence, and the last paragraph flagged as a footer.
fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut at = 0u32;
    for word in TEXT.split(' ') {
        let w = word.trim_end_matches('.');
        tokens.push(Span::new(at, at + w.len() as u32));
        if w.len() < word.len() {
            tokens.push(Span::new(at + w.len() as u32, at + word.len() as u32));
        }
        at += word.len() as u32 + 1;
    }
    let mut st =
        Structure { spans: Spans::new(ViewId::SOURCE), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    for (a, b) in [(0, 14), (15, 29), (30, 46)] {
        st.spans.push(Span::new(a, b));
        st.kind.push(Block::Paragraph);
        st.parent.push(None);
        st.depth.push(0);
    }
    let flags = Boilerplate::<Structure>::new(vec![None, None, Some(BoilerplateKind::Footer)], vec![0.0, 0.0, 1.0]);
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, st, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, flags, Provenance::by("t")).unwrap();
    doc
}

fn lengths<L: Layer>(doc: &Document, name: &'static str) -> Vec<u32> {
    doc.get_key(&Key::<TermStats<L>>::named(name)).unwrap().length.clone()
}

#[test]
fn boilerplate_tokens_count_in_no_unit() {
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .given(Key::<Boilerplate<Structure>>::DEFAULT)
        .add(TermIndexer)
        .step(Bound::new(TermCounter::<Structure>::new()).write(0, "all"))
        .step(Bound::new(TermCounter::<View>::new()).write(0, "all"))
        .step(Bound::new(ContentTermCounter::<Structure>::new()).write(0, "content"))
        .step(Bound::new(ContentTermCounter::<View>::new()).write(0, "content"))
        .build()
        .unwrap();
    let mut docs = [document()];
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let doc = &docs[0];
    assert_eq!(lengths::<Structure>(doc, "all"), [3, 3, 3]);
    assert_eq!(lengths::<Structure>(doc, "content"), [3, 3, 0]);
    assert_eq!(lengths::<View>(doc, "all"), [9]);
    assert_eq!(lengths::<View>(doc, "content"), [6]);
    let content = doc.get_key(&Key::<TermStats<View>>::named("content")).unwrap();
    let vocabulary = doc.get::<Vocabulary>().unwrap();
    assert_eq!(content.tf(Ix::new(0), vocabulary.lookup("Cats").unwrap()), 2);
    assert_eq!(content.tf(Ix::new(0), vocabulary.lookup("footer").unwrap()), 0);
    // The footer's terms are in no content unit.
    assert_eq!(content.df[vocabulary.lookup("footer").unwrap().get()], 0);
}

#[test]
fn boilerplate_in_another_view_is_refused() {
    let mut doc = document();
    let source = doc.get::<View>().unwrap().clone();
    let (text, alignment) = ViewBuilder::new(TEXT).finish();
    doc.add(&Key::named("copy"), View::derived(&source, text, alignment), Provenance::by("t")).unwrap();
    let copy = doc.get_key(&Key::<View>::named("copy")).unwrap().id();
    let st = Structure { spans: Spans::new(copy), kind: Vec::new(), parent: Vec::new(), depth: Vec::new() };
    doc.add(&Key::named("other"), st, Provenance::by("t")).unwrap();
    doc.add(
        &Key::named("other"),
        Boilerplate::<Structure>::new(Vec::new(), Vec::new()),
        Provenance { reads: vec![Key::<Structure>::named("other").id()], ..Provenance::by("t") },
    )
    .unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Structure>::named("other"))
        .given(Key::<Boilerplate<Structure>>::named("other"))
        .add(TermIndexer)
        .step(Bound::new(ContentTermCounter::<View>::new()).read(4, "other").read(5, "other"))
        .build()
        .unwrap();
    let mut docs = [doc];
    let err = pipeline.run(&mut docs).remove(0).unwrap_err().to_string();
    assert!(err.contains("the boilerplate is in view"), "{err}");
}
