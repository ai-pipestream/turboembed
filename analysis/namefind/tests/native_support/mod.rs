#![allow(dead_code)]
use turbo_analysis::{Annotator, Document, Layer, Span, Spans, View, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_namefind::{Dictionary, DictionaryNameFinder};

pub fn finder(extra: usize, kind: &str) -> DictionaryNameFinder {
    let mut entries = vec![vec!["Zoë".to_owned()], vec!["李明".to_owned()]];
    entries.extend((0..extra).map(|i| vec![format!("Extra{i}")]));
    let entries = entries.into_iter().map(|tokens| (kind.to_owned(), tokens));
    DictionaryNameFinder::new(Dictionary::from_entries(true, entries)).unwrap()
}

pub fn input() -> (Document, Tokens) {
    let doc = Document::new("Zoë met 李明.");
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 4), (5, 8), (9, 15)] {
        spans.push(Span::new(a, b));
    }
    (doc, Tokens { spans, sentence: None })
}

pub fn entities(finder: &DictionaryNameFinder) -> Entities {
    let (doc, tokens) = input();
    let (out,) = finder.annotate((doc.get::<View>().unwrap(), &tokens)).unwrap();
    out.check().unwrap();
    assert_eq!(out.spans.columns(), (&[0, 9][..], &[4, 15][..]));
    assert_eq!(out.tokens.iter().map(|r| (r.start.get(), r.end.get())).collect::<Vec<_>>(), [(0, 1), (2, 3)]);
    out
}
