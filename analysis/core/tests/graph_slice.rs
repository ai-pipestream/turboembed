//! Byte-range subgraph slicing (#1199 part 2): source and view
//! slicing through alignments, span and relation layer slicing
//! with index remaps, and the refusal for layers whose codecs do
//! not support slicing.

use std::sync::Arc;

use turbo_analysis::export::{Codecs, ExportError, export, import};
use turbo_analysis::*;
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, Sentences, Tokens};

fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut codecs);
    codecs.register::<Edges<Tokens, Tokens>>().register::<Groups<Tokens>>();
    codecs
}

/// "aa bb cc dd" with two sentences, four tokens, an edge and a
/// group spanning the middle, and a derived view replacing "cc"
/// with "C".
fn document() -> Document {
    let text = "aa bb cc dd";
    let mut doc = Document::new(text);
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 5));
    sentences.push(Span::new(6, 11));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test:sentences")).unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 2), (3, 5), (6, 8), (9, 11)] {
        spans.push(Span::new(a, b));
    }
    doc.add(
        &Key::DEFAULT,
        Tokens { spans, sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1)]) },
        Provenance::by("test:tokens"),
    )
    .unwrap();
    let kinds: Arc<TagSet> = TagSet::new("rel", &["next"]);
    let next = kinds.tag("next").unwrap();
    doc.add(
        &Key::DEFAULT,
        Edges::<Tokens, Tokens>::new(
            kinds.clone(),
            [
                Edge::new(Ix::new(0), Ix::new(1), next, 1.0),
                Edge::new(Ix::new(1), Ix::new(2), next, 1.0),
                Edge::new(Ix::new(2), Ix::new(3), next, 1.0),
            ],
        ),
        Provenance::by("test:edges"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Groups::<Tokens>::new([vec![Ix::new(0), Ix::new(1)], vec![Ix::new(1), Ix::new(2), Ix::new(3)]]),
        Provenance::by("test:groups"),
    )
    .unwrap();
    let v0 = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(text);
    builder.keep(6);
    builder.replace(8, "C");
    let (v1_text, a1) = builder.finish();
    assert_eq!(v1_text, "aa bb C dd");
    doc.add(&Key::<View>::named("clean"), View::derived(&v0, v1_text, a1), Provenance::by("test:view")).unwrap();
    doc
}

#[test]
fn slice_rebases_source_views_and_layers_and_remaps_relations() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    // Keep "bb cc" (3..8 of the source).
    let sliced = graph.slice_source_span(3..8, &codecs()).unwrap();
    assert_eq!(sliced.source.text, "bb cc");
    let out = import(&sliced, &codecs()).unwrap();
    assert_eq!(out.source(), "bb cc");

    let sentences: Vec<Span> = out.get::<Sentences>().unwrap().spans.iter().map(|(_, s)| s).collect();
    assert_eq!(sentences, vec![Span::new(0, 2), Span::new(3, 5)]);
    let tokens = out.get::<Tokens>().unwrap();
    let token_spans: Vec<Span> = tokens.spans.iter().map(|(_, s)| s).collect();
    assert_eq!(token_spans, vec![Span::new(0, 2), Span::new(3, 5)]);
    assert_eq!(tokens.sentence.as_deref(), Some([Ix::new(0), Ix::new(1)].as_slice()));

    // Only the 1 -> 2 edge survives, remapped to 0 -> 1.
    let edges: Vec<(usize, usize)> =
        out.get::<Edges<Tokens, Tokens>>().unwrap().iter().map(|(_, e)| (e.from.get(), e.to.get())).collect();
    assert_eq!(edges, vec![(0, 1)]);
    // The first group keeps only token 1 (now 0); the second keeps
    // 1 and 2 (now 0 and 1).
    let groups: Vec<Vec<usize>> =
        out.get::<Groups<Tokens>>().unwrap().iter().map(|(_, m)| m.iter().map(|i| i.get()).collect()).collect();
    assert_eq!(groups, vec![vec![0], vec![0, 1]]);

    // The derived view sliced through its alignment: "bb C".
    let clean = out.get_key(&Key::<View>::named("clean")).unwrap();
    assert_eq!(clean.text(), "bb C");
}

#[test]
fn slice_of_an_empty_range_and_full_range() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let full = graph.slice_source_span(0..11, &codecs()).unwrap();
    let out = import(&full, &codecs()).unwrap();
    assert_eq!(out.source(), "aa bb cc dd");
    assert_eq!(out.get::<Tokens>().unwrap().spans.len(), 4);
    let empty = graph.slice_source_span(4..4, &codecs()).unwrap();
    let out = import(&empty, &codecs()).unwrap();
    assert_eq!(out.source(), "");
    assert_eq!(out.get::<Tokens>().unwrap().spans.len(), 0);
}

#[test]
fn slice_refuses_unsupported_layers_and_bad_ranges() {
    let mut doc = document();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(3, 5));
    doc.add(
        &Key::DEFAULT,
        Abbreviations { spans, form: vec![AbbreviationForm::Short] },
        Provenance::by("test:abbreviations"),
    )
    .unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    let err = graph.slice_source_span(3..8, &codecs()).unwrap_err();
    assert!(matches!(err, ExportError::InvalidLayer { .. }), "abbreviations refuse: {err:?}");
    assert!(err.to_string().contains("abbreviations"), " names the layer: {err}");

    let graph = export(&document(), &codecs()).unwrap();
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 4..3;
    assert!(graph.slice_source_span(reversed, &codecs()).is_err());
    assert!(graph.slice_source_span(0..99, &codecs()).is_err());
    // Off a character boundary in the source text.
    let mut doc = Document::new("aé b");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 1));
    doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("test:sentences")).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    assert!(graph.slice_source_span(0..2, &codecs()).is_err());
}
