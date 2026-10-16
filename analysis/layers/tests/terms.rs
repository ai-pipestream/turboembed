use turbo_analysis::{Anchor, Document, Ix, Key, Layer, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Stems, Strs, TermSlot, Terms, Tokens};

#[test]
fn stems_and_terms_are_aligned_to_tokens_and_validate_columns() {
    assert_eq!(Stems::ID, "turbo:stems");
    assert_eq!(Stems::anchor(), Anchor::aligned::<Tokens>());
    assert_eq!(Terms::ID, "turbo:terms");
    assert_eq!(Terms::anchor(), Anchor::aligned::<Tokens>());

    let mut surface = Strs::default();
    surface.push("Cafés");
    surface.push("were");
    let mut folded = Strs::default();
    folded.push("cafes");
    folded.push("were");
    let mut stem = Strs::default();
    stem.push("cafe");
    stem.push("were");
    let mut lemma = Strs::default();
    lemma.push("café");
    lemma.push("be");

    let terms = Terms { surface, folded, stem: Some(stem), lemma: Some(lemma) };
    assert_eq!(terms.len(), 2);
    assert_eq!(terms.check(), Ok(()));
    assert_eq!(
        terms.get(Ix::new(0)),
        TermSlot { surface: "Cafés", folded: "cafes", stem: Some("cafe"), lemma: Some("café") }
    );
    assert_eq!(
        terms.iter().collect::<Vec<_>>(),
        [
            TermSlot { surface: "Cafés", folded: "cafes", stem: Some("cafe"), lemma: Some("café") },
            TermSlot { surface: "were", folded: "were", stem: Some("were"), lemma: Some("be") },
        ]
    );

    let mut doc = Document::new("Cafés were");
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    tok_spans.push(Span::new(0, 6));
    tok_spans.push(Span::new(7, 11));
    doc.add(&Key::DEFAULT, Tokens { spans: tok_spans, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, terms.clone(), Provenance::by("test")).unwrap();

    let mut bad_folded = terms.clone();
    bad_folded.folded.push("extra");
    assert_eq!(bad_folded.check().unwrap_err(), "2 surface forms and 3 folded forms");

    let mut bad_stem = terms.clone();
    bad_stem.stem.as_mut().unwrap().push("extra");
    assert_eq!(bad_stem.check().unwrap_err(), "2 surface forms and 3 stems");

    let mut bad_lemma = terms;
    bad_lemma.lemma.as_mut().unwrap().push("extra");
    assert_eq!(bad_lemma.check().unwrap_err(), "2 surface forms and 3 lemmas");
}
