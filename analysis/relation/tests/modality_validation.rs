mod modality_support;

use turbo_analysis::{Analyzer, DocumentGraph, Key, Layer, Span, Spans, ViewId, export, import};
use turbo_analysis_layers::{Dependencies, Modality, Tokens};
use turbo_relation::{ModalityAnnotator, ModalityConfig};

fn graph() -> DocumentGraph {
    let d = modality_support::tokens_only("é Users must encrypt data .");
    let a = Analyzer::builder().given(Key::<Tokens>::DEFAULT).add(ModalityAnnotator::new()).build().unwrap();
    let mut docs = vec![d];
    for result in a.run(&mut docs) {
        result.unwrap();
    }
    let m = docs[0].require(&Key::<Modality>::DEFAULT).unwrap();
    assert_eq!(m.len(), 1);
    assert!(m.tag(0).cue.is_some());
    export(&docs[0], &turbo_relation::export::codecs()).unwrap()
}

#[test]
fn cycle_refusal_works_through_analyzer() {
    for (text, heads) in [("not a", "2:advmod 1:advmod"), ("Users may buy", "2:nsubj 3:aux 1:conj")] {
        let d = modality_support::parsed(text, heads);
        let a = Analyzer::builder()
            .given(Key::<Tokens>::DEFAULT)
            .given(Key::<Dependencies>::DEFAULT)
            .add(ModalityAnnotator::new())
            .build()
            .unwrap();
        let mut docs = vec![d];
        for result in a.run(&mut docs) {
            let error = result.unwrap_err().to_string();
            assert!(error.contains("cycle"), "{error}");
        }
    }
}

#[test]
fn valid_modality_roundtrip_and_stale_reach_refusal() {
    let codecs = turbo_relation::export::codecs();
    let back = import(&DocumentGraph::from_bytes(&graph().to_bytes().unwrap()).unwrap(), &codecs).unwrap();
    let original = back.require(&Key::<Modality>::DEFAULT).unwrap();
    let mut m = original.clone();
    let mut spans = Spans::new(ViewId(0));
    spans.push(Span::new(0, "é Users must encrypt data .".len() as u32));
    m.spans = spans;
    assert!(m.check().unwrap_err().contains("stale reach"));
}

#[test]
fn utf8_interior_cue_is_refused() {
    let mut g = graph();
    let layer = g.layers.iter_mut().find(|l| l.layer == Modality::NAME.to_string()).unwrap();
    let body = layer.body.as_mut().unwrap();
    let n = body.len();
    assert_eq!(body[n - 13], 1, "the final row must have a cue");
    body[n - 12..n - 8].copy_from_slice(&1_u32.to_le_bytes());
    body[n - 8..n - 4].copy_from_slice(&2_u32.to_le_bytes());
    let bytes = g.to_bytes().unwrap();
    let result = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &turbo_relation::export::codecs());
    if let Ok(ref d) = result {
        let cue = d.require(&Key::<Modality>::DEFAULT).unwrap().tag(0).cue.unwrap();
        eprintln!(
            "accepted cue {}..{}; start is a UTF-8 boundary: {}",
            cue.start,
            cue.end,
            d.text(ViewId(0)).is_char_boundary(cue.start as usize)
        );
    }
    assert!(result.is_err(), "a cue starting inside é imported");
}

#[test]
fn large_window_does_not_overflow() {
    let d = modality_support::tokens_only("Users must encrypt data .");
    let annotator =
        ModalityAnnotator::with_config(ModalityConfig { max_window_tokens: usize::MAX, ..Default::default() });
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| annotator.modality(&d)));
    assert!(result.is_ok(), "a valid large window panicked");
    let m = result.unwrap().unwrap();
    assert!(m.tag_for_span(modality_support::token(&d, 2)).is_some(), "the predicate was lost");
}

#[test]
fn storage_checks_cues_without_requiring_them_inside_the_scope() {
    use turbo_analysis::{Document, Provenance};
    use turbo_analysis_layers::ModalityTag;
    for cue in [Span::new(1, 2), Span::new(0, 1), Span::new(0, 99)] {
        let mut d = Document::new("é abc");
        let mut m = Modality::new(ViewId::SOURCE);
        m.push(Span::new(3, 6), ModalityTag { cue: Some(cue), confidence: 1.0, ..Default::default() });
        let err = d.add(&Key::DEFAULT, m, Provenance::by("test:modality")).unwrap_err();
        assert!(err.to_string().contains("cue"), "{err}");
    }
    let mut d = Document::new("é abc");
    let mut m = Modality::new(ViewId::SOURCE);
    m.push(Span::new(3, 6), ModalityTag { cue: Some(Span::new(0, 2)), confidence: 1.0, ..Default::default() });
    d.add(&Key::DEFAULT, m, Provenance::by("test:modality")).unwrap();
    let codecs = turbo_relation::export::codecs();
    let restored =
        import(&DocumentGraph::from_bytes(&export(&d, &codecs).unwrap().to_bytes().unwrap()).unwrap(), &codecs)
            .unwrap();
    assert_eq!(restored.get::<Modality>().unwrap().tag(0).cue, Some(Span::new(0, 2)));
}

#[test]
fn windows_clamp_to_text_without_changing_zero_or_ordinary_windows() {
    let d = modality_support::tokens_only("Users must encrypt data .");
    let run = |window| {
        ModalityAnnotator::with_config(ModalityConfig { max_window_tokens: window, ..Default::default() })
            .modality(&d)
            .unwrap()
    };
    assert_eq!(run(usize::MAX), run(128));
    let zero = run(0);
    assert!(zero.tag_for_span(modality_support::token(&d, 1)).is_some());
    assert!(zero.tag_for_span(modality_support::token(&d, 2)).is_none());
}
