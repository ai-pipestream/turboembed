use turbo_analysis::{Document, DocumentGraph, Ix, Key, Layer, Provenance, Span, ViewId, export, import};
use turbo_analysis_layers::{Quotations, QuoteKind};

fn quotes(rows: &[&[(u32, u32)]]) -> Quotations {
    let mut q = Quotations::new(ViewId::SOURCE);
    for row in rows {
        let fragments: Vec<_> = row.iter().map(|&(start, end)| Span::new(start, end)).collect();
        q.spans.push_fragments(&fragments);
        q.speaker.push(None);
        q.cue_verb.push(None);
        q.speaker_chain.push(Some(u32::MAX));
        q.kind.push(QuoteKind::Mixed);
        q.confidence.push(0.5);
    }
    q.check().unwrap();
    q
}

fn checked(q: Quotations) -> Quotations {
    let mut doc = Document::new("é-x-é");
    doc.add(&Key::DEFAULT, q, Provenance::by("test:quote-equality")).unwrap();
    doc.require(&Key::<Quotations>::DEFAULT).unwrap().clone()
}

#[test]
fn equal_envelopes_do_not_hide_different_quotation_content() {
    let variants = [
        checked(quotes(&[&[(0, 7)]])),
        checked(quotes(&[&[(0, 2), (5, 7)]])),
        checked(quotes(&[&[(0, 3), (5, 7)]])),
        checked(quotes(&[&[(0, 2), (3, 4), (5, 7)]])),
    ];
    for (i, left) in variants.iter().enumerate() {
        for (j, right) in variants.iter().enumerate() {
            assert_eq!(left.get(0).unwrap().content, right.get(0).unwrap().content);
            assert_eq!(left == right, i == j, "content variants {i} and {j}");
        }
    }
}

#[test]
fn later_rows_also_compare_their_fragments() {
    let full = checked(quotes(&[&[(0, 2)], &[(3, 7)]]));
    let gap = checked(quotes(&[&[(0, 2)], &[(3, 4), (5, 7)]]));
    assert_ne!(full, gap);
    assert_ne!(gap, full);
}

#[test]
fn binary_round_trip_preserves_named_fragmented_content_and_equality() {
    let mut doc = Document::new("é-x-é");
    let continuous = quotes(&[&[(0, 7)]]);
    let fragmented = quotes(&[&[(0, 2), (5, 7)]]);
    doc.add(&Key::DEFAULT, continuous.clone(), Provenance::by("test:continuous")).unwrap();
    let key = Key::<Quotations>::named("gold");
    doc.add(&key, fragmented.clone(), Provenance::by("test:fragmented")).unwrap();
    let codecs = turbo_analysis_layers::export::codecs();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let restored = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    let actual = restored.require(&key).unwrap();
    assert_eq!(actual.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), [Span::new(0, 2), Span::new(5, 7)]);
    assert_eq!(actual, &fragmented);
    assert_eq!(restored.require(&Key::<Quotations>::DEFAULT).unwrap(), &continuous);
    assert_ne!(actual, restored.require(&Key::<Quotations>::DEFAULT).unwrap());
}

#[test]
fn equality_keeps_existing_row_and_float_semantics() {
    let original = checked(quotes(&[&[(0, 2), (5, 7)]]));
    assert_eq!(original, original.clone());
    assert_eq!(Quotations::default(), Quotations::default());
    assert_ne!(original, Quotations::default());
    for field in 0..5 {
        let mut changed = original.clone();
        match field {
            0 => changed.speaker[0] = Some(Span::new(3, 4)),
            1 => changed.cue_verb[0] = Some(Span::new(3, 4)),
            2 => changed.speaker_chain[0] = None,
            3 => changed.kind[0] = QuoteKind::Direct,
            4 => changed.confidence[0] = 0.75,
            _ => unreachable!(),
        }
        assert_ne!(original, checked(changed));
    }
    let mut positive = original.clone();
    positive.confidence[0] = 0.0;
    let mut negative = original;
    negative.confidence[0] = -0.0;
    assert_eq!(positive, negative);
}
