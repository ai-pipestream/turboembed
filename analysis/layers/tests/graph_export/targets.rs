use super::*;

#[test]
fn target_identity_and_word_subspan_refusals() {
    // A word subspan split across a UTF-8 character must not import.
    let mut doc = Document::new("é");
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 2)]), sentence: None }, Provenance::by("test")).unwrap();
    let mut words = Words::new();
    words.push_word(IxRange::new(0, 1), None, Some(Span::new(0, 2)));
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    let body = graph.layers.iter_mut().find(|l| l.layer == "turbo:words").unwrap().body.as_mut().unwrap();
    let at = body.len() - 4;
    body[at..].copy_from_slice(&1u32.to_le_bytes());
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("boundaries"), "{detail}"),
        Ok(_) => panic!("a subspan splitting a character imported"),
        Err(other) => panic!("a split subspan failed as {other}"),
    }

    // An encoded target instance the provenance does not bind to.
    let doc = document();
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("alt", &[0]), 1);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("a payload naming another target instance imported"),
        Err(other) => panic!("a rebound target failed as {other}"),
    }

    // An empty sentence column naming a missing target.
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("missing", &[]), 0);
    assert!(import(&graph, &codecs()).is_err(), "an empty column naming a missing target imported");

    // A token pointing at a sentence past the sentence layer.
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("", &[9]), 1);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::DanglingEndpoint { .. })));

    // A structure block that is its own parent.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.usize(1).unwrap();
    w.u8(2);
    w.usize(1).unwrap();
    w.u8(1);
    w.u32(0);
    w.usize(1).unwrap();
    w.u8(0);
    splice(&mut graph, "turbo:structure", w.finish(), 1);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::Malformed(_))));

    // A document date naming a temporal mention past the temporals.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.u8(0);
    w.u8(1);
    w.str("turbo:temporal").unwrap();
    w.str("").unwrap();
    w.u32(9);
    splice(&mut graph, "turbo:document-date", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a document date naming a missing mention imported");
}

#[test]
fn keyed_default_word_target_must_not_rebind_through_provenance() {
    let mut doc = Document::new("ab");
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 1)]), sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::named("alt"), Tokens { spans: spans(&[(1, 2)]), sentence: None }, Provenance::by("test")).unwrap();
    let mut words = Words::on(Key::named("alt"));
    words.push_word(IxRange::new(0, 1), None, None);
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Tokens>::named("alt").id()];
    doc.add(&Key::DEFAULT, words, provenance).unwrap();

    // The valid document, with its explicit alt key, round-trips.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // The same body with only the encoded target instance emptied
    // would validate against the default tokens and then rebind to
    // alt through provenance, so it must be refused.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.str("opennlp:tokens").unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.u8(0);
    w.u8(0);
    splice(&mut graph, "turbo:words", w.finish(), 1);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("an empty keyed target rebound through provenance imported"),
        Err(other) => panic!("an empty keyed target failed as {other}"),
    }
}

#[test]
fn keyed_default_opinion_target_must_not_rebind_through_provenance() {
    let mut doc = Document::new("ab");
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 1)]), sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::named("alt"), Tokens { spans: spans(&[(1, 2)]), sentence: None }, Provenance::by("test")).unwrap();
    let mut opinions = Opinions::on(Key::named("alt"));
    opinions.tokens = vec![IxRange::new(0, 1)];
    opinions.polarity = vec![0.5];
    opinions.intensity = vec![1.0];
    opinions.negated = vec![false];
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Tokens>::named("alt").id()];
    doc.add(&Key::DEFAULT, opinions, provenance).unwrap();

    // The valid document, with its explicit alt key, round-trips.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // The same body with only the encoded token target emptied must
    // be refused rather than rebound to alt through provenance.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.str("opennlp:tokens").unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.usize(1).unwrap();
    w.f64(0.5);
    w.usize(1).unwrap();
    w.f64(1.0);
    w.usize(1).unwrap();
    w.u8(0);
    splice(&mut graph, "turbo:opinions", w.finish(), 1);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("an empty keyed opinion target rebound through provenance imported"),
        Err(other) => panic!("an empty keyed opinion target failed as {other}"),
    }
}
