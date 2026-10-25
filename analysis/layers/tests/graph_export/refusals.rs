use super::*;

#[test]
fn vector_dimension_mismatches_are_refused() {
    let mut doc = document();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            Vec::new(),
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        Provenance::by("test"),
    )
    .unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    // Claim 5 values for 2 rows of dimension 2.
    let mut w = Writer::new();
    w.u32(2);
    w.usize(2).unwrap();
    w.usize(5).unwrap();
    for _ in 0..5 {
        w.u32(0);
    }
    w.usize(0).unwrap();
    w.str("m").unwrap();
    w.str("r").unwrap();
    w.str("s").unwrap();
    splice(&mut graph, "opennlp:embeddings", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("vector values"), "{detail}"),
        Ok(_) => panic!("a vector block with a value count mismatch imported"),
        Err(other) => panic!("a vector block mismatch failed as {other}"),
    }
}

#[test]
fn sentiment_family_refusals() {
    let mut doc = document();
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Money {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["500"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Numbers {
            spans: spans(&[(0, 5)]),
            numeral_start: vec![0],
            numeral_end: vec![5],
            negative: vec![false],
            coefficient: strs(&["42"]),
            scale: vec![0],
            multiplier: vec![Multiplier::One],
            marker: vec![ScaleMarker::None],
            notation: vec![NumberNotation::LatinUs],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Phrases {
            spans: spans(&[(0, 5)]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: tagset("phrase-kinds", &["NP"]),
            head: vec![Ix::new(0)],
            parent: vec![None],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            2,
            vec![0.0; 4],
            vec![],
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        by,
    )
    .unwrap();

    // A twelve byte vector body claiming billions of values must be
    // a typed refusal, not an allocation.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.u32(1);
    w.usize(u32::MAX as usize).unwrap();
    w.usize(u32::MAX as usize).unwrap();
    splice(&mut graph, "opennlp:embeddings", w.finish(), u32::MAX);
    assert!(import(&graph, &codecs()).is_err(), "a vector body claiming billions of values imported");

    // An extreme number span must be refused, never panicking in the
    // layer's own checks.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(u32::MAX, u32::MAX)]);
    splice(&mut graph, "turbo:numbers", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a number at the end of the address space imported");

    // A schema-1 turbo:numbers record must be refused by schema version.
    let mut graph = export(&doc, &codecs()).unwrap();
    for record in &mut graph.layers {
        if record.layer == "turbo:numbers" {
            record.schema = 1;
        }
    }
    match import(&graph, &codecs()) {
        Err(ExportError::IncompatibleSchema { layer, exported, supported }) => {
            assert_eq!((layer.as_str(), exported, supported), ("turbo:numbers", 1, 2));
        }
        Ok(_) => panic!("a schema-1 turbo:numbers record imported"),
        Err(other) => panic!("a schema-1 turbo:numbers record failed as {other}"),
    }

    // A phrase head outside its token range.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.str("opennlp:tokens").unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.usize(1).unwrap();
    w.str("NP").unwrap();
    w.usize(1).unwrap();
    w.u16(0);
    w.usize(1).unwrap();
    w.u32(5);
    w.usize(1).unwrap();
    w.u8(0);
    splice(&mut graph, "opennlp:phrases", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a phrase head outside its tokens imported");

    // An unknown currency code.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.usize(1).unwrap();
    w.u8(0);
    w.usize(1).unwrap();
    w.str("1").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.usize(1).unwrap();
    w.str("not-a-currency").unwrap();
    splice(&mut graph, "turbo:money", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "an unknown currency code imported");
}

#[test]
fn quantities_schema_1_is_refused() {
    let mut doc = document();
    let mut q = Quantities::new(ViewId::SOURCE);
    q.push(turbo_analysis_layers::QuantityMention {
        span: Span::new(0, 5),
        negative: false,
        coefficient: "5",
        scale: 0,
        unit: "km",
    });
    doc.add(&Key::DEFAULT, q, Provenance::by("test")).unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    for layer in &mut graph.layers {
        if layer.layer == "turbo:quantities" {
            layer.schema = 1;
        }
    }
    assert!(matches!(
        import(&graph, &codecs()),
        Err(ExportError::IncompatibleSchema { ref layer, exported: 1, supported: 2 }) if layer == "turbo:quantities"
    ));
}
