use super::*;

#[test]
fn core_pipeline_layers_round_trip() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();

    assert_eq!(back.get::<Tokens>().unwrap().sentence.as_ref().unwrap()[3], Ix::new(1));
    assert_eq!(back.get::<Words>().unwrap().tokens[2], IxRange::new(2, 3));
    let words = back.get::<Words>().unwrap();
    assert_eq!(words.form.as_ref().unwrap()[0].as_deref(), Some("hello"));
    assert_eq!(words.subspan.as_ref().unwrap()[1], Some(Span::new(6, 11)));
    assert_eq!(back.get::<Subwords>().unwrap().ids, [10, 11, 12]);
    assert_eq!(back.get::<Subwords>().unwrap().model.id, "m");
    assert_eq!(back.get::<PosTags>().unwrap().tags[3], Tag(1));
    assert_eq!(back.get::<Lemmas>().unwrap().lemma.get(1), "world");
    assert_eq!(back.get::<Terms>().unwrap().folded.get(0), "hello");
    assert_eq!(back.get::<Dependencies<Tokens>>().unwrap().head[0], Some(Ix::new(1)));
    assert_eq!(back.get::<Dependencies<Words>>().unwrap().head[5], Some(Ix::new(4)));
    let entities = back.get::<Entities>().unwrap();
    assert_eq!(entities.spans.fragments(Ix::new(0)).count(), 2, "the discontinuous entity kept its fragments");
    assert_eq!(entities.tokens[0], IxRange::new(0, 5));
    assert_eq!(back.get::<Structure>().unwrap().parent[2], Some(Ix::new(0)));
    assert_eq!(back.get::<Boundaries>().unwrap().reason[1], Cut::Inside(Block::Table));
    let chunks = back.get::<Chunks>().unwrap();
    assert_eq!(
        (chunks.parent[1], chunks.sentences[2], chunks.over_budget[2]),
        (Some(Ix::new(0)), IxRange::new(1, 2), true)
    );
    assert_eq!(back.get::<Language>().unwrap().best(), Some(("eng", 0.9)));
    assert_eq!(back.get::<LanguageSpans>().unwrap().get(Ix::new(0)).script, Script::Latin);
    assert_eq!(back.get::<Temporals>().unwrap().year, [2026]);
    assert_eq!(back.get::<DocumentDate>().unwrap().mention, Some(Ix::new(0)));
    assert_eq!(back.get::<Categories<turbo_analysis::View>>().unwrap().best_category(0), "a");
    assert_eq!(back.get::<Categories<Sentences>>().unwrap().best_category(1), "b");
    // The abstaining sentence round-trips: scored, no best category.
    let abstained = back.get_key(&Key::<Categories<Sentences>>::named("abstained")).unwrap();
    let row = abstained.get(0).unwrap();
    assert_eq!(row.best_tag, Tag::NONE);
    assert_eq!(row.best, "");
    assert_eq!(row.scores, [0.2, 0.3]);
    // The multi-label sentence keeps its full selected set across the
    // wire, not just the best category; the abstaining one stays empty.
    let multi = back.get_key(&Key::<Categories<Sentences>>::named("multi")).unwrap();
    assert_eq!(multi.category, [Tag(0), Tag::NONE]);
    assert_eq!(multi.selected(0), [Tag(0), Tag(1)]);
    assert!(multi.selected(1).is_empty());
    assert_eq!(multi.get(0).unwrap().selected, [Tag(0), Tag(1)]);

    // Re-export is byte stable.
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn subword_and_level_vectors_round_trip() {
    let mut doc = document();
    let model = ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() };
    doc.add(
        &Key::DEFAULT,
        Levels { spans: spans(&[(0, 21)]), unit: vec![Unit::Document], parent: vec![None] },
        Provenance::by("test"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Subwords>::new(2, vec![0.25, -0.5, 0.75, 1.0, -0.25, 0.5], vec![], model.clone()),
        Provenance::by("test"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Vectors::<Levels>::new(2, vec![1.0, 2.0], vec![], model), Provenance::by("test")).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.get::<Vectors<Subwords>>().unwrap().data, [0.25, -0.5, 0.75, 1.0, -0.25, 0.5]);
    assert_eq!(back.get::<Vectors<Levels>>().unwrap().data, [1.0, 2.0]);
}

#[test]
fn layers_on_a_derived_view_round_trip() {
    use turbo_analysis::{Alignment, View, ViewBuilder};
    let mut doc = Document::decoded(
        b"Caf\xe9 moo.".to_vec(),
        "Café moo.",
        Alignment::from_edits(vec![turbo_analysis::text::Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }])
            .unwrap(),
    )
    .unwrap();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(5);
    builder.replace(6, "");
    let (clean, alignment) = builder.finish();
    doc.add(&Key::<View>::named("clean"), View::derived(&source, clean, alignment), Provenance::by("test")).unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    let mut sentences = Spans::new(clean_id);
    sentences.push(Span::new(0, 9));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let mut tokens = Spans::new(clean_id);
    tokens.push(Span::new(0, 5));
    tokens.push(Span::new(5, 8));
    tokens.push(Span::new(8, 9));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: Some(vec![Ix::new(0); 3]) }, Provenance::by("test"))
        .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let tokens = back.get::<Tokens>().unwrap();
    assert_eq!(tokens.spans.view(), back.get_key(&Key::<View>::named("clean")).unwrap().id());
    assert_eq!(back.text(tokens.spans.view()), "Cafémoo.");
    let in_source = back.to_source(tokens.spans.view(), tokens.spans.get(Ix::new(1)));
    assert_eq!(&back.source()[in_source.range()], "moo");
}

#[test]
fn corrupted_payloads_are_refused() {
    let base = export(&document(), &codecs()).unwrap();

    // A dependency head cycle, valid column lengths or not.
    let mut graph = base.clone();
    let mut w = Writer::new();
    put_tagset(&mut w, &tagset("rel", &["root", "dep"])).unwrap();
    w.usize(2).unwrap();
    w.u8(1);
    w.u32(1);
    w.u8(1);
    w.u32(0);
    w.usize(2).unwrap();
    w.u16(1);
    w.u16(1);
    splice(&mut graph, "opennlp:dependencies", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("cycle"), "{detail}"),
        Ok(_) => panic!("a dependency cycle imported"),
        Err(other) => panic!("a dependency cycle failed as {other}"),
    }

    // A tag outside its decoded set.
    let mut graph = base.clone();
    let mut w = Writer::new();
    put_tagset(&mut w, &tagset("pos", &["NN"])).unwrap();
    w.usize(6).unwrap();
    for _ in 0..6 {
        w.u16(9);
    }
    w.u8(0);
    splice(&mut graph, "opennlp:pos", w.finish(), 6);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::Malformed(_))));

    // A terms column shorter than the surface column.
    let mut graph = base.clone();
    let mut w = Writer::new();
    for col in [2usize, 1] {
        w.usize(col).unwrap();
        for i in 0..col {
            w.str(&format!("t{i}")).unwrap();
        }
    }
    w.u8(0);
    w.u8(0);
    splice(&mut graph, "turbo:terms", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("folded"), "{detail}"),
        Ok(_) => panic!("a short folded column imported"),
        Err(other) => panic!("a short folded column failed as {other}"),
    }
}
