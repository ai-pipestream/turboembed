use turbo_analysis::*;
use turbo_analysis_layers::{Block, Structure};
use turbo_normalize::{NormalizeStep, Whitespace};
use turbo_structure::{StructureAnnotator, StructureProjector, Syntax};

const TEXT: &str = "SUMMARY\n\nCafé and trees.\n\nDETAILS\n\nMore text.\n";

#[test]
fn source_structure_survives_normalization_with_provenance() {
    let pipeline = Analyzer::builder()
        .step(Bound::new(StructureAnnotator::new(Syntax::Plain)).write(0, "source"))
        .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "clean"))
        .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
        .build()
        .unwrap();
    let mut docs = [Document::new(TEXT)];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let original = doc.get_key(&Key::<Structure>::named("source")).unwrap();
    let projected = doc.get::<Structure>().unwrap();
    assert_eq!(original.kind.iter().filter(|&&k| k == Block::Heading).count(), 2);
    assert_eq!(projected.kind, original.kind);
    assert_eq!(projected.parent, original.parent);
    assert_eq!(projected.depth, original.depth);
    let clean = doc.get_key(&Key::<View>::named("clean")).unwrap();
    assert_eq!(projected.spans.view(), clean.id());
    for ((_, a), (_, b)) in original.spans.iter().zip(projected.spans.iter()) {
        assert_eq!(doc.to_source(projected.spans.view(), b), a);
    }
    let provenance = doc.provenance(Key::<Structure>::DEFAULT.id()).unwrap();
    assert!(provenance.reads.contains(&Key::<Structure>::named("source").id()));
}

#[test]
fn unrelated_views_crossed_edges_and_erased_blocks_are_refused() {
    let mut doc = Document::new("First. Second.");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, 6));
    spans.push(Span::new(7, 14));
    let structure = Structure { spans, kind: vec![Block::Paragraph; 2], parent: vec![None; 2], depth: vec![0; 2] };
    let source = doc.get::<View>().unwrap();
    assert!(StructureProjector.annotate((source, &structure)).is_err());
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(5);
    builder.replace(8, "x");
    let (text, alignment) = builder.finish();
    let edited = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("cross"), edited, Provenance::by("test")).unwrap();
    let cross = doc.get_key(&Key::<View>::named("cross")).unwrap();
    assert!(StructureProjector.annotate((cross, &structure)).err().unwrap().to_string().contains("edge inside"));
    let mut builder = ViewBuilder::new(doc.get::<View>().unwrap().text());
    builder.replace(6, "");
    let (text, alignment) = builder.finish();
    let edited = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("erased"), edited, Provenance::by("test")).unwrap();
    let erased = doc.get_key(&Key::<View>::named("erased")).unwrap();
    assert!(StructureProjector.annotate((erased, &structure)).err().unwrap().to_string().contains("erased"));

    let mut empty_spans = Spans::new(ViewId::SOURCE);
    empty_spans.push(Span::new(0, 0));
    empty_spans.push(Span::new(0, 6));
    let with_empty = Structure {
        spans: empty_spans,
        kind: vec![Block::Heading, Block::Paragraph],
        parent: vec![None; 2],
        depth: vec![0; 2],
    };
    let builder = ViewBuilder::new(doc.get::<View>().unwrap().text());
    let (text, alignment) = builder.finish();
    let identity = View::derived(doc.get::<View>().unwrap(), text, alignment);
    doc.add(&Key::named("identity"), identity, Provenance::by("test")).unwrap();
    let identity = doc.get_key(&Key::<View>::named("identity")).unwrap();
    let (projected,) = StructureProjector.annotate((identity, &with_empty)).unwrap();
    assert_eq!(projected.spans.get(Ix::new(0)), Span::new(0, 0));
    assert_eq!(projected.spans.get(Ix::new(1)), Span::new(0, 6));
}

#[test]
fn projected_sections_constrain_chunks_in_the_clean_view() {
    use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences, Subwords};
    use turbo_chunker::{Budget, Chunker};
    let pipeline = Analyzer::builder()
        .step(Bound::new(StructureAnnotator::new(Syntax::Plain)).write(0, "source"))
        .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "clean"))
        .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
        .build()
        .unwrap();
    let mut docs = [Document::new(TEXT)];
    pipeline.run(&mut docs).remove(0).unwrap();
    let doc = &mut docs[0];
    let view = doc.get_key(&Key::<View>::named("clean")).unwrap();
    let mut sentences = Spans::new(view.id());
    for s in ["SUMMARY", "Café and trees.", "DETAILS", "More text."] {
        let at = view.text().find(s).unwrap();
        sentences.push(Span::new(at as u32, (at + s.len()) as u32));
    }
    let mut words = Spans::new(view.id());
    for word in view.text().split_whitespace() {
        let at = word.as_ptr() as usize - view.text().as_ptr() as usize;
        words.push(Span::new(at as u32, (at + word.len()) as u32));
    }
    let model = ModelRef { id: "fixture".into(), revision: "1".into(), sha256: "fixture".into() };
    let ids = vec![1; words.len()];
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: words, ids, model, token: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Boundaries { cost: vec![1; 4], reason: vec![Cut::Sentence; 4] }, Provenance::by("test"))
        .unwrap();
    let chunker = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Boundaries>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Chunker { budget: Budget { max: 100, target: 100, size_weight: 1, overhead: 0 } })
        .build()
        .unwrap();
    chunker.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];
    let chunks = doc.get::<Chunks>().unwrap();
    let leaves: Vec<_> = chunks
        .spans
        .iter()
        .filter(|(i, _)| chunks.level[i.get()] == 0)
        .map(|(_, s)| &doc.text(chunks.spans.view())[s.range()])
        .collect();
    assert_eq!(leaves, ["SUMMARY Café and trees.", "DETAILS More text."]);
}

#[test]
fn discontinuous_blocks_project_each_fragment_to_the_child_view() {
    let text = "Start  here.  [note]  Resume  there.  After.";
    let mut doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push_fragments(&[Span::new(0, 12), Span::new(22, 36)]);
    spans.push(Span::new(38, 44));
    let structure = Structure {
        spans,
        kind: vec![Block::Paragraph, Block::Paragraph],
        parent: vec![None, None],
        depth: vec![0, 0],
    };
    doc.add(&Key::<Structure>::named("source"), structure, Provenance::by("test")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<Structure>::named("source"))
        .step(Bound::new(NormalizeStep::new(Whitespace::collapse())).write(0, "clean"))
        .step(Bound::new(StructureProjector).read(0, "clean").read(1, "source"))
        .build()
        .unwrap();
    let mut docs = [doc];
    assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
    let doc = &docs[0];
    let projected = doc.get::<Structure>().unwrap();
    let clean_text = doc.text(projected.spans.view());
    assert!(projected.spans.is_discontinuous(Ix::new(0)));
    let frags: Vec<&str> = projected.spans.fragments(Ix::new(0)).map(|f| &clean_text[f.range()]).collect();
    assert_eq!(frags, ["Start here.", "Resume there."]);
    assert_eq!(&clean_text[projected.spans.get(Ix::new(1)).range()], "After.");
}

#[test]
fn discontinuous_fragments_coalesce_when_normalization_deletes_the_gap() {
    let text = "left GAP right";
    let doc = Document::new(text);
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push_fragments(&[Span::new(0, 4), Span::new(9, 14)]);
    let structure = Structure { spans, kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] };
    let mut builder = ViewBuilder::new(text);
    builder.keep(4);
    builder.replace(9, "");
    let (clean_text, align) = builder.finish();
    let clean = View::derived(doc.get::<View>().unwrap(), clean_text, align);
    let (projected,) = StructureProjector.annotate((&clean, &structure)).unwrap();
    assert!(!projected.spans.is_discontinuous(Ix::new(0)));
    assert_eq!(&clean.text()[projected.spans.get(Ix::new(0)).range()], "leftright");
}
