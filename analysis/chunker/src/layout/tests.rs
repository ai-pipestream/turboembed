use turbo_analysis::export::{export, import};
use turbo_analysis::{Analyzer, Bound, Document, Ix, Key, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Cut, LayoutBoxes, PageBoundingBox, Sentences, Structure, Subwords};

use super::*;
use crate::{Budget, Chunker, Costs, STRUCTURE, StructureBoundaries, Weight};

#[test]
fn layout_boundaries_detect_page_breaks_column_jumps_and_visual_gaps() {
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 10));
    sentences.push(Span::new(11, 20));
    sentences.push(Span::new(21, 30));
    sentences.push(Span::new(31, 40));
    let sentences = Sentences { spans: sentences };

    let mut layout = LayoutBoxes::new(ViewId::SOURCE);
    // Sentence 0: page 1, left column, y = 10..30
    layout.push(Span::new(0, 10), 1, 50.0, 10.0, 250.0, 30.0);
    // Sentence 1: page 1, left column, large vertical gap (y = 60..80, gap = 30 >= 18)
    layout.push(Span::new(11, 20), 1, 50.0, 60.0, 250.0, 80.0);
    // Sentence 2: page 1, right column jump (x0 = 300 > 250, y0 = 15 < 60)
    layout.push(Span::new(21, 30), 1, 300.0, 15.0, 500.0, 35.0);
    // Sentence 3: page 2 (page break 1 -> 2)
    layout.push(Span::new(31, 40), 2, 50.0, 10.0, 250.0, 30.0);

    let costs = LayoutCosts::default();
    let bounds = layout_boundaries(&sentences, Some(&layout), costs).unwrap();
    assert_eq!(bounds.cost, vec![costs.visual_gap_bonus, costs.column_jump_bonus, costs.page_break_bonus, 0]);
    assert_eq!(bounds.reason, vec![Cut::Paragraph, Cut::Paragraph, Cut::Section, Cut::Sentence]);
}

#[test]
fn page_break_bonus_shifts_dp_chunker_cut_and_updates_config_hash() {
    let text = "One. Two. Three. Four.";
    let mut doc = Document::new(text);
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, 4));
    sent_spans.push(Span::new(5, 9));
    sent_spans.push(Span::new(10, 16));
    sent_spans.push(Span::new(17, 22));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("test")).unwrap();

    let mut sub_spans = Spans::new(ViewId::SOURCE);
    // 4 sentences, 10 tokens each -> 40 tokens total
    for s in [Span::new(0, 4), Span::new(5, 9), Span::new(10, 16), Span::new(17, 22)] {
        for _ in 0..10 {
            sub_spans.push(s);
        }
    }
    let model = ModelRef { id: "m".into(), revision: "1".into(), sha256: "0".into() };
    doc.add(&Key::DEFAULT, Subwords { spans: sub_spans, ids: vec![1; 40], model, token: None }, Provenance::by("test"))
        .unwrap();

    let mut layout = LayoutBoxes::new(ViewId::SOURCE);
    // Sentences 0, 1, 2 on page 1; sentence 3 on page 2 -> page break after sentence 2 (gap 2)
    // Wait: let's put page break after sentence 0 (gap 0) vs target = 20 tokens!
    // With target = 20 and equal sentence costs, DP cuts after sentence 1 (20 + 20 tokens).
    // With a strong page_break_bonus at gap 0 (page 1 -> page 2 after sentence 0),
    // DP prefers cutting at the page break after sentence 0!
    layout.push(Span::new(0, 4), 1, 10.0, 10.0, 100.0, 30.0);
    layout.push(Span::new(5, 9), 2, 10.0, 10.0, 100.0, 30.0);
    layout.push(Span::new(10, 16), 2, 10.0, 35.0, 100.0, 55.0);
    layout.push(Span::new(17, 22), 2, 10.0, 60.0, 100.0, 80.0);
    doc.add(&Key::DEFAULT, layout, Provenance::by("test")).unwrap();

    let step_a = LayoutBoundaries { costs: LayoutCosts { page_break_bonus: 0, ..LayoutCosts::default() } };
    let step_b = LayoutBoundaries { costs: LayoutCosts { page_break_bonus: -2000, ..LayoutCosts::default() } };
    assert_ne!(step_a.config_hash(), step_b.config_hash());

    let weigh_a = WeighLayoutBoundaries { base: Weight::ONE, layout: Weight(0) };
    let weigh_b = WeighLayoutBoundaries { base: Weight::ONE, layout: Weight::ONE };
    assert_ne!(weigh_a.config_hash(), weigh_b.config_hash());

    let budget = Budget { max: 35, target: 20, size_weight: 400, overhead: 0 };
    let run_pipeline = |doc: Document, page_break_bonus: i32| -> Document {
        let pipeline = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Subwords>::DEFAULT)
            .given(Key::<LayoutBoxes>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
            .step(
                LayoutBoundaries {
                    costs: LayoutCosts {
                        page_break_bonus,
                        column_jump_bonus: 0,
                        visual_gap_bonus: 0,
                        visual_gap_threshold: 18.0,
                    },
                }
                .bound(),
            )
            .step(WeighLayoutBoundaries::default().bound())
            .add(Chunker { budget })
            .add(ChunkCitationAnnotator)
            .build()
            .unwrap();
        let mut docs = [doc];
        assert_eq!(pipeline.run(&mut docs)[0], Ok(()));
        docs.into_iter().next().unwrap()
    };

    let without_bonus = run_pipeline(doc.clone(), 0);
    let chunks_no_bonus = without_bonus.get::<Chunks>().unwrap();
    let starts_no_bonus: Vec<u32> = chunks_no_bonus.spans.iter().map(|(_, s)| s.start).collect();
    assert_eq!(starts_no_bonus, vec![0, 10]);

    let with_bonus = run_pipeline(doc, -2000);
    let chunks_bonus = with_bonus.get::<Chunks>().unwrap();
    let starts_bonus: Vec<u32> = chunks_bonus.spans.iter().map(|(_, s)| s.start).collect();
    assert_eq!(starts_bonus, vec![0, 5]);

    let citations = with_bonus.get::<ChunkCitations>().unwrap();
    assert_eq!(citations.len(), 2);
    assert_eq!(citations.page_span(Ix::new(0)), Some((1, 1)));
    assert_eq!(citations.page_boxes(Ix::new(0)), &[PageBoundingBox::new(1, 10.0, 10.0, 100.0, 30.0)]);
    assert_eq!(citations.page_span(Ix::new(1)), Some((2, 2)));
    assert_eq!(citations.page_boxes(Ix::new(1)), &[PageBoundingBox::new(2, 10.0, 10.0, 100.0, 80.0)]);

    // Verify binary codec round-trip for both LayoutBoxes and ChunkCitations.
    let mut codecs = turbo_analysis_layers::export::codecs();
    crate::export::register_codecs(&mut codecs);
    let bytes = export(&with_bonus, &codecs).unwrap();
    let restored = import(&bytes, &codecs).unwrap();
    assert_eq!(restored.get::<ChunkCitations>().unwrap(), citations);
    let restored_layout = restored.get::<LayoutBoxes>().unwrap();
    assert_eq!(
        restored_layout.iter().collect::<Vec<_>>(),
        with_bonus.get::<LayoutBoxes>().unwrap().iter().collect::<Vec<_>>()
    );
}
