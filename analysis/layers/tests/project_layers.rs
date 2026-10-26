//! Cross-view bi-directional layer projection tests across all span-anchored
//! layers in `turbo-analysis-layers`.

use std::sync::Arc;

use turbo_analysis::{
    Analyzer, CodePointOffset, CodePointSpan, Document, Edge, Edges, Groups, Ix, IxRange, Key, Layer, ModelRef,
    ProjectView, Provenance, Span, Spans, Tag, TagSet, Utf16Offset, Utf16Span, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    AbbreviationForm, Abbreviations, AnchoredLayerKind, ArgumentRole, Aspect, AspectTarget, Aspects, Block, Categories,
    ChunkTableHeaders, Chunks, ColumnAlign, Containment, ContainmentChain, Currency, Cut, Date, DeonticKind, Edus,
    Entities, EpistemicStatus, EventArgument, EventFrame, Events, ExpansionRow, KeyValuePair, KeyValuePairs,
    LanguageSpans, Levels, LexicalExpansions, Matches, Modality, ModalityPolarity, ModalityTag, Modifier, Money,
    MoneyMention, Multiplier, NumberMention, NumberNotation, Numbers, Opinion, Opinions, OrdinalBase, OrdinalMention,
    Ordinals, Origin, PhraseChunks, Phrases, Pii, PiiConfidence, PiiKind, PiiMention, PlaceAncestor, PolarityScores,
    QualityFlags, QualityMetrics, Quantities, QuantityMention, Quotation, Quotations, QuoteKind, ScaleMarker, SenseRow,
    Sentences, SpanQuality, Structure, Subwords, TableCell, Tables, TemporalMention, TemporalValue, Temporals,
    TextQuality, TimeGrain, TimeKind, TimeMention, Times, TokenVectorWindows, Tokens, Unit, Vectors, WORDNET_POS,
    WordNetSenses, Words,
};
use turbo_unicode::Script;

fn build_three_view_doc() -> (Document, ViewId, ViewId) {
    let src = "<p>Café \u{FB01}re $10 5km 1st 2026-05-01</p>";
    assert_eq!(src.len(), 41);
    let mut doc = Document::new(src);
    let v0 = doc.get::<View>().unwrap().clone();

    let mut b1 = ViewBuilder::new(v0.text());
    b1.replace(3, "");
    b1.keep(37);
    b1.replace(41, "");
    let (t1, a1) = b1.finish();
    doc.add(&Key::<View>::named("child"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
    let v1 = doc.get_key(&Key::<View>::named("child")).unwrap().clone();

    let mut b2 = ViewBuilder::new(v1.text());
    b2.keep(6);
    b2.replace(9, "fi");
    let (t2, a2) = b2.finish();
    doc.add(&Key::<View>::named("grandchild"), View::derived(&v1, t2, a2), Provenance::by("test")).unwrap();
    let v2 = doc.get_key(&Key::<View>::named("grandchild")).unwrap().clone();

    (doc, v1.id(), v2.id())
}

fn spans_from<L>(view: ViewId, ranges: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(view);
    for &(a, b) in ranges {
        s.push(Span::new(a, b));
    }
    s
}

#[test]
fn segmentation_syntax_and_companion_layers_project_bidirectionally() {
    let (mut doc, child_id, grand_id) = build_three_view_doc();
    let tok_ranges = [(0, 5), (6, 10), (11, 14), (15, 18), (19, 22), (23, 33)];

    let structure = Structure {
        spans: spans_from(grand_id, &[(0, 33), (0, 33)]),
        kind: vec![Block::Section, Block::Paragraph],
        parent: vec![None, Some(Ix::new(0))],
        depth: vec![1, 0],
    };
    let sentences = Sentences { spans: spans_from(grand_id, &[(0, 33)]) };
    let tokens = Tokens { spans: spans_from(grand_id, &tok_ranges), sentence: Some(vec![Ix::new(0); 6]) };
    let model = ModelRef { id: "test-model".into(), revision: "1".into(), sha256: "0".repeat(64) };
    let subwords = Subwords {
        spans: spans_from(grand_id, &tok_ranges),
        ids: vec![101, 102, 103, 104, 105, 106],
        model: model.clone(),
        token: Some((0..6).map(Ix::new).collect()),
    };
    let chunks = Chunks {
        spans: spans_from(grand_id, &[(0, 33)]),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![6],
        over_budget: vec![false],
        prompt_overhead: 0,
    };
    let levels = Levels {
        spans: spans_from(grand_id, &[(0, 33), (0, 33), (0, 33)]),
        unit: vec![Unit::Document, Unit::Section(Ix::new(0)), Unit::Chunk(Ix::new(0))],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(1))],
    };

    let mut ent_spans = Spans::new(grand_id);
    ent_spans.push(Span::new(0, 5));
    ent_spans.try_push_fragments(&[Span::new(6, 10), Span::new(15, 18)]).unwrap();
    let entities = Entities {
        spans: ent_spans,
        tokens: vec![IxRange::new(0, 1), IxRange::new(1, 4)],
        kind: vec![Tag(0), Tag(1)],
        kinds: TagSet::new("ner", &["LOC", "MISC"]),
        score: vec![0.95, 0.85],
    };

    let phrase_kinds = TagSet::new("phrases", &["NP"]);
    let phrase_chunks = PhraseChunks {
        spans: spans_from(grand_id, &[(0, 10)]),
        tokens: vec![IxRange::new(0, 2)],
        kind: vec![Tag(0)],
        kinds: Arc::clone(&phrase_kinds),
    };
    let phrases = Phrases {
        spans: spans_from(grand_id, &[(0, 10)]),
        tokens: vec![IxRange::new(0, 2)],
        kind: vec![Tag(0)],
        kinds: phrase_kinds,
        head: vec![Ix::new(1)],
        parent: vec![None],
    };

    let struct_src = doc.project_layer(&doc.project_layer(&structure, child_id).unwrap(), ViewId::SOURCE).unwrap();
    assert_eq!(struct_src.spans.get(Ix::new(0)), Span::new(3, 37));
    assert_eq!(doc.project_layer(&struct_src, grand_id).unwrap().spans.get(Ix::new(0)), Span::new(0, 33));

    let sents_src = doc.project_layer(&doc.project_layer(&sentences, child_id).unwrap(), ViewId::SOURCE).unwrap();
    assert_eq!(sents_src.spans.get(Ix::new(0)), Span::new(3, 37));

    let toks_src = doc.project_layer(&doc.project_layer(&tokens, child_id).unwrap(), ViewId::SOURCE).unwrap();
    assert_eq!(toks_src.len(), 6);
    assert_eq!(toks_src.spans.get(Ix::new(0)), Span::new(3, 8));
    assert_eq!(toks_src.spans.get(Ix::new(1)), Span::new(9, 14));
    assert_eq!(toks_src.spans.get(Ix::new(2)), Span::new(15, 18));
    assert_eq!(toks_src.spans.get(Ix::new(5)), Span::new(27, 37));

    let fire_u16 = doc.view_span_to_utf16(ViewId::SOURCE, toks_src.spans.get(Ix::new(1))).unwrap();
    let fire_cp = doc.view_span_to_code_point(ViewId::SOURCE, toks_src.spans.get(Ix::new(1))).unwrap();
    assert_eq!(fire_u16, Utf16Span::new(Utf16Offset(8), Utf16Offset(11)));
    assert_eq!(fire_cp, CodePointSpan::new(CodePointOffset(8), CodePointOffset(11)));

    let subs_src = doc.project_layer(&subwords, ViewId::SOURCE).unwrap();
    assert_eq!(subs_src.spans.get(Ix::new(1)), Span::new(9, 14));
    let chunks_src = doc.project_layer(&chunks, ViewId::SOURCE).unwrap();
    assert_eq!(chunks_src.spans.get(Ix::new(0)), Span::new(3, 37));
    let levels_src = doc.project_layer(&levels, ViewId::SOURCE).unwrap();
    assert_eq!(levels_src.spans.get(Ix::new(2)), Span::new(3, 37));

    let ents_src = doc.project_layer(&entities, ViewId::SOURCE).unwrap();
    let ent1_frags: Vec<Span> = ents_src.spans.fragments(Ix::new(1)).collect();
    assert_eq!(ent1_frags, vec![Span::new(9, 14), Span::new(19, 22)]);

    let pc_src = doc.project_layer(&phrase_chunks, ViewId::SOURCE).unwrap();
    assert_eq!(pc_src.spans.get(Ix::new(0)), Span::new(3, 14));
    let p_src = doc.project_layer(&phrases, ViewId::SOURCE).unwrap();
    assert_eq!(p_src.spans.get(Ix::new(0)), Span::new(3, 14));

    let prov = || Provenance::by("test");
    doc.add(&Key::DEFAULT, struct_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, sents_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, toks_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, subs_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, chunks_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, levels_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, ents_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, pc_src, prov()).unwrap();
    doc.add(&Key::DEFAULT, p_src, prov()).unwrap();

    doc.add(&Key::DEFAULT, Vectors::<Chunks>::new(2, vec![1.0, 0.0], Vec::new(), model.clone()), prov()).unwrap();
    doc.add(&Key::DEFAULT, Vectors::<Sentences>::new(2, vec![0.0, 1.0], Vec::new(), model), prov()).unwrap();

    let cat_labels = TagSet::new("topics", &["finance"]);
    let mut sent_cats = Categories::<Sentences>::new(Arc::clone(&cat_labels));
    sent_cats.push(Tag(0), &[0.9]);
    let mut chunk_cats = Categories::<Chunks>::new(cat_labels);
    chunk_cats.push(Tag(0), &[0.9]);
    doc.add(&Key::DEFAULT, sent_cats, prov()).unwrap();
    doc.add(&Key::DEFAULT, chunk_cats, prov()).unwrap();

    let edges = Edges::<Entities, Entities>::new(
        TagSet::new("rel", &["related"]),
        [Edge { from: Ix::new(0), to: Ix::new(1), kind: Tag(0), score: 1.0 }],
    );
    doc.add(&Key::DEFAULT, edges, prov()).unwrap();
    doc.add(&Key::DEFAULT, Groups::<Entities>::new([[Ix::new(0), Ix::new(1)]]), prov()).unwrap();
}

#[test]
fn numeric_temporal_pii_sentiment_discourse_and_table_layers_project() {
    let (mut doc, child_id, grand_id) = build_three_view_doc();
    let tok_ranges = [(0, 5), (6, 10), (11, 14), (15, 18), (19, 22), (23, 33)];
    let prov = || Provenance::by("test");
    let norm_sents_key = Key::<Sentences>::named("norm");
    let norm_toks_key = Key::<Tokens>::named("norm");

    doc.add(&norm_sents_key, Sentences { spans: spans_from(grand_id, &[(0, 33)]) }, prov()).unwrap();
    let mut tok_prov = prov();
    tok_prov.reads.push(norm_sents_key.id());
    doc.add(
        &norm_toks_key,
        Tokens { spans: spans_from(grand_id, &tok_ranges), sentence: Some(vec![Ix::new(0); 6]) },
        tok_prov,
    )
    .unwrap();

    let project_toks =
        ProjectView::new(norm_toks_key, Key::<Tokens>::DEFAULT).unwrap().require(Key::<Sentences>::DEFAULT);
    let analyzer = Analyzer::builder()
        .given(norm_sents_key)
        .given(norm_toks_key)
        .step(ProjectView::new(norm_sents_key, Key::<Sentences>::DEFAULT).unwrap())
        .step(project_toks)
        .build()
        .unwrap();
    assert!(analyzer.run(std::slice::from_mut(&mut doc))[0].is_ok());
    assert_eq!(doc.made_on(Key::<Tokens>::DEFAULT.id()), &[Key::<Sentences>::DEFAULT.id()]);

    let mut numbers = Numbers::new(grand_id);
    numbers.push(NumberMention {
        span: Span::new(15, 16),
        numeral_end: 16,
        negative: false,
        coefficient: "5",
        scale: 0,
        multiplier: Multiplier::One,
        marker: ScaleMarker::None,
        notation: NumberNotation::LatinUs,
    });
    let numbers_src = doc.project_layer(&numbers, ViewId::SOURCE).unwrap();
    assert_eq!(
        (numbers_src.spans.get(Ix::new(0)), numbers_src.numeral(Ix::new(0))),
        (Span::new(19, 20), Span::new(19, 20))
    );
    doc.add(&Key::DEFAULT, numbers_src, prov()).unwrap();

    let mut money = Money::new(grand_id);
    money.push(MoneyMention {
        span: Span::new(11, 14),
        negative: false,
        coefficient: "10",
        scale: 0,
        currency: Currency::parse("USD").unwrap(),
    });
    let money_src = doc.project_layer(&money, ViewId::SOURCE).unwrap();
    assert_eq!(money_src.spans.get(Ix::new(0)), Span::new(15, 18));
    doc.add(&Key::DEFAULT, money_src, prov()).unwrap();

    let mut quantities = Quantities::new(grand_id);
    quantities.push(QuantityMention {
        span: Span::new(15, 18),
        negative: false,
        coefficient: "5",
        scale: 0,
        unit: "km",
    });
    let quantities_src = doc.project_layer(&quantities, ViewId::SOURCE).unwrap();
    assert_eq!(quantities_src.spans.get(Ix::new(0)), Span::new(19, 22));
    doc.add(&Key::DEFAULT, quantities_src, prov()).unwrap();

    let mut ordinals = Ordinals::new(grand_id);
    ordinals.push(OrdinalMention { span: Span::new(19, 22), base: OrdinalBase::Start, offset: 1 });
    let ordinals_src = doc.project_layer(&ordinals, ViewId::SOURCE).unwrap();
    assert_eq!(ordinals_src.spans.get(Ix::new(0)), Span::new(23, 26));
    doc.add(&Key::DEFAULT, ordinals_src, prov()).unwrap();

    let mut times = Times::new(grand_id);
    times.push(TimeMention {
        span: Span::new(23, 33),
        kind: TimeKind::Date,
        grain: TimeGrain::Day,
        modifier: Modifier::NONE,
        start: Some(1_777_593_600_000),
        end: None,
        duration_ms: None,
        ambiguous: false,
        timex: "2026-05-01",
    });
    let times_src = doc.project_layer(&times, ViewId::SOURCE).unwrap();
    assert_eq!(times_src.spans.get(Ix::new(0)), Span::new(27, 37));
    doc.add(&Key::DEFAULT, times_src, prov()).unwrap();

    let mut temporals = Temporals::new(grand_id);
    temporals.push(TemporalMention {
        span: Span::new(23, 33),
        value: TemporalValue::Day(Date::new(2026, 5, 1).unwrap()),
        origin: Origin::Absolute,
    });
    let temporals_src = doc.project_layer(&temporals, ViewId::SOURCE).unwrap();
    assert_eq!(temporals_src.spans.get(Ix::new(0)), Span::new(27, 37));
    doc.add(&Key::DEFAULT, temporals_src, prov()).unwrap();

    let mut pii = Pii::new(grand_id);
    pii.push(PiiMention {
        span: Span::new(23, 33),
        kind: PiiKind::Custom("date_id"),
        confidence: PiiConfidence::Format,
        normalized: "2026-05-01",
    });
    let pii_src = doc.project_layer(&pii, ViewId::SOURCE).unwrap();
    assert_eq!(pii_src.spans.get(Ix::new(0)), Span::new(27, 37));
    doc.add(&Key::DEFAULT, pii_src, prov()).unwrap();

    let mut quotes = Quotations::new(grand_id);
    quotes.push(Quotation {
        content: Span::new(11, 22),
        speaker: Some(Span::new(0, 5)),
        cue_verb: Some(Span::new(6, 10)),
        speaker_chain: None,
        kind: QuoteKind::Direct,
        confidence: 0.9,
    });
    let quotes_src = doc.project_layer(&quotes, ViewId::SOURCE).unwrap();
    let q0 = quotes_src.get(0).unwrap();
    assert_eq!(
        (q0.content, q0.speaker, q0.cue_verb),
        (Span::new(15, 26), Some(Span::new(3, 8)), Some(Span::new(9, 14)))
    );
    doc.add(&Key::DEFAULT, quotes_src, prov()).unwrap();

    let opinions = Opinions::from_items_on(
        norm_toks_key,
        doc.require(&norm_toks_key).unwrap(),
        [Opinion { tokens: IxRange::new(0, 2), polarity: 0.8, intensity: 1.0, negated: false }],
    )
    .unwrap();
    let opinions_src = doc.project_layer(&opinions, ViewId::SOURCE).unwrap();
    assert_eq!(opinions_src.tokens_key.id(), Key::<Tokens>::DEFAULT.id());
    assert_eq!(opinions_src.span(Ix::new(0), doc.get::<Tokens>().unwrap()), Span::new(3, 14));
    doc.add(&Key::DEFAULT, opinions_src, prov()).unwrap();

    let aspects = Aspects::from_items_on(
        norm_toks_key,
        norm_sents_key,
        Key::<Opinions>::DEFAULT,
        doc.require(&norm_toks_key).unwrap(),
        doc.require(&norm_sents_key).ok(),
        doc.get::<Opinions>(),
        TagSet::new("aspect-cats", &["quality"]),
        [Aspect {
            target: AspectTarget::Term { tokens: IxRange::new(0, 1), sentence: Some(Ix::new(0)) },
            category: Some(Tag(0)),
            polarity: PolarityScores { positive: 0.8, negative: 0.1, neutral: 0.1 },
            opinions: &[Ix::new(0)],
        }],
    )
    .unwrap();
    let aspects_src = doc.project_layer(&aspects, ViewId::SOURCE).unwrap();
    assert_eq!(aspects_src.tokens_key.id(), Key::<Tokens>::DEFAULT.id());
    assert_eq!(aspects_src.sentences_key.id(), Key::<Sentences>::DEFAULT.id());
    assert_eq!(
        aspects_src.span(Ix::new(0), doc.get::<Tokens>().unwrap(), doc.get::<Sentences>()),
        Some(Span::new(3, 8))
    );
    doc.add(&Key::DEFAULT, aspects_src, prov()).unwrap();

    let mut matches = Matches::new(grand_id, TagSet::new("rules", &["R1"]));
    matches.push(Span::new(0, 5), IxRange::new(0, 1), Tag(0), 0);
    assert_eq!(doc.project_layer(&matches, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 8));

    let mut lang_spans = LanguageSpans::new(grand_id, TagSet::new("langs", &["fra"]));
    lang_spans.push(Span::new(0, 5), Tag(0), Script::Latin, 0.9);
    assert_eq!(doc.project_layer(&lang_spans, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 8));

    let mut abbr = Abbreviations::new(grand_id);
    abbr.push(Span::new(0, 5), AbbreviationForm::Long);
    abbr.push(Span::new(6, 10), AbbreviationForm::Short);
    assert_eq!(doc.project_layer(&abbr, ViewId::SOURCE).unwrap().spans.get(Ix::new(1)), Span::new(9, 14));

    let mut senses = WordNetSenses::new(grand_id, Arc::from("oewn:2024"));
    senses.push(
        Span::new(6, 10),
        SenseRow {
            token: Ix::new(1),
            synset_id: "0001-n",
            sense_id: "fire%1",
            pos: WORDNET_POS[0],
            lemma: "fire",
            synset_lemmas: &[String::from("fire")],
            gloss: "flame",
            sense_rank: 0,
        },
    );
    assert_eq!(doc.project_layer(&senses, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(9, 14));

    let mut exps = LexicalExpansions::new(grand_id, Arc::from("oewn:2024"));
    exps.push(
        Span::new(6, 10),
        ExpansionRow {
            token: Ix::new(1),
            term: "blaze",
            relation: 0,
            depth: 0,
            sense_rank: 0,
            weight: 1.0,
            synset_id: "0001-n",
        },
    );
    assert_eq!(doc.project_layer(&exps, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(9, 14));

    let mut tables = Tables::new(grand_id);
    tables.spans.push(Span::new(0, 5));
    tables.cells.push(TableCell {
        table_id: 0,
        row_ix: 0,
        col_ix: 0,
        row_span: 1,
        col_span: 1,
        is_header: true,
        align: ColumnAlign::Left,
    });
    tables.table_bounds.push(Span::new(0, 10));
    tables.rows.push(Span::new(0, 10));
    tables.row_table.push(0);
    tables.row_ix.push(0);
    let tables_src = doc.project_layer(&tables, ViewId::SOURCE).unwrap();
    assert_eq!(
        (tables_src.spans.get(Ix::new(0)), tables_src.table_bounds.get(Ix::new(0))),
        (Span::new(3, 8), Span::new(3, 14))
    );

    let mut kv = KeyValuePairs::new(grand_id);
    kv.spans.push(Span::new(0, 10));
    kv.pairs.push(KeyValuePair { key: Span::new(0, 5), separator: Span::new(5, 6), value: Span::new(6, 10) });
    let kv_src = doc.project_layer(&kv, ViewId::SOURCE).unwrap();
    assert_eq!((kv_src.pairs[0].key, kv_src.pairs[0].value), (Span::new(3, 8), Span::new(9, 14)));

    let g_chunks = Chunks {
        spans: spans_from(grand_id, &[(0, 10)]),
        level: vec![0],
        parent: vec![None],
        sentences: vec![IxRange::new(0, 1)],
        cut: vec![Cut::DocumentStart],
        cost: vec![0],
        model_tokens: vec![2],
        over_budget: vec![false],
        prompt_overhead: 0,
    };
    let cth = ChunkTableHeaders::build(&g_chunks, &tables, doc.text(grand_id)).unwrap();
    assert_eq!(doc.project_layer(&cth, ViewId::SOURCE).unwrap().contexts[0].header_cells[0], Span::new(3, 8));

    let mut edus = Edus::new(grand_id);
    edus.push(Span::new(0, 10), Ix::new(0));
    assert_eq!(doc.project_layer(&edus, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 14));

    let metrics = QualityMetrics { composite_score: 0.9, flags: QualityFlags::CLEAN, ..QualityMetrics::default() };
    let tq = TextQuality::new(grand_id, doc.text(grand_id).len() as u32, metrics);
    let tq_src = doc.project_layer(&tq, ViewId::SOURCE).unwrap();
    assert_eq!(tq_src.spans.get(Ix::new(0)), Span::new(0, doc.text(ViewId::SOURCE).len() as u32));
    doc.add(&Key::DEFAULT, tq_src, Provenance::by("test")).unwrap();

    let mut sq = SpanQuality::new(grand_id);
    sq.push(Span::new(0, 10), metrics);
    assert_eq!(doc.project_layer(&sq, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 14));

    let mut events = Events::new(grand_id);
    events.push(EventFrame {
        trigger: Span::new(6, 10),
        trigger_lemma: Some("fire"),
        sentence_ix: 0,
        arguments: &[EventArgument {
            span: Span::new(0, 5),
            role: ArgumentRole::Agent,
            anchored_layer: AnchoredLayerKind::None,
            anchored_index: None,
        }],
        confidence: 0.9,
    });
    let events_src = doc.project_layer(&events, ViewId::SOURCE).unwrap();
    assert_eq!((events_src.get(0).trigger, events_src.get(0).arguments[0].span), (Span::new(9, 14), Span::new(3, 8)));

    let mut modality = Modality::new(grand_id);
    modality.push(
        Span::new(0, 10),
        ModalityTag {
            polarity: ModalityPolarity::Affirmed,
            epistemic: EpistemicStatus::Asserted,
            deontic: DeonticKind::None,
            cue: Some(Span::new(6, 10)),
            confidence: 0.9,
        },
    );
    let modality_src = doc.project_layer(&modality, ViewId::SOURCE).unwrap();
    assert_eq!(modality_src.tag_for_span(Span::new(3, 14)).unwrap().cue, Some(Span::new(9, 14)));

    let mut cont = Containment::new(grand_id);
    let chain = ContainmentChain::new(vec![PlaceAncestor::new("1", "Paris", "ADM2").unwrap()]).unwrap();
    cont.push(Span::new(0, 5), &chain);
    assert_eq!(doc.project_layer(&cont, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 8));

    let tvw = TokenVectorWindows { spans: spans_from(grand_id, &[(0, 33)]) };
    assert_eq!(doc.project_layer(&tvw, ViewId::SOURCE).unwrap().spans.get(Ix::new(0)), Span::new(3, 37));

    let words = Words {
        tokens: vec![IxRange::new(1, 2)],
        form: None,
        subspan: Some(vec![Some(Span::new(6, 10))]),
        tokens_key: Key::<Tokens>::named("norm"),
    };
    assert_eq!(doc.project_layer(&words, child_id).unwrap().subspan_of(Ix::new(0)), Some(Span::new(6, 11)));
}
