use super::*;

#[tokio::test]
async fn named_entity_salience_survives_both_views_and_all_offset_modes() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = consolidated();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(2, "e");
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("ascii"), View::derived(&source, text, alignment), Provenance::by("context:view"))
        .unwrap();
    let mut default = DocumentEntities::new(ViewId::SOURCE);
    default.entities.push(doc.get_key(&Key::<DocumentEntities>::named("rollup")).unwrap().entities[0].clone());
    doc.add(
        &Key::<DocumentEntities>::DEFAULT,
        default,
        Provenance { on: vec![Key::<Entities>::named("gold").id()], ..Provenance::by("context:default") },
    )
    .unwrap();
    let salience = EntitySalience {
        scores: [0.2, 0.9]
            .into_iter()
            .enumerate()
            .map(|(i, score)| EntitySalienceScore {
                entity_ix: i as u32,
                salience: score,
                rank: 1 - i as u32,
                position_score: 0.1,
                syntax_score: 0.2,
                frequency_score: 0.3,
                centrality_score: 0.4,
            })
            .collect(),
    };
    doc.add(
        &Key::<EntitySalience>::named("scorer"),
        salience,
        Provenance { on: vec![Key::<DocumentEntities>::named("rollup").id()], ..Provenance::by("context:salience") },
    )
    .unwrap();
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = project(&mut client, &doc, mode, view).await;
            let layer = out.layers.iter().find(|l| l.id == EntitySalience::NAME.to_string()).unwrap();
            assert_eq!(layer.annotations.len(), 2, "nonempty salience was erased");
            assert_eq!(layer.indexes_layer[0].instance, "rollup");
            let end = if view == 0 && mode == OffsetMode::Utf8 { 2 } else { 1 };
            assert_eq!(layer.annotations[0].spans, [SpanFragment { start: 0, end }]);
            let last = match mode {
                OffsetMode::Utf8 => {
                    if view == 0 {
                        7
                    } else {
                        6
                    }
                }
                OffsetMode::Utf16 => 4,
                _ => 3,
            };
            assert_eq!(layer.annotations[1].spans, [SpanFragment { start: last, end: last + 1 }]);
            for (i, annotation) in layer.annotations.iter().enumerate() {
                let Some(Value::EntitySalience(score)) = &annotation.value else { panic!("salience payload") };
                assert_eq!((score.entity_index, score.rank), (i as u32, 1 - i as u32));
                assert_eq!(score.salience, [0.2, 0.9][i]);
                assert_eq!(
                    (score.position_score, score.syntax_score, score.frequency_score, score.centrality_score),
                    (0.1, 0.2, 0.3, 0.4)
                );
            }
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
            for field in [
                "index",
                "rank",
                "duplicate-rank",
                "salience",
                "position",
                "syntax",
                "frequency",
                "centrality",
                "target",
                "empty",
            ] {
                let mut bad = out.clone();
                let layer = bad.layers.iter_mut().find(|l| l.id == EntitySalience::NAME.to_string()).unwrap();
                if field == "target" {
                    layer.indexes_layer[0].instance.clear();
                }
                if field == "empty" {
                    layer.annotations.clear();
                    invalid(client.round_trip_typed_document(bad).await.unwrap_err());
                    continue;
                }
                let Some(Value::EntitySalience(score)) = &mut layer.annotations[0].value else { panic!("salience") };
                match field {
                    "index" => score.entity_index = 99,
                    "rank" => score.rank = 99,
                    "duplicate-rank" => score.rank = 0,
                    "salience" => score.salience = f32::NAN,
                    "position" => score.position_score = 1.1,
                    "syntax" => score.syntax_score = f32::NAN,
                    "frequency" => score.frequency_score = -0.1,
                    "centrality" => score.centrality_score = f32::INFINITY,
                    _ => {}
                }
                invalid(client.round_trip_typed_document(bad).await.unwrap_err());
            }
        }
    }
    server.stop().await;
}

#[tokio::test]
async fn named_chunk_table_headers_survive_both_views_and_all_offset_modes() {
    let (server, mut client) = common::serve_no_admin().await;
    let mut doc = Document::new("xxé A");
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.replace(2, "");
    let (text, alignment) = builder.finish();
    doc.add(&Key::<View>::named("trim"), View::derived(&source, text, alignment), Provenance::by("context:view"))
        .unwrap();
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans(ViewId(1), &[(0, 4)]) },
        Provenance::by("context:sentences"),
    )
    .unwrap();
    let chunks = Chunks {
        spans: spans(ViewId(1), &[(0, 3), (3, 4)]),
        level: vec![0; 2],
        parent: vec![None; 2],
        sentences: vec![IxRange::new(0, 1); 2],
        cut: vec![Cut::DocumentStart, Cut::Sentence],
        cost: vec![0; 2],
        model_tokens: vec![1; 2],
        over_budget: vec![false; 2],
        prompt_overhead: 0,
    };
    let mut tables = Tables::new(ViewId(1));
    tables.table_bounds.push(Span::new(0, 4));
    for (i, (start, end)) in [(0, 2), (3, 4)].into_iter().enumerate() {
        tables.spans.push(Span::new(start, end));
        tables.cells.push(TableCell {
            table_id: 0,
            row_ix: i as u16,
            col_ix: 0,
            row_span: 1,
            col_span: 1,
            is_header: i == 0,
            align: ColumnAlign::Left,
        });
        tables.rows.push(Span::new(if i == 0 { 0 } else { 3 }, if i == 0 { 3 } else { 4 }));
        tables.row_table.push(0);
        tables.row_ix.push(i as u16);
    }
    let headers = ChunkTableHeaders::build(&chunks, &tables, "é A").unwrap();
    doc.add(
        &Key::<Chunks>::named("body"),
        chunks,
        Provenance { reads: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("context:chunks") },
    )
    .unwrap();
    doc.add(&Key::<Tables>::named("grid"), tables, Provenance::by("context:tables")).unwrap();
    doc.add(
        &Key::<ChunkTableHeaders>::named("headers"),
        headers,
        Provenance {
            on: vec![Key::<Chunks>::named("body").id(), Key::<Tables>::named("grid").id()],
            ..Provenance::by("context:headers")
        },
    )
    .unwrap();
    for view in [0, 1] {
        for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
            let out = project(&mut client, &doc, mode, view).await;
            let layer = out.layers.iter().find(|l| l.id == ChunkTableHeaders::NAME.to_string()).unwrap();
            assert_eq!(layer.annotations.len(), 2, "nonempty table context was erased");
            assert_eq!(layer.indexes_layer[0].instance, "body");
            assert_eq!(layer.indexes_layer[1].instance, "grid");
            for (i, annotation) in layer.annotations.iter().enumerate() {
                let Some(Value::ChunkTableHeader(header)) = &annotation.value else { panic!("header payload") };
                assert_eq!((header.chunk_index, header.table_id, header.is_continuation), (i as u32, 0, i > 0));
                assert_eq!(header.header_prefix, "| é |\n| :--- |\n");
                let start = if view == 0 { 2 } else { 0 };
                assert_eq!(
                    header.header_cells,
                    [SpanFragment { start, end: start + if mode == OffsetMode::Utf8 { 2 } else { 1 } }]
                );
                let size = if mode == OffsetMode::Utf8 { 3 } else { 2 };
                let span = if i == 0 {
                    SpanFragment { start, end: start + size }
                } else {
                    SpanFragment { start: start + size, end: start + size + 1 }
                };
                assert_eq!(annotation.spans, [span]);
            }
            assert_eq!(client.round_trip_typed_document(out.clone()).await.unwrap().into_inner(), out);
            for field in ["chunk", "table", "span", "target"] {
                let mut bad = out.clone();
                let layer = bad.layers.iter_mut().find(|l| l.id == ChunkTableHeaders::NAME.to_string()).unwrap();
                if field == "target" {
                    layer.indexes_layer[1].instance = "absent".into();
                }
                let Some(Value::ChunkTableHeader(header)) = &mut layer.annotations[0].value else { panic!("header") };
                match field {
                    "chunk" => header.chunk_index = 99,
                    "table" => header.table_id = 99,
                    "span" => header.header_cells[0] = SpanFragment { start: 999, end: 1000 },
                    _ => {}
                }
                invalid(client.round_trip_typed_document(bad).await.unwrap_err());
            }
        }
    }
    server.stop().await;
}
