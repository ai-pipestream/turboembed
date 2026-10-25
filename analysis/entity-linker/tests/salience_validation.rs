#[cfg(test)]
mod tests {
    use std::path::Path;
    use turbo_analysis::export::{DocumentGraph, export, import};
    use turbo_analysis::{Analyzer, Annotator, Bound, Document, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
    use turbo_analysis_embed::SubwordAnnotator;
    use turbo_analysis_layers::{Block, Chunks, DocumentEntities, Entities, EntitySalience, Sentences, Structure};
    use turbo_chunker::{
        Budget, ChunkEntities, ChunkEntityHeaders, Chunker, Costs, HeaderPolicy, SalienceHeaders, StructureBoundaries,
    };
    use turbo_entity_linker::{EntityConsolidator, EntitySalienceAnnotator, SalienceConfig};
    use turbo_maxent::Dictionary;
    use turbo_namefind::DictionaryNameFinder;
    use turbo_normalize::{NormalizeStep, UnicodeForm};
    use turbo_sentdetect::NewlineDetector;
    use turbo_tokenize::{SimpleTokenizer, TokenizerOptions};

    const TEXT: &str = "Acme builds machines.\nAcme sells machines.\nAcme hires staff.\nBeta meets Acme.\nBeta visits town.\nBeta leaves town.\n";
    fn pipeline(config: SalienceConfig) -> Document {
        pipeline_text(TEXT, config)
    }
    fn pipeline_text(text: &str, config: SalienceConfig) -> Document {
        pipeline_named(text, config, false)
    }
    fn pipeline_named(text: &str, config: SalienceConfig, named: bool) -> Document {
        let (entity_name, document_name, salience_name) =
            if named { ("names", "document", "salience") } else { ("", "", "") };
        let dictionary = Dictionary::read("<dictionary case_sensitive='true'><entry><token>Acme</token></entry><entry><token>Beta</token></entry></dictionary>".as_bytes()).unwrap();
        let model = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
        let analyzer = Analyzer::builder()
            .add(NewlineDetector::default())
            .add(SimpleTokenizer::new(TokenizerOptions::default()))
            .step(Bound::new(DictionaryNameFinder::new(dictionary, "ORG")).write(0, entity_name))
            .step(Bound::new(EntityConsolidator::new()).read(1, entity_name).write(0, document_name))
            .step(
                Bound::new(EntitySalienceAnnotator::with_config(config)).read(1, document_name).write(0, salience_name),
            )
            .add(SubwordAnnotator::open(&model).unwrap())
            .add(StructureBoundaries { costs: Costs::default() })
            .add(Chunker { budget: Budget { max: 10, target: 8, size_weight: 1000, overhead: 0 } })
            .step(
                Bound::new(ChunkEntityHeaders {
                    policy: HeaderPolicy { max_carried: 0 },
                    salience: SalienceHeaders { top_document_entities: 1, order_by_salience: true },
                    ..Default::default()
                })
                .read(3, entity_name)
                .read(7, document_name)
                .read(8, salience_name),
            )
            .build()
            .unwrap();
        let mut docs = [Document::new(text)];
        assert_eq!(analyzer.run(&mut docs)[0], Ok(()));
        docs.into_iter().next().unwrap()
    }
    fn codecs() -> turbo_analysis::export::Codecs {
        let mut codecs = turbo_entity_linker::codecs();
        turbo_chunker::export::register_codecs(&mut codecs);
        codecs
    }
    #[test]
    fn production_pipeline_and_graph_roundtrip() {
        let d = pipeline(SalienceConfig::default());
        let s = d.get::<EntitySalience>().unwrap();
        assert_eq!(s.scores.len(), 2);
        assert_eq!(s.top_k(1)[0].entity_ix, 0);
        let headers = d.get::<ChunkEntities>().unwrap();
        let ents = d.get::<Entities>().unwrap();
        let chunks = d.get::<Chunks>().unwrap();
        assert!(chunks.spans.len() > 1);
        let last = Ix::new(chunks.spans.len() as u32 - 1);
        let rendered = headers.header(last, ents, TEXT).unwrap();
        assert!(rendered.contains("in scope: Acme"), "{rendered}");
        println!("actual pipeline: {} chunks, final header {rendered}", chunks.spans.len());
        let graph = export(&d, &codecs()).unwrap();
        let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
        assert_eq!(export(&restored, &codecs()).unwrap(), graph);
    }
    #[test]
    fn finite_range_and_score_rank_corruption_controls() {
        let d = pipeline(SalienceConfig::default());
        let graph = export(&d, &codecs()).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.1, 1.1] {
            for component in 0..5 {
                let mut bad = graph.clone();
                let body =
                    bad.layers.iter_mut().find(|l| l.layer == EntitySalience::ID).unwrap().body.as_mut().unwrap();
                let at = 12 + component * 4;
                body[at..at + 4].copy_from_slice(&value.to_bits().to_le_bytes());
                assert!(import(&DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap(), &codecs()).is_err());
            }
        }
        for (offset, value) in [(4, 2u32), (8, 2), (36, 0)] {
            let mut bad = graph.clone();
            let body = bad.layers.iter_mut().find(|l| l.layer == EntitySalience::ID).unwrap().body.as_mut().unwrap();
            body[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(import(&DocumentGraph::from_bytes(&bad.to_bytes().unwrap()).unwrap(), &codecs()).is_err());
        }
    }
    #[test]
    fn optional_wrong_view_is_rejected() {
        let mut d = pipeline(SalienceConfig::default());
        let normalizer = Analyzer::builder()
            .step(Bound::new(NormalizeStep::new(UnicodeForm::nfc())).write(0, "nfc"))
            .build()
            .unwrap();
        assert_eq!(normalizer.run(std::slice::from_mut(&mut d))[0], Ok(()));
        let view = d.get_key(&Key::<View>::named("nfc")).unwrap();
        let mut spans = Spans::new(view.id());
        spans.push(Span::new(0, TEXT.len() as u32));
        let wrong = Sentences { spans };
        let result = EntitySalienceAnnotator::new().annotate((
            d.get::<View>().unwrap(),
            d.get::<DocumentEntities>().unwrap(),
            Some(&wrong),
            None,
        ));
        assert!(result.is_err(), "wrong-view sentences silently discarded");
    }
    #[test]
    fn missing_salience_tail_is_rejected_from_graph() {
        let d = pipeline(SalienceConfig::default());
        let mut graph = export(&d, &codecs()).unwrap();
        let layer = graph.layers.iter_mut().find(|l| l.layer == EntitySalience::ID).unwrap();
        let body = layer.body.as_mut().unwrap();
        body[..4].copy_from_slice(&1u32.to_le_bytes());
        body.truncate(32);
        layer.len = 1;
        assert!(
            import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).is_err(),
            "partial salience layer accepted for two entities"
        );
    }
    #[test]
    fn source_bounds_are_checked_on_graph_import() {
        let d = pipeline(SalienceConfig::default());
        let mut graph = export(&d, &codecs()).unwrap();
        let layer = graph.layers.iter_mut().find(|l| l.layer == DocumentEntities::ID).unwrap();
        let body = layer.body.as_mut().unwrap();
        // Locate the first mention span by its exact initial source coordinates.
        let needle = [0u32.to_le_bytes(), 4u32.to_le_bytes()].concat();
        let at = body.windows(8).position(|b| b == needle.as_slice()).unwrap();
        body[at + 4..at + 8].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(
            import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).is_err(),
            "outside consolidated mention accepted"
        );
    }
    #[test]
    fn invalid_negative_weight_is_rejected() {
        let d = pipeline(SalienceConfig::default());
        let config = SalienceConfig { position_weight: -10.0, pagerank_damping: 1.5, ..Default::default() };
        let result = EntitySalienceAnnotator::with_config(config).salience(&d);
        assert!(result.is_err(), "invalid weights and damping silently clamped into valid-looking salience");
    }

    fn corrupt_mention(d: &Document, mutate: impl FnOnce(&mut [u8], usize)) -> DocumentGraph {
        let mut graph = export(d, &codecs()).unwrap();
        let span = d.get::<DocumentEntities>().unwrap().entities[0].mentions[0].span;
        let needle = [span.start.to_le_bytes(), span.end.to_le_bytes()].concat();
        let body = graph.layers.iter_mut().find(|l| l.layer == DocumentEntities::ID).unwrap().body.as_mut().unwrap();
        let at = body.windows(8).position(|b| b == needle.as_slice()).unwrap();
        mutate(body, at);
        DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn consolidated_mentions_validate_utf8_source_rows_and_view_ids() {
        let d = pipeline_text(&("é\n".to_owned() + TEXT), SalienceConfig::default());
        for kind in 0..3 {
            let bad = corrupt_mention(&d, |body, at| match kind {
                0 => body[at..at + 4].copy_from_slice(&1u32.to_le_bytes()),
                1 => body[at + 9..at + 13].copy_from_slice(&5u32.to_le_bytes()),
                _ => body[..2].copy_from_slice(&u16::MAX.to_le_bytes()),
            });
            assert!(import(&bad, &codecs()).is_err(), "source corruption {kind}");
        }
    }

    #[test]
    fn reversed_mention_coordinates_return_an_error_without_panicking() {
        let d = pipeline(SalienceConfig::default());
        let bad = corrupt_mention(&d, |body, at| body[at..at + 4].copy_from_slice(&5u32.to_le_bytes()));
        let result = std::panic::catch_unwind(|| import(&bad, &codecs()));
        assert!(result.is_ok(), "graph import panicked on reversed mention coordinates");
        assert!(result.unwrap().is_err());
    }

    #[test]
    fn heading_fragments_do_not_award_a_bonus_to_mentions_in_their_gap() {
        let mut d = pipeline(SalienceConfig::default());
        let original = d.get::<EntitySalience>().unwrap().clone();
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push_fragments(&[Span::new(0, 4), Span::new(TEXT.len() as u32 - 5, TEXT.len() as u32)]);
        let headings = Structure { spans, kind: vec![Block::Heading], parent: vec![None], depth: vec![1] };
        d.add(&Key::DEFAULT, headings, Provenance::by("test:source-structure")).unwrap();
        let scored = EntitySalienceAnnotator::new().salience(&d).unwrap();
        assert_eq!(scored.scores[1].position_score, original.scores[1].position_score);
    }

    #[test]
    fn enabled_headers_refuse_a_mismatched_document_entity_view() {
        let d = pipeline(SalienceConfig::default());
        let mut wrong = d.get::<DocumentEntities>().unwrap().clone();
        wrong.view = ViewId(99);
        let run = |salience| {
            ChunkEntityHeaders { salience, ..Default::default() }.annotate((
                d.get::<Sentences>().unwrap(),
                d.get::<Chunks>().unwrap(),
                d.get::<View>().unwrap(),
                d.get::<Entities>(),
                None,
                None,
                None,
                Some(&wrong),
                d.get::<EntitySalience>(),
            ))
        };
        assert!(run(SalienceHeaders { top_document_entities: 1, order_by_salience: true }).is_err());
        assert!(run(SalienceHeaders::default()).is_ok(), "disabled behavior is unchanged");
    }

    #[test]
    fn configuration_extremes_are_explicit_and_weight_scaling_is_stable() {
        let d = pipeline(SalienceConfig::default());
        for kind in 0..10 {
            let mut c = SalienceConfig::default();
            match kind {
                0 => c.position_weight = f32::NAN,
                1 => c.syntax_weight = f32::INFINITY,
                2 => c.frequency_weight = -0.1,
                3 => {
                    c.position_weight = 0.0;
                    c.syntax_weight = 0.0;
                    c.frequency_weight = 0.0;
                    c.centrality_weight = 0.0;
                }
                4 => c.position_decay = -0.1,
                5 => c.heading_bonus = f32::NAN,
                6 => c.pagerank_damping = f32::NAN,
                7 => c.pagerank_damping = -0.1,
                8 => c.pagerank_damping = 1.1,
                _ => c.pagerank_iters = 0,
            }
            assert!(EntitySalienceAnnotator::with_config(c).salience(&d).is_err(), "invalid configuration {kind}");
        }
        let base = EntitySalienceAnnotator::new().salience(&d).unwrap();
        for factor in [1e-30, 1e30] {
            let c = SalienceConfig {
                position_weight: 0.3 * factor,
                syntax_weight: 0.3 * factor,
                frequency_weight: 0.2 * factor,
                centrality_weight: 0.2 * factor,
                ..Default::default()
            };
            let result = EntitySalienceAnnotator::with_config(c).salience(&d).unwrap();
            result.check().unwrap();
            for (a, b) in base.scores.iter().zip(&result.scores) {
                assert_eq!(a.rank, b.rank);
                assert!((a.salience - b.salience).abs() < 1e-6);
            }
        }
    }
    #[test]
    fn named_source_entities_and_salience_dependencies_round_trip() {
        let d = pipeline_named(TEXT, SalienceConfig::default(), true);
        assert!(d.get::<Entities>().is_none());
        let entities = d.get_key(&Key::<Entities>::named("names")).unwrap();
        let salience = d.get_key(&Key::<EntitySalience>::named("salience")).unwrap();
        assert_eq!(salience.scores.len(), 2);
        let h = d.get::<ChunkEntities>().unwrap();
        let last = Ix::new(d.get::<Chunks>().unwrap().spans.len() as u32 - 1);
        assert!(h.header(last, entities, TEXT).unwrap().contains("in scope: Acme"));
        let graph = export(&d, &codecs()).unwrap();
        let restored = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
        assert_eq!(export(&restored, &codecs()).unwrap(), graph);
    }
}
