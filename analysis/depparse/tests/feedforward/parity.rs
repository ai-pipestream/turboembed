use super::support::*;
use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;
fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-6 * b.abs().max(1e-8), "{a} != {b}");
}
#[test]
fn java_seeded_training_weights_context_scores_and_greedy_beam_match() {
    for (dropout, bytes, text) in [
        (false, include_bytes!("data/plain.bin").as_slice(), include_str!("data/plain.tsv")),
        (true, include_bytes!("data/dropout.bin").as_slice(), include_str!("data/dropout.tsv")),
    ] {
        let model = FeedforwardDependencyTrainer::train(&samples(), &settings(dropout)).unwrap();
        let java = FeedforwardDependencyModel::from_java_bytes(bytes).unwrap();
        assert_eq!(model.vocabulary(), java.vocabulary());
        assert_eq!(model.transitions(), java.transitions());
        let a = model.weights();
        let b = java.weights();
        for (a, b) in [
            (&a.embeddings, &b.embeddings),
            (&a.hidden_weights, &b.hidden_weights),
            (&a.hidden_bias, &b.hidden_bias),
            (&a.output_weights, &b.output_weights),
            (&a.output_bias, &b.output_bias),
        ] {
            for (&a, &b) in a.iter().zip(b) {
                close(a as f64, b as f64);
            }
        }
        assert_eq!(java.to_java_bytes().unwrap(), bytes);
        assert_eq!(
            model.to_bytes().unwrap(),
            FeedforwardDependencyTrainer::train(&samples(), &settings(dropout)).unwrap().to_bytes().unwrap()
        );
        let mut state = ArcStandardState::<Tokens>::new(3).unwrap();
        let steps = ArcStandardOracle::transitions(samples()[0].graph()).unwrap();
        let mut rows = text.lines();
        for step in steps {
            let context = FeedforwardContext::extract(
                &state,
                &["I", "see", "birds"],
                &["PRON", "VERB", "NOUN"],
                FeatureLayout::OpenNlp14,
            )
            .unwrap();
            let actual: Vec<_> = context
                .words
                .iter()
                .chain(&context.pos)
                .chain(&context.labels)
                .map(|s| s.as_deref().unwrap_or("<NULL>"))
                .collect();
            assert_eq!(actual, rows.next().unwrap().split('\t').skip(1).collect::<Vec<_>>());
            let ids = model.feature_ids(&context).unwrap();
            assert_eq!(format!("I\t{ids:?}"), rows.next().unwrap());
            let expected: Vec<f64> = rows
                .next()
                .unwrap()
                .trim_start_matches("S\t[")
                .trim_end_matches(']')
                .split(", ")
                .map(|v| v.parse().unwrap())
                .collect();
            for (a, b) in model.score(&ids).unwrap().into_iter().zip(expected) {
                close(a, b);
            }
            state.apply(&step).unwrap();
        }
        for beam in [1, 4] {
            let parser = FeedforwardDependencyParser::new(model.clone(), beam).unwrap();
            let graph: DependencyGraph = parser.parse(&["I", "see", "birds"], &["PRON", "VERB", "NOUN"]).unwrap();
            for i in 0..3 {
                let arc = graph.arc(Ix::new(i)).unwrap();
                assert_eq!(
                    format!("G\t{beam}\t{i}\t{}\t{}", arc.head.map_or(-1, |h| h.get() as i32), arc.relation),
                    rows.next().unwrap()
                );
            }
        }
    }
}
#[test]
fn normalize_matches_java_fixture() {
    let base = minimal(FeatureLayout::OpenNlp14);
    let mut vocabulary = base.vocabulary().clone();
    let mut expected = Vec::new();
    for line in include_str!("data/normalize.tsv").lines() {
        let (word, normalized) = line.split_once('\t').unwrap();
        let id = if let Some(&id) = vocabulary.words.get(normalized) {
            id
        } else {
            let id = vocabulary.len();
            vocabulary.words.insert(normalized.into(), id);
            id
        };
        expected.push((word, id));
    }
    let mut weights = base.weights().clone();
    weights.embeddings.resize(vocabulary.len(), 0.0);
    let model = FeedforwardDependencyModel::new(
        vocabulary,
        base.transitions().to_vec(),
        base.layout(),
        base.activation(),
        weights,
    )
    .unwrap();
    for (word, id) in expected {
        let ctx =
            FeedforwardContext::extract(&ArcStandardState::<Tokens>::new(1).unwrap(), &[word], &["X"], base.layout())
                .unwrap();
        assert_eq!(model.feature_ids(&ctx).unwrap()[3], id, "{word}");
    }
}
#[test]
fn scoring_uses_double_products_and_cache_preserves_cancellation() {
    for (embedding, weight, bias, scale, pre) in [
        (2f32.powi(80), 2f32.powi(80), 0.0, 2f32.powi(-120), 2f64.powi(160)),
        (2f32.powi(-80), 2f32.powi(-80), 0.0, 2f32.powi(120), 2f64.powi(-160)),
        (1.0, 1.0, -1.0, 1.0, 0.0),
    ] {
        let base = minimal(FeatureLayout::OpenNlp14);
        let mut w = base.weights().clone();
        w.embeddings.fill(embedding);
        w.hidden_weights[0] = weight;
        w.hidden_bias[0] = bias;
        w.output_weights = vec![0.0, -scale, scale];
        let mut model = FeedforwardDependencyModel::new(
            base.vocabulary().clone(),
            base.transitions().to_vec(),
            base.layout(),
            base.activation(),
            w,
        )
        .unwrap();
        let ids = vec![0; base.layout().feature_count()];
        let expected = pre * pre * pre * f64::from(scale);
        assert_eq!(model.score(&ids).unwrap(), vec![0.0, -expected, expected]);
        let bytes = model.to_bytes().unwrap();
        assert_eq!(model.precompute_words(&[("missing".into(), 10), ("*UNK*".into(), 100)], 14), 14);
        assert_eq!(model.score(&ids).unwrap(), vec![0.0, -expected, expected]);
        assert_eq!(model.to_bytes().unwrap(), bytes);
    }
}

#[test]
fn cache_matches_direct_scoring_for_trained_model_at_all_oracle_states() {
    let mut direct = FeedforwardDependencyTrainer::train(&samples(), &settings(true)).unwrap();
    direct.precompute_words(&[], 0);
    let mut cached = direct.clone();
    let count = cached
        .precompute_words(&[("birds".into(), 3), ("I".into(), 2), ("see".into(), 1), ("birds".into(), 1)], 10_000);
    assert_eq!(count, 3 * direct.layout().positions());
    for sample in samples() {
        let mut state = ArcStandardState::<Tokens>::new(sample.tokens().len()).unwrap();
        let words = sample.tokens().iter().map(String::as_str).collect::<Vec<_>>();
        let tags = sample.tags().iter().map(String::as_str).collect::<Vec<_>>();
        for step in ArcStandardOracle::transitions(sample.graph()).unwrap() {
            let context = FeedforwardContext::extract(&state, &words, &tags, direct.layout()).unwrap();
            let ids = direct.feature_ids(&context).unwrap();
            assert_eq!(direct.score(&ids).unwrap(), cached.score(&ids).unwrap());
            state.apply(&step).unwrap();
        }
    }
    assert_eq!(cached.precompute_words(&[("birds".into(), 2)], 0), 0);
}
