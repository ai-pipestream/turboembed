use super::support::*;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;

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
