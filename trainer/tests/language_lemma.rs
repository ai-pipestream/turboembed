use turbo::manifest::{Decode, HeadKind, HeadSpec, LanguageInput, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, LemmaExample, Lemmatizer, fit_lemmatizer, mine_trees};

fn spec() -> HeadSpec {
    let labels = mine_trees([("walked", "walk"), ("went", "go")], 1);
    HeadSpec {
        name: "lemma".into(),
        layer: "turboembed:lemmas".into(),
        kind: HeadKind::EditTreeLemmatizer,
        labels,
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::EditTree,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        language: Some(LanguageInput { languages: vec!["es".into(), "ca".into()], dim: 2 }),
        seen: None,
        span: None,
        coref: None,
        seen_lookup: None,
        relation: None,
        layer_mix: None,
        license_class: None,
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.lemma".into(),
    }
}

#[test]
fn lemma_language_gradients_match_finite_differences() {
    let spec = spec();
    let model = Lemmatizer::<f32>::new(&spec, 2, &mut Rng::new(11)).unwrap();
    let params = model.params().to_vec();
    let features = [0.4, -0.6, 0.2, 0.1];
    let gold = [Some(0), Some(1)];
    let mut analytic = model.zero_grads();
    let mut work = Default::default();
    model.loss_language(&features, &[], Some(0), &gold, &mut analytic, &mut work).unwrap();
    assert!(analytic[0][..2].iter().any(|value| value.abs() > 0.0));
    assert_eq!(analytic[0][2..], [0.0, 0.0]);

    for i in 0..params.len() {
        for j in 0..params[i].len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i][j] += 0.001;
            minus[i][j] -= 0.001;
            let plus = Lemmatizer::with_params(&spec, 2, plus).unwrap();
            let minus = Lemmatizer::with_params(&spec, 2, minus).unwrap();
            let (mut plus_work, mut minus_work) = (Default::default(), Default::default());
            let (mut plus_grads, mut minus_grads) = (plus.zero_grads(), minus.zero_grads());
            let lp = plus.loss_language(&features, &[], Some(0), &gold, &mut plus_grads, &mut plus_work).unwrap();
            let lm = minus.loss_language(&features, &[], Some(0), &gold, &mut minus_grads, &mut minus_work).unwrap();
            let numeric = (lp - lm) / 0.002;
            assert!(
                (numeric - f64::from(analytic[i][j])).abs() < 0.005,
                "tensor {i} value {j}: {numeric} vs {}",
                analytic[i][j]
            );
        }
    }
}

#[test]
fn lemma_and_core_select_the_same_language_and_require_one() {
    let spec = spec();
    let model = Lemmatizer::<f32>::new(&spec, 2, &mut Rng::new(11)).unwrap();
    let core = model.to_core().unwrap();
    let features = [0.4, -0.6, 0.2, 0.1];
    for (row, language) in ["es", "ca"].into_iter().enumerate() {
        let mut trainer_work = Default::default();
        model.forward_language(&features, &[], Some(row as u32), &mut trainer_work).unwrap();
        let mut core_work = Default::default();
        let mut scores = Vec::new();
        core.scores_language(&features, &[0, 1], None, Some(language), &mut scores, &mut core_work).unwrap();
        for (actual, expected) in trainer_work.scores().iter().zip(&scores) {
            assert!((actual - expected).abs() < 1e-6, "{language}: {actual} vs {expected}");
        }
    }
    let example = LemmaExample {
        features: features.into(),
        tags: Vec::new(),
        forms: vec!["walked".into(), "went".into()],
        lemmas: vec!["walk".into(), "go".into()],
        language: None,
    };
    let error =
        fit_lemmatizer(&spec, 2, vec![example.clone()], vec![example], &Config { epochs: 1, ..Config::default() })
            .unwrap_err();
    assert!(error.message.contains("needs a language"), "{error}");
}
