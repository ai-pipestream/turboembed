use turbo::head::TokenTagger;
use turbo::manifest::{Decode, HeadKind, HeadSpec, LanguageInput, Scheme, WordPooling};
use turbo_trainer::{Config, Example, Tagger, fit};

fn spec() -> HeadSpec {
    HeadSpec {
        name: "upos".into(),
        layer: "turboembed:pos".into(),
        kind: HeadKind::TokenTagger,
        labels: vec!["A".into(), "B".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Argmax,
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
        tensor_prefix: "heads.upos".into(),
    }
}

fn params() -> Vec<Vec<f32>> {
    vec![vec![0.1, -0.2, 0.3, 0.2], vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.4, 0.1, -0.3], vec![0.05, -0.05]]
}

#[test]
fn language_and_head_gradients_match_finite_differences() {
    let spec = spec();
    let model = Tagger::with_params(&spec, 2, params()).unwrap();
    let features = [0.4, -0.6];
    let mut analytic = model.zero_grads();
    let mut work = Default::default();
    model.loss_language(&features, None, Some(0), &[1], &mut analytic, &mut work).unwrap();
    assert!(analytic[0][..2].iter().any(|value| value.abs() > 0.0));
    assert_eq!(analytic[0][2..], [0.0, 0.0]);

    for i in 0..params().len() {
        for j in 0..params()[i].len() {
            let mut plus = params();
            let mut minus = params();
            plus[i][j] += 0.001;
            minus[i][j] -= 0.001;
            let plus = Tagger::with_params(&spec, 2, plus).unwrap();
            let minus = Tagger::with_params(&spec, 2, minus).unwrap();
            let mut plus_work = Default::default();
            let mut minus_work = Default::default();
            let mut plus_grads = plus.zero_grads();
            let mut minus_grads = minus.zero_grads();
            let lp = plus.loss_language(&features, None, Some(0), &[1], &mut plus_grads, &mut plus_work).unwrap();
            let lm = minus.loss_language(&features, None, Some(0), &[1], &mut minus_grads, &mut minus_work).unwrap();
            let numeric = (lp - lm) / 0.002;
            assert!(
                (numeric - f64::from(analytic[i][j])).abs() < 0.002,
                "tensor {i} value {j}: {numeric} vs {}",
                analytic[i][j]
            );
        }
    }
}

#[test]
fn training_and_core_scoring_use_the_example_language() {
    let spec = spec();
    let example = |language| Example {
        features: vec![1.0, 0.0],
        labels: vec![language],
        forms: Vec::new(),
        language: Some(language),
    };
    let train: Vec<Example> = (0..40).flat_map(|i| [example(i % 2), example(i % 2)]).collect();
    let dev: Vec<Example> = (0..8).map(|i| example(i % 2)).collect();
    let config = Config { epochs: 80, batch: 4, lr: 0.03, patience: 0, threads: 1, ..Config::default() };
    let (model, report) = fit(&spec, 2, &train, &dev, &config).unwrap();
    assert_eq!(report.dev[report.best_epoch], 1.0, "{report:?}");
    let core: TokenTagger = model.to_token_tagger().unwrap();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Default::default());
    for (language, name) in ["es", "ca"].into_iter().enumerate() {
        core.scores_language(&[1.0, 0.0], &[0], None, Some(name), &mut scores, &mut work).unwrap();
        core.decode(&scores, 1, &mut tags).unwrap();
        assert_eq!(tags, [language as u32], "{name}");
    }

    let missing: Vec<Example> = dev.into_iter().map(|example| Example { language: None, ..example }).collect();
    let error = fit(&spec, 2, &train, &missing, &Config { epochs: 1, ..config }).unwrap_err();
    assert!(error.message.contains("needs a language"), "{error}");
}
