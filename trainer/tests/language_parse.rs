use turbo::manifest::{Biaffine, Decode, HeadKind, HeadSpec, LanguageInput, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, ParseExample, Parser, fit_parser};

fn spec() -> HeadSpec {
    HeadSpec {
        name: "deps".into(),
        layer: "turboembed:dependencies".into(),
        kind: HeadKind::BiaffineParser,
        labels: vec!["root".into(), "dep".into()],
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode: Decode::Mst,
        biaffine: Some(Biaffine { arc: 2, rel: 2 }),
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
        tensor_prefix: "heads.deps".into(),
    }
}

#[test]
fn parser_language_gradients_match_finite_differences() {
    let spec = spec();
    let model = Parser::<f32>::new(&spec, 2, &mut Rng::new(7)).unwrap();
    let mut params = model.params().to_vec();
    let last = params.len() - 2;
    for tensor in &mut params[last..] {
        for (i, value) in tensor.iter_mut().enumerate() {
            *value = 0.01 * (i + 1) as f32;
        }
    }
    let model = Parser::with_params(&spec, 2, params.clone()).unwrap();
    let features = [0.4, -0.6, 0.2, 0.1];
    let (heads, rels) = ([0, 1], [0, 1]);
    let mut analytic = model.zero_grads();
    let mut work = Default::default();
    model.loss_language(&features, Some(0), &heads, &rels, &mut analytic, &mut work).unwrap();
    assert!(analytic[0][..2].iter().any(|value| value.abs() > 0.0));
    assert_eq!(analytic[0][2..], [0.0, 0.0]);

    for i in 0..params.len() {
        for j in 0..params[i].len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i][j] += 0.001;
            minus[i][j] -= 0.001;
            let plus = Parser::with_params(&spec, 2, plus).unwrap();
            let minus = Parser::with_params(&spec, 2, minus).unwrap();
            let (mut plus_work, mut minus_work) = (Default::default(), Default::default());
            let (mut plus_grads, mut minus_grads) = (plus.zero_grads(), minus.zero_grads());
            let lp = plus.loss_language(&features, Some(0), &heads, &rels, &mut plus_grads, &mut plus_work).unwrap();
            let lm = minus.loss_language(&features, Some(0), &heads, &rels, &mut minus_grads, &mut minus_work).unwrap();
            let numeric = (lp - lm) / 0.002;
            assert!(
                (numeric - f64::from(analytic[i][j])).abs() < 0.01,
                "tensor {i} value {j}: {numeric} vs {}",
                analytic[i][j]
            );
        }
    }
}

#[test]
fn parser_and_core_select_the_same_language_and_require_one() {
    let spec = spec();
    let model = Parser::<f32>::new(&spec, 2, &mut Rng::new(7)).unwrap();
    let core = model.to_core().unwrap();
    let features = [0.4, -0.6, 0.2, 0.1];
    for (row, language) in ["es", "ca"].into_iter().enumerate() {
        let mut trainer_work = Default::default();
        model.forward_language(&features, Some(row as u32), &mut trainer_work).unwrap();
        let mut core_work = Default::default();
        let mut arcs = Vec::new();
        core.arcs_language(&features, &[0, 1], Some(language), &mut arcs, &mut core_work).unwrap();
        assert_eq!(trainer_work.arcs(), arcs.as_slice(), "{language}");
    }
    let example = ParseExample { features: features.into(), heads: vec![0, 1], rels: vec![0, 1], language: None };
    let error = fit_parser(
        &spec,
        2,
        std::slice::from_ref(&example),
        std::slice::from_ref(&example),
        &Config { epochs: 1, ..Config::default() },
    )
    .unwrap_err();
    assert!(error.message.contains("needs a language"), "{error}");
}
