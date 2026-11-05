use turbo::head::tagger::NO_WORD;
use turbo::manifest::{Biaffine, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::encoder::{Encoder, Geometry};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Draw, TunedParseExample, TunedParser, fit_tuned_parser};

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

fn example() -> TunedParseExample {
    TunedParseExample {
        ids: vec![0, 1, 2, 3, 4, 5],
        words: vec![NO_WORD, 0, 0, 1, 2, NO_WORD],
        heads: vec![0, 1, 0],
        rels: vec![0, 1, 0],
    }
}

fn encoder() -> Encoder<f32> {
    let geo = Geometry {
        layers: 1,
        hidden: 4,
        heads: 1,
        intermediate: 8,
        eps: 1e-5,
        window: usize::MAX,
        global_every: 0,
        max_positions: 16,
        global_theta: 10_000.0,
        local_theta: 10_000.0,
    };
    Encoder::random(geo, 8, &mut Rng::new(7)).unwrap()
}

#[test]
fn tuned_parser_gradients_match_finite_differences() {
    let spec = spec();
    let model = TunedParser::new(encoder(), &spec, &mut Rng::new(11)).unwrap();
    let params = model.params();
    let example = example();
    let mut analytic = model.zero_grads();
    let mut work = Default::default();
    model.loss_at(&example, Draw::at(0), &mut analytic, &mut work).unwrap();

    for i in 0..params.len() {
        for j in 0..params[i].len() {
            let mut plus = params.clone();
            let mut minus = params.clone();
            plus[i][j] += 0.001;
            minus[i][j] -= 0.001;
            let mut a = model.clone();
            let mut b = model.clone();
            a.set_params(plus).unwrap();
            b.set_params(minus).unwrap();
            let (mut aw, mut bw) = (Default::default(), Default::default());
            let (mut ag, mut bg) = (a.zero_grads(), b.zero_grads());
            let la = a.loss_at(&example, Draw::at(0), &mut ag, &mut aw).unwrap();
            let lb = b.loss_at(&example, Draw::at(0), &mut bg, &mut bw).unwrap();
            let numeric = (la - lb) / 0.002;
            assert!(
                (numeric - f64::from(analytic[i][j])).abs() < 0.05,
                "tensor {i} value {j}: {numeric} vs {}",
                analytic[i][j]
            );
        }
    }
}

#[test]
fn tuned_parser_training_is_deterministic_and_scores_dev() {
    let spec = spec();
    let train: Vec<TunedParseExample> = (0..8).map(|_| example()).collect();
    let config = Config { epochs: 2, batch: 2, threads: 2, ..Config::default() };
    let (a, ra) = fit_tuned_parser(encoder(), &spec, &train, &train, &config).unwrap();
    let (b, rb) = fit_tuned_parser(encoder(), &spec, &train, &train, &config).unwrap();
    assert_eq!(a.params(), b.params());
    assert_eq!(ra.dev, rb.dev);
    assert!(a.score(&train).unwrap().words > 0);
}
