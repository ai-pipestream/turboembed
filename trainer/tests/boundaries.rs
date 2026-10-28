//! The public trainer APIs refuse what they cannot train on or score
//! honestly: shapes the head does not have, a lattice of another label
//! count, nonfinite scores, settings that move against the gradient, and
//! gold labels that do not match the words. A one-step run trains.

use turbo::head::crf::allowed;
use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;
use turbo_trainer::{Config, Example, Tagger, crf, fit, xent};

fn spec(labels: &[&str], scheme: Scheme, decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "probe".into(),
        layer: "turboembed:probe".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme,
        word_pooling: WordPooling::First,
        adapter: None,
        decode,
        artifact: "weights".into(),
        tensor_prefix: "probe".into(),
    }
}

#[test]
fn a_tagger_keeps_the_shapes_it_was_made_with() {
    let s = spec(&["A", "B"], Scheme::Tags, Decode::Argmax);
    assert!(Tagger::<f32>::new(&s, 0, &mut Rng::new(1)).is_err(), "zero width");
    assert!(Tagger::<f32>::with_params(&s, usize::MAX, vec![vec![], vec![0.0, 0.0]]).is_err(), "overflowing width");
    let mut t = Tagger::<f32>::with_params(&s, 1, vec![vec![0.0, 0.0], vec![1.0, 0.0]]).unwrap();
    assert!(t.set_params(vec![vec![0.0, 0.0], vec![]]).is_err());
    assert!(t.set_params(vec![vec![0.0, 0.0]]).is_err());
    assert_eq!(t.params()[1], [1.0, 0.0]);
    let n = t.forward(&[1.0], &mut Work::default()).unwrap();
    assert_eq!(n, 1);
    let mut wrong = vec![vec![0.0f32; 2]];
    let mut opt = turbo_trainer::adamw::AdamW::new(Default::default(), &[2, 2]).unwrap();
    assert!(t.step(&mut opt, &mut wrong, &[true, false]).is_err());
}

#[test]
fn the_losses_refuse_mismatched_shapes_and_nonfinite_scores() {
    // A one-label lattice with two-label parameters.
    let (_, one) = allowed(Scheme::Tags, &["A".to_string()]).unwrap();
    let (tr, st, en) = (vec![0.0f64; 4], vec![0.0; 2], vec![0.0; 2]);
    let (mut gt, mut gs, mut ge, mut demit) = (vec![0.0; 4], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
    let p = crf::Params { transitions: &tr, start: &st, end: &en };
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &one, &[0.0, 0.0], &[0], 2, &mut demit, g).is_err());
    // The right lattice but nonfinite emissions, then short gradients.
    let (_, two) = allowed(Scheme::Tags, &["A".to_string(), "B".to_string()]).unwrap();
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &two, &[f64::NAN, 0.0], &[0], 2, &mut demit, g).is_err());
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &two, &[0.0, 0.0], &[0], 2, &mut demit[..1], g).is_err());
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &two, &[0.0, 0.0], &[0], 2, &mut demit, g).unwrap() > 0.0);

    let mut d = vec![0.0f64; 2];
    assert!(xent::loss(&[f64::NAN, 0.0], &[0], 2, &mut d).is_err());
    assert_eq!(d, [0.0, 0.0], "a refused loss leaves the gradients alone");
    assert!(xent::loss(&[0.0, 0.0, 0.0], &[0], 2, &mut d).is_err());
    assert!(xent::loss(&[0.0, 0.0], &[0], 2, &mut d[..1]).is_err());
}

fn one_example() -> Vec<Example> {
    vec![Example { features: vec![1.0], labels: vec![1] }]
}

#[test]
fn fit_refuses_settings_that_move_against_the_gradient() {
    let s = spec(&["A", "B"], Scheme::Tags, Decode::Argmax);
    let c = Config { epochs: 2, batch: 1, lr: -0.1, ..Config::default() };
    assert!(fit(&s, 1, &one_example(), &one_example(), &c).is_err());
    let c = Config { epochs: 2, batch: 1, lr: f64::NAN, ..Config::default() };
    assert!(fit(&s, 1, &one_example(), &one_example(), &c).is_err());
}

#[test]
fn score_refuses_gold_that_does_not_match_the_words() {
    let s = spec(&["A", "B"], Scheme::Tags, Decode::Argmax);
    let t = Tagger::<f32>::with_params(&s, 1, vec![vec![0.0, 0.0], vec![1.0, 0.0]]).unwrap();
    let core = t.to_token_tagger().unwrap();
    let extra = [Example { features: vec![1.0], labels: vec![0, 1] }];
    assert!(turbo_trainer::fit::score(&core, &extra).is_err());
    let out_of_range = [Example { features: vec![1.0], labels: vec![2] }];
    assert!(turbo_trainer::fit::score(&core, &out_of_range).is_err());
    let fine = [Example { features: vec![1.0], labels: vec![0] }];
    assert_eq!(turbo_trainer::fit::score(&core, &fine).unwrap(), 1.0);
}

#[test]
fn one_step_and_short_schedules_move_every_trained_weight() {
    let s = spec(&["A", "B"], Scheme::Tags, Decode::Argmax);
    for (epochs, warmup) in [(1, 0.0), (2, 0.0), (3, 0.5)] {
        let c = Config { epochs, batch: 1, lr: 0.1, weight_decay: 0.0, warmup, patience: 0, ..Config::default() };
        let start = Tagger::<f32>::new(&s, 1, &mut Rng::new(c.seed)).unwrap();
        let (trained, report) = fit(&s, 1, &one_example(), &one_example(), &c).unwrap();
        assert!(report.train_loss[0] > 0.0);
        for (i, (a, b)) in start.params().iter().zip(trained.params()).enumerate() {
            assert!(a.iter().zip(b).all(|(x, y)| x != y), "{epochs} epochs: tensor {i} {a:?} -> {b:?}");
        }
    }
}
