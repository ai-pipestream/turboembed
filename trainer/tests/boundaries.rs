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

#[test]
fn an_overflowing_loss_leaves_the_gradients_as_they_were() {
    // Finite scores whose loss overflows: [MAX, -MAX] against label 1.
    let mut d = vec![7.0f64, 7.0];
    assert!(xent::loss(&[f64::MAX, -f64::MAX], &[1], 2, &mut d).is_err());
    assert_eq!(d, [7.0, 7.0]);
    let mut d = vec![7.0f64, 7.0];
    assert!(xent::loss(&[0.0, 0.0], &[2], 2, &mut d).is_err(), "a gold label out of range");
    assert_eq!(d, [7.0, 7.0]);

    let s = spec(&["A", "B"], Scheme::Tags, Decode::Crf);
    let (_, lattice) = allowed(s.scheme, &s.labels).unwrap();
    // Scores a common offset does not explain: [MAX, -MAX] against label 1.
    let (start, end, tr) = (vec![0.0; 2], vec![0.0; 2], vec![0.0; 4]);
    let p = crf::Params { transitions: &tr, start: &start, end: &end };
    let (mut gt, mut gs, mut ge, mut de) = (vec![7.0; 4], vec![7.0; 2], vec![7.0; 2], vec![7.0; 2]);
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &lattice, &[f64::MAX, -f64::MAX], &[1], 2, &mut de, g).is_err());
    assert!([&gt, &gs, &ge, &de].iter().all(|v| v.iter().all(|&x| x == 7.0)), "{gt:?} {gs:?} {ge:?} {de:?}");
}

#[test]
fn a_common_score_offset_changes_neither_loss_nor_gradient() {
    let (_, two) = allowed(Scheme::Tags, &["A".to_string(), "B".to_string()]).unwrap();
    for offset in [0.0f32, 1e20, -1e20, 3e38] {
        let emit = [offset, offset];
        let mut d = vec![0.0f32; 2];
        let l = xent::loss(&emit, &[0], 2, &mut d).unwrap();
        assert!((l - std::f64::consts::LN_2).abs() < 1e-12 && d == [-0.5, 0.5], "xent at {offset}: {l} {d:?}");

        // The same offset on the emissions, the start and end scores, and
        // the transitions, over two words.
        let (tr, st, en) = (vec![offset; 4], vec![offset; 2], vec![offset; 2]);
        let p = crf::Params { transitions: &tr, start: &st, end: &en };
        let (mut gt, mut gs, mut ge, mut de) = (vec![0.0f32; 4], vec![0.0; 2], vec![0.0; 2], vec![0.0; 4]);
        let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
        let l = crf::nll(p, &two, &[offset; 4], &[0, 1], 2, &mut de, g).unwrap();
        assert!((l - 4f64.ln()).abs() < 1e-12, "crf at {offset}: {l}");
        assert_eq!((de, gs, ge), (vec![-0.5, 0.5, 0.5, -0.5], vec![-0.5, 0.5], vec![0.5, -0.5]), "crf at {offset}");
        assert_eq!(gt, [0.25, -0.75, 0.25, 0.25], "crf at {offset}");
    }
}

/// Codex's cases: large start and end scores that cancel along every
/// legal path, and a large start score only a forbidden path could take.
/// Neither may swallow the normaliser.
#[test]
fn large_scores_that_cancel_or_are_forbidden_leave_the_loss_whole() {
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    let (_, tags) = allowed(Scheme::Tags, &names(&["A", "B"])).unwrap();
    let (_, bio) = allowed(Scheme::Bio, &names(&["O", "B-X", "I-X"])).unwrap();
    // Lattice, start, end, emissions, and the start, end and emission
    // gradients wanted.
    type Case<'a> = (&'a turbo::head::crf::Allowed, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);
    let cases: [Case; 3] = [
        (&tags, vec![1e20, 0.0], vec![0.0, 1e20], vec![0.0; 2], vec![-0.5, 0.5]),
        (&tags, vec![0.0, 1e20], vec![1e20, 0.0], vec![0.0; 2], vec![-0.5, 0.5]),
        (&bio, vec![0.0, 0.0, 1e20], vec![0.0; 3], vec![0.0; 3], vec![-0.5, 0.5, 0.0]),
    ];
    for (k, (lattice, st, en, emit, want)) in cases.into_iter().enumerate() {
        let labels = st.len();
        let tr = vec![0.0f32; labels * labels];
        let p = crf::Params { transitions: &tr, start: &st, end: &en };
        let (mut gt, mut gs, mut ge, mut de) =
            (vec![0.0f32; labels * labels], vec![0.0; labels], vec![0.0; labels], vec![0.0; labels]);
        let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
        let l = crf::nll(p, lattice, &emit, &[0], labels, &mut de, g).unwrap();
        assert!((l - std::f64::consts::LN_2).abs() < 1e-12, "case {k}: loss {l}");
        assert_eq!((&de, &gs, &ge), (&want, &want, &want), "case {k}");
    }
}
