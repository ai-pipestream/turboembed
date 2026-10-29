//! A head over a learned mix of layers (#1616): the mix's and the head's
//! gradients against central finite differences in f64, a one-layer mix
//! is the plain head, and a run learns to weight the layer that holds the
//! labels.

use turbo::manifest::{Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::mix::{MixExample, MixTagger, fit_mix};
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;
use turbo_trainer::{Config, Example, Tagger, fit};

const EPS: f64 = 1e-6;

fn spec(scheme: Scheme, adapter: Option<(u32, u32)>, hash: bool) -> HeadSpec {
    let labels: &[&str] = if scheme == Scheme::Tags { &["A", "B", "C"] } else { &["O", "B-X", "I-X"] };
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: if scheme == Scheme::Tags { Decode::Argmax } else { Decode::Crf },
        biaffine: None,
        pos_input: None,
        hash_embed: hash.then_some(HashEmbed { dim: 2, rows: [5, 3, 4, 3] }),
        license_class: None,
        span: None,
        coref: None,
        artifact: "weights".into(),
        tensor_prefix: "t".into(),
    }
}

#[test]
fn mix_and_head_gradients_match_finite_differences() {
    let (n, hidden, layers) = (4, 3, 3);
    let forms = ["Ann", "met", "bob", "here"];
    for (scheme, adapter, hash) in
        [(Scheme::Tags, None, false), (Scheme::Bio, Some((2, 1)), false), (Scheme::Tags, Some((2, 1)), true)]
    {
        let s = spec(scheme, adapter, hash);
        let mut rng = Rng::new(5);
        let mut m = MixTagger::<f64>::new(&s, hidden, layers, &mut rng).unwrap();
        // Off the uniform start, so every logit's gradient differs.
        let mut p = m.params().to_vec();
        let k = p.len() - 2;
        p[k] = vec![0.3, -0.2, 0.1];
        p[k + 1] = vec![1.4];
        m.set_params(p).unwrap();
        let x: Vec<Vec<f64>> = (0..layers).map(|_| (0..n * hidden).map(|_| rng.uniform(1.0)).collect()).collect();
        let gold = [0u32, 1, 2, 0];
        let f = if hash { Some(&forms[..]) } else { None };
        let mut grads = m.zero_grads();
        m.loss(&x, f, &gold, &mut grads, &mut Work::default()).unwrap();
        for t in [k, k + 1, k - 1] {
            for i in 0..m.params()[t].len() {
                let at = |d: f64| {
                    let mut q = m.clone();
                    let mut p = q.params().to_vec();
                    p[t][i] += d;
                    q.set_params(p).unwrap();
                    q.loss(&x, f, &gold, &mut q.zero_grads(), &mut Work::default()).unwrap()
                };
                let numeric = (at(EPS) - at(-EPS)) / (2.0 * EPS);
                let tol = 1e-6 * (1.0 + numeric.abs().max(grads[t][i].abs()));
                assert!((numeric - grads[t][i]).abs() <= tol, "{scheme:?} {t}[{i}]: {numeric} vs {}", grads[t][i]);
            }
        }
    }
}

#[test]
fn a_one_layer_mix_is_the_plain_head() {
    let s = spec(Scheme::Bio, Some((2, 1)), false);
    let (hidden, n) = (3, 4);
    let mut rng = Rng::new(7);
    let x: Vec<f64> = (0..n * hidden).map(|_| rng.uniform(1.0)).collect();
    let m = MixTagger::<f64>::new(&s, hidden, 1, &mut Rng::new(1)).unwrap();
    let t = Tagger::<f64>::new(&s, hidden, &mut Rng::new(1)).unwrap();
    assert_eq!(m.tagger().params(), t.params());
    let gold = [1u32, 2, 0, 1];
    let (mut gm, mut gt) = (m.zero_grads(), t.zero_grads());
    let lm = m.loss(std::slice::from_ref(&x), None, &gold, &mut gm, &mut Work::default()).unwrap();
    let lt = t.loss(&x, &gold, &mut gt, &mut Work::default()).unwrap();
    assert_eq!(lm, lt);
    assert_eq!(&gm[..gt.len()], &gt[..]);
    // The mix's refusals.
    let two = [x.clone(), x.clone()];
    assert!(m.loss(&two, None, &gold, &mut m.zero_grads(), &mut Work::default()).is_err());
    assert!(MixTagger::<f64>::new(&s, hidden, 0, &mut Rng::new(1)).is_err());
}

/// Sequences whose labels layer `signal` holds (the label's one-hot in
/// its first features) and the other layers do not (noise).
fn data(seed: u64, count: usize, layers: usize, signal: usize, hidden: usize) -> Vec<MixExample> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| {
            let labels: Vec<u32> = (0..5).map(|_| (rng.uniform(1.5) + 1.5) as u32 % 3).collect();
            let layers = (0..layers)
                .map(|l| {
                    labels
                        .iter()
                        .flat_map(|&y| {
                            let noise: Vec<f32> = (0..hidden).map(|_| rng.uniform(1.0) as f32).collect();
                            (0..hidden).map(
                                move |j| if l == signal && j < 3 { (j as u32 == y) as u32 as f32 } else { noise[j] },
                            )
                        })
                        .collect()
                })
                .collect();
            MixExample { layers, labels, forms: Vec::new() }
        })
        .collect()
}

#[test]
fn a_run_weights_the_layer_that_holds_the_labels() {
    let s = spec(Scheme::Tags, None, false);
    let (hidden, layers, signal) = (4, 3, 1);
    let train = data(1, 120, layers, signal, hidden);
    let dev = data(2, 40, layers, signal, hidden);
    let c = Config { epochs: 30, batch: 8, lr: 0.05, patience: 0, threads: 1, ..Config::default() };
    let (m, report) = fit_mix(&s, hidden, layers, &train, &dev, &c, &mut |_, _, _| Ok(())).unwrap();
    let (a, _) = m.weights();
    assert!(a[signal] > 0.5 && a[signal] > a[0] && a[signal] > a[2], "{a:?}");
    assert!(report.dev[report.best_epoch] > 0.95, "{:?}", report.dev);
    // The best single noisy layer does worse.
    let only = |l: usize, v: &[MixExample]| -> Vec<Example> {
        v.iter()
            .map(|e| Example { features: e.layers[l].clone(), labels: e.labels.clone(), forms: Vec::new() })
            .collect()
    };
    let (_, r0) = fit(&s, hidden, &only(0, &train), &only(0, &dev), &c).unwrap();
    assert!(r0.dev[r0.best_epoch] < 0.6, "{:?}", r0.dev);
    // Two runs, one mix.
    let (again, _) = fit_mix(&s, hidden, layers, &train, &dev, &c, &mut |_, _, _| Ok(())).unwrap();
    assert_eq!(again.params(), m.params());
}
