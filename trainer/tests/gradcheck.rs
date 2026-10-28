//! Every op's gradients, and the whole tagger loss's, against central
//! finite differences, all in f64.

use turbo::head::crf::allowed;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Tagger, crf, linear, lstm, xent};

const EPS: f64 = 1e-6;

fn random(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(scale)).collect()
}

/// Checks `analytic` against the central difference of `f` in each
/// coordinate of `x`.
fn check(what: &str, x: &mut [f64], analytic: &[f64], f: &mut dyn FnMut(&[f64]) -> f64) {
    assert_eq!(x.len(), analytic.len(), "{what}");
    for i in 0..x.len() {
        let keep = x[i];
        x[i] = keep + EPS;
        let up = f(x);
        x[i] = keep - EPS;
        let down = f(x);
        x[i] = keep;
        let numeric = (up - down) / (2.0 * EPS);
        let tol = 1e-6 * (1.0 + numeric.abs().max(analytic[i].abs()));
        assert!((numeric - analytic[i]).abs() <= tol, "{what}[{i}]: numeric {numeric}, analytic {}", analytic[i]);
    }
}

#[test]
fn linear_gradients_match_finite_differences() {
    let mut rng = Rng::new(1);
    let (rows, inp, out) = (3, 4, 5);
    let mut w = random(&mut rng, out * inp, 0.5);
    let mut b = random(&mut rng, out, 0.5);
    let mut x = random(&mut rng, rows * inp, 1.0);
    let r = random(&mut rng, rows * out, 1.0);
    let loss = |w: &[f64], b: &[f64], x: &[f64]| {
        let mut y = vec![0.0; rows * out];
        linear::forward(w, b, x, inp, out, &mut y);
        y.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>()
    };
    let (mut dw, mut db, mut dx) = (vec![0.0; w.len()], vec![0.0; b.len()], vec![0.0; x.len()]);
    linear::backward(&w, &x, &r, inp, out, &mut dw, &mut db, Some(&mut dx));
    let (b0, x0) = (b.clone(), x.clone());
    check("w", &mut w, &dw, &mut |w| loss(w, &b0, &x0));
    let w0 = w.clone();
    check("b", &mut b, &db, &mut |b| loss(&w0, b, &x0));
    check("x", &mut x, &dx, &mut |x| loss(&w0, &b0, x));
}

#[test]
fn lstm_gradients_match_finite_differences_in_both_directions() {
    for reverse in [false, true] {
        let mut rng = Rng::new(2 + reverse as u64);
        let (n, inp, h) = (4, 3, 2);
        let (stride, offset) = (2 * h + 1, 1);
        let mut w = random(&mut rng, 4 * h * inp, 0.6);
        let mut u = random(&mut rng, 4 * h * h, 0.6);
        let mut b = random(&mut rng, 4 * h, 0.3);
        let mut x = random(&mut rng, n * inp, 1.0);
        let r = random(&mut rng, n * stride, 1.0);
        let loss = |w: &[f64], u: &[f64], b: &[f64], x: &[f64]| {
            let mut out = vec![0.0; n * stride];
            let mut cache = lstm::Cache::default();
            lstm::forward(lstm::Dir { w, u, b, reverse }, x, inp, h, &mut out, stride, offset, &mut cache);
            out.iter().zip(&r).map(|(a, b)| a * b).sum::<f64>()
        };
        let mut out = vec![0.0; n * stride];
        let mut cache = lstm::Cache::default();
        let d = lstm::Dir { w: &w, u: &u, b: &b, reverse };
        lstm::forward(d, &x, inp, h, &mut out, stride, offset, &mut cache);
        let (mut gw, mut gu, mut gb, mut dx) =
            (vec![0.0; w.len()], vec![0.0; u.len()], vec![0.0; b.len()], vec![0.0; x.len()]);
        let g = lstm::Grad { w: &mut gw, u: &mut gu, b: &mut gb };
        lstm::backward(d, &x, inp, h, &r, stride, offset, &cache, g, &mut dx);
        let (u0, b0, x0) = (u.clone(), b.clone(), x.clone());
        check("w", &mut w, &gw, &mut |w| loss(w, &u0, &b0, &x0));
        let w0 = w.clone();
        check("u", &mut u, &gu, &mut |u| loss(&w0, u, &b0, &x0));
        check("b", &mut b, &gb, &mut |b| loss(&w0, &u0, b, &x0));
        check("x", &mut x, &dx, &mut |x| loss(&w0, &u0, &b0, x));
    }
}

#[test]
fn crf_gradients_match_finite_differences_under_each_scheme() {
    let cases: [(Scheme, &[&str], &[u32]); 3] = [
        (Scheme::Tags, &["A", "B", "C"], &[0, 2, 2, 1]),
        (Scheme::Bio, &["O", "B-X", "I-X", "B-Y"], &[1, 2, 0, 3, 1]),
        (Scheme::Bilou, &["O", "B-X", "I-X", "L-X", "U-X"], &[1, 2, 3, 4, 0]),
    ];
    for (k, (scheme, labels, gold)) in cases.into_iter().enumerate() {
        let names: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
        let (_, allow) = allowed(scheme, &names).unwrap();
        let (n, l) = (gold.len(), labels.len());
        let mut rng = Rng::new(10 + k as u64);
        let mut emit = random(&mut rng, n * l, 1.0);
        let mut t = random(&mut rng, l * l, 0.5);
        let mut s = random(&mut rng, l, 0.5);
        let mut e = random(&mut rng, l, 0.5);
        let loss = |emit: &[f64], t: &[f64], s: &[f64], e: &[f64]| {
            let (mut de, mut gt, mut gs, mut ge) = (vec![0.0; n * l], vec![0.0; l * l], vec![0.0; l], vec![0.0; l]);
            let p = crf::Params { transitions: t, start: s, end: e };
            let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
            crf::nll(p, &allow, emit, gold, l, &mut de, g).unwrap()
        };
        let (mut de, mut gt, mut gs, mut ge) = (vec![0.0; n * l], vec![0.0; l * l], vec![0.0; l], vec![0.0; l]);
        let p = crf::Params { transitions: &t, start: &s, end: &e };
        let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
        let value = crf::nll(p, &allow, &emit, gold, l, &mut de, g).unwrap();
        assert!(value > 0.0, "{scheme:?}");
        let (t0, s0, e0) = (t.clone(), s.clone(), e.clone());
        check("emit", &mut emit, &de, &mut |x| loss(x, &t0, &s0, &e0));
        let m0 = emit.clone();
        check("transitions", &mut t, &gt, &mut |x| loss(&m0, x, &s0, &e0));
        check("start", &mut s, &gs, &mut |x| loss(&m0, &t0, x, &e0));
        check("end", &mut e, &ge, &mut |x| loss(&m0, &t0, &s0, x));
    }
}

#[test]
fn crf_refuses_a_gold_sequence_the_scheme_forbids() {
    let names: Vec<String> = ["O", "B-X", "I-X"].iter().map(|s| s.to_string()).collect();
    let (_, allow) = allowed(Scheme::Bio, &names).unwrap();
    let z = vec![0.0f64; 9];
    let (mut de, mut gt, mut gs, mut ge) = (vec![0.0; 6], vec![0.0; 9], vec![0.0; 3], vec![0.0; 3]);
    let p = crf::Params { transitions: &z, start: &z[..3], end: &z[..3] };
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    assert!(crf::nll(p, &allow, &z[..6], &[0, 2], 3, &mut de, g).is_err());
}

#[test]
fn cross_entropy_gradients_match_finite_differences() {
    let mut rng = Rng::new(20);
    let (n, l) = (3, 4);
    let gold = [3, 0, 1];
    let mut s = random(&mut rng, n * l, 2.0);
    let mut ds = vec![0.0; n * l];
    xent::loss(&s, &gold, l, &mut ds).unwrap();
    check("scores", &mut s, &ds, &mut |s| xent::loss(s, &gold, l, &mut vec![0.0; n * l]).unwrap());
}

fn spec(labels: &[&str], scheme: Scheme, adapter: Option<(u32, u32)>, decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        biaffine: None,
        hash_embed: None,
        artifact: "weights".into(),
        tensor_prefix: "t".into(),
    }
}

#[test]
fn whole_tagger_loss_gradients_match_finite_differences() {
    let cases = [
        (spec(&["A", "B", "C"], Scheme::Tags, None, Decode::Argmax), vec![2u32, 0, 1, 1]),
        (spec(&["A", "B", "C"], Scheme::Tags, Some((3, 2)), Decode::Argmax), vec![2, 0, 1, 1]),
        (spec(&["O", "B-X", "I-X", "L-X", "U-X"], Scheme::Bilou, Some((2, 1)), Decode::Crf), vec![1, 2, 3, 0]),
    ];
    let hidden = 4;
    for (k, (s, gold)) in cases.into_iter().enumerate() {
        let mut rng = Rng::new(30 + k as u64);
        let mut model = Tagger::<f64>::new(&s, hidden, &mut rng).unwrap();
        // Random CRF scores and biases too, so no gradient is checked at zero.
        for i in 0..model.params().len() {
            model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.3));
        }
        let x = random(&mut rng, gold.len() * hidden, 1.0);
        let mut grads = model.zero_grads();
        let mut work = Default::default();
        model.loss(&x, &gold, &mut grads, &mut work).unwrap();
        for (i, g) in grads.iter().enumerate() {
            let mut p = model.params()[i].clone();
            let name = model.names()[i].clone();
            check(&format!("case {k} {name}"), &mut p, g, &mut |v| {
                let mut m = model.clone();
                m.param_mut(i).copy_from_slice(v);
                m.loss(&x, &gold, &mut m.zero_grads(), &mut Default::default()).unwrap()
            });
        }
    }
}
