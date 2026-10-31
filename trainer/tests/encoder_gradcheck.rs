//! The encoder's ops and the whole encoder against central finite
//! differences, all in f64 (#1519).

use turbo_trainer::encoder::{Encoder, Geometry, Lora, Work, attention, gated, lora, matmul, norm, rope};
use turbo_trainer::rng::Rng;

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

fn weighted(y: &[f64], r: &[f64]) -> f64 {
    y.iter().zip(r).map(|(a, b)| a * b).sum()
}

#[test]
fn matmul_gradients_match_finite_differences() {
    let mut rng = Rng::new(11);
    let (rows, inp, out) = (3, 4, 5);
    let mut w = random(&mut rng, out * inp, 0.5);
    let mut x = random(&mut rng, rows * inp, 1.0);
    let r = random(&mut rng, rows * out, 1.0);
    let loss = |w: &[f64], x: &[f64]| {
        let mut y = vec![0.0; rows * out];
        matmul::forward(x, w, inp, out, &mut y);
        weighted(&y, &r)
    };
    let (mut dw, mut dx) = (vec![0.0; w.len()], vec![0.0; x.len()]);
    matmul::grad_w(&x, &r, inp, out, &mut dw);
    matmul::grad_x(&w, &r, inp, out, &mut dx);
    let (w0, x0) = (w.clone(), x.clone());
    check("w", &mut w, &dw, &mut |w| loss(w, &x0));
    check("x", &mut x, &dx, &mut |x| loss(&w0, x));
}

#[test]
fn layer_norm_gradients_match_with_and_without_a_bias() {
    for bias in [false, true] {
        let mut rng = Rng::new(12 + bias as u64);
        let (rows, width, eps) = (3, 6, 1e-5);
        let mut x = random(&mut rng, rows * width, 1.5);
        let mut w = random(&mut rng, width, 1.0);
        let mut b = if bias { random(&mut rng, width, 0.5) } else { Vec::new() };
        let r = random(&mut rng, rows * width, 1.0);
        let loss = |x: &[f64], w: &[f64], b: &[f64]| {
            let (mut y, mut c) = (vec![0.0; rows * width], norm::Cache::default());
            norm::forward(x, w, b, eps, &mut y, &mut c);
            weighted(&y, &r)
        };
        let (mut y, mut c) = (vec![0.0; rows * width], norm::Cache::default());
        norm::forward(&x, &w, &b, eps, &mut y, &mut c);
        let (mut dx, mut dw, mut db) = (vec![0.0; x.len()], vec![0.0; width], vec![0.0; b.len()]);
        norm::backward(&x, &w, &c, &r, &mut dx, Some(&mut dw), bias.then_some(&mut db[..]));
        let (x0, w0, b0) = (x.clone(), w.clone(), b.clone());
        check("x", &mut x, &dx, &mut |x| loss(x, &w0, &b0));
        check("w", &mut w, &dw, &mut |w| loss(&x0, w, &b0));
        if bias {
            check("b", &mut b, &db, &mut |b| loss(&x0, &w0, b));
        }
    }
}

#[test]
fn rope_backward_is_the_transpose_of_forward() {
    let mut rng = Rng::new(13);
    let (rows, ld, cols, d) = (5, 10, 8, 4);
    let table = rope::Table::<f64>::new(160_000.0, 10_000.0, 8, d).unwrap();
    for global in [true, false] {
        let mut x = random(&mut rng, rows * ld, 1.0);
        let r = random(&mut rng, rows * ld, 1.0);
        let mut loss = |x: &[f64]| {
            let mut y = x.to_vec();
            table.apply(global, &mut y, ld, cols);
            weighted(&y, &r)
        };
        let mut dx = r.clone();
        table.apply_grad(global, &mut dx, ld, cols);
        check("x", &mut x, &dx, &mut loss);
    }
}

#[test]
fn attention_gradients_match_for_global_and_local_layers() {
    for window in [usize::MAX, 1] {
        let mut rng = Rng::new(14);
        let (n, hidden, heads) = (5, 8, 2);
        let mut qkv = random(&mut rng, n * 3 * hidden, 1.0);
        let r = random(&mut rng, n * hidden, 1.0);
        let mut loss = |qkv: &[f64]| {
            let (mut ctx, mut c) = (vec![0.0; n * hidden], attention::Cache::default());
            attention::forward(qkv, hidden, heads, window, &mut ctx, &mut c);
            weighted(&ctx, &r)
        };
        let (mut ctx, mut c) = (vec![0.0; n * hidden], attention::Cache::default());
        attention::forward(&qkv, hidden, heads, window, &mut ctx, &mut c);
        let mut dqkv = vec![0.0; qkv.len()];
        attention::backward(&qkv, hidden, heads, window, &c, &r, &mut dqkv);
        check("qkv", &mut qkv, &dqkv, &mut loss);
        // On a local layer query 0 reads keys 0 and 1 alone: moving key
        // 2's values changes its context on the global layer only.
        let mut moved = qkv.clone();
        moved[2 * 3 * hidden + 2 * hidden] += 1.0;
        let mut ctx2 = vec![0.0; n * hidden];
        attention::forward(&moved, hidden, heads, window, &mut ctx2, &mut c);
        assert_eq!(ctx[..hidden] == ctx2[..hidden], window == 1, "window {window}");
    }
}

#[test]
fn gated_mlp_gradients_match_finite_differences() {
    let mut rng = Rng::new(15);
    let (rows, inter) = (3, 4);
    let mut pre = random(&mut rng, rows * 2 * inter, 2.0);
    let r = random(&mut rng, rows * inter, 1.0);
    let mut loss = |pre: &[f64]| {
        let mut act = vec![0.0; rows * inter];
        gated::forward(pre, inter, &mut act);
        weighted(&act, &r)
    };
    let mut dpre = vec![0.0; pre.len()];
    gated::backward(&pre, inter, &r, &mut dpre);
    check("pre", &mut pre, &dpre, &mut loss);
}

#[test]
fn lora_gradients_match_finite_differences() {
    let mut rng = Rng::new(16);
    let (rows, inp, out, rank, s) = (3, 4, 5, 2, 1.5);
    let mut x = random(&mut rng, rows * inp, 1.0);
    let (mut a, mut b) = (random(&mut rng, rank * inp, 0.5), random(&mut rng, out * rank, 0.5));
    let r = random(&mut rng, rows * out, 1.0);
    let loss = |x: &[f64], a: &[f64], b: &[f64]| {
        let (mut y, mut h) = (vec![0.0; rows * out], Vec::new());
        lora::forward(x, a, b, inp, rank, s, &mut y, &mut h);
        weighted(&y, &r)
    };
    let (mut y, mut h) = (vec![0.0; rows * out], Vec::new());
    lora::forward(&x, &a, &b, inp, rank, s, &mut y, &mut h);
    let (mut dx, mut da, mut db) = (vec![0.0; x.len()], vec![0.0; a.len()], vec![0.0; b.len()]);
    lora::backward(&x, &a, &b, &h, inp, rank, s, &r, &mut da, &mut db, Some(&mut dx));
    let (x0, a0, b0) = (x.clone(), a.clone(), b.clone());
    check("x", &mut x, &dx, &mut |x| loss(x, &a0, &b0));
    check("a", &mut a, &da, &mut |a| loss(&x0, a, &b0));
    check("b", &mut b, &db, &mut |b| loss(&x0, &a0, b));
    // A fresh adapter changes nothing.
    let (a1, b1) = lora::init::<f64>(inp, out, rank, &mut rng);
    let mut y1 = vec![0.0; rows * out];
    lora::forward(&x, &a1, &b1, inp, rank, s, &mut y1, &mut h);
    assert!(y1.iter().all(|&v| v == 0.0));
}

/// Three layers over two heads: layers 0 and 2 global, 1 local with one
/// key each side, so both bases and the window are in the gradient.
fn geometry() -> Geometry {
    Geometry {
        layers: 3,
        hidden: 8,
        heads: 2,
        intermediate: 6,
        eps: 1e-5,
        window: 1,
        global_every: 2,
        max_positions: 16,
        global_theta: 160_000.0,
        local_theta: 10_000.0,
    }
}

/// The loss of `e` with its weights replaced by `params`.
fn encoder_loss(e: &Encoder<f64>, params: &[Vec<f64>], ids: &[u32], r: &[f64]) -> f64 {
    let mut m = e.clone();
    m.set_params(params.to_vec()).unwrap();
    let mut w = Work::default();
    m.forward(ids, &mut w).unwrap();
    weighted(&w.out, r)
}

fn check_encoder(e: &Encoder<f64>, ids: &[u32], seed: u64) {
    let mut rng = Rng::new(seed);
    let r = random(&mut rng, ids.len() * e.geometry().hidden, 1.0);
    let mut w = Work::default();
    e.forward(ids, &mut w).unwrap();
    let mut g = e.grads();
    e.backward(&w, &r, &mut g).unwrap();
    let mut params = e.params().to_vec();
    for (s, trainable) in e.trainable().into_iter().enumerate() {
        if !trainable {
            assert!(g[s].iter().all(|&v| v == 0.0), "slot {s} is frozen");
            continue;
        }
        let mut p = std::mem::take(&mut params[s]);
        let analytic = g[s].clone();
        check(&format!("slot {s}"), &mut p, &analytic, &mut |p| {
            let mut all = params.clone();
            all[s] = p.to_vec();
            encoder_loss(e, &all, ids, &r)
        });
        params[s] = p;
    }
}

#[test]
fn encoder_gradients_match_finite_differences() {
    let mut rng = Rng::new(17);
    let e = Encoder::<f64>::random(geometry(), 12, &mut rng).unwrap();
    // A repeated id, so the word table's gradient sums two rows.
    check_encoder(&e, &[3, 7, 1, 7, 9], 18);
}

#[test]
fn adapter_gradients_match_and_the_base_is_frozen() {
    let mut rng = Rng::new(19);
    let e = Encoder::<f64>::random(geometry(), 12, &mut rng).unwrap();
    // B starts at zero, which would hide A's gradient: move both first.
    let mut e = e.with_lora(Lora { rank: 2, alpha: 4.0 }, &mut rng).unwrap();
    let mut params = e.params().to_vec();
    for (s, t) in e.trainable().into_iter().enumerate() {
        if t {
            params[s] = random(&mut rng, params[s].len(), 0.5);
        }
    }
    e.set_params(params).unwrap();
    check_encoder(&e, &[2, 5, 5, 0, 11, 4], 20);
}

#[test]
fn merged_adapters_give_the_plain_encoder_the_same_states() {
    let mut rng = Rng::new(21);
    let geo = geometry();
    let mut e =
        Encoder::<f64>::random(geo, 12, &mut rng).unwrap().with_lora(Lora { rank: 2, alpha: 4.0 }, &mut rng).unwrap();
    let mut params = e.params().to_vec();
    for (s, t) in e.trainable().into_iter().enumerate() {
        if t {
            params[s] = random(&mut rng, params[s].len(), 0.5);
        }
    }
    e.set_params(params).unwrap();
    let plain = Encoder::<f64>::assemble(geo, 12, e.merged()).unwrap();
    let ids = [1, 2, 3, 4, 5, 6, 7];
    let (mut a, mut b) = (Work::default(), Work::default());
    e.forward(&ids, &mut a).unwrap();
    plain.forward(&ids, &mut b).unwrap();
    for (x, y) in a.out.iter().zip(&b.out) {
        assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    }
}

#[test]
fn the_encoder_refuses_what_it_cannot_run() {
    let mut rng = Rng::new(22);
    let e = Encoder::<f64>::random(geometry(), 12, &mut rng).unwrap();
    let mut w = Work::default();
    assert!(e.forward(&[], &mut w).is_err());
    assert!(e.forward(&[12], &mut w).is_err());
    assert!(e.forward(&[1; 17], &mut w).is_err());
    e.forward(&[1, 2], &mut w).unwrap();
    let mut g = e.grads();
    assert!(e.backward(&w, &[0.0; 8], &mut g).is_err());
    let odd = Geometry { heads: 3, ..geometry() };
    assert!(Encoder::<f64>::random(odd, 12, &mut rng).is_err());
}
