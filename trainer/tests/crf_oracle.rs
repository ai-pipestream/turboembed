//! The CRF against brute force: for 1 to 5 words over TAGS lattices of 2
//! to 4 labels, BIO and BILOU, every legal path is enumerated in f64,
//! and log Z, the loss, each place's marginals and the gradient of every
//! emission, start, end and transition score come from that enumeration.
//! The forward-backward path must match it on random scores and on the
//! adversarial ones: one path dominating by 1e3, all scores equal, large
//! negative transitions, the gold path the worst, the gold path far
//! below the rest (at the edge of `SCORE_RANGE`), and a large score every
//! path shares. The same inputs in f32 match too, and the loss's
//! gradient matches finite differences.

use turbo::head::crf::{Allowed, allowed};
use turbo::manifest::Scheme;
use turbo_trainer::crf;
use turbo_trainer::logspace::{logsumexp, softmax};
use turbo_trainer::rng::Rng;

/// One CRF problem: scores in f64, the gold path, and its lattice.
#[derive(Clone)]
struct Case {
    labels: usize,
    n: usize,
    emit: Vec<f64>,
    tr: Vec<f64>,
    start: Vec<f64>,
    end: Vec<f64>,
}

/// What a run gives: the loss and the gradients of emit, tr, start, end.
#[derive(Debug)]
struct Out {
    loss: f64,
    grads: [Vec<f64>; 4],
}

fn paths(a: &Allowed, labels: usize, n: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut p = vec![0usize; n];
    loop {
        let legal = a.may_start(p[0]) && a.may_end(p[n - 1]) && p.windows(2).all(|w| a.may_follow(w[0], w[1]));
        if legal {
            out.push(p.clone());
        }
        // Next sequence, as a counter in base `labels`.
        let mut i = 0;
        loop {
            if i == n {
                return out;
            }
            p[i] += 1;
            if p[i] < labels {
                break;
            }
            p[i] = 0;
            i += 1;
        }
    }
}

fn score(c: &Case, p: &[usize]) -> f64 {
    let l = c.labels;
    let mut s = c.start[p[0]] + c.end[p[c.n - 1]];
    for t in 0..c.n {
        s += c.emit[t * l + p[t]];
        if t > 0 {
            s += c.tr[p[t - 1] * l + p[t]];
        }
    }
    s
}

/// Brute force: path probabilities from a softmax over every legal path.
fn oracle(c: &Case, all: &[Vec<usize>], gold: &[usize]) -> Out {
    let l = c.labels;
    let scores: Vec<f64> = all.iter().map(|p| score(c, p)).collect();
    let gold_score = score(c, gold);
    // log Z less the gold score, from the largest path, without rounding
    // a large shared score into the difference.
    let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let loss = (m - gold_score) + scores.iter().map(|s| (s - m).exp()).sum::<f64>().ln();
    let mut prob = scores.clone();
    softmax(&mut prob);
    let (mut de, mut dt, mut ds, mut dn) = (vec![0.0; c.n * l], vec![0.0; l * l], vec![0.0; l], vec![0.0; l]);
    for (p, &w) in all.iter().zip(&prob) {
        ds[p[0]] += w;
        dn[p[c.n - 1]] += w;
        for t in 0..c.n {
            de[t * l + p[t]] += w;
            if t > 0 {
                dt[p[t - 1] * l + p[t]] += w;
            }
        }
    }
    ds[gold[0]] -= 1.0;
    dn[gold[c.n - 1]] -= 1.0;
    for t in 0..c.n {
        de[t * l + gold[t]] -= 1.0;
        if t > 0 {
            dt[gold[t - 1] * l + gold[t]] -= 1.0;
        }
    }
    Out { loss, grads: [de, dt, ds, dn] }
}

/// The fast path, in f64.
fn fast(c: &Case, a: &Allowed, gold: &[usize]) -> Out {
    let l = c.labels;
    let p = crf::Params { transitions: &c.tr, start: &c.start, end: &c.end };
    let (mut gt, mut gs, mut ge, mut de) = (vec![0.0; l * l], vec![0.0; l], vec![0.0; l], vec![0.0; c.n * l]);
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    let y: Vec<u32> = gold.iter().map(|&v| v as u32).collect();
    let loss = crf::nll(p, a, &c.emit, &y, l, &mut de, g).unwrap();
    Out { loss, grads: [de, gt, gs, ge] }
}

/// The fast path in f32, on the case's scores rounded to f32.
fn fast32(c: &Case, a: &Allowed, gold: &[usize]) -> Out {
    let l = c.labels;
    let f = |v: &[f64]| v.iter().map(|&x| x as f32).collect::<Vec<f32>>();
    let (emit, tr, st, en) = (f(&c.emit), f(&c.tr), f(&c.start), f(&c.end));
    let p = crf::Params { transitions: &tr, start: &st, end: &en };
    let (mut gt, mut gs, mut ge, mut de) = (vec![0f32; l * l], vec![0f32; l], vec![0f32; l], vec![0f32; c.n * l]);
    let g = crf::Grad { transitions: &mut gt, start: &mut gs, end: &mut ge };
    let y: Vec<u32> = gold.iter().map(|&v| v as u32).collect();
    let loss = crf::nll(p, a, &emit, &y, l, &mut de, g).unwrap();
    let w = |v: Vec<f32>| v.into_iter().map(f64::from).collect();
    Out { loss, grads: [w(de), w(gt), w(gs), w(ge)] }
}

/// The case with every score rounded to f32, for the f32 comparison.
fn rounded(c: &Case) -> Case {
    let r = |v: &[f64]| v.iter().map(|&x| x as f32 as f64).collect();
    Case { emit: r(&c.emit), tr: r(&c.tr), start: r(&c.start), end: r(&c.end), ..c.clone() }
}

const NAMES: [&str; 4] = ["emit", "transitions", "start", "end"];

fn compare(what: &str, got: &Out, want: &Out, tol: f64) {
    let lt = tol * (1.0 + want.loss.abs());
    assert!((got.loss - want.loss).abs() <= lt, "{what}: loss {} vs oracle {}", got.loss, want.loss);
    for (k, (g, w)) in got.grads.iter().zip(&want.grads).enumerate() {
        for (i, (a, b)) in g.iter().zip(w).enumerate() {
            assert!((a - b).abs() <= tol, "{what}: {}[{i}] {a} vs oracle {b}\n{got:?}\n{want:?}", NAMES[k]);
        }
    }
}

/// The lattices: TAGS over 2 to 4 labels, BIO and BILOU over one type.
fn lattices() -> Vec<(String, usize, Allowed)> {
    let mut out = Vec::new();
    for l in 2..=4 {
        let names: Vec<String> = (0..l).map(|i| format!("T{i}")).collect();
        out.push((format!("tags{l}"), l, allowed(Scheme::Tags, &names).unwrap().1));
    }
    for (scheme, names) in
        [(Scheme::Bio, &["O", "B-X", "I-X"][..]), (Scheme::Bilou, &["O", "B-X", "I-X", "L-X", "U-X"])]
    {
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        out.push((format!("{scheme:?}"), names.len(), allowed(scheme, &names).unwrap().1));
    }
    out
}

fn random(rng: &mut Rng, labels: usize, n: usize, scale: f64) -> Case {
    let mut v = |k: usize| (0..k).map(|_| rng.uniform(scale)).collect::<Vec<f64>>();
    Case { labels, n, emit: v(n * labels), tr: v(labels * labels), start: v(labels), end: v(labels) }
}

#[test]
fn forward_backward_matches_enumerating_every_path() {
    let mut rng = Rng::new(1509);
    let mut checked = 0;
    for (name, l, a) in lattices() {
        for n in 1..=5 {
            let all = paths(&a, l, n);
            assert!(!all.is_empty(), "{name} {n}");
            let pick = |rng: &mut Rng| all[rng.below(all.len())].clone();
            let mut cases: Vec<(&str, Case, Vec<usize>)> = Vec::new();
            for _ in 0..3 {
                let c = random(&mut rng, l, n, 2.0);
                cases.push(("random", c, pick(&mut rng)));
            }
            cases.push(("all equal", Case { emit: vec![0.0; n * l], ..random(&mut rng, l, n, 0.0) }, pick(&mut rng)));
            // One path dominating every other by 1e3, gold elsewhere.
            let mut c = random(&mut rng, l, n, 1.0);
            let q = pick(&mut rng);
            (0..n).for_each(|t| c.emit[t * l + q[t]] += 1e3);
            cases.push(("one path dominates", c, pick(&mut rng)));
            // Large negative transitions on half the pairs.
            let mut c = random(&mut rng, l, n, 1.0);
            c.tr.iter_mut().for_each(|t| {
                if rng.below(2) == 0 {
                    *t -= 1e4;
                }
            });
            cases.push(("large negative transitions", c, pick(&mut rng)));
            // The gold path the worst legal path.
            let c = random(&mut rng, l, n, 3.0);
            let worst = all.iter().min_by(|p, q| score(&c, p).total_cmp(&score(&c, q))).unwrap().clone();
            cases.push(("gold is the worst path", c, worst));
            // The gold path far below the rest, and every other path tied.
            let g = pick(&mut rng);
            let mut c = Case { emit: vec![0.0; n * l], ..random(&mut rng, l, n, 0.0) };
            (0..n).for_each(|t| c.emit[t * l + g[t]] = -1e6);
            cases.push(("gold far below a tie", c, g));
            // A large score every path shares, on every component: 2^30,
            // on scores in steps of 2^-16, so every path sum (at most 12 terms
            // under 2^31) is exact in f64
            // and the oracle is exact too.
            let mut c = random(&mut rng, l, n, 1.0);
            for v in [&mut c.emit, &mut c.tr, &mut c.start, &mut c.end] {
                v.iter_mut().for_each(|x| *x = (*x * 65536.0).round() / 65536.0 + 1073741824.0);
            }
            cases.push(("large shared score", c, pick(&mut rng)));

            for (what, c, gold) in cases {
                let label = format!("{name}, {n} words, {what}");
                compare(&label, &fast(&c, &a, &gold), &oracle(&c, &all, &gold), 1e-9);
                let r = rounded(&c);
                compare(&format!("{label}, f32"), &fast32(&c, &a, &gold), &oracle(&r, &all, &gold), 1e-6);
                // Each place's marginals sum to one: the emission
                // gradient of a word sums to zero.
                let out = fast(&c, &a, &gold);
                for t in 0..n {
                    let s: f64 = out.grads[0][t * l..][..l].iter().sum();
                    assert!(s.abs() <= 1e-9, "{label}: word {t} marginals sum to {}", s + 1.0);
                }
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 5 * 5 * 9);
}

#[test]
fn the_loss_gradient_matches_finite_differences() {
    const EPS: f64 = 1e-6;
    let mut rng = Rng::new(1518);
    for (name, l, a) in lattices() {
        for n in 1..=3 {
            let all = paths(&a, l, n);
            let c = random(&mut rng, l, n, 1.5);
            let gold = all[rng.below(all.len())].clone();
            let out = fast(&c, &a, &gold);
            assert!((logsumexp(all.iter().map(|p| score(&c, p))) - score(&c, &gold) - out.loss).abs() < 1e-9);
            for k in 0..4 {
                for i in 0..out.grads[k].len() {
                    let at = |d: f64| {
                        let mut m = c.clone();
                        [&mut m.emit, &mut m.tr, &mut m.start, &mut m.end][k][i] += d;
                        fast(&m, &a, &gold).loss
                    };
                    let numeric = (at(EPS) - at(-EPS)) / (2.0 * EPS);
                    let g = out.grads[k][i];
                    assert!(
                        (numeric - g).abs() <= 1e-6,
                        "{name} {n}: {}[{i}] numeric {numeric} analytic {g}",
                        NAMES[k]
                    );
                }
            }
        }
    }
}
