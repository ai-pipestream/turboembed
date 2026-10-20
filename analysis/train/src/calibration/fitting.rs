use super::Samples;
use turbo_calibration::{Result, Transform};
#[derive(Clone, Copy, Debug)]
pub enum Method {
    Temperature,
    Platt,
    Isotonic,
}

pub fn fit_transform(method: Method, samples: &Samples<'_>) -> Result<Transform> {
    let k = samples.probabilities.first().ok_or("no calibration samples")?.len();
    samples.check(k)?;
    let transform = match method {
        Method::Temperature => temperature(samples),
        Method::Platt => platt(samples),
        Method::Isotonic => isotonic(samples),
    };
    transform.check()?;
    Ok(transform)
}
fn loss(t: f64, s: &Samples<'_>) -> f64 {
    let mut out = vec![0.0; s.probabilities[0].len()];
    let mut sum = 0.0;
    for (p, &y) in s.probabilities.iter().zip(s.gold) {
        Transform::Temperature(t).apply(p, &mut out).unwrap();
        sum -= out[y].max(1e-300).ln() / s.gold.len() as f64;
    }
    sum
}
fn temperature(s: &Samples<'_>) -> Transform {
    // Bounded deterministic search in log temperature, including T=1.
    let (mut lo, mut hi) = (-6.0f64, 6.0f64);
    for _ in 0..80 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if loss(a.exp(), s) <= loss(b.exp(), s) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let t = ((lo + hi) / 2.0).exp();
    Transform::Temperature(if loss(t, s) <= loss(1.0, s) { t } else { 1.0 })
}
fn platt(s: &Samples<'_>) -> Transform {
    let (mut a, mut b) = (1.0, 0.0);
    let n = s.gold.len() as f64 * s.probabilities[0].len() as f64;
    for _ in 0..2000 {
        let (mut da, mut db) = (0.0, 0.0);
        for (row, &y) in s.probabilities.iter().zip(s.gold) {
            for (j, &p) in row.iter().enumerate() {
                let p = p.clamp(1e-15, 1.0 - 1e-15);
                let x = p.ln() - (-p).ln_1p();
                let z: f64 = a * x + b;
                let pred = if z >= 0.0 {
                    1.0 / (1.0 + (-z).exp())
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                };
                let e = pred - f64::from(j == y);
                da += e * x / n;
                db += e / n;
            }
        }
        a = (a - 0.01 * da).clamp(0.0, 1e6);
        b = (b - 0.01 * db).clamp(-1e6, 1e6);
    }
    Transform::Platt { slope: a, intercept: b }
}
fn isotonic(s: &Samples<'_>) -> Transform {
    let mut pairs: Vec<_> = s
        .probabilities
        .iter()
        .zip(s.gold)
        .flat_map(|(p, &y)| p.iter().enumerate().map(move |(i, &v)| (v, f64::from(i == y))))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Equal x must belong to one group before PAVA; tie order cannot
    // change the fitted function.
    let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
    let mut i = 0;
    while i < pairs.len() {
        let x = pairs[i].0;
        let (mut sum, mut count) = (0.0, 0);
        while i < pairs.len() && pairs[i].0 == x {
            sum += pairs[i].1;
            count += 1;
            i += 1;
        }
        blocks.push((x, sum, count));
        while blocks.len() > 1 {
            let n = blocks.len();
            let (a, b) = (blocks[n - 2], blocks[n - 1]);
            if a.1 / a.2 as f64 <= b.1 / b.2 as f64 {
                break;
            }
            blocks.pop();
            blocks[n - 2] = (b.0, a.1 + b.1, a.2 + b.2);
        }
    }
    Transform::Isotonic {
        upper: blocks.iter().map(|b| b.0).collect(),
        value: blocks.iter().map(|b| b.1 / b.2 as f64).collect(),
    }
}
