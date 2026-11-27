//! Bounded calibration-only chooser for exact same-boundary city / first-name
//! alternatives. All feature fitting and threshold selection use publisher
//! train, split on the pinned persona order. Never a serving policy.

use serde_json::json;

pub const WIDTH: usize = 38;

#[derive(Clone, Copy)]
pub struct Flip {
    pub a: usize,
    pub b: usize,
    pub to_city: bool,
    pub features: [f64; WIDTH],
}

impl Flip {
    pub fn direction(&self) -> usize {
        usize::from(!self.to_city)
    }

    pub fn native(&self) -> &'static str {
        if self.to_city { "first_name" } else { "city" }
    }

    pub fn other(&self) -> &'static str {
        if self.to_city { "city" } else { "first_name" }
    }
}

/// Distinct anchored confidence scores plus a fixed signed 32-lane
/// projection of frozen contextual states. No token list or gold features.
pub fn features(a: usize, b: usize, states: &[f32], city: f32, first: f32, outside: f32) -> [f64; WIDTH] {
    let hidden = 768usize;
    let all = states.len() / hidden;
    assert_eq!(states.len(), all * hidden);
    let from = a.saturating_sub(1);
    let to = (b + 1).min(all);
    let mut out = [0.0; WIDTH];
    let log = |x: f32| f64::from(x.max(1e-12)).ln();
    out[0] = log(city);
    out[1] = log(first);
    out[2] = log(outside);
    out[3] = log(city) - log(first);
    out[4] = (b - a) as f64;
    out[5] = f64::from(a == 0);
    for token in from..to {
        let weight = if (a..b).contains(&token) { 1.0 } else { 0.25 };
        for (i, &x) in states[token * hidden..(token + 1) * hidden].iter().enumerate() {
            let h = (i as u32).wrapping_mul(0x9e37_79b9).rotate_left(7);
            let lane = ((h >> 10) as usize) % 32;
            let sign = if h & 1 == 0 { 1.0 } else { -1.0 };
            out[6 + lane] += f64::from(x) * sign * weight / ((b - a) as f64);
        }
    }
    assert!(out.iter().all(|v| v.is_finite()));
    out
}

pub type Sample = ([f64; WIDTH], bool);

pub struct Fit {
    center: [f64; WIDTH],
    scale: [f64; WIDTH],
    coeffs: [f64; WIDTH],
    bias: f64,
    bar: Option<f64>,
    pub receipt: serde_json::Value,
}

fn logit(x: f64) -> f64 {
    1.0 / (1.0 + (-x.clamp(-32.0, 32.0)).exp())
}

impl Fit {
    fn score(&self, x: &[f64; WIDTH]) -> f64 {
        logit(self.bias + (0..WIDTH).map(|i| self.coeffs[i] * (x[i] - self.center[i]) * self.scale[i]).sum::<f64>())
    }

    pub fn admits(&self, x: &[f64; WIDTH]) -> bool {
        self.bar.is_some_and(|bar| self.score(x) >= bar)
    }
}

pub fn fit(train: &[Sample], calibrate: &[Sample]) -> Fit {
    assert!(!train.is_empty());
    let n = train.len() as f64;
    let center = std::array::from_fn(|i| train.iter().map(|(x, _)| x[i]).sum::<f64>() / n);
    let scale = std::array::from_fn(|i| {
        let var = train.iter().map(|(x, _)| (x[i] - center[i]).powi(2)).sum::<f64>() / n;
        1.0 / var.sqrt().max(0.05)
    });
    let mut model = Fit { center, scale, coeffs: [0.0; WIDTH], bias: 0.0, bar: None, receipt: json!(null) };
    let mut samples = Vec::with_capacity(train.len());
    for (x, y) in train {
        let z: [f64; WIDTH] = std::array::from_fn(|i| (x[i] - center[i]) * scale[i]);
        samples.push((z, *y));
    }
    for _ in 0..200 {
        let mut gradient = [0.0; WIDTH];
        let mut gb = 0.0;
        for (x, y) in &samples {
            let p = logit(model.bias + (0..WIDTH).map(|i| model.coeffs[i] * x[i]).sum::<f64>());
            let e = p - f64::from(*y);
            gb += e;
            for (g, value) in gradient.iter_mut().zip(x) {
                *g += e * value;
            }
        }
        for (c, g) in model.coeffs.iter_mut().zip(gradient) {
            *c -= 0.18 * (g / n + 0.03 * *c);
        }
        model.bias -= 0.18 * gb / n;
    }
    let mut ranked: Vec<_> = calibrate.iter().map(|(x, y)| (model.score(x), *y)).collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut correct, mut best_count, mut best_correct) = (0, 0, 0);
    for (i, &(score, good)) in ranked.iter().enumerate() {
        correct += usize::from(good);
        let next_distinct = ranked.get(i + 1).is_none_or(|r| r.0 < score);
        if next_distinct && correct >= 10 && correct * 10 >= (i + 1) * 9 {
            model.bar = Some(score);
            best_count = i + 1;
            best_correct = correct;
        }
    }
    model.receipt = json!({
        "fit_eligible":train.len(),
        "fit_positive":train.iter().filter(|e|e.1).count(),
        "calibration_eligible":calibrate.len(),
        "calibration_positive":calibrate.iter().filter(|e|e.1).count(),
        "calibration_predictions":best_count,
        "calibration_correct":best_correct,
        "calibration_minimum_precision":0.90,
        "calibration_minimum_correct":10,
        "locked_bar":model.bar,
    });
    model
}
