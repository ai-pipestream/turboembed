#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub tp: usize,
    pub fp: usize,
    pub tn: usize,
    pub r#fn: usize,
}

impl Counts {
    pub fn precision(self) -> f64 {
        ratio(self.tp, self.tp + self.fp)
    }

    pub fn recall(self) -> f64 {
        ratio(self.tp, self.tp + self.r#fn)
    }

    pub fn accuracy(self) -> f64 {
        ratio(self.tp + self.tn, self.tp + self.fp + self.tn + self.r#fn)
    }

    pub fn f1(self) -> f64 {
        ratio(2 * self.tp, 2 * self.tp + self.fp + self.r#fn)
    }
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 { 0.0 } else { numerator as f64 / denominator as f64 }
}

pub fn score(scores: &[f64], labels: &[bool], threshold: f64) -> Counts {
    let mut out = Counts::default();
    for (&score, &label) in scores.iter().zip(labels) {
        match (score >= threshold, label) {
            (true, true) => out.tp += 1,
            (true, false) => out.fp += 1,
            (false, false) => out.tn += 1,
            (false, true) => out.r#fn += 1,
        }
    }
    out
}

pub fn best_threshold(scores: &[f64], labels: &[bool]) -> f64 {
    let mut best = (0.5, f64::NEG_INFINITY, f64::NEG_INFINITY);
    for step in 1..20 {
        let threshold = step as f64 * 0.05;
        let counts = score(scores, labels, threshold);
        let candidate = (threshold, counts.f1(), counts.precision());
        if candidate.1 > best.1 || (candidate.1 == best.1 && candidate.2 > best.2) {
            best = candidate;
        }
    }
    best.0
}

pub fn report(name: &str, threshold: f64, counts: Counts) {
    println!(
        "{name}: threshold={threshold:.2} TP={} FP={} TN={} FN={} precision={:.6} recall={:.6} F1={:.6}",
        counts.tp,
        counts.fp,
        counts.tn,
        counts.r#fn,
        counts.precision(),
        counts.recall(),
        counts.f1()
    );
}
