//! The best labelling of a run of units: a Viterbi search over the
//! candidate languages, where each unit adds its own score for a
//! language and every change of language between two units costs what
//! the gap between them says it should.

/// What separates two units, from weakest to strongest. A switch of
/// language is cheapest where the text already breaks.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gap {
    /// Whitespace only (or nothing, inside a token).
    Space,
    /// Punctuation: a comma, a colon, a quotation mark, a bracket, a dash.
    Punctuation,
    /// A sentence boundary or a line break.
    Sentence,
}

/// The cost of a switch at each kind of gap, in the units of the scores
/// (natural log probability).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SwitchCost {
    pub space: f64,
    pub punctuation: f64,
    pub sentence: f64,
}

impl SwitchCost {
    pub fn at(&self, gap: Gap) -> f64 {
        match gap {
            Gap::Space => self.space,
            Gap::Punctuation => self.punctuation,
            Gap::Sentence => self.sentence,
        }
    }
}

/// Buffers for [`decode`], kept between calls.
#[derive(Debug, Default)]
pub struct Decoder {
    score: Vec<f64>,
    next: Vec<f64>,
    /// For each unit after the first and each label, the label of the
    /// unit before it on the best path.
    back: Vec<u32>,
}

impl Decoder {
    /// Labels `n` units with one of `k` labels each, into `out`.
    /// `emission[u * k + l]` is unit `u`'s score for label `l` and
    /// `gaps[u]` is the gap before unit `u` (`gaps[0]` is not read). Ties
    /// go to the lower label and, between staying and switching, to
    /// staying.
    pub fn decode(
        &mut self,
        n: usize,
        k: usize,
        emission: &[f64],
        gaps: &[Gap],
        cost: &SwitchCost,
        out: &mut Vec<u32>,
    ) {
        assert_eq!(emission.len(), n * k);
        assert!(gaps.len() >= n);
        out.clear();
        if n == 0 || k == 0 {
            return;
        }
        self.score.clear();
        self.score.extend_from_slice(&emission[..k]);
        self.back.clear();
        self.back.resize((n - 1) * k, 0);
        for u in 1..n {
            let best = argmax(&self.score);
            let switched = self.score[best] - cost.at(gaps[u]);
            self.next.clear();
            for l in 0..k {
                let (from, s) = if self.score[l] >= switched { (l, self.score[l]) } else { (best, switched) };
                self.back[(u - 1) * k + l] = from as u32;
                self.next.push(s + emission[u * k + l]);
            }
            std::mem::swap(&mut self.score, &mut self.next);
        }
        out.resize(n, 0);
        let mut l = argmax(&self.score);
        for u in (0..n).rev() {
            out[u] = l as u32;
            if u > 0 {
                l = self.back[(u - 1) * k + l] as usize;
            }
        }
    }
}

/// The first index of the largest value.
fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const COST: SwitchCost = SwitchCost { space: 3.0, punctuation: 1.0, sentence: 0.5 };

    fn run(emission: &[[f64; 2]], gaps: &[Gap]) -> Vec<u32> {
        let flat: Vec<f64> = emission.iter().flatten().copied().collect();
        let mut out = Vec::new();
        Decoder::default().decode(emission.len(), 2, &flat, gaps, &COST, &mut out);
        out
    }

    #[test]
    fn one_unit_against_the_run_does_not_pay_for_two_switches() {
        let e = [[1.0, 0.0], [1.0, 0.0], [0.0, 2.0], [1.0, 0.0]];
        assert_eq!(run(&e, &[Gap::Space; 4]), [0, 0, 0, 0]);
    }

    #[test]
    fn a_cheap_gap_lets_the_same_evidence_switch() {
        let e = [[2.0, 0.0], [2.0, 0.0], [0.0, 2.0], [0.0, 2.0]];
        assert_eq!(run(&e, &[Gap::Space; 4]), [0, 0, 1, 1]);
        let e = [[1.0, 0.0], [0.0, 1.2], [1.0, 0.0]];
        assert_eq!(run(&e, &[Gap::Space; 3]), [0, 0, 0]);
        assert_eq!(run(&e, &[Gap::Space, Gap::Sentence, Gap::Sentence]), [0, 1, 0]);
    }

    #[test]
    fn ties_stay_and_go_to_the_lower_label() {
        assert_eq!(run(&[[0.0, 0.0], [0.0, 0.0]], &[Gap::Space; 2]), [0, 0]);
        // [0, 0], [1, 1] and [1, 0] all score 3: the last unit's tie goes
        // to label 0, which got there by staying.
        let e = [[0.0, 3.0], [3.0, 0.0]];
        assert_eq!(run(&e, &[Gap::Space; 2]), [0, 0]);
    }

    #[test]
    fn nothing_to_label_is_nothing() {
        let mut out = vec![7];
        Decoder::default().decode(0, 3, &[], &[], &COST, &mut out);
        assert!(out.is_empty());
    }
}
