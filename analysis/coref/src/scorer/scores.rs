//! The accumulated scores (`CorefScores` and `CorefScores.Score`).

/// How far a derived value may lie from the value it is derived from.
const DERIVED_TOLERANCE: f64 = 1e-12;

/// One precision, recall and F1 triple, each in `[0, 1]`, F1 the
/// harmonic mean of the other two (zero when both are).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Score {
    precision: f64,
    recall: f64,
    f1: f64,
}

fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

fn harmonic_mean(precision: f64, recall: f64) -> f64 {
    if precision + recall == 0.0 { 0.0 } else { 2.0 * precision * recall / (precision + recall) }
}

fn unit(value: f64, name: &str) -> Result<(), String> {
    if (0.0..=1.0).contains(&value) { Ok(()) } else { Err(format!("{name} must lie in [0, 1]: {value}")) }
}

impl Score {
    /// A validated triple: each value in `[0, 1]` and `f1` the harmonic
    /// mean of `precision` and `recall`.
    pub fn new(precision: f64, recall: f64, f1: f64) -> Result<Score, String> {
        unit(precision, "precision")?;
        unit(recall, "recall")?;
        unit(f1, "f1")?;
        let expected = harmonic_mean(precision, recall);
        if (f1 - expected).abs() > DERIVED_TOLERANCE {
            return Err(format!("f1 must equal the harmonic mean of precision and recall: {expected}"));
        }
        Ok(Score { precision, recall, f1 })
    }

    /// A triple from a numerator and denominator per side, a zero
    /// denominator giving zero.
    pub fn of(
        precision_numerator: f64,
        precision_denominator: f64,
        recall_numerator: f64,
        recall_denominator: f64,
    ) -> Score {
        let precision = ratio(precision_numerator, precision_denominator);
        let recall = ratio(recall_numerator, recall_denominator);
        Score { precision, recall, f1: harmonic_mean(precision, recall) }
    }

    pub fn precision(&self) -> f64 {
        self.precision
    }

    pub fn recall(&self) -> f64 {
        self.recall
    }

    pub fn f1(&self) -> f64 {
        self.f1
    }
}

/// The scores a [`CorefScorer`](super::CorefScorer) has accumulated.
///
/// The CoNLL average is the mean of the MUC, B-cubed and entity-based
/// CEAF F1 scores, the official figure of the CoNLL-2011 and CoNLL-2012
/// shared tasks.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CorefScores {
    muc: Score,
    b_cubed: Score,
    ceaf_m: Score,
    ceaf_e: Score,
    mentions: Score,
    conll: f64,
}

impl CorefScores {
    /// Validated scores: `conll` in `[0, 1]` and the mean of the MUC,
    /// B-cubed and CEAF-e F1 scores.
    pub fn new(
        muc: Score,
        b_cubed: Score,
        ceaf_m: Score,
        ceaf_e: Score,
        mentions: Score,
        conll: f64,
    ) -> Result<CorefScores, String> {
        unit(conll, "conll")?;
        let expected = (muc.f1 + b_cubed.f1 + ceaf_e.f1) / 3.0;
        if (conll - expected).abs() > DERIVED_TOLERANCE {
            return Err(format!("conll must equal the coreference F1 average: {expected}"));
        }
        Ok(CorefScores { muc, b_cubed, ceaf_m, ceaf_e, mentions, conll })
    }

    pub(super) fn derived(muc: Score, b_cubed: Score, ceaf_m: Score, ceaf_e: Score, mentions: Score) -> CorefScores {
        let conll = (muc.f1 + b_cubed.f1 + ceaf_e.f1) / 3.0;
        CorefScores { muc, b_cubed, ceaf_m, ceaf_e, mentions, conll }
    }

    /// The link-based MUC score.
    pub fn muc(&self) -> Score {
        self.muc
    }

    /// The mention-based B-cubed score.
    pub fn b_cubed(&self) -> Score {
        self.b_cubed
    }

    /// The mention-based CEAF score.
    pub fn ceaf_m(&self) -> Score {
        self.ceaf_m
    }

    /// The entity-based CEAF score.
    pub fn ceaf_e(&self) -> Score {
        self.ceaf_e
    }

    /// Mention detection, comparing the mention sets alone.
    pub fn mentions(&self) -> Score {
        self.mentions
    }

    /// The CoNLL average F1.
    pub fn conll(&self) -> f64 {
        self.conll
    }
}
