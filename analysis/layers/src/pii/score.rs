//! An optional numeric score on a mention. Absent is not zero: a rule
//! detector records no score, a learned head records one with its model
//! and whether it is calibrated.

use turbo_analysis::Fnv1a;

use super::PiiDetectorRef;

/// What a [`PiiScore::value`] is. [`PiiScoreKind::id`] is the stable
/// number shared by the codec and the proto enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiScoreKind {
    /// A probability in `0..=1`.
    Probability,
    /// A raw logit, any finite value.
    Logit,
    /// A heuristic risk score in `0..=1`, labelled as such: not a
    /// probability.
    Heuristic,
}

impl PiiScoreKind {
    pub const ALL: [PiiScoreKind; 3] = [PiiScoreKind::Probability, PiiScoreKind::Logit, PiiScoreKind::Heuristic];

    /// The stable id: probability 1, logit 2, heuristic 3.
    pub const fn id(self) -> u8 {
        match self {
            PiiScoreKind::Probability => 1,
            PiiScoreKind::Logit => 2,
            PiiScoreKind::Heuristic => 3,
        }
    }

    pub const fn from_id(id: u32) -> Option<PiiScoreKind> {
        match id {
            1 => Some(PiiScoreKind::Probability),
            2 => Some(PiiScoreKind::Logit),
            3 => Some(PiiScoreKind::Heuristic),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            PiiScoreKind::Probability => "probability",
            PiiScoreKind::Logit => "logit",
            PiiScoreKind::Heuristic => "heuristic",
        }
    }
}

/// A numeric score with its meaning and its source.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PiiScore {
    pub value: f32,
    pub kind: PiiScoreKind,
    /// The model that scored, an entry of the layer's [`super::PiiDetectors`].
    pub model: PiiDetectorRef,
    /// Whether `value` was calibrated against held-out data.
    pub calibrated: bool,
}

impl PiiScore {
    /// Finite, and in `0..=1` unless a logit.
    pub fn check(&self) -> Result<(), String> {
        if !self.value.is_finite() {
            return Err("pii score is not finite".into());
        }
        if self.kind != PiiScoreKind::Logit && !(0.0..=1.0).contains(&self.value) {
            return Err(format!("pii {} score is outside 0..=1", self.kind.name()));
        }
        Ok(())
    }

    pub fn fold_hash(&self, h: &mut Fnv1a) {
        h.f32(self.value).u8(self.kind.id()).u16(self.model.index()).bool(self.calibrated);
    }
}
