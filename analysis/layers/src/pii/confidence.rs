use super::PiiEvidence;

/// How much the text supports a mention's kind: a summary derived from the
/// mention's [`PiiEvidence`] by [`PiiConfidence::derive`], never a policy
/// order. "Has a label" and "a checksum passed" are independent facts.
/// [`PiiConfidence::id`] is the stable number shared by the codec, the
/// proto enum, the C `TURBO_PII_CONFIDENCE_*` constants and Java.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PiiConfidence {
    /// No passed checksum and no label: the shape alone.
    Format,
    /// A context label within the window names the kind (`Passport:`,
    /// `SSN`), and no checksum passed.
    Labelled,
    /// A checksum or check digit over the value passed: Luhn, mod-97,
    /// mod-11, ISO 7064, base58check, bech32, EIP-55, the ABA weights.
    Checksum,
    /// A context label within the window names the kind (`IMEI:`, `SIN`),
    /// and a check digit passed.
    LabelledChecksum,
}

impl PiiConfidence {
    /// Every tier, in id order.
    pub const ALL: [PiiConfidence; 4] =
        [PiiConfidence::Format, PiiConfidence::Checksum, PiiConfidence::LabelledChecksum, PiiConfidence::Labelled];

    /// The stable id: format 1, checksum 2, labelled-checksum 3, labelled 4.
    /// 0 is the proto's `UNSPECIFIED` and is never a tier.
    pub const fn id(self) -> u8 {
        match self {
            PiiConfidence::Format => 1,
            PiiConfidence::Checksum => 2,
            PiiConfidence::LabelledChecksum => 3,
            PiiConfidence::Labelled => 4,
        }
    }

    /// The tier with stable id `id`.
    pub const fn from_id(id: u32) -> Option<PiiConfidence> {
        match id {
            1 => Some(PiiConfidence::Format),
            2 => Some(PiiConfidence::Checksum),
            3 => Some(PiiConfidence::LabelledChecksum),
            4 => Some(PiiConfidence::Labelled),
            _ => None,
        }
    }

    /// The tier named `name`.
    pub fn parse(name: &str) -> Option<PiiConfidence> {
        PiiConfidence::ALL.into_iter().find(|c| c.name() == name)
    }

    pub const fn name(self) -> &'static str {
        match self {
            PiiConfidence::Format => "format",
            PiiConfidence::Labelled => "labelled",
            PiiConfidence::Checksum => "checksum",
            PiiConfidence::LabelledChecksum => "labelled-checksum",
        }
    }

    /// The derivation table: a passed checksum fact and a label fact,
    /// each present or not, give the four tiers. A failed checksum counts
    /// for nothing.
    pub fn derive(evidence: &[PiiEvidence]) -> PiiConfidence {
        let checksum = evidence.iter().any(|e| e.is_checksum_passed());
        let label = evidence.iter().any(|e| e.is_label());
        match (checksum, label) {
            (false, false) => PiiConfidence::Format,
            (false, true) => PiiConfidence::Labelled,
            (true, false) => PiiConfidence::Checksum,
            (true, true) => PiiConfidence::LabelledChecksum,
        }
    }
}
