/// How much the text supports a mention's kind.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PiiConfidence {
    /// The shape alone: the syntax of an address, a key or a token, a
    /// registered domain or calling code, allocated number ranges. No
    /// check digit backs it.
    Format,
    /// A context label within the window names the kind (`Passport:`,
    /// `SSN`), and the value matches the kind's format without a check
    /// digit.
    Labelled,
    /// A checksum or check digit over the value passes: Luhn, mod-97,
    /// mod-11, ISO 7064, base58check, bech32, EIP-55, the ABA weights.
    Checksum,
    /// A context label within the window names the kind (`IMEI:`, `SIN`),
    /// and its check digit passes.
    LabelledChecksum,
}

impl PiiConfidence {
    pub const fn name(self) -> &'static str {
        match self {
            PiiConfidence::Format => "format",
            PiiConfidence::Labelled => "labelled",
            PiiConfidence::Checksum => "checksum",
            PiiConfidence::LabelledChecksum => "labelled-checksum",
        }
    }

    /// Raises `Format` to `Labelled` and `Checksum` to `LabelledChecksum`.
    pub const fn with_label(self) -> PiiConfidence {
        match self {
            PiiConfidence::Format | PiiConfidence::Labelled => PiiConfidence::Labelled,
            PiiConfidence::Checksum | PiiConfidence::LabelledChecksum => PiiConfidence::LabelledChecksum,
        }
    }
}
