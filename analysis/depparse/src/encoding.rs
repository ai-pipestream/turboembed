//! Explicit relation encoding, separate from the label text.
use crate::{Result, error};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LabelEncoding {
    /// Ordinary labels, including arbitrary carets and backslashes.
    #[default]
    Plain,
    /// Original UD release directional paths, with no escaped custom labels.
    LegacyPseudo,
    /// Every base/path label is escaped before directional paths are added.
    EscapedV1,
}
impl LabelEncoding {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::LegacyPseudo => "pseudo-legacy",
            Self::EscapedV1 => "pseudo-escaped-v1",
        }
    }
    pub(crate) fn read(s: &str) -> Result<Self> {
        match s {
            "plain" => Ok(Self::Plain),
            "pseudo-legacy" => Ok(Self::LegacyPseudo),
            "pseudo-escaped-v1" => Ok(Self::EscapedV1),
            _ => Err(error("unsupported dependency label encoding")),
        }
    }
    /// Only the exact eight unmarked artifacts published in
    /// models-ud-depparse-2.16-1 use the historical automatic compatibility path.
    /// Other unmarked models retain ordinary labels.
    pub(crate) fn historical(bytes: &[u8]) -> Self {
        if RELEASE_SHA256.contains(&turbo_maxent::sha256(bytes).to_hex().as_str()) {
            Self::LegacyPseudo
        } else {
            Self::Plain
        }
    }
}
const RELEASE_SHA256: &[&str] = &[
    // turbo-ca-ud-ancora-2.16-depparse-feedforward-1
    "8af62c85bd4f3ab42498ee8b217911ff55101df2caab554391120ab2dbb45455",
    // turbo-ca-ud-ancora-2.16-depparse-maxent-1
    "33e2ec6844fabfa173bf0e0eafca62eaf259f95622e1c149a2f15197782c27f6",
    // turbo-es-ud-ancora-2.16-depparse-feedforward-1
    "68aea2dfec51c015ce8ac99ff4c15875d84a364ef9bfe9fe3b5a4976c0b0f8a7",
    // turbo-es-ud-ancora-2.16-depparse-maxent-1
    "6d399e33ee858d0ab87ebc6fa9438a445f614b6729f8a609ab5ffeabfa538f63",
    // turbo-fi-ud-ftb-2.16-depparse-feedforward-1
    "3df4a7db92fb89759aa964153cc7a55154779cf25a56250cc8b2b2924ce9870d",
    // turbo-fi-ud-ftb-2.16-depparse-maxent-1
    "755ef066e10c65adc3d17f93bb15b35ebee78fbd189fd56297f12409bf5f782d",
    // turbo-it-ud-markit-2.16-depparse-feedforward-1
    "397fc29dd5ee2ac576400f38742a38023d4b23abe09df351dd18c449094f2b3e",
    // turbo-it-ud-markit-2.16-depparse-maxent-1
    "2f8539971c1d4434bde2b5e16aff8348c8ef7841d2a5576b180ed23e8f70da93",
];
