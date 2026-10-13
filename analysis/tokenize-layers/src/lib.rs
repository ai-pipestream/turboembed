//! The tokenizers' own layer: the word type of each token the UAX #29
//! word tokenizer writes. It lives in this small crate so a consumer of
//! word types does not depend on the tokenizers that make them.

use turbo_analysis::{Anchor, Layer};
use turbo_analysis_layers::Tokens;

/// What kind of word a token is, as OpenNLP's
/// `opennlp.tools.tokenize.uax29.WordType` defines it. The variants are
/// in the Java order, so the discriminant is the Java ordinal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum WordType {
    /// Holds at least one letter, possibly mixed with digits and connectors.
    Alphanumeric,
    /// Digits and numeric connectors only.
    Numeric,
    /// Holds a Han ideograph.
    Ideographic,
    /// Hiragana.
    Hiragana,
    /// Katakana.
    Katakana,
    /// Hangul.
    Hangul,
    /// A Southeast Asian script that needs dictionary segmentation: Thai,
    /// Lao, Myanmar, Khmer, Tai Le, New Tai Lue, Tai Viet.
    SoutheastAsian,
    /// An emoji, an emoji sequence or a regional indicator flag.
    Emoji,
}

impl WordType {
    /// Every value, in discriminant order.
    pub const ALL: [WordType; 8] = [
        WordType::Alphanumeric,
        WordType::Numeric,
        WordType::Ideographic,
        WordType::Hiragana,
        WordType::Katakana,
        WordType::Hangul,
        WordType::SoutheastAsian,
        WordType::Emoji,
    ];

    /// The Java enum constant's name, as golden files print it.
    pub const fn java_name(self) -> &'static str {
        match self {
            WordType::Alphanumeric => "ALPHANUMERIC",
            WordType::Numeric => "NUMERIC",
            WordType::Ideographic => "IDEOGRAPHIC",
            WordType::Hiragana => "HIRAGANA",
            WordType::Katakana => "KATAKANA",
            WordType::Hangul => "HANGUL",
            WordType::SoutheastAsian => "SOUTHEAST_ASIAN",
            WordType::Emoji => "EMOJI",
        }
    }

    /// The value with the given Java name.
    pub fn from_java_name(name: &str) -> Option<WordType> {
        WordType::ALL.into_iter().find(|t| t.java_name() == name)
    }
}

/// The word type of every token: item i is the type of token i.
pub struct WordTypes {
    pub kind: Vec<WordType>,
}

impl Layer for WordTypes {
    const ID: &'static str = "turbo:word-types";

    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }

    fn len(&self) -> usize {
        self.kind.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_are_java_ordinals() {
        for (i, t) in WordType::ALL.into_iter().enumerate() {
            assert_eq!(t as usize, i);
            assert_eq!(WordType::from_java_name(t.java_name()), Some(t));
        }
        assert_eq!(WordType::SoutheastAsian.java_name(), "SOUTHEAST_ASIAN");
        assert_eq!(WordType::from_java_name("Emoji"), None);
        assert_eq!(std::mem::size_of::<WordType>(), 1);
    }

    #[test]
    fn aligned_to_tokens() {
        assert_eq!(WordTypes::anchor(), Anchor::aligned::<Tokens>());
        let types = WordTypes { kind: vec![WordType::Numeric, WordType::Emoji] };
        assert_eq!(types.len(), 2);
        assert_eq!(WordTypes::ID, "turbo:word-types");
    }
}
