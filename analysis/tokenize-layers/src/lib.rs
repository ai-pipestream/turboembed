//! The tokenizers' own layers: the word type of each token the UAX #29
//! word tokenizer writes, and the readings, pronunciations and lemmas of
//! the tokens a MeCab dictionary tokenizer writes. They live in this small
//! crate so a consumer of them does not depend on the tokenizers that make
//! them.

pub mod export;

use turbo_analysis::{Anchor, Layer};
use turbo_analysis_layers::{Strs, Tokens};

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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
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

    fn check(&self) -> Result<(), String> {
        if u32::try_from(self.kind.len()).is_err() {
            return Err(format!("{} word types do not fit in u32", self.kind.len()));
        }
        Ok(())
    }
}

/// What a MeCab dictionary says of each token, from the feature columns of
/// the token's entry: item i is token i's. A column is `None` when the
/// dictionary's format has no such column (mecab-ko-dic has no lemma), and
/// an empty string where the token's entry leaves it out: an unknown word,
/// whose `unk.def` template has fewer columns, or the dictionary's `*`.
#[derive(Clone, Debug, Default)]
pub struct Readings {
    /// How the token is read: katakana for a Japanese dictionary, Hangul
    /// for a Korean one.
    pub reading: Option<Strs>,
    /// How the token is pronounced, in katakana, where it differs from the
    /// reading by long vowels (`ー`) and the like.
    pub pronunciation: Option<Strs>,
    /// The base form of an inflected token, as written.
    pub lemma: Option<Strs>,
}

fn strs_eq(a: &Option<Strs>, b: &Option<Strs>) -> bool {
    match (a, b) {
        (Some(x), Some(y)) => x.len() == y.len() && x.iter().eq(y.iter()),
        (None, None) => true,
        _ => false,
    }
}

impl PartialEq for Readings {
    fn eq(&self, other: &Self) -> bool {
        strs_eq(&self.reading, &other.reading)
            && strs_eq(&self.pronunciation, &other.pronunciation)
            && strs_eq(&self.lemma, &other.lemma)
    }
}

impl Eq for Readings {}

impl Readings {
    fn columns(&self) -> impl Iterator<Item = &Strs> {
        [&self.reading, &self.pronunciation, &self.lemma].into_iter().flatten()
    }
}

impl Layer for Readings {
    const ID: &'static str = "turbo:readings";

    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }

    /// The length of the first column present.
    fn len(&self) -> usize {
        self.columns().next().map_or(0, Strs::len)
    }

    /// At least one column, all of one length that fits in `u32`.
    fn check(&self) -> Result<(), String> {
        let n = self.len();
        if self.columns().next().is_none() {
            return Err("a reading, pronunciation or lemma column".into());
        }
        if self.columns().any(|c| c.len() != n) {
            return Err("the reading, pronunciation and lemma columns of one length".into());
        }
        if u32::try_from(n).is_err() {
            return Err(format!("{n} readings do not fit in u32"));
        }
        Ok(())
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
        assert_eq!(types.clone(), types);
        assert_eq!(WordTypes::default().len(), 0);
        assert!(types.check().is_ok());
        assert_eq!(types.len(), 2);
        assert_eq!(WordTypes::ID, "turbo:word-types");
    }

    fn strs(items: &[&str]) -> Strs {
        let mut s = Strs::default();
        items.iter().for_each(|i| s.push(i));
        s
    }

    #[test]
    fn readings_are_aligned_to_tokens_and_checked() {
        assert_eq!(Readings::ID, "turbo:readings");
        assert_eq!(Readings::anchor(), Anchor::aligned::<Tokens>());
        let r = Readings { reading: None, pronunciation: Some(strs(&["a", ""])), lemma: Some(strs(&["x", "y"])) };
        assert_eq!(r.clone(), r);
        assert_eq!(r.len(), 2);
        assert!(r.check().is_ok());
        let r = Readings { reading: Some(strs(&["a"])), pronunciation: None, lemma: Some(strs(&["x", "y"])) };
        assert!(r.check().is_err());
        let r = Readings::default();
        assert_eq!(r.len(), 0);
        assert!(r.check().is_err());
    }
}
