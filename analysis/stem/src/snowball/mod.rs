//! The Snowball stemmers OpenNLP ships (`opennlp.tools.stemmer.snowball`).
//!
//! The code in `algorithms/` is generated from the Snowball sources at the
//! commit OpenNLP's Java classes were generated from, by the Snowball
//! compiler's Rust backend changed to work on UTF-16 code units as the Java
//! backend does; `snowball/README.md` in this crate says how and why.

use std::num::NonZeroU32;

use crate::stemmer::Stemmer;

pub(crate) mod runtime;

// The generated code's own style: labeled loops that run once, explicit
// returns and casts, references to statics.
#[allow(
    clippy::collapsible_match,
    clippy::identity_op,
    clippy::needless_borrow,
    clippy::needless_return,
    clippy::never_loop,
    clippy::redundant_static_lifetimes,
    clippy::single_match,
    clippy::unnecessary_cast
)]
mod algorithms {
    pub mod arabic;
    pub mod catalan;
    pub mod danish;
    pub mod dutch;
    pub mod english;
    pub mod finnish;
    pub mod french;
    pub mod german;
    pub mod greek;
    pub mod hungarian;
    pub mod indonesian;
    pub mod irish;
    pub mod italian;
    pub mod norwegian;
    pub mod porter;
    pub mod portuguese;
    pub mod romanian;
    pub mod russian;
    pub mod spanish;
    pub mod swedish;
    pub mod turkish;
}

use runtime::SnowballEnv;

/// `SnowballStemmer.ALGORITHM`: one Snowball stemmer per language, and
/// Porter's original English algorithm as Snowball writes it. The variants
/// are in the Java order, so the discriminant is the Java ordinal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SnowballAlgorithm {
    Arabic,
    Danish,
    Dutch,
    Catalan,
    English,
    Finnish,
    French,
    German,
    Greek,
    Hungarian,
    Indonesian,
    Irish,
    Italian,
    Norwegian,
    /// Snowball's `porter.sbl`. OpenNLP's `PorterStemmer` is a separate,
    /// hand-written implementation: [`crate::PorterStemmer`].
    Porter,
    Portuguese,
    Romanian,
    Russian,
    Spanish,
    Swedish,
    Turkish,
}

impl SnowballAlgorithm {
    /// Every value, in discriminant order.
    pub const ALL: [SnowballAlgorithm; 21] = [
        SnowballAlgorithm::Arabic,
        SnowballAlgorithm::Danish,
        SnowballAlgorithm::Dutch,
        SnowballAlgorithm::Catalan,
        SnowballAlgorithm::English,
        SnowballAlgorithm::Finnish,
        SnowballAlgorithm::French,
        SnowballAlgorithm::German,
        SnowballAlgorithm::Greek,
        SnowballAlgorithm::Hungarian,
        SnowballAlgorithm::Indonesian,
        SnowballAlgorithm::Irish,
        SnowballAlgorithm::Italian,
        SnowballAlgorithm::Norwegian,
        SnowballAlgorithm::Porter,
        SnowballAlgorithm::Portuguese,
        SnowballAlgorithm::Romanian,
        SnowballAlgorithm::Russian,
        SnowballAlgorithm::Spanish,
        SnowballAlgorithm::Swedish,
        SnowballAlgorithm::Turkish,
    ];

    /// The Java enum constant's name.
    pub const fn java_name(self) -> &'static str {
        match self {
            SnowballAlgorithm::Arabic => "ARABIC",
            SnowballAlgorithm::Danish => "DANISH",
            SnowballAlgorithm::Dutch => "DUTCH",
            SnowballAlgorithm::Catalan => "CATALAN",
            SnowballAlgorithm::English => "ENGLISH",
            SnowballAlgorithm::Finnish => "FINNISH",
            SnowballAlgorithm::French => "FRENCH",
            SnowballAlgorithm::German => "GERMAN",
            SnowballAlgorithm::Greek => "GREEK",
            SnowballAlgorithm::Hungarian => "HUNGARIAN",
            SnowballAlgorithm::Indonesian => "INDONESIAN",
            SnowballAlgorithm::Irish => "IRISH",
            SnowballAlgorithm::Italian => "ITALIAN",
            SnowballAlgorithm::Norwegian => "NORWEGIAN",
            SnowballAlgorithm::Porter => "PORTER",
            SnowballAlgorithm::Portuguese => "PORTUGUESE",
            SnowballAlgorithm::Romanian => "ROMANIAN",
            SnowballAlgorithm::Russian => "RUSSIAN",
            SnowballAlgorithm::Spanish => "SPANISH",
            SnowballAlgorithm::Swedish => "SWEDISH",
            SnowballAlgorithm::Turkish => "TURKISH",
        }
    }

    /// The value with the given Java name, which must match exactly.
    pub fn from_java_name(name: &str) -> Option<SnowballAlgorithm> {
        SnowballAlgorithm::ALL.into_iter().find(|a| a.java_name() == name)
    }

    /// The generated stemmer's entry point.
    fn program(self) -> fn(&mut SnowballEnv) -> bool {
        use algorithms::*;
        match self {
            SnowballAlgorithm::Arabic => arabic::stem,
            SnowballAlgorithm::Danish => danish::stem,
            SnowballAlgorithm::Dutch => dutch::stem,
            SnowballAlgorithm::Catalan => catalan::stem,
            SnowballAlgorithm::English => english::stem,
            SnowballAlgorithm::Finnish => finnish::stem,
            SnowballAlgorithm::French => french::stem,
            SnowballAlgorithm::German => german::stem,
            SnowballAlgorithm::Greek => greek::stem,
            SnowballAlgorithm::Hungarian => hungarian::stem,
            SnowballAlgorithm::Indonesian => indonesian::stem,
            SnowballAlgorithm::Irish => irish::stem,
            SnowballAlgorithm::Italian => italian::stem,
            SnowballAlgorithm::Norwegian => norwegian::stem,
            SnowballAlgorithm::Porter => porter::stem,
            SnowballAlgorithm::Portuguese => portuguese::stem,
            SnowballAlgorithm::Romanian => romanian::stem,
            SnowballAlgorithm::Russian => russian::stem,
            SnowballAlgorithm::Spanish => spanish::stem,
            SnowballAlgorithm::Swedish => swedish::stem,
            SnowballAlgorithm::Turkish => turkish::stem,
        }
    }
}

/// `SnowballStemmer`: one Snowball algorithm, applied `repeat` times to
/// each word.
///
/// As in OpenNLP, the word is set once and the algorithm then runs
/// `repeat` times on the state the previous run left, including its
/// cursor and limits, not on a fresh copy of the previous run's output.
/// The second run therefore does not always stem the first run's stem as
/// a new word would be stemmed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnowballStemmer {
    algorithm: SnowballAlgorithm,
    repeat: NonZeroU32,
}

impl SnowballStemmer {
    /// The algorithm applied once, as `new SnowballStemmer(algorithm)`.
    pub const fn new(algorithm: SnowballAlgorithm) -> SnowballStemmer {
        SnowballStemmer { algorithm, repeat: NonZeroU32::MIN }
    }

    /// The algorithm applied `repeat` times, as
    /// `new SnowballStemmer(algorithm, repeat)`.
    pub const fn with_repeat(algorithm: SnowballAlgorithm, repeat: NonZeroU32) -> SnowballStemmer {
        SnowballStemmer { algorithm, repeat }
    }

    pub const fn algorithm(&self) -> SnowballAlgorithm {
        self.algorithm
    }

    pub const fn repeat(&self) -> NonZeroU32 {
        self.repeat
    }
}

impl Stemmer for SnowballStemmer {
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>) {
        let mut buf = std::mem::take(out);
        buf.clear();
        buf.extend_from_slice(word);
        let mut env = SnowballEnv::new(buf);
        let program = self.algorithm.program();
        for _ in 0..self.repeat.get() {
            program(&mut env);
        }
        *out = env.into_current();
    }

    fn config_hash(&self) -> u64 {
        crate::Algorithm::Snowball(*self).config_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_are_java_ordinals() {
        for (i, a) in SnowballAlgorithm::ALL.into_iter().enumerate() {
            assert_eq!(a as usize, i);
            assert_eq!(SnowballAlgorithm::from_java_name(a.java_name()), Some(a));
        }
        assert_eq!(SnowballAlgorithm::from_java_name("english"), None);
    }

    #[test]
    fn stems_a_few_words() {
        let english = SnowballStemmer::new(SnowballAlgorithm::English);
        assert_eq!(english.stem("generalizations"), "general");
        assert_eq!(english.stem("running"), "run");
        let german = SnowballStemmer::new(SnowballAlgorithm::German);
        assert_eq!(german.stem("häuser"), "haus");
    }
}
