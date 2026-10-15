//! Coarse parts of speech recognized by WordNet lexicons (`WordNetPOS`).

use crate::error::WordNetError;

/// Coarse parts of speech recognized by a WordNet lexicon.
///
/// Adjective satellites (`s` in WNDB and WN-LMF) are normalized to
/// [`WordNetPos::Adjective`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WordNetPos {
    /// Noun (`n`).
    Noun,
    /// Verb (`v`).
    Verb,
    /// Adjective (`a` or satellite `s`).
    Adjective,
    /// Adverb (`r`).
    Adverb,
}

impl WordNetPos {
    /// All four WordNet parts of speech in declaration order.
    pub const ALL: [Self; 4] = [Self::Noun, Self::Verb, Self::Adjective, Self::Adverb];

    /// Returns the single-character WNDB / WN-LMF part-of-speech code (`n`, `v`, `a`, `r`).
    #[inline]
    pub const fn code(self) -> char {
        match self {
            Self::Noun => 'n',
            Self::Verb => 'v',
            Self::Adjective => 'a',
            Self::Adverb => 'r',
        }
    }

    /// Returns the WNDB file suffix (`noun`, `verb`, `adj`, `adv`).
    #[inline]
    pub const fn file_suffix(self) -> &'static str {
        match self {
            Self::Noun => "noun",
            Self::Verb => "verb",
            Self::Adjective => "adj",
            Self::Adverb => "adv",
        }
    }

    /// Resolves a WNDB or WN-LMF single-character part-of-speech code (`n`, `v`, `a`, `s`, `r`).
    ///
    /// Adjective satellites (`s`) normalize to [`WordNetPos::Adjective`].
    #[inline]
    pub const fn from_code(code: char) -> Option<Self> {
        match code {
            'n' => Some(Self::Noun),
            'v' => Some(Self::Verb),
            'a' | 's' => Some(Self::Adjective),
            'r' => Some(Self::Adverb),
            _ => None,
        }
    }

    /// Returns the Java enum name (`NOUN`, `VERB`, `ADJECTIVE`, `ADVERB`).
    #[inline]
    pub const fn java_name(self) -> &'static str {
        match self {
            Self::Noun => "NOUN",
            Self::Verb => "VERB",
            Self::Adjective => "ADJECTIVE",
            Self::Adverb => "ADVERB",
        }
    }

    /// Resolves a WNDB or WN-LMF single-character part-of-speech code, returning an error if unknown.
    #[inline]
    pub fn try_from_code(code: char) -> Result<Self, WordNetError> {
        Self::from_code(code)
            .ok_or_else(|| WordNetError::invalid(format!("Unknown WordNet POS code: '{code}'")))
    }
}

impl std::fmt::Display for WordNetPos {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.java_name())
    }
}
