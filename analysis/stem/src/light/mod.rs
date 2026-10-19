//! The light and minimal stemmers (`opennlp.tools.stemmer.light`), which
//! OpenNLP adapted from Lucene: short lists of endings, removed without a
//! dictionary.
//!
//! Each expects lower-case, NFC input, as OpenNLP documents: none folds
//! case, so `Häuser` keeps its capital, and a decomposed accent is two
//! code units the rules do not look through. Most of them also fold some
//! accented letters to plain ones (German ä to a, French é to e), each its
//! own set.

mod english_minimal;
mod finnish_light;
mod french_light;
mod french_minimal;
mod german_light;
mod german_minimal;
mod hungarian_light;
mod italian_light;
mod norwegian_light;
mod norwegian_minimal;
mod portuguese_light;
mod russian_light;
mod spanish_light;
mod spanish_minimal;
mod swedish_light;
mod swedish_minimal;
mod util;

use crate::stemmer::Stemmer;

/// The written standards of Norwegian a Norwegian stemmer removes the
/// endings of. OpenNLP's constructors take a non-empty set of
/// `NorwegianVariety`; these are its three possible values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NorwegianStandard {
    Bokmaal,
    Nynorsk,
    Both,
}

impl NorwegianStandard {
    pub const fn bokmaal(self) -> bool {
        matches!(self, NorwegianStandard::Bokmaal | NorwegianStandard::Both)
    }

    pub const fn nynorsk(self) -> bool {
        matches!(self, NorwegianStandard::Nynorsk | NorwegianStandard::Both)
    }
}

/// One light or minimal stemmer: the OpenNLP class of the same name.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum LightStemmer {
    EnglishMinimal,
    FinnishLight,
    FrenchLight,
    FrenchMinimal,
    GermanLight,
    GermanMinimal,
    HungarianLight,
    ItalianLight,
    NorwegianLight(NorwegianStandard),
    NorwegianMinimal(NorwegianStandard),
    PortugueseLight,
    RussianLight,
    SpanishLight,
    SpanishMinimal,
    SwedishLight,
    SwedishMinimal,
}

impl LightStemmer {
    /// The Java class name.
    pub const fn java_name(self) -> &'static str {
        match self {
            LightStemmer::EnglishMinimal => "EnglishMinimalStemmer",
            LightStemmer::FinnishLight => "FinnishLightStemmer",
            LightStemmer::FrenchLight => "FrenchLightStemmer",
            LightStemmer::FrenchMinimal => "FrenchMinimalStemmer",
            LightStemmer::GermanLight => "GermanLightStemmer",
            LightStemmer::GermanMinimal => "GermanMinimalStemmer",
            LightStemmer::HungarianLight => "HungarianLightStemmer",
            LightStemmer::ItalianLight => "ItalianLightStemmer",
            LightStemmer::NorwegianLight(_) => "NorwegianLightStemmer",
            LightStemmer::NorwegianMinimal(_) => "NorwegianMinimalStemmer",
            LightStemmer::PortugueseLight => "PortugueseLightStemmer",
            LightStemmer::RussianLight => "RussianLightStemmer",
            LightStemmer::SpanishLight => "SpanishLightStemmer",
            LightStemmer::SpanishMinimal => "SpanishMinimalStemmer",
            LightStemmer::SwedishLight => "SwedishLightStemmer",
            LightStemmer::SwedishMinimal => "SwedishMinimalStemmer",
        }
    }

    /// Stems the word in `s` in place and returns the stem's length; the
    /// Java classes' `stem(char[], int)`.
    fn stem_in_place(self, s: &mut [u16]) -> usize {
        let len = s.len();
        match self {
            LightStemmer::EnglishMinimal => english_minimal::stem(s, len),
            LightStemmer::FinnishLight => finnish_light::stem(s, len),
            LightStemmer::FrenchLight => french_light::stem(s, len),
            LightStemmer::FrenchMinimal => french_minimal::stem(s, len),
            LightStemmer::GermanLight => german_light::stem(s, len),
            LightStemmer::GermanMinimal => german_minimal::stem(s, len),
            LightStemmer::HungarianLight => hungarian_light::stem(s, len),
            LightStemmer::ItalianLight => italian_light::stem(s, len),
            LightStemmer::NorwegianLight(standard) => norwegian_light::stem(s, len, standard),
            LightStemmer::NorwegianMinimal(standard) => norwegian_minimal::stem(s, len, standard),
            LightStemmer::PortugueseLight => portuguese_light::stem(s, len),
            LightStemmer::RussianLight => russian_light::stem(s, len),
            LightStemmer::SpanishLight => spanish_light::stem(s, len),
            LightStemmer::SpanishMinimal => spanish_minimal::stem(s, len),
            LightStemmer::SwedishLight => swedish_light::stem(s, len),
            LightStemmer::SwedishMinimal => swedish_minimal::stem(s, len),
        }
    }
}

impl Stemmer for LightStemmer {
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>) {
        out.clear();
        out.extend_from_slice(word);
        let len = self.stem_in_place(out);
        out.truncate(len);
    }

    fn config_hash(&self) -> u64 {
        crate::Algorithm::Light(*self).config_hash()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standards() {
        assert!(NorwegianStandard::Both.bokmaal() && NorwegianStandard::Both.nynorsk());
        assert!(!NorwegianStandard::Bokmaal.nynorsk() && !NorwegianStandard::Nynorsk.bokmaal());
    }

    #[test]
    fn stems_a_few_words() {
        assert_eq!(LightStemmer::EnglishMinimal.stem("ponies"), "pony");
        assert_eq!(LightStemmer::EnglishMinimal.stem("aies"), "aies");
        assert_eq!(LightStemmer::GermanMinimal.stem("häuser"), "haus");
        assert_eq!(LightStemmer::NorwegianMinimal(NorwegianStandard::Nynorsk).stem("gutar"), "gut");
        assert_eq!(LightStemmer::NorwegianMinimal(NorwegianStandard::Bokmaal).stem("gutar"), "gutar");
        assert_eq!(LightStemmer::SpanishMinimal.stem("crisis"), "crisis");
        assert_eq!(LightStemmer::SpanishMinimal.stem("relojes"), "reloj");
    }
}
