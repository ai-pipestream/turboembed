//! The rules of one language: its terminators, its abbreviations and the
//! words that start a sentence after one.

use super::tables;

/// What one language changes in the rule splitter.
#[derive(Debug)]
pub struct LanguageRules {
    /// Characters that end a sentence in this language although their
    /// `Sentence_Break` value says otherwise; they are read as `STerm`.
    pub enders: &'static [char],
    /// Characters whose `Sentence_Break` value is `ATerm` or `STerm` but
    /// that do not end a sentence in this language; they are read as
    /// `SContinue`.
    pub not_enders: &'static [char],
    /// Abbreviations as the CLDR sentence break suppressions list them:
    /// case-sensitive, with their final full stop, sorted by bytes.
    pub suppressions: &'static [&'static str],
    /// Abbreviations without their final full stop, in lower case,
    /// sorted by bytes; a word matches in any case.
    pub abbreviations: &'static [&'static str],
    /// Words that start a sentence, case-sensitive, sorted by bytes. A
    /// full stop that an abbreviation or an initial would keep inside the
    /// sentence ends it when one of these follows.
    pub starters: &'static [&'static str],
    /// Whether a full stop after a number of one to three digits marks an
    /// ordinal (`am 12. Juni`) and ends no sentence when a capital letter,
    /// a digit or a dash follows.
    pub ordinal_stop: bool,
    /// The sources of the data, for the documentation and provenance.
    pub sources: &'static [Source],
}

/// Where a language's data comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Source {
    /// The CLDR 48.2 sentence break suppressions (Unicode License v3).
    Cldr,
    /// The abbreviation lists of Wikimedia sentencex (MIT).
    Sentencex,
    /// The terminators are tailored as sentencex tailors them for this
    /// language.
    Terminators,
}

const fn rules(
    suppressions: &'static [&'static str],
    abbreviations: &'static [&'static str],
    sources: &'static [Source],
) -> LanguageRules {
    LanguageRules {
        enders: &[],
        not_enders: &[],
        suppressions,
        abbreviations,
        starters: &[],
        ordinal_stop: false,
        sources,
    }
}

/// The rules of a language that has none of its own: the defaults of
/// every language and no abbreviations.
pub static ROOT: LanguageRules = rules(&[], &[], &[]);

pub static AM: LanguageRules = rules(&[], tables::SENTENCEX_AM, &[Source::Sentencex]);
pub static AR: LanguageRules = rules(&[], tables::SENTENCEX_AR, &[Source::Sentencex]);
pub static BG: LanguageRules = rules(&[], tables::SENTENCEX_BG, &[Source::Sentencex]);
pub static DA: LanguageRules = rules(&[], tables::SENTENCEX_DA, &[Source::Sentencex]);
/// German: the CLDR suppressions, and ordinals written with a full stop
/// before a capitalized noun (`am 12. Juni`).
pub static DE: LanguageRules = LanguageRules { ordinal_stop: true, ..rules(tables::CLDR_DE, &[], &[Source::Cldr]) };
pub static ES: LanguageRules = rules(tables::CLDR_ES, &[], &[Source::Cldr]);
pub static FR: LanguageRules = rules(tables::CLDR_FR, &[], &[Source::Cldr]);
pub static IT: LanguageRules = rules(tables::CLDR_IT, &[], &[Source::Cldr]);
pub static KK: LanguageRules = rules(&[], tables::SENTENCEX_KK, &[Source::Sentencex]);
pub static PL: LanguageRules = rules(&[], tables::SENTENCEX_PL, &[Source::Sentencex]);
pub static PT: LanguageRules = rules(tables::CLDR_PT, &[], &[Source::Cldr]);
pub static RU: LanguageRules = rules(tables::CLDR_RU, &[], &[Source::Cldr]);
pub static UK: LanguageRules = rules(&[], tables::SENTENCEX_UK, &[Source::Sentencex]);

/// English: the CLDR suppressions and sentencex's sentence starters.
pub static EN: LanguageRules = LanguageRules {
    starters: tables::SENTENCEX_STARTERS_EN,
    sources: &[Source::Cldr, Source::Sentencex],
    ..rules(tables::CLDR_EN, &[], &[])
};

/// Greek: the semicolon and U+037E GREEK QUESTION MARK end a question.
pub static EL: LanguageRules =
    LanguageRules { enders: &[';', '\u{37E}'], sources: &[Source::Terminators], ..rules(&[], &[], &[]) };

/// Armenian: the full stop is `։` (U+0589), which Unicode already makes a
/// terminator, and is often typed as a colon. The ASCII full stop and
/// U+2024 stand for the Armenian `mijaket`, which does not end a sentence.
pub static HY: LanguageRules = LanguageRules {
    enders: &[':'],
    not_enders: &['.', '\u{2024}'],
    sources: &[Source::Terminators],
    ..rules(&[], &[], &[])
};

/// Burmese: U+104F MYANMAR SYMBOL GENITIVE ends a sentence in formal text.
pub static MY: LanguageRules =
    LanguageRules { enders: &['\u{104F}'], sources: &[Source::Terminators], ..rules(&[], &[], &[]) };

impl LanguageRules {
    /// Whether `token`, a word with its final full stop, is a CLDR
    /// suppression.
    pub(crate) fn is_suppression(&self, token: &str) -> bool {
        self.suppressions.binary_search(&token).is_ok()
    }

    /// Whether `word`, already in lower case and without its final full
    /// stop, is an abbreviation.
    pub(crate) fn is_abbreviation(&self, word: &str) -> bool {
        self.abbreviations.binary_search(&word).is_ok()
    }

    /// Whether `word` starts sentences.
    pub(crate) fn is_starter(&self, word: &str) -> bool {
        self.starters.binary_search(&word).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::languages::LANGUAGES;

    #[test]
    fn lists_are_sorted_and_unique() {
        for l in LANGUAGES.iter().filter_map(|l| l.rules.map(|r| (l.code, r))) {
            let (code, r) = l;
            for list in [r.suppressions, r.abbreviations, r.starters] {
                assert!(list.windows(2).all(|w| w[0] < w[1]), "{code}");
            }
            assert!(r.suppressions.iter().all(|s| s.ends_with('.')), "{code}");
            assert!(r.abbreviations.iter().all(|s| !s.is_empty() && !s.ends_with('.')), "{code}");
            assert!(r.abbreviations.iter().all(|s| s.to_lowercase() == *s), "{code}");
            assert!(!r.sources.is_empty(), "{code}");
        }
    }

    #[test]
    fn lookups() {
        assert!(DE.is_suppression("z.B.") || DE.is_suppression("Dr."));
        assert!(EN.is_suppression("Mr."));
        assert!(!EN.is_suppression("mr."));
        assert!(EN.is_starter("The") && !EN.is_starter("the"));
        assert!(PL.is_abbreviation("np"));
        assert!(!ROOT.is_abbreviation("np"));
    }
}
