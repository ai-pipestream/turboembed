//! The stopword step's own layer and the languages it has lists for. It
//! lives in this small crate so a consumer of the flags (term vectors,
//! BM25) does not depend on the step that writes them.

use turbo_analysis::{Anchor, Ix, Layer};
use turbo_analysis_layers::Tokens;

/// A language OpenNLP bundles a stopword list for. The variants are in the
/// order `StopwordLists.supportedLanguages()` lists them, so the
/// discriminant is the position there.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum StopwordLanguage {
    Bulgarian,
    Danish,
    German,
    English,
    Spanish,
    Finnish,
    French,
    Italian,
    Dutch,
    Portuguese,
    Russian,
}

impl StopwordLanguage {
    /// Every value, in discriminant order.
    pub const ALL: [StopwordLanguage; 11] = [
        StopwordLanguage::Bulgarian,
        StopwordLanguage::Danish,
        StopwordLanguage::German,
        StopwordLanguage::English,
        StopwordLanguage::Spanish,
        StopwordLanguage::Finnish,
        StopwordLanguage::French,
        StopwordLanguage::Italian,
        StopwordLanguage::Dutch,
        StopwordLanguage::Portuguese,
        StopwordLanguage::Russian,
    ];

    /// The ISO 639-1 code, which names the list in OpenNLP (`en.txt`).
    pub const fn iso639_1(self) -> &'static str {
        match self {
            StopwordLanguage::Bulgarian => "bg",
            StopwordLanguage::Danish => "da",
            StopwordLanguage::German => "de",
            StopwordLanguage::English => "en",
            StopwordLanguage::Spanish => "es",
            StopwordLanguage::Finnish => "fi",
            StopwordLanguage::French => "fr",
            StopwordLanguage::Italian => "it",
            StopwordLanguage::Dutch => "nl",
            StopwordLanguage::Portuguese => "pt",
            StopwordLanguage::Russian => "ru",
        }
    }

    /// The ISO 639-2 terminologic code, which `Locale.getISO3Language`
    /// gives and OpenNLP's language detector writes (`eng`).
    pub const fn iso639_3(self) -> &'static str {
        match self {
            StopwordLanguage::Bulgarian => "bul",
            StopwordLanguage::Danish => "dan",
            StopwordLanguage::German => "deu",
            StopwordLanguage::English => "eng",
            StopwordLanguage::Spanish => "spa",
            StopwordLanguage::Finnish => "fin",
            StopwordLanguage::French => "fra",
            StopwordLanguage::Italian => "ita",
            StopwordLanguage::Dutch => "nld",
            StopwordLanguage::Portuguese => "por",
            StopwordLanguage::Russian => "rus",
        }
    }

    /// The list `StopwordLists.forLanguage(code)` returns, or `None` where
    /// it throws. It takes the ISO 639-1 code, the ISO 639-2 terminologic
    /// code, or the bibliographic codes `dut`, `fre` and `ger`, all in
    /// lower case only: OpenNLP lowercases the code after checking it
    /// against a set of lowercase codes, so `EN` is refused.
    pub fn from_code(code: &str) -> Option<StopwordLanguage> {
        let bibliographic = match code {
            "dut" => Some(StopwordLanguage::Dutch),
            "fre" => Some(StopwordLanguage::French),
            "ger" => Some(StopwordLanguage::German),
            _ => None,
        };
        bibliographic.or_else(|| Self::ALL.into_iter().find(|l| l.iso639_1() == code || l.iso639_3() == code))
    }
}

/// Whether each token is a stopword of one bundled list: item i is token
/// i. Tokens are flagged, never removed; what OpenNLP's
/// `StopwordFilter.filter` would keep is [`Stopwords::kept`].
pub struct Stopwords {
    /// The list the flags come from.
    pub language: StopwordLanguage,
    pub stop: Vec<bool>,
}

impl Stopwords {
    /// The tokens that are not stopwords, in order. For the bundled lists,
    /// which hold single words only, these are the tokens OpenNLP's
    /// `StopwordFilter.filter`, `StopwordFilteringTokenizer` and
    /// `StopwordFilterStream` keep.
    pub fn kept(&self) -> impl Iterator<Item = Ix<Tokens>> + '_ {
        self.stop.iter().enumerate().filter(|&(_, &s)| !s).map(|(i, _)| Ix::new(i as u32))
    }
}

impl Layer for Stopwords {
    const ID: &'static str = "turbo:stopwords";

    fn anchor() -> Anchor {
        Anchor::aligned::<Tokens>()
    }

    fn len(&self) -> usize {
        self.stop.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_follow_the_supported_languages() {
        let codes: Vec<&str> = StopwordLanguage::ALL.iter().map(|l| l.iso639_1()).collect();
        assert_eq!(codes, ["bg", "da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "ru"]);
        for (i, l) in StopwordLanguage::ALL.into_iter().enumerate() {
            assert_eq!(l as usize, i);
        }
        assert_eq!(std::mem::size_of::<StopwordLanguage>(), 1);
    }

    #[test]
    fn codes_resolve_as_opennlp_resolves_them() {
        for l in StopwordLanguage::ALL {
            assert_eq!(StopwordLanguage::from_code(l.iso639_1()), Some(l));
            assert_eq!(StopwordLanguage::from_code(l.iso639_3()), Some(l));
        }
        assert_eq!(StopwordLanguage::from_code("ger"), Some(StopwordLanguage::German));
        assert_eq!(StopwordLanguage::from_code("dut"), Some(StopwordLanguage::Dutch));
        assert_eq!(StopwordLanguage::from_code("fre"), Some(StopwordLanguage::French));
        for refused in ["EN", "Eng", "en-US", "", "sv", "und", "x-unspecified", "eng "] {
            assert_eq!(StopwordLanguage::from_code(refused), None, "{refused}");
        }
    }

    #[test]
    fn aligned_to_tokens_and_keeps_the_unflagged() {
        assert_eq!(Stopwords::anchor(), Anchor::aligned::<Tokens>());
        assert_eq!(Stopwords::ID, "turbo:stopwords");
        let flags = Stopwords { language: StopwordLanguage::English, stop: vec![true, false, true, false] };
        assert_eq!(flags.len(), 4);
        assert_eq!(flags.kept().map(Ix::get).collect::<Vec<_>>(), [1, 3]);
    }
}
