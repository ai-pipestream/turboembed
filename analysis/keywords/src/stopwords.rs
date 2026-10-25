//! The stopword seam YAKE excludes candidates with: one of OpenNLP's
//! eleven bundled lists, or the caller's own words.
//!
//! YAKE checks the folded term, so a caller list holds folded forms; the
//! bundled lists lowercase the term they are asked about, exactly
//! `StopwordFilter.isStopword`.

use std::fmt;
use std::sync::Arc;

use turbo_stopword::{StopwordLanguage, StopwordList};

/// Why a caller stopword list was refused.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StopwordError {
    /// The caller handed an empty list.
    Empty,
}

impl fmt::Display for StopwordError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            StopwordError::Empty => "a stopword list with no words",
        })
    }
}

impl std::error::Error for StopwordError {}

/// Which words YAKE excludes from candidates (their edges, that is;
/// interior stop words stay, with the correction the scoring applies).
#[derive(Clone, Debug)]
pub enum Stopwords {
    /// One of the lists OpenNLP bundles ([`StopwordList::bundled`]).
    Bundled(StopwordLanguage),
    /// The caller's own words, sorted by bytes with duplicates dropped,
    /// held shared so the filter clones cheaply.
    List(Arc<[Box<str>]>),
}

impl Stopwords {
    pub fn bundled(language: StopwordLanguage) -> Stopwords {
        Stopwords::Bundled(language)
    }

    /// The caller's list, from folded forms. An empty iterator is refused.
    pub fn custom<'w>(words: impl IntoIterator<Item = &'w str>) -> Result<Stopwords, StopwordError> {
        let mut list: Vec<Box<str>> = words.into_iter().map(str::to_owned).map(Box::from).collect();
        if list.is_empty() {
            return Err(StopwordError::Empty);
        }
        list.sort();
        list.dedup();
        Ok(Stopwords::List(list.into()))
    }

    /// Whether `folded` is a stopword of this filter.
    pub fn contains(&self, folded: &str) -> bool {
        match self {
            Stopwords::Bundled(language) => StopwordList::bundled(*language).contains(folded),
            Stopwords::List(words) => words.binary_search_by(|w| w.as_bytes().cmp(folded.as_bytes())).is_ok(),
        }
    }

    /// A hash covering which words this filter holds, for a provenance
    /// config hash.
    pub fn config_hash(&self) -> u64 {
        match self {
            Stopwords::Bundled(language) => 0x11 | ((*language as u64) << 8),
            Stopwords::List(words) => {
                let mut h = turbo_analysis::Fnv1a::with_seed(0x22);
                for w in words.iter() {
                    h.str(w);
                }
                h.finish()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_delegates_to_opennlps_list() {
        let en = Stopwords::bundled(StopwordLanguage::English);
        assert!(en.contains("the"));
        assert!(!en.contains("cat"));
        let de = Stopwords::bundled(StopwordLanguage::German);
        assert!(de.contains("der"));
        assert!(!de.contains("the"));
    }

    #[test]
    fn custom_lists_sort_dedup_and_refuse_empty() {
        assert!(matches!(Stopwords::custom(Vec::<&str>::new()), Err(StopwordError::Empty)));
        let list = Stopwords::custom(["gamma", "alpha", "gamma"]).unwrap();
        assert!(list.contains("alpha"));
        assert!(list.contains("gamma"));
        assert!(!list.contains("beta"));
        assert_ne!(list.config_hash(), Stopwords::bundled(StopwordLanguage::English).config_hash());
        assert_ne!(Stopwords::custom(["a"]).unwrap().config_hash(), Stopwords::custom(["b"]).unwrap().config_hash());
    }
}
