/// Default aspect category labels used when no custom `TagSet` is supplied.
pub const DEFAULT_ASPECT_CATEGORIES: &[&str] = &[
    "battery",
    "display",
    "performance",
    "build",
    "software",
    "camera",
    "audio",
    "service",
    "food",
    "price",
    "ambience",
    "general",
];

/// Domain opinion words and predicate verbs that often carry aspect-specific
/// sentiment in product, software, and hospitality reviews even when absent or
/// weakly weighted in general-purpose social lexicons. In byte order by word,
/// for binary search.
pub(super) const DOMAIN_OPINION_LEXICON: &[(&str, f64)] = &[
    ("affordable", 2.2),
    ("attentive", 2.4),
    ("authentic", 2.0),
    ("awful", -2.7),
    ("bargain", 2.2),
    ("bland", -2.1),
    ("bloated", -2.0),
    ("blurry", -2.1),
    ("bright", 1.9),
    ("broken", -2.4),
    ("buggy", -2.3),
    ("clean", 2.0),
    ("clunky", -2.1),
    ("cold", -1.6),
    ("compact", 1.8),
    ("courteous", 2.3),
    ("cozy", 2.2),
    ("cracked", -2.4),
    ("cracks", -2.4),
    ("cramped", -1.9),
    ("crashed", -2.5),
    ("crashes", -2.5),
    ("crisp", 2.3),
    ("defective", -2.5),
    ("delicious", 2.7),
    ("dim", -1.8),
    ("dirty", -2.3),
    ("disappointing", -2.3),
    ("disconnected", -2.0),
    ("disconnects", -2.1),
    ("drained", -2.1),
    ("drains", -2.2),
    ("durable", 2.2),
    ("expensive", -1.8),
    ("fast", 2.0),
    ("filthy", -2.7),
    ("flavorful", 2.5),
    ("flawless", 2.7),
    ("flickered", -2.1),
    ("flickers", -2.1),
    ("flimsy", -2.2),
    ("fragile", -2.0),
    ("freezes", -2.3),
    ("fresh", 2.1),
    ("friendly", 2.3),
    ("froze", -2.3),
    ("frozen", -2.0),
    ("generous", 2.1),
    ("glare", -1.7),
    ("grainy", -2.0),
    ("greasy", -2.0),
    ("helpful", 2.2),
    ("horrible", -2.8),
    ("inattentive", -2.2),
    ("inexpensive", 2.0),
    ("intuitive", 2.3),
    ("juicy", 2.1),
    ("lagged", -2.1),
    ("laggy", -2.2),
    ("lags", -2.1),
    ("lasting", 1.9),
    ("lasts", 1.9),
    ("lightweight", 2.0),
    ("loud", -1.5),
    ("lukewarm", -1.9),
    ("mediocre", -1.9),
    ("muffled", -1.9),
    ("mushy", -1.9),
    ("noisy", -1.9),
    ("overcooked", -2.2),
    ("overheated", -2.3),
    ("overheats", -2.3),
    ("overpriced", -2.4),
    ("polite", 2.1),
    ("pricey", -1.7),
    ("prompt", 2.1),
    ("quick", 1.9),
    ("quiet", 1.9),
    ("reasonable", 1.9),
    ("reliable", 2.3),
    ("responsive", 2.3),
    ("ripoff", -2.7),
    ("rude", -2.5),
    ("scratch", -2.0),
    ("scratched", -2.2),
    ("scratches", -2.2),
    ("seamless", 2.4),
    ("sharp", 2.0),
    ("silent", 2.0),
    ("sleek", 2.1),
    ("slow", -1.9),
    ("slowest", -2.3),
    ("sluggish", -2.2),
    ("smooth", 2.1),
    ("snappy", 2.3),
    ("soggy", -2.2),
    ("solid", 2.0),
    ("spacious", 2.1),
    ("spotless", 2.5),
    ("stable", 2.0),
    ("stale", -2.2),
    ("stalled", -2.1),
    ("stalls", -2.1),
    ("steep", -1.5),
    ("sturdy", 2.2),
    ("stutters", -2.0),
    ("tasteless", -2.3),
    ("tasty", 2.3),
    ("tender", 2.1),
    ("terrible", -2.8),
    ("tinny", -2.0),
    ("undercooked", -2.3),
    ("underwhelming", -2.1),
    ("unresponsive", -2.4),
    ("useless", -2.5),
    ("vibrant", 2.3),
    ("vivid", 2.2),
    ("welcoming", 2.3),
    ("wobbles", -1.8),
    ("wobbly", -1.9),
    ("zippy", 2.2),
];

mod nouns;
mod words;
pub(super) use nouns::*;
pub(super) use words::*;

pub(super) fn clean_word_lower(raw: &str) -> String {
    let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '-');
    if trimmed.contains('\u{2019}') {
        // The tables use ASCII apostrophes; a tokenizer that keeps
        // contractions whole may write them with U+2019 instead.
        trimmed.replace('\u{2019}', "'").to_ascii_lowercase()
    } else {
        trimmed.to_ascii_lowercase()
    }
}

/// The domain opinion valence of `word`, if it is in [`DOMAIN_OPINION_LEXICON`].
pub(super) fn domain_opinion_valence(word: &str) -> Option<f64> {
    DOMAIN_OPINION_LEXICON.binary_search_by_key(&word, |&(w, _)| w).ok().map(|i| DOMAIN_OPINION_LEXICON[i].1)
}

/// The canonical aspect category of a single-word aspect term.
pub(super) fn single_word_aspect_category(term: &str) -> Option<&'static str> {
    SINGLE_WORD_ASPECT_TERMS.binary_search_by_key(&term, |&(t, _)| t).ok().map(|i| SINGLE_WORD_ASPECT_TERMS[i].1)
}

/// Whether `word` is a syntactic noun or discourse marker that is never a
/// valid aspect target on its own.
pub(super) fn is_aspect_stopword(word: &str) -> bool {
    ASPECT_STOPWORDS.binary_search(&word).is_ok()
}

pub(super) fn lookup_category_hint(tokens_lower: &[String]) -> Option<&'static str> {
    for &(pat, cat) in MULTIWORD_ASPECT_TERMS {
        if pat.len() == tokens_lower.len() && pat.iter().zip(tokens_lower).all(|(&p, t)| p == t.as_str()) {
            return Some(cat);
        }
    }
    for t in tokens_lower.iter().rev() {
        if let Some(cat) = single_word_aspect_category(t) {
            return Some(cat);
        }
    }
    None
}

pub(super) fn is_pure_aspect_noun(lower: &str) -> bool {
    matches!(
        lower,
        "support"
            | "value"
            | "service"
            | "party"
            | "security"
            | "care"
            | "comfort"
            | "credit"
            | "interest"
            | "share"
            | "shares"
            | "gift"
            | "energy"
            | "power"
            | "resolution"
            | "vision"
            | "clarity"
            | "intelligence"
            | "grace"
            | "spirit"
            | "harmony"
            | "treat"
            | "feeling"
            | "novel"
            | "play"
    )
}

pub(super) fn is_negation_word(w: &str) -> bool {
    NEGATION_CUES.binary_search(&w).is_ok() || w.ends_with("n't")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_tables_are_in_strict_byte_order() {
        let assert_strict_byte_order = |keys: &[&str]| {
            for w in keys.windows(2) {
                assert!(w[0] < w[1], "table is not in strict byte order: {:?} then {:?}", w[0], w[1]);
            }
        };
        assert_strict_byte_order(&DOMAIN_OPINION_LEXICON.iter().map(|&(w, _)| w).collect::<Vec<_>>());
        assert_strict_byte_order(&SINGLE_WORD_ASPECT_TERMS.iter().map(|&(t, _)| t).collect::<Vec<_>>());
        assert_strict_byte_order(ASPECT_STOPWORDS);
        assert_strict_byte_order(NEGATION_CUES);
    }

    #[test]
    fn indexed_lookups_match_every_table_entry() {
        for &(w, valence) in DOMAIN_OPINION_LEXICON {
            assert_eq!(domain_opinion_valence(w), Some(valence), "domain valence of {w:?}");
        }
        for &(term, category) in SINGLE_WORD_ASPECT_TERMS {
            assert_eq!(single_word_aspect_category(term), Some(category), "aspect category of {term:?}");
        }
        for &w in ASPECT_STOPWORDS {
            assert!(is_aspect_stopword(w), "aspect stopword {w:?}");
        }
        for &w in NEGATION_CUES {
            assert!(is_negation_word(w), "negation cue {w:?}");
        }
        // Keys not in a table find nothing; the "n't" suffix rule is separate.
        assert_eq!(domain_opinion_valence("zzz-not-a-word"), None);
        assert_eq!(single_word_aspect_category("zzz-not-a-word"), None);
        assert!(!is_aspect_stopword("zzz-not-a-word"));
        assert!(!is_negation_word("zzz-not-a-word"));
        assert!(is_negation_word("mustn't"));
    }
}

#[cfg(test)]
mod clean_tests {
    #[test]
    fn clean_word_lower_normalizes_curly_apostrophes() {
        assert_eq!(super::clean_word_lower("ISN\u{2019}T"), "isn't");
        assert_eq!(super::clean_word_lower("DON'T"), "don't");
        assert_eq!(super::clean_word_lower("  Great! "), "great");
    }
}
