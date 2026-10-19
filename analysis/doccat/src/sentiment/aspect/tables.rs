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

/// Known multi-word aspect terms (lowercased token sequences) with their
/// canonical category name in [`DEFAULT_ASPECT_CATEGORIES`].
pub(super) const MULTIWORD_ASPECT_TERMS: &[(&[&str], &str)] = &[
    (&["battery", "life"], "battery"),
    (&["battery", "longevity"], "battery"),
    (&["battery", "capacity"], "battery"),
    (&["charging", "speed"], "battery"),
    (&["charging", "port"], "battery"),
    (&["power", "adapter"], "battery"),
    (&["power", "cord"], "battery"),
    (&["power", "brick"], "battery"),
    (&["screen", "resolution"], "display"),
    (&["screen", "brightness"], "display"),
    (&["display", "quality"], "display"),
    (&["viewing", "angles"], "display"),
    (&["viewing", "angle"], "display"),
    (&["refresh", "rate"], "display"),
    (&["color", "accuracy"], "display"),
    (&["touch", "screen"], "display"),
    (&["boot", "time"], "performance"),
    (&["load", "times"], "performance"),
    (&["load", "time"], "performance"),
    (&["processing", "speed"], "performance"),
    (&["frame", "rate"], "performance"),
    (&["fan", "noise"], "performance"),
    (&["build", "quality"], "build"),
    (&["track", "pad"], "build"),
    (&["touch", "pad"], "build"),
    (&["key", "travel"], "build"),
    (&["keyboard", "backlight"], "build"),
    (&["fingerprint", "reader"], "build"),
    (&["hinge", "design"], "build"),
    (&["user", "interface"], "software"),
    (&["operating", "system"], "software"),
    (&["camera", "quality"], "camera"),
    (&["camera", "lens"], "camera"),
    (&["night", "mode"], "camera"),
    (&["photo", "quality"], "camera"),
    (&["video", "quality"], "camera"),
    (&["sound", "quality"], "audio"),
    (&["speaker", "volume"], "audio"),
    (&["audio", "quality"], "audio"),
    (&["noise", "cancellation"], "audio"),
    (&["customer", "service"], "service"),
    (&["customer", "support"], "service"),
    (&["wait", "staff"], "service"),
    (&["waiting", "time"], "service"),
    (&["wait", "time"], "service"),
    (&["delivery", "time"], "service"),
    (&["tech", "support"], "service"),
    (&["portion", "size"], "food"),
    (&["portion", "sizes"], "food"),
    (&["wine", "list"], "food"),
    (&["menu", "selection"], "food"),
    (&["breakfast", "buffet"], "food"),
    (&["price", "point"], "price"),
    (&["value", "for", "money"], "price"),
    (&["dining", "room"], "ambience"),
    (&["seating", "area"], "ambience"),
    (&["background", "music"], "ambience"),
];

/// Single-word aspect head lexicon mapped to canonical category names in
/// [`DEFAULT_ASPECT_CATEGORIES`]. In byte order by term, for binary search.
pub(super) const SINGLE_WORD_ASPECT_TERMS: &[(&str, &str)] = &[
    ("ambiance", "ambience"),
    ("ambience", "ambience"),
    ("app", "software"),
    ("appetizer", "food"),
    ("appetizers", "food"),
    ("atmosphere", "ambience"),
    ("audio", "audio"),
    ("autofocus", "camera"),
    ("bartender", "service"),
    ("bass", "audio"),
    ("battery", "battery"),
    ("beer", "food"),
    ("bezel", "display"),
    ("bill", "price"),
    ("bloatware", "software"),
    ("body", "build"),
    ("breakfast", "food"),
    ("brightness", "display"),
    ("build", "build"),
    ("burger", "food"),
    ("camera", "camera"),
    ("case", "build"),
    ("charger", "battery"),
    ("charging", "battery"),
    ("chassis", "build"),
    ("check", "price"),
    ("cocktails", "food"),
    ("coffee", "food"),
    ("cooling", "performance"),
    ("cost", "price"),
    ("cpu", "performance"),
    ("crust", "food"),
    ("decor", "ambience"),
    ("delivery", "service"),
    ("design", "build"),
    ("dessert", "food"),
    ("dinner", "food"),
    ("dish", "food"),
    ("dishes", "food"),
    ("display", "display"),
    ("drinks", "food"),
    ("driver", "software"),
    ("drivers", "software"),
    ("entree", "food"),
    ("espresso", "food"),
    ("fan", "performance"),
    ("fee", "price"),
    ("fees", "price"),
    ("firmware", "software"),
    ("food", "food"),
    ("gpu", "performance"),
    ("headphones", "audio"),
    ("hinge", "build"),
    ("host", "service"),
    ("hostess", "service"),
    ("ingredients", "food"),
    ("interface", "software"),
    ("interior", "ambience"),
    ("keyboard", "build"),
    ("keys", "build"),
    ("latency", "performance"),
    ("lens", "camera"),
    ("lighting", "ambience"),
    ("lunch", "food"),
    ("management", "service"),
    ("manager", "service"),
    ("meal", "food"),
    ("meals", "food"),
    ("memory", "performance"),
    ("menu", "food"),
    ("mic", "audio"),
    ("microphone", "audio"),
    ("monitor", "display"),
    ("mouse", "build"),
    ("music", "ambience"),
    ("os", "software"),
    ("panel", "display"),
    ("pasta", "food"),
    ("patio", "ambience"),
    ("performance", "performance"),
    ("photos", "camera"),
    ("pictures", "camera"),
    ("pixels", "display"),
    ("pizza", "food"),
    ("portion", "food"),
    ("portions", "food"),
    ("ports", "build"),
    ("price", "price"),
    ("prices", "price"),
    ("pricing", "price"),
    ("processor", "performance"),
    ("ram", "performance"),
    ("reception", "service"),
    ("resolution", "display"),
    ("room", "ambience"),
    ("rooms", "ambience"),
    ("salad", "food"),
    ("sauce", "food"),
    ("screen", "display"),
    ("seating", "ambience"),
    ("sensor", "camera"),
    ("server", "service"),
    ("service", "service"),
    ("software", "software"),
    ("sound", "audio"),
    ("soup", "food"),
    ("speaker", "audio"),
    ("speakers", "audio"),
    ("speed", "performance"),
    ("ssd", "performance"),
    ("staff", "service"),
    ("steak", "food"),
    ("storage", "performance"),
    ("support", "service"),
    ("sushi", "food"),
    ("table", "ambience"),
    ("tables", "ambience"),
    ("thermals", "performance"),
    ("touchpad", "build"),
    ("trackpad", "build"),
    ("ui", "software"),
    ("updates", "software"),
    ("value", "price"),
    ("view", "ambience"),
    ("waiter", "service"),
    ("waitress", "service"),
    ("webcam", "camera"),
    ("weight", "build"),
    ("wine", "food"),
];

/// Implicit aspect category cues for opinion words when no explicit aspect term
/// is present in the sentence.
pub(super) const IMPLICIT_OPINION_CATEGORY_CUES: &[(&str, &str)] = &[
    ("expensive", "price"),
    ("overpriced", "price"),
    ("pricey", "price"),
    ("affordable", "price"),
    ("inexpensive", "price"),
    ("cheap", "price"),
    ("bargain", "price"),
    ("ripoff", "price"),
    ("delicious", "food"),
    ("tasty", "food"),
    ("flavorful", "food"),
    ("yummy", "food"),
    ("soggy", "food"),
    ("bland", "food"),
    ("stale", "food"),
    ("tasteless", "food"),
    ("overcooked", "food"),
    ("undercooked", "food"),
    ("greasy", "food"),
    ("rude", "service"),
    ("attentive", "service"),
    ("courteous", "service"),
    ("inattentive", "service"),
    ("polite", "service"),
    ("welcoming", "service"),
    ("cozy", "ambience"),
    ("cramped", "ambience"),
    ("noisy", "ambience"),
    ("spacious", "ambience"),
    ("spotless", "ambience"),
    ("filthy", "ambience"),
    ("snappy", "performance"),
    ("sluggish", "performance"),
    ("laggy", "performance"),
    ("zippy", "performance"),
    ("buggy", "software"),
    ("intuitive", "software"),
    ("flimsy", "build"),
    ("sturdy", "build"),
    ("sleek", "build"),
    ("lightweight", "build"),
    ("blurry", "camera"),
    ("tinny", "audio"),
];

/// Words that are syntactic nouns or pronouns or discourse markers but never
/// valid aspect targets on their own. In byte order, for binary search.
pub(super) const ASPECT_STOPWORDS: &[&str] = &[
    "all",
    "any",
    "anyone",
    "anything",
    "bit",
    "bits",
    "both",
    "day",
    "days",
    "each",
    "everyone",
    "everything",
    "fact",
    "few",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "it",
    "its",
    "kind",
    "least",
    "less",
    "little",
    "lot",
    "lots",
    "many",
    "me",
    "mine",
    "month",
    "months",
    "more",
    "most",
    "much",
    "my",
    "myself",
    "nobody",
    "none",
    "nothing",
    "one",
    "ones",
    "other",
    "others",
    "our",
    "ours",
    "part",
    "parts",
    "people",
    "person",
    "place",
    "point",
    "reason",
    "she",
    "some",
    "someone",
    "something",
    "sort",
    "that",
    "their",
    "theirs",
    "them",
    "these",
    "they",
    "thing",
    "things",
    "this",
    "those",
    "time",
    "times",
    "today",
    "tomorrow",
    "type",
    "us",
    "way",
    "ways",
    "we",
    "week",
    "weeks",
    "what",
    "whatever",
    "which",
    "who",
    "whom",
    "year",
    "years",
    "yesterday",
    "you",
    "your",
    "yours",
];

/// Negation cues used to detect negated opinions. In byte order, for binary
/// search.
pub(super) const NEGATION_CUES: &[&str] = &[
    "aren't",
    "arent",
    "barely",
    "cannot",
    "cant",
    "couldn't",
    "couldnt",
    "didn't",
    "didnt",
    "doesn't",
    "doesnt",
    "don't",
    "dont",
    "hadn't",
    "hadnt",
    "hardly",
    "hasn't",
    "hasnt",
    "haven't",
    "havent",
    "isn't",
    "isnt",
    "lack",
    "lacked",
    "lacking",
    "lacks",
    "neither",
    "never",
    "no",
    "nobody",
    "none",
    "nor",
    "not",
    "nothing",
    "nowhere",
    "rarely",
    "scarcely",
    "seldom",
    "shouldn't",
    "shouldnt",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
    "without",
    "won't",
    "wont",
    "wouldn't",
    "wouldnt",
];

/// Degree booster and dampener modifiers with their intensity multiplier delta.
pub(super) const DEGREE_MODIFIERS: &[(&str, f64)] = &[
    ("very", 0.30),
    ("extremely", 0.45),
    ("incredibly", 0.45),
    ("exceptionally", 0.45),
    ("remarkably", 0.40),
    ("totally", 0.30),
    ("completely", 0.35),
    ("absolutely", 0.40),
    ("really", 0.25),
    ("so", 0.25),
    ("too", 0.25),
    ("super", 0.35),
    ("highly", 0.30),
    ("quite", 0.15),
    ("pretty", 0.12),
    ("fairly", 0.10),
    ("rather", 0.12),
    ("slightly", -0.35),
    ("somewhat", -0.25),
    ("marginally", -0.40),
    ("a_bit", -0.30),
];

/// Adversative and clause-splitting conjunctions.
pub(super) const ADVERSATIVE_CONJUNCTIONS: &[&str] = &[
    "but",
    "however",
    "yet",
    "although",
    "though",
    "even",
    "whereas",
    "while",
    "except",
    "despite",
    "nevertheless",
    "nonetheless",
    "unfortunately",
    "sadly",
];

/// Determiners and leading articles stripped from the front of noun-phrase
/// aspect candidates.
pub(super) const LEADING_DETERMINERS: &[&str] = &[
    "the", "a", "an", "this", "that", "these", "those", "my", "our", "your", "his", "her", "its", "their", "some",
    "any", "each", "every", "another", "both", "all",
];

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
