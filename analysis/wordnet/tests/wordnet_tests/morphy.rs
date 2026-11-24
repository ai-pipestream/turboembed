//! The Morphy exception lists and lemmatizer, and lexical expansion.

use super::*;

// ============================================================================
// MorphyExceptionsTest & MorphyLemmatizerTest
// ============================================================================

#[test]
fn test_morphy_exceptions_lookup_and_validation() {
    let exc = morphy_exceptions_fixture();
    assert_eq!(&["mouse"], exc.lookup("mice", WordNetPos::Noun));
    assert_eq!(&["mouse"], exc.lookup("Mice", WordNetPos::Noun));
    assert_eq!(&["mouse"], exc.lookup("MICE", WordNetPos::Noun));
    assert_eq!(&["go"], exc.lookup("went", WordNetPos::Verb));
    assert_eq!(&["good"], exc.lookup("better", WordNetPos::Adjective));
    assert_eq!(&["well"], exc.lookup("best", WordNetPos::Adverb));
    assert!(exc.lookup("went", WordNetPos::Noun).is_empty());
    assert!(exc.lookup("dog", WordNetPos::Noun).is_empty());
    assert_eq!(&["axis", "ax"], exc.lookup("axes", WordNetPos::Noun));
}

#[test]
fn test_morphy_lemmatizer() {
    let morphy = MorphyLemmatizer::new(wndb_fixture(), morphy_exceptions_fixture());
    let one = |token: &str, tag: &str| morphy.lemmatize(&[token], &[tag]).unwrap()[0].clone();

    let cases = [
        ("mice", "NN", "mouse"),
        ("Mice", "NNS", "mouse"),
        ("men", "NNS", "man"),
        ("ran", "VBD", "run"),
        ("running", "VBG", "run"),
        ("went", "VBD", "go"),
        ("gone", "VBN", "go"),
        ("best", "RBS", "well"),
        ("dogs", "NNS", "dog"),
        ("boxes", "NNS", "box"),
        ("berries", "NNS", "berry"),
        ("runs", "NNS", "run"),
        ("runs", "VBZ", "run"),
        ("walked", "VBD", "walk"),
        ("walking", "VBG", "walk"),
        ("walks", "VBZ", "walk"),
        ("moved", "VBD", "move"),
        ("taller", "JJR", "tall"),
        ("tallest", "JJS", "tall"),
        ("larger", "JJR", "large"),
        ("dog", "NN", "dog"),
        ("quickly", "RB", "quickly"),
        ("dogs", "n", "dog"),
        ("walked", "v", "walk"),
        ("taller", "a", "tall"),
        ("best", "r", "well"),
        ("oxen", "NNS", "ox"),
        ("better", "JJR", "good"),
    ];
    for (token, tag, expected) in cases {
        assert_eq!(expected, one(token, tag), "{token}/{tag}");
    }

    let unknowns = [
        ("dogged", "VBD"),
        ("boxes", "VBZ"),
        ("glarbs", "NNS"),
        ("walk", "NN"),
        ("dog", "DT"),
        ("dog", "XYZ"),
        ("dogs", ""),
        ("was", "AUX"),
        ("taller", "AUX"),
    ];
    for (token, tag) in unknowns {
        assert_eq!("O", one(token, tag), "{token}/{tag}");
    }

    assert_eq!(
        vec!["O", "mouse", "run", "quickly"],
        morphy.lemmatize(&["The", "mice", "ran", "quickly"], &["DT", "NNS", "VBD", "RB"]).unwrap()
    );

    let candidates = morphy.lemmatize_candidates(&["glarbs", "berries", "axes"], &["NNS", "NNS", "NNS"]).unwrap();
    assert_eq!(vec![vec!["O"], vec!["berry"], vec!["axis", "ax"]], candidates);
    assert_eq!("axis", one("axes", "NNS"));

    assert_eq!(Some(WordNetPos::Noun), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("NNP"));
    assert_eq!(Some(WordNetPos::Verb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("VBZ"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("JJ"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("RBR"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("a"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("s"));
    assert_eq!(Some(WordNetPos::Adjective), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("ADJ"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("ADV"));
    assert_eq!(Some(WordNetPos::Adverb), MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag("r"));
    for tag in ["DT", "", "AUX", "ADP", "SCONJ", "SYM", "ß", "ſ"] {
        assert_eq!(None, MorphyLemmatizer::<InMemoryWordNetLexicon>::pos_from_tag(tag), "{tag}");
    }
}

// ============================================================================
// LexicalExpanderTest & LexicalExpanderLexiconTest
// ============================================================================

fn find_expansion<'a>(expansions: &'a [Expansion], term: &str) -> Option<&'a Expansion> {
    expansions.iter().find(|e| e.term() == term)
}

#[test]
fn test_lexical_expander_over_fixtures_and_morphy() {
    let lmf = lmf_fixture();
    let wndb = wndb_fixture();

    let exp_lmf = LexicalExpander::builder(&lmf).build().expand("dog", WordNetPos::Noun).unwrap();
    let dom = find_expansion(&exp_lmf, "domestic dog").unwrap();
    assert_eq!(ExpansionKind::Synonym, dom.kind());
    assert_eq!(1.0, dom.weight());
    assert_eq!(0, dom.sense_rank());

    let canid = find_expansion(&exp_lmf, "canid").unwrap();
    assert_eq!(ExpansionKind::Hypernym, canid.kind());
    assert_eq!(1, canid.depth());
    assert_eq!(0.5, canid.weight());

    let exp_multi = LexicalExpander::builder(&wndb).build().expand("domestic_dog", WordNetPos::Noun).unwrap();
    assert_eq!(
        vec![
            Expansion::new("dog", ExpansionKind::Synonym, 0, 0, 1.0).unwrap(),
            Expansion::new("canid", ExpansionKind::Hypernym, 1, 0, 0.5).unwrap(),
        ],
        exp_multi
    );

    let morphy = MorphyLemmatizer::new(lmf.clone(), morphy_exceptions_fixture());
    let expander = LexicalExpander::builder(&lmf).lemmatizer(morphy).build();

    let dogs = expander.expand("dogs", WordNetPos::Noun).unwrap();
    assert_eq!(ExpansionKind::Synonym, find_expansion(&dogs, "dog").unwrap().kind());
    assert!(find_expansion(&dogs, "canid").is_some());

    let mice = expander.expand("mice", WordNetPos::Noun).unwrap();
    assert_eq!(ExpansionKind::Synonym, find_expansion(&mice, "mouse").unwrap().kind());
    assert!(find_expansion(&mice, "rodent").is_some());
}
