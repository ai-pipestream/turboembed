//! The SymSpell engine, its edit distances, the dictionary and model
//! readers, and parity with OpenNLP's results (`tests/data/spellcheck/`,
//! written by OpenNLP's classes on the JVM; see the README there).

#[path = "spellcheck/compound.rs"]
mod compound;
#[path = "spellcheck/model.rs"]
mod model;

use std::fs;

use turbo_normalize::Dictionary;
use turbo_normalize::spellcheck::{
    ConfigError, DamerauOsa, EditDistance, FrequencyDictionary, Levenshtein, LookupError, MAX_COUNT, MAX_ENTRIES,
    MalformedReason, Metric, ModelError, Suggestion, SymSpell, SymSpellConfig, SymSpellModel, Verbosity,
};

fn data(name: &str) -> String {
    format!("{}/tests/data/spellcheck/{name}", env!("CARGO_MANIFEST_DIR"))
}

/// The engine OpenNLP's `TinyDictionary` builds: maximum distance 2, the
/// tiny unigram and bigram fixtures.
fn tiny() -> SymSpell {
    let mut engine = SymSpell::new(SymSpellConfig::builder().max_dictionary_edit_distance(2).build().unwrap());
    let unigrams = fs::read_to_string(data("frequency_dictionary_tiny.txt")).unwrap();
    let bigrams = fs::read_to_string(data("frequency_bigramdictionary_tiny.txt")).unwrap();
    FrequencyDictionary::load_unigrams(&mut engine, &unigrams).unwrap();
    FrequencyDictionary::load_bigrams(&mut engine, &bigrams).unwrap();
    engine
}

fn with(words: &[(&str, u64)]) -> SymSpell {
    with_metric(words, Metric::DamerauOsa)
}

fn with_metric(words: &[(&str, u64)], metric: Metric) -> SymSpell {
    let config = SymSpellConfig::builder().max_dictionary_edit_distance(2).metric(metric).build().unwrap();
    let mut engine = SymSpell::new(config);
    for &(w, c) in words {
        engine.add(w, c);
    }
    engine
}

fn terms<'a>(items: &[Suggestion<'a>]) -> Vec<&'a str> {
    items.iter().map(|s| s.term).collect()
}

fn top<'a>(engine: &'a SymSpell, term: &str, max: usize) -> Vec<Suggestion<'a>> {
    engine.lookup(term, Verbosity::Top, max).unwrap()
}

fn compound(engine: &SymSpell, input: &str) -> String {
    engine.lookup_compound(input, 2).unwrap().term
}

// ---------------------------------------------------------------------
// Edit distances (DamerauOSADistanceTest, EditDistanceTest).
// ---------------------------------------------------------------------

#[test]
fn damerau_osa_distances() {
    let d = DamerauOsa;
    assert_eq!(d.distance("kitten", "kitten", 2), Some(0));
    assert_eq!(d.distance("cat", "car", 2), Some(1));
    assert_eq!(d.distance("ca", "ac", 2), Some(1));
    assert_eq!(d.distance("converse", "covnerse", 2), Some(1));
    assert_eq!(d.distance("kitten", "sitting", 5), Some(3));
    assert_eq!(d.distance("flaw", "lawn", 5), Some(2));
    assert_eq!(d.distance("kitten", "sitting", 2), None);
    assert_eq!(d.distance("abc", "xyz", 2), None);
    assert_eq!(d.distance("", "", 2), Some(0));
    assert_eq!(d.distance("", "abc", 5), Some(3));
    assert_eq!(d.distance("abc", "", 5), Some(3));
    assert_eq!(d.distance("\u{1F600}", "\u{1F601}", 2), Some(1));
    assert_eq!(d.distance("café", "cafe", 2), Some(1));
}

#[test]
fn both_metrics_agree_without_transpositions() {
    for (a, b, expected) in [
        ("kitten", "kitten", 0),
        ("cat", "car", 1),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("sunday", "saturday", 3),
        ("abc", "abcd", 1),
        ("abcd", "abc", 1),
    ] {
        assert_eq!(DamerauOsa.distance(a, b, 9), Some(expected), "{a} {b}");
        assert_eq!(Levenshtein.distance(a, b, 9), Some(expected), "{a} {b}");
    }
}

#[test]
fn a_transposition_is_one_edit_or_two() {
    for (a, b) in [("ca", "ac"), ("converse", "covnerse"), ("abcd", "abdc"), ("teh", "the"), ("hte", "the")] {
        assert_eq!(DamerauOsa.distance(a, b, 4), Some(1), "{a} {b}");
        assert_eq!(Levenshtein.distance(a, b, 4), Some(2), "{a} {b}");
    }
    assert_eq!(DamerauOsa.distance("ca", "ac", 1), Some(1));
    assert_eq!(Levenshtein.distance("ca", "ac", 1), None);
}

#[test]
fn the_bound_cuts_off_above_and_keeps_at_the_maximum() {
    for m in [Metric::DamerauOsa, Metric::Levenshtein] {
        assert_eq!(m.distance("kitten", "sitting", 2), None);
        assert_eq!(m.distance("abc", "xyz", 2), None);
        assert_eq!(m.distance("kitten", "sitting", 3), Some(3));
        assert_eq!(m.distance("word", "word", 0), Some(0));
        assert_eq!(m.distance("word", "ward", 0), None);
        for max in 0..=6 {
            assert_eq!(m.distance("a", "aaaaaa", max), (max >= 5).then_some(5), "{m:?} max {max}");
        }
        assert_eq!(m.distance("", "", 2), Some(0));
        assert_eq!(m.distance("", "abc", 5), Some(3));
        assert_eq!(m.distance("abc", "", 5), Some(3));
    }
    assert_eq!(DamerauOsa.distance("abcd", "abdc", 4), DamerauOsa.distance("abdc", "abcd", 4));
    assert_eq!(Levenshtein.distance("flaw", "lawn", 4), Levenshtein.distance("lawn", "flaw", 4));
}

#[test]
fn metric_ids_are_the_model_format_ids() {
    assert_eq!(Metric::DamerauOsa.id(), "damerau-osa");
    assert_eq!(Metric::Levenshtein.id(), "levenshtein");
    assert_eq!(Metric::from_id("levenshtein"), Some(Metric::Levenshtein));
    assert_eq!(Metric::from_id("jaro"), None);
}

// ---------------------------------------------------------------------
// Suggestion order (SuggestItemTest) and configuration (SymSpellConfigTest).
// ---------------------------------------------------------------------

#[test]
fn suggestions_order_by_distance_then_count_then_term() {
    let s = |term, edit_distance, count| Suggestion { term, edit_distance, count };
    assert!(s("the", 1, 10) < s("then", 2, 1_000_000));
    assert!(s("the", 1, 1_000_000) < s("she", 1, 5));
    assert_eq!(s("word", 1, 42).cmp(&s("word", 1, 42)), std::cmp::Ordering::Equal);
    let mut items = vec![s("c", 2, 999), s("a", 0, 1), s("b1", 1, 5), s("b2", 1, 500)];
    items.sort();
    assert_eq!(terms(&items), ["a", "b2", "b1", "c"]);
    // Distinct terms never compare equal; the order is antisymmetric.
    assert!(s("cat", 1, 5) > s("car", 1, 5));
    assert!(s("car", 1, 5) < s("cat", 1, 5));
    let set: std::collections::BTreeSet<_> = [s("cat", 1, 5), s("car", 1, 5)].into_iter().collect();
    assert_eq!(set.len(), 2);
    // Java compares UTF-16 code units: a supplementary character (a
    // surrogate, 0xD800 and up) sorts before U+E000..U+FFFF.
    assert!(s("\u{10000}", 1, 5) < s("\u{FF21}", 1, 5));
}

#[test]
fn configuration_defaults_and_checks() {
    let c = SymSpellConfig::default();
    assert_eq!(c.max_dictionary_edit_distance(), 2);
    assert_eq!(c.prefix_length(), 7);
    assert_eq!(c.count_threshold(), 1);
    assert_eq!(c.metric(), Metric::DamerauOsa);
    assert_eq!(c.corpus_word_count(), 0);
    assert_eq!(SymSpellConfig::builder().build().unwrap(), c);
    assert_eq!(SymSpellConfig::builder().corpus_word_count(1_000_000).build().unwrap().corpus_word_count(), 1_000_000);
    let custom = SymSpellConfig::builder()
        .max_dictionary_edit_distance(3)
        .prefix_length(5)
        .count_threshold(10)
        .metric(Metric::Levenshtein)
        .build()
        .unwrap();
    assert_eq!(
        (custom.max_dictionary_edit_distance(), custom.prefix_length(), custom.count_threshold(), custom.metric()),
        (3, 5, 10, Metric::Levenshtein)
    );
    assert_eq!(SymSpellConfig::builder().prefix_length(0).build(), Err(ConfigError::PrefixLengthZero));
    assert_eq!(SymSpellConfig::builder().count_threshold(0).build(), Err(ConfigError::CountThresholdZero));
    let e = SymSpellConfig::builder().max_dictionary_edit_distance(2).prefix_length(2).build().unwrap_err();
    assert_eq!(e.to_string(), "prefixLength (2) must be > maxDictionaryEditDistance (2)");
    assert_eq!(SymSpellConfig::builder().corpus_word_count(MAX_COUNT + 1).build(), Err(ConfigError::CountTooLarge));
}

// ---------------------------------------------------------------------
// SymSpellTest.
// ---------------------------------------------------------------------

fn basic() -> SymSpell {
    with(&[
        ("the", 23135851162),
        ("of", 13151942776),
        ("members", 1226734),
        ("member", 7560493),
        ("hello", 4569100),
        ("world", 5705975),
        ("a", 9081174698),
    ])
}

#[test]
fn exact_match_has_zero_distance() {
    let e = basic();
    let r = top(&e, "members", 2);
    assert_eq!((r[0].term, r[0].edit_distance), ("members", 0));
}

#[test]
fn corrects_a_single_edit() {
    let e = basic();
    let r = top(&e, "membrs", 2);
    assert_eq!(r[0].term, "members");
    assert!(r[0].edit_distance <= 2);
}

#[test]
fn verbosity_all_returns_every_candidate_sorted() {
    let e = basic();
    let r = e.lookup("membe", Verbosity::All, 2).unwrap();
    assert!(r.len() >= 2);
    assert!(r.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn far_terms_have_no_suggestion_and_the_default_lookup_is_top() {
    let e = basic();
    assert!(top(&e, "zzzzzzzz", 2).is_empty());
    assert_eq!(e.lookup_top("helo")[0].term, "hello");
}

#[test]
fn compound_splits_and_merges() {
    let e = with(&[("hello", 4569100), ("world", 5705975)]);
    let r = e.lookup_compound("helloworld", 2).unwrap();
    assert_eq!(r.term, "hello world");
    let e = with(&[("members", 1226734)]);
    assert_eq!(compound(&e, "mem bers"), "members");
}

// ---------------------------------------------------------------------
// SymSpellLookupTest.
// ---------------------------------------------------------------------

fn ascii() -> SymSpell {
    with(&[("bank", 100), ("bonk", 80), ("band", 90), ("member", 7560493), ("members", 1226734), ("the", 23135851162)])
}

#[test]
fn top_returns_one_best_suggestion() {
    let e = ascii();
    let r = top(&e, "bnk", 2);
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].term, r[0].edit_distance), ("bank", 1));
}

#[test]
fn closest_returns_every_suggestion_at_the_smallest_distance() {
    let e = ascii();
    let r = e.lookup("bnk", Verbosity::Closest, 2).unwrap();
    assert_eq!(terms(&r), ["bank", "bonk"]);
    assert!(r.iter().all(|s| s.edit_distance == 1));
}

#[test]
fn all_returns_every_candidate_within_the_distance() {
    let e = ascii();
    let r = e.lookup("bnk", Verbosity::All, 2).unwrap();
    assert_eq!(terms(&r), ["bank", "bonk", "band"]);
}

#[test]
fn all_keeps_the_exact_match_and_its_neighbours_closest_stops() {
    let mut e = ascii();
    e.add("ban", 50);
    let all = e.lookup("ban", Verbosity::All, 2).unwrap();
    assert_eq!((all[0].term, all[0].edit_distance), ("ban", 0));
    assert!(all.len() > 1);
    let closest = e.lookup("ban", Verbosity::Closest, 2).unwrap();
    assert_eq!(terms(&closest), ["ban"]);
}

#[test]
fn distance_zero_returns_only_exact_matches() {
    let e = ascii();
    let r = top(&e, "member", 0);
    assert_eq!(r.len(), 1);
    assert_eq!((r[0].term, r[0].edit_distance), ("member", 0));
    assert!(top(&e, "membe", 0).is_empty());
}

#[test]
fn a_smaller_distance_excludes_farther_candidates() {
    let e = ascii();
    assert_eq!(terms(&e.lookup("membe", Verbosity::All, 1).unwrap()), ["member"]);
    let wide = e.lookup("membe", Verbosity::All, 2).unwrap();
    assert!(terms(&wide).contains(&"members"));
}

#[test]
fn a_distance_above_the_index_is_refused() {
    let e = ascii();
    let err = e.lookup("bank", Verbosity::Top, 3).unwrap_err();
    assert_eq!(err, LookupError { requested: 3, max_dictionary_edit_distance: 2 });
    assert!(err.to_string().contains("maxEditDistance"));
    assert!(e.lookup_compound("bank", 3).is_err());
    // OpenNLP checks the distance only when it looks a token up, so it
    // answers a blank input with any distance; here the check comes first.
    assert!(e.lookup_compound("", 3).is_err());
    assert!(e.lookup_compound(" \t", 3).is_err());
}

#[test]
fn typos_beyond_the_prefix_of_a_long_word_are_corrected() {
    let mut e =
        SymSpell::new(SymSpellConfig::builder().max_dictionary_edit_distance(2).prefix_length(7).build().unwrap());
    e.add("definitely", 6655443);
    assert_eq!(top(&e, "definitelyy", 2)[0].term, "definitely");
    assert_eq!(top(&e, "definately", 2)[0].term, "definitely");
}

#[test]
fn exact_match_under_every_verbosity_and_empty_or_unknown_terms() {
    let e = ascii();
    for v in [Verbosity::Top, Verbosity::Closest, Verbosity::All] {
        let r = e.lookup("the", v, 2).unwrap();
        assert_eq!((r[0].term, r[0].edit_distance), ("the", 0), "{v:?}");
    }
    assert!(e.lookup("zzzzzzzz", Verbosity::All, 2).unwrap().is_empty());
    assert!(top(&e, "", 2).is_empty());
}

#[test]
fn accented_non_latin_and_emoji_terms_by_code_point() {
    let e = with(&[("café", 100), ("naïve", 50), ("straße", 40), ("日本語", 30)]);
    assert_eq!(top(&e, "cafe", 2)[0].term, "café");
    assert_eq!(top(&e, "naive", 2)[0].term, "naïve");
    assert_eq!(top(&e, "strase", 2)[0].term, "straße");
    assert_eq!(top(&e, "日本誤", 2)[0].term, "日本語");
    let e = with(&[("a\u{1F600}b", 100)]);
    let r = top(&e, "a\u{1F601}b", 2);
    assert_eq!((r[0].term, r[0].edit_distance), ("a\u{1F600}b", 1));
}

// ---------------------------------------------------------------------
// Counts, thresholds and the dictionary interface.
// ---------------------------------------------------------------------

#[test]
fn counts_accumulate_and_index_at_the_threshold() {
    let config = SymSpellConfig::builder().count_threshold(10).build().unwrap();
    let mut e = SymSpell::new(config);
    assert!(!e.add("word", 4));
    assert_eq!(e.word_count(), 1);
    assert!(!e.contains("word"));
    assert!(top(&e, "wrd", 2).is_empty());
    assert!(e.add("word", 6));
    assert!(e.contains("word"));
    assert_eq!(top(&e, "wrd", 2)[0].term, "word");
    // Already indexed: still true, and the count keeps growing.
    assert!(e.add("word", 1));
    assert_eq!(e.count("word"), Some(11));
    // A zero count adds nothing; counts saturate at Long.MAX_VALUE.
    assert!(!e.add("zero", 0));
    assert_eq!(e.count("zero"), None);
    e.add("big", MAX_COUNT);
    e.add("big", MAX_COUNT);
    assert_eq!(e.count("big"), Some(MAX_COUNT));
    assert_eq!(e.corpus_word_count(), MAX_COUNT);
}

#[test]
fn the_noise_scorer_sees_indexed_words_only() {
    let config = SymSpellConfig::builder().count_threshold(3).build().unwrap();
    let mut e = SymSpell::new(config);
    e.add("known", 5);
    e.add("rare", 2);
    let d: &dyn Dictionary = &e;
    assert!(d.contains("known"));
    assert!(!d.contains("rare"));
    assert!(!d.contains("unknown"));
}

#[test]
fn bigrams_accumulate() {
    let mut e = SymSpell::default();
    e.add_bigram("the", "bank", 3);
    e.add_bigram("the", "bank", 4);
    e.add_bigram("a", "bank", 1);
    assert_eq!(e.bigram("the", "bank"), Some(7));
    assert_eq!(e.bigram_count(), 2);
    assert_eq!(e.max_bigram_count(), 7);
}

#[test]
fn the_fingerprint_follows_every_addition_in_order() {
    let mut a = SymSpell::default();
    let mut b = SymSpell::default();
    assert_eq!(a.fingerprint(), b.fingerprint());
    a.add("x", 1);
    a.add("y", 1);
    b.add("y", 1);
    b.add("x", 1);
    assert_ne!(a.fingerprint(), b.fingerprint());
    let mut c = SymSpell::default();
    c.add("x", 1);
    c.add("y", 1);
    assert_eq!(a.fingerprint(), c.fingerprint());
    let other = SymSpell::new(SymSpellConfig::builder().prefix_length(5).build().unwrap());
    assert_ne!(other.fingerprint(), SymSpell::default().fingerprint());
}

// ---------------------------------------------------------------------
// SymSpellEditDistanceMetricTest.
// ---------------------------------------------------------------------

fn metric_engine(metric: Metric) -> SymSpell {
    with_metric(
        &[("the", 23135851162), ("and", 12997637966), ("receive", 56473829), ("form", 100000), ("from", 1838693953)],
        metric,
    )
}

#[test]
fn the_metric_decides_whether_a_transposition_is_in_reach() {
    let d = metric_engine(Metric::DamerauOsa);
    let r = top(&d, "teh", 1);
    assert_eq!((r[0].term, r[0].edit_distance), ("the", 1));
    let l = metric_engine(Metric::Levenshtein);
    assert!(top(&l, "teh", 1).is_empty());
    let r = top(&l, "teh", 2);
    assert_eq!((r[0].term, r[0].edit_distance), ("the", 2));
    let (a, b) = (top(&d, "fonm", 2)[0], top(&l, "fonm", 2)[0]);
    assert_eq!((a.term, b.term), ("form", "form"));
    assert_eq!(a.edit_distance, b.edit_distance);
}
