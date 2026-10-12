//! The SymSpell engine, its edit distances, the dictionary and model
//! readers, and parity with OpenNLP's results (`tests/data/spellcheck/`,
//! written by OpenNLP's classes on the JVM; see the README there).

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

// ---------------------------------------------------------------------
// SymSpellCompoundTest.
// ---------------------------------------------------------------------

#[test]
fn compound_merges_run_on_words() {
    let t = tiny();
    assert_eq!(compound(&t, "helloworld"), "hello world");
    assert_eq!(compound(&t, "spellchecker"), "spell checker");
    let mut e = with(&[("hello", 4569100), ("world", 5705975)]);
    e.add_bigram("hello", "world", 5432109);
    assert_eq!(compound(&e, "helloworld"), "hello world");
    assert_eq!(compound(&e, "hel loworld"), "hello world");
}

#[test]
fn compound_corrects_mixed_sentences() {
    let t = tiny();
    assert_eq!(compound(&t, "teh quikc broen fxo"), "the quick brown fox");
    let e = with(&[("the", 23135851162), ("quick", 119476788)]);
    assert_eq!(compound(&e, "thequikc"), "the quick");
    let r = t.lookup_compound("the quick brown fox", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("the quick brown fox", 0));
}

#[test]
fn compound_edge_cases() {
    let t = tiny();
    assert_eq!(compound(&t, ""), "");
    assert_eq!(compound(&t, "   "), "");
    assert_eq!(compound(&t, "zzzzqqqq"), "zzzzqqqq");
}

#[test]
fn compound_splits_on_unicode_white_space() {
    let t = tiny();
    let r = t.lookup_compound("hello\u{A0}world", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("hello world", 1));
    for sep in ['\u{A0}', '\u{85}', '\u{2028}'] {
        assert_eq!(compound(&t, &format!("hello{sep}world{sep}hello")), "hello world hello", "{sep:?}");
    }
    let r = t.lookup_compound("\u{A0}", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("", 0));
    let r = t.lookup_compound("\u{A0}hello world\u{A0}\u{85}", 2).unwrap();
    assert_eq!((r.term.as_str(), r.edit_distance), ("hello world", 0));
}

#[test]
fn compound_parts_say_where_each_word_came_from() {
    let t = tiny();
    let input = "  teh quikc  helloworld spell checker hel lo";
    let r = t.lookup_compound(input, 2).unwrap();
    let parts: Vec<(&str, &str)> =
        r.parts.iter().map(|p| (&input[p.source.clone()], &r.term[p.output.clone()])).collect();
    assert_eq!(r.term, "the quick hello world spell checker hello");
    assert_eq!(
        parts,
        [
            ("teh", "the"),
            ("quikc", "quick"),
            ("helloworld", "hello world"),
            ("spell", "spell"),
            ("checker", "checker"),
            ("hel lo", "hello"),
        ]
    );
    // The outputs are the phrase with single spaces between them.
    let rebuilt: Vec<&str> = r.parts.iter().map(|p| &r.term[p.output.clone()]).collect();
    assert_eq!(rebuilt.join(" "), r.term);
}

// ---------------------------------------------------------------------
// FrequencyDictionaryLoaderTest.
// ---------------------------------------------------------------------

fn unigrams(text: &str) -> Result<Vec<(String, u64)>, turbo_normalize::spellcheck::MalformedLine> {
    let mut out = Vec::new();
    FrequencyDictionary::read_unigrams(text, |w, c| out.push((w.to_owned(), c)))?;
    Ok(out)
}

fn bigrams(text: &str) -> Result<Vec<(String, u64)>, turbo_normalize::spellcheck::MalformedLine> {
    let mut out = Vec::new();
    FrequencyDictionary::read_bigrams(text, |a, b, c| out.push((format!("{a} {b}"), c)))?;
    Ok(out)
}

fn pairs(rows: &[(&str, u64)]) -> Vec<(String, u64)> {
    rows.iter().map(|&(w, c)| (w.to_owned(), c)).collect()
}

#[test]
fn the_loader_skips_blank_and_comment_lines() {
    let mut n = 0;
    let read = FrequencyDictionary::read_unigrams("the\t100\n\n# a comment\n   \nworld\t50\n", |_, _| n += 1).unwrap();
    assert_eq!((read, n), (2, 2));
    assert_eq!(unigrams("the\t100\n\n# a comment\n   \nworld\t50\n").unwrap(), pairs(&[("the", 100), ("world", 50)]));
    let e = unigrams("the\tnotanumber\n").unwrap_err();
    assert_eq!(e.line_number, 1);
}

#[test]
fn a_unigram_word_ends_at_a_tab_or_space_only() {
    for (line, word) in [
        ("a 5", "a"),
        ("a\t5", "a"),
        ("a \t  \t 5", "a"),
        (" a 5", "a"),
        ("\t\ta 5", "a"),
        ("a 5 \t", "a"),
        (" a  5 ", "a"),
        ("a\u{B}b 5", "a\u{B}b"),
        ("a\u{C}b 5", "a\u{C}b"),
        ("a\u{A0}b 5", "a\u{A0}b"),
        ("a\u{3000}b 5", "a\u{3000}b"),
        ("\u{1F600} 5", "\u{1F600}"),
        ("中文 5", "中文"),
        ("cafe\u{301} 5", "cafe\u{301}"),
    ] {
        assert_eq!(unigrams(&format!("{line}\n")).unwrap(), pairs(&[(word, 5)]), "{line:?}");
    }
}

#[test]
fn bigram_columns_split_on_tab_and_space_runs() {
    assert_eq!(bigrams("the  world\t3\nhello \t there   4\n").unwrap(), pairs(&[("the world", 3), ("hello there", 4)]));
    assert_eq!(bigrams("a b\u{A0}c 5\n").unwrap(), pairs(&[("a b\u{A0}c", 5)]));
}

#[test]
fn lines_of_java_whitespace_are_skipped() {
    for blank in ["\t\t", " \t \t ", "\u{B}\u{C}", "\u{3000}", "\u{2028}"] {
        let text = format!("the\t100\n{blank}\nworld 5\n");
        assert_eq!(unigrams(&text).unwrap(), pairs(&[("the", 100), ("world", 5)]), "{blank:?}");
    }
}

#[test]
fn skipped_lines_count_toward_the_line_number() {
    let e = unigrams("the\t100\n\t\t\n# note\n\nworld\n").unwrap_err();
    assert_eq!(e.line_number, 5);
}

#[test]
fn counts_accepted() {
    for (count, expected) in [("007", 7), ("0", 0), ("9223372036854775807", 9223372036854775807)] {
        assert_eq!(unigrams(&format!("the\t{count}\n")).unwrap(), pairs(&[("the", expected)]));
    }
}

#[test]
fn malformed_unigram_lines_name_the_reason() {
    for (line, reason) in [
        ("the", "expected 'word<sep>count'"),
        ("the\u{A0}100", "expected 'word<sep>count'"),
        ("\u{A0}", "expected 'word<sep>count'"),
        ("\u{A0}\u{A0}", "expected 'word<sep>count'"),
        ("the\t-5", "count must not be negative"),
        ("the\t-99999999999999999999", "count must not be negative"),
        ("the\t9223372036854775808", "count is not an integer"),
        ("the\t5\u{A0}", "count is not an integer"),
        ("the\t5\u{1}", "count is not an integer"),
        ("the\t\u{1}5", "count is not an integer"),
        ("the\t5.0", "count is not an integer"),
        ("the\t1e3", "count is not an integer"),
        ("the\t99999999999999999999", "count is not an integer"),
        ("the\t-", "count is not an integer"),
        ("the\t+", "count is not an integer"),
        ("the\t+5", "count is not an integer"),
        ("the\t+-5", "count is not an integer"),
        ("the\t-0", "count must not be negative"),
    ] {
        let e = unigrams(&format!("{line}\n")).unwrap_err();
        assert_eq!(e.line_number, 1, "{line:?}");
        assert!(e.to_string().contains(&format!("({reason})")), "{line:?}: {e}");
    }
}

#[test]
fn counts_in_any_decimal_digits() {
    for (count, value) in [
        ("\u{665}", 5),
        ("\u{FF15}", 5),
        ("5\u{665}", 55),
        ("\u{661}\u{662}\u{663}", 123),
        ("\u{FF19}\u{FF12}\u{FF12}\u{FF13}", 9223),
        ("\u{1D7CE}", 0),
        ("0", 0),
        ("007", 7),
        ("\u{660}\u{665}", 5),
    ] {
        assert_eq!(unigrams(&format!("caf\u{E9}\t{count}\n")).unwrap(), pairs(&[("caf\u{E9}", value)]));
        assert_eq!(bigrams(&format!("\u{4E2D}\u{6587} caf\u{E9}\t{count}\n")).unwrap(), pairs(&[("中文 café", value)]));
    }
}

#[test]
fn counts_that_are_not_digits_are_refused() {
    for (count, reason) in [
        ("+\u{FF15}", MalformedReason::CountNotInteger),
        ("-\u{660}", MalformedReason::CountNegative),
        ("-\u{665}", MalformedReason::CountNegative),
        ("+5", MalformedReason::CountNotInteger),
        ("-", MalformedReason::CountNotInteger),
        ("\u{1F600}", MalformedReason::CountNotInteger),
        ("\u{665}\u{301}", MalformedReason::CountNotInteger),
        ("\u{B2}", MalformedReason::CountNotInteger),
    ] {
        assert_eq!(unigrams(&format!("caf\u{E9}\t{count}\n")).unwrap_err().reason, reason, "{count:?}");
        assert_eq!(bigrams(&format!("中文 café\t{count}\n")).unwrap_err().reason, reason, "{count:?}");
    }
}

#[test]
fn other_loader_cases() {
    let e = bigrams("the\t5\n").unwrap_err();
    assert_eq!(e.line_number, 1);
    assert!(e.to_string().contains("(expected 'w1<sep>w2<sep>count')"));
    assert_eq!(unigrams("the 100 extra\n").unwrap(), pairs(&[("the", 100)]));
    assert_eq!(unigrams("\u{FEFF}# note\nthe 5\n").unwrap(), pairs(&[("the", 5)]));
    // The message is OpenNLP's, with the tab of the line written as \t.
    let e = unigrams("the\tx\n").unwrap_err();
    assert_eq!(e.to_string(), "malformed dictionary line 1 (count is not an integer): \"the\\tx\"");
}

#[test]
fn loaders_feed_an_engine() {
    let mut e = SymSpell::new(SymSpellConfig::default());
    let u =
        FrequencyDictionary::load_unigrams(&mut e, &fs::read_to_string(data("frequency_dictionary_tiny.txt")).unwrap())
            .unwrap();
    let b = FrequencyDictionary::load_bigrams(
        &mut e,
        &fs::read_to_string(data("frequency_bigramdictionary_tiny.txt")).unwrap(),
    )
    .unwrap();
    assert_eq!(u as usize, e.word_count());
    assert_eq!(b as usize, e.bigram_count());
    assert_eq!(top(&e, "teh", 2)[0].term, "the");
}

// ---------------------------------------------------------------------
// The binary model (SymSpellModelSerializationTest,
// SymSpellModelSerializerLimitsTest) and bytes written by OpenNLP.
// ---------------------------------------------------------------------

fn java_model(name: &str) -> (Vec<u8>, SymSpellModel) {
    let bytes = fs::read(data(name)).unwrap();
    let model = SymSpellModel::read(&bytes).unwrap();
    (bytes, model)
}

#[test]
fn models_written_by_opennlp_read_and_write_back_byte_for_byte() {
    let (bytes, model) = java_model("model-tiny.bin");
    assert_eq!((model.language(), model.name(), model.version()), ("en", "spellcheck-tiny", "1.0"));
    assert_eq!(*model.config(), SymSpellConfig::default());
    assert_eq!(model.unigrams().len(), 63 + 4);
    assert_eq!(model.unigrams()[0], ("the".to_owned(), 23135851162));
    assert!(model.unigrams().contains(&("a\u{0}b".to_owned(), 7)));
    assert!(model.unigrams().contains(&("\u{10428}word".to_owned(), 1000)));
    assert!(model.unigrams().contains(&("maximum".to_owned(), MAX_COUNT)));
    assert_eq!(model.bigrams().len(), 20);
    assert_eq!(model.write().unwrap(), bytes);

    let (bytes, model) = java_model("model-levenshtein.bin");
    let c = model.config();
    assert_eq!(
        (c.max_dictionary_edit_distance(), c.prefix_length(), c.count_threshold(), c.metric(), c.corpus_word_count()),
        (1, 4, 3, Metric::Levenshtein, 1_000_000_000)
    );
    assert_eq!((model.language(), model.name(), model.version()), ("de", "spellcheck-levenshtein", "2.5"));
    assert_eq!(model.write().unwrap(), bytes);
    let engine = model.build();
    // Below the count threshold of 3: kept, not indexed.
    assert_eq!(engine.word_count(), 8);
    assert!(!engine.contains("bonk") && !engine.contains("bandana") && engine.contains("bandanas"));
    assert_eq!(engine.bigram("river", "bank"), Some(30));
}

#[test]
fn round_trips_keep_the_model_and_its_suggestions() {
    let (_, model) = java_model("model-tiny.bin");
    let again = SymSpellModel::read(&model.write().unwrap()).unwrap();
    assert_eq!(again, model);
    let (a, b) = (model.build(), again.build());
    for (typo, word) in [
        ("teh", "the"),
        ("recieve", "receive"),
        ("wrold", "world"),
        ("quikc", "quick"),
        ("becuse", "because"),
        ("calender", "calendar"),
    ] {
        let x = a.lookup(typo, Verbosity::All, 2).unwrap();
        assert_eq!(x, b.lookup(typo, Verbosity::All, 2).unwrap());
        assert_eq!(x[0].term, word);
    }
    assert_eq!(b.lookup_compound("helloworld", 2).unwrap().term, "hello world");
    // A fixed corpus size survives; the derived one stays 0.
    let fixed = SymSpellConfig::builder().corpus_word_count(987_654_321).build().unwrap();
    let m = SymSpellModel::new("en", "n", "1", fixed, model.unigrams().to_vec(), model.bigrams().to_vec()).unwrap();
    assert_eq!(SymSpellModel::read(&m.write().unwrap()).unwrap().config().corpus_word_count(), 987_654_321);
    assert_eq!(again.config().corpus_word_count(), 0);
}

#[test]
fn repeated_rows_keep_the_first_position_and_the_last_count() {
    let m = SymSpellModel::new(
        "en",
        "n",
        "1",
        SymSpellConfig::default(),
        [("a".to_owned(), 1), ("b".to_owned(), 2), ("a".to_owned(), 3)],
        [("x y".to_owned(), 1)],
    )
    .unwrap();
    assert_eq!(m.unigrams(), [("a".to_owned(), 3), ("b".to_owned(), 2)]);
    for bad in ["xy", " y", "x "] {
        let e = SymSpellModel::new("en", "n", "1", SymSpellConfig::default(), [], [(bad.to_owned(), 1)]).unwrap_err();
        assert_eq!(e, ModelError::InvalidBigramKey(bad.to_owned()));
    }
    let e = SymSpellModel::new(" \t", "n", "1", SymSpellConfig::default(), [], []).unwrap_err();
    assert_eq!(e, ModelError::Blank("language"));
}

/// A header as OpenNLP's limits test writes it, up to the unigram count
/// and optionally a bigram count.
fn header(unigram_count: i32, bigram_count: Option<i32>) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0x5359_4D53u32.to_be_bytes());
    b.extend_from_slice(&1u32.to_be_bytes());
    for s in ["en", "test", "1.0"] {
        b.extend_from_slice(&(s.len() as u16).to_be_bytes());
        b.extend_from_slice(s.as_bytes());
    }
    b.extend_from_slice(&2i32.to_be_bytes());
    b.extend_from_slice(&7i32.to_be_bytes());
    b.extend_from_slice(&1i64.to_be_bytes());
    b.extend_from_slice(&11u16.to_be_bytes());
    b.extend_from_slice(b"damerau-osa");
    b.extend_from_slice(&0i64.to_be_bytes());
    b.extend_from_slice(&unigram_count.to_be_bytes());
    if let Some(n) = bigram_count {
        b.extend_from_slice(&n.to_be_bytes());
    }
    b
}

#[test]
fn row_counts_outside_the_limit_are_refused() {
    let over = MAX_ENTRIES as i32 + 1;
    for (bytes, field) in [
        (header(i32::MAX, None), "unigram count"),
        (header(over, None), "unigram count"),
        (header(-1, None), "unigram count"),
        (header(0, Some(i32::MAX)), "bigram count"),
        (header(0, Some(over)), "bigram count"),
        (header(0, Some(-1)), "bigram count"),
    ] {
        let e = SymSpellModel::read(&bytes).unwrap_err();
        assert!(matches!(e, ModelError::EntryCount { field: f, .. } if f == field), "{e:?}");
        assert!(e.to_string().contains(field));
    }
    assert!(SymSpellModel::read(&header(0, Some(0))).is_ok());
    assert!(matches!(SymSpellModel::read_with_limit(&header(5, None), 4), Err(ModelError::EntryCount { .. })));
}

#[test]
fn corrupt_streams_are_refused() {
    let good = header(0, Some(0));
    let mut bad_magic = good.clone();
    bad_magic[0] = 0;
    assert_eq!(SymSpellModel::read(&bad_magic), Err(ModelError::BadMagic(0x0059_4D53)));
    assert!(SymSpellModel::read(&bad_magic).unwrap_err().to_string().contains("magic was 0x00594D53"));
    let mut bad_version = good.clone();
    bad_version[7] = 2;
    assert_eq!(SymSpellModel::read(&bad_version), Err(ModelError::UnsupportedVersion(2)));
    for cut in 0..good.len() {
        assert_eq!(SymSpellModel::read(&good[..cut]), Err(ModelError::Truncated), "cut at {cut}");
    }
    let mut trailing = good.clone();
    trailing.push(0);
    assert_eq!(SymSpellModel::read(&trailing), Err(ModelError::TrailingBytes(1)));
    // The prefix length (7) set to the maximum edit distance (2).
    let mut config = good.clone();
    let at = 8 + 4 + 6 + 5 + 4 + 3;
    config[at] = 2;
    assert!(matches!(SymSpellModel::read(&config), Err(ModelError::CorruptConfig(_))));
    let mut metric = good.clone();
    let id = metric.windows(11).position(|w| w == b"damerau-osa").unwrap();
    metric[id] = b'x';
    assert!(SymSpellModel::read(&metric).unwrap_err().to_string().contains("unknown edit-distance id: xamerau-osa"));
}

#[test]
fn modified_utf8_and_vlong_rows() {
    let config = SymSpellConfig::default();
    let words = ["\u{0}", "é", "€", "\u{10428}", "plain"];
    let rows: Vec<(String, u64)> =
        words.iter().zip([0, 127, 128, 1 << 35, MAX_COUNT]).map(|(w, c)| ((*w).to_owned(), c)).collect();
    let m = SymSpellModel::new("en", "n", "1", config, rows.clone(), []).unwrap();
    let bytes = m.write().unwrap();
    // U+0000 is two bytes, a supplementary character two three-byte
    // surrogates.
    let find = |needle: &[u8]| bytes.windows(needle.len()).any(|w| w == needle);
    assert!(find(&[0x00, 0x02, 0xC0, 0x80, 0x00]));
    assert!(find(&[0x00, 0x06, 0xED, 0xA0, 0x81, 0xED, 0xB0, 0xA8]));
    // 127 is one byte, 128 two.
    assert!(find(&[0x00, 0x02, 0xC3, 0xA9, 0x7F]));
    assert!(find(&[0x00, 0x03, 0xE2, 0x82, 0xAC, 0x80, 0x01]));
    assert_eq!(SymSpellModel::read(&bytes).unwrap().unigrams(), rows.as_slice());

    // A lone surrogate cannot be a Rust string.
    let mut lone = header(1, None);
    lone.extend_from_slice(&[0x00, 0x03, 0xED, 0xA0, 0x81, 0x01]);
    lone.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&lone), Err(ModelError::MalformedUtf));
    // A continuation byte where a character starts.
    let mut bad = header(1, None);
    bad.extend_from_slice(&[0x00, 0x01, 0x80, 0x01]);
    bad.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&bad), Err(ModelError::MalformedUtf));
    // Eleven bytes of continuation is too long for a vlong.
    let mut long = header(1, None);
    long.extend_from_slice(&[0x00, 0x01, b'a']);
    long.extend_from_slice(&[0x80; 10]);
    long.push(0);
    assert_eq!(SymSpellModel::read(&long), Err(ModelError::VLongTooLong));
    // Ten bytes that set the top bit: Long.MIN_VALUE in Java.
    let mut negative = header(1, None);
    negative.extend_from_slice(&[0x00, 0x01, b'a']);
    negative.extend_from_slice(&[0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01]);
    negative.extend_from_slice(&0i32.to_be_bytes());
    assert_eq!(SymSpellModel::read(&negative), Err(ModelError::NegativeCount));
    // A string longer than a UTF field holds.
    let m = SymSpellModel::new("en", "n", "1", config, [("x".repeat(65536), 1)], []).unwrap();
    assert_eq!(m.write(), Err(ModelError::StringTooLong));
}

// ---------------------------------------------------------------------
// Parity with OpenNLP over generated terms and phrases.
// ---------------------------------------------------------------------

/// Undoes the escaping of expected.txt: `\u{hex}` is a code point.
fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(i) = rest.find("\\u{") {
        out.push_str(&rest[..i]);
        let end = rest[i..].find('}').unwrap() + i;
        out.push(char::from_u32(u32::from_str_radix(&rest[i + 3..end], 16).unwrap()).unwrap());
        rest = &rest[end + 1..];
    }
    out.push_str(rest);
    out
}

/// `term,distance,count` as expected.txt writes a suggestion.
fn item(s: &str) -> (String, usize, u64) {
    let f: Vec<&str> = s.split(',').collect();
    (unescape(f[0]), f[1].parse().unwrap(), f[2].parse().unwrap())
}

fn expected(kind: &str) -> Vec<Vec<String>> {
    fs::read_to_string(data("expected.txt"))
        .unwrap()
        .lines()
        .filter(|l| l.starts_with(&format!("{kind}\t")))
        .map(|l| l.split('\t').map(str::to_owned).collect())
        .collect()
}

fn engines() -> [(&'static str, SymSpell); 2] {
    [("tiny", java_model("model-tiny.bin").1.build()), ("levenshtein", java_model("model-levenshtein.bin").1.build())]
}

#[test]
fn lookups_match_opennlp() {
    let engines = engines();
    let rows = expected("lookup");
    assert!(rows.len() > 2000);
    for row in rows {
        let engine = &engines.iter().find(|(n, _)| *n == row[1]).unwrap().1;
        let verbosity = match row[2].as_str() {
            "TOP" => Verbosity::Top,
            "CLOSEST" => Verbosity::Closest,
            _ => Verbosity::All,
        };
        let term = unescape(&row[4]);
        let want: Vec<(String, usize, u64)> =
            if row.len() < 6 || row[5].is_empty() { Vec::new() } else { row[5].split('|').map(item).collect() };
        let got: Vec<(String, usize, u64)> = engine
            .lookup(&term, verbosity, row[3].parse().unwrap())
            .unwrap()
            .iter()
            .map(|s| (s.term.to_owned(), s.edit_distance, s.count))
            .collect();
        assert_eq!(got, want, "{row:?}");
    }
}

#[test]
fn compound_lookups_match_opennlp() {
    let engines = engines();
    let rows = expected("compound");
    assert!(rows.len() > 1500);
    for row in rows {
        let engine = &engines.iter().find(|(n, _)| *n == row[1]).unwrap().1;
        let input = unescape(&row[3]);
        let r = engine.lookup_compound(&input, row[2].parse().unwrap()).unwrap();
        assert_eq!((r.term.clone(), r.edit_distance, r.count), item(&row[4]), "{row:?}");
        // Every part points at whole tokens of the input, in order, and
        // the outputs are the phrase with single spaces between them.
        let mut at = 0;
        for p in &r.parts {
            assert!(p.source.start >= at && p.source.end > p.source.start, "{row:?}");
            let src = &input[p.source.clone()];
            assert!(!src.starts_with(char::is_whitespace) && !src.ends_with(char::is_whitespace));
            at = p.source.end;
        }
        let outputs: Vec<&str> = r.parts.iter().map(|p| &r.term[p.output.clone()]).collect();
        assert_eq!(outputs.join(" "), r.term);
    }
}
