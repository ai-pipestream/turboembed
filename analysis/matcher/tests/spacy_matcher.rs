//! spaCy's `Matcher` tests, ported as fixtures: the same patterns over the
//! same tokens give the same matches. The cases come from
//! `spacy/tests/matcher/test_matcher_logic.py` and `test_matcher_api.py`
//! of spaCy 3.8.7 (MIT License, Copyright (C) 2016-2024 ExplosionAI GmbH,
//! 2016 spaCy GmbH, 2015 Matthew Honnibal); each test names the one it
//! ports. spaCy builds a `Doc` from words joined by spaces, and so do
//! these tests.
//!
//! Not ported: callbacks, pickling, `Span` inputs, warnings and pattern
//! validation of spaCy's dictionary syntax, which have no counterpart in
//! typed patterns; `MORPH` set operators, `DEP`, `ENT_IOB`,
//! `IS_LEFT_PUNCT`, `IS_RIGHT_PUNCT`, `LIKE_URL` and `LIKE_EMAIL`, which
//! the matcher does not test (the crate documentation lists them).

mod common;

use common::{doc, find, found, spans, split};
use turbo_matcher::{Attr, Cmp, Flag, Greedy, Match, MatchError, Matcher, Op, Pattern, PatternError, Tok};

fn matcher(rules: &[(&str, Vec<Pattern>)]) -> Matcher {
    let mut b = Matcher::builder();
    for (label, patterns) in rules {
        b.add(label, patterns.clone(), None).unwrap();
    }
    b.build().unwrap()
}

fn greedy(label: &str, patterns: Vec<Pattern>, g: Greedy) -> Matcher {
    let mut b = Matcher::builder();
    b.add(label, patterns, Some(g)).unwrap();
    b.build().unwrap()
}

/// A pattern over `ORTH` from spaCy's test shorthand: `a+ a b? c* d{2,3}`.
fn ops(spec: &str) -> Pattern {
    spec.split(' ')
        .map(|part| {
            let (c, op) = part.split_at(1);
            let tok = Tok::text(c);
            match op {
                "" => tok,
                "+" => tok.one_or_more(),
                "*" => tok.zero_or_more(),
                "?" => tok.optional(),
                rep => {
                    let inner = rep.strip_prefix('{').and_then(|r| r.strip_suffix('}')).unwrap();
                    let (min, max) = match inner.split_once(',') {
                        Some((a, b)) => (a.parse().unwrap_or(0), b.parse().ok()),
                        None => (inner.parse().unwrap(), inner.parse().ok()),
                    };
                    tok.repeat(min, max)
                }
            }
        })
        .collect()
}

fn chars(s: &str) -> common::Doc {
    let words: Vec<String> = s.chars().map(String::from).collect();
    doc(&words.iter().map(String::as_str).collect::<Vec<_>>())
}

// test_matcher_logic.py

#[test]
fn issue118_overlapping_matches() {
    let d = split("how many points did lebron james score against the boston celtics last night");
    for patterns in [
        vec![vec![Tok::lower("celtics")], vec![Tok::lower("boston"), Tok::lower("celtics")]],
        vec![vec![Tok::lower("boston"), Tok::lower("celtics")], vec![Tok::lower("celtics")]],
    ] {
        assert_eq!(spans(&matcher(&[("BostonCeltics", patterns)]), &d), vec![(9, 11), (10, 11)]);
    }
    for patterns in [
        vec![vec![Tok::lower("boston")], vec![Tok::lower("boston"), Tok::lower("celtics")]],
        vec![vec![Tok::lower("boston"), Tok::lower("celtics")], vec![Tok::lower("boston")]],
    ] {
        assert_eq!(spans(&matcher(&[("BostonCeltics", patterns)]), &d), vec![(9, 10), (9, 11)]);
    }
}

#[test]
fn issue242_overlapping_multi_word_phrases() {
    let d = split("There are different food safety standards in different countries .");
    let m = matcher(&[(
        "FOOD",
        vec![vec![Tok::lower("food"), Tok::lower("safety")], vec![Tok::lower("safety"), Tok::lower("standards")]],
    )]);
    assert_eq!(spans(&m, &d), vec![(3, 5), (4, 6)]);
}

#[test]
fn issue587_patterns_added_one_by_one() {
    let d = doc(&["a", "b", ";", "c"]);
    let mut b = Matcher::builder();
    b.add("TEST1", vec![vec![Tok::text("a"), Tok::text("b")]], None).unwrap();
    let p2 = vec![Tok::text("a"), Tok::text("b"), Tok::any().flag(Flag::Punct, true), Tok::text("c")];
    let p3 = vec![Tok::text("a"), Tok::text("b"), Tok::any().flag(Flag::Punct, true), Tok::text("d")];
    b.add("TEST2", vec![p2], None).unwrap();
    b.add("TEST3", vec![p3], None).unwrap();
    assert_eq!(find(&b.build().unwrap(), &d).len(), 2);
}

#[test]
fn issue590_overlapping_matches_of_one_label() {
    let d = doc(&["n", "=", "1", ";", "a", ":", "5", "%"]);
    let mut b = Matcher::builder();
    let alpha = || Tok::any().flag(Flag::Alpha, true);
    let num = || Tok::any().flag(Flag::LikeNum, true);
    b.add("ab", vec![vec![alpha(), Tok::text(":"), num(), Tok::text("%")]], None).unwrap();
    b.add("ab", vec![vec![alpha(), Tok::text("="), num()]], None).unwrap();
    assert_eq!(find(&b.build().unwrap(), &d).len(), 2);
}

#[test]
fn issue850_ambiguous_variable_length() {
    let d = doc(&["bob", "and", "and", "frank"]);
    let m = matcher(&[("FarAway", vec![vec![Tok::lower("bob"), Tok::any().zero_or_more(), Tok::lower("frank")]])]);
    assert_eq!(spans(&m, &d), vec![(0, 4)]);
    let m =
        matcher(&[("FarAway", vec![vec![Tok::lower("bob"), Tok::lower("and").zero_or_more(), Tok::lower("frank")]])]);
    assert_eq!(spans(&m, &d), vec![(0, 4)]);
}

#[test]
fn issue1434_optional_at_end_of_short_doc() {
    let m = matcher(&[("MyMatcher", vec![vec![Tok::text("Hello"), Tok::any().flag(Flag::Alpha, true).optional()]])]);
    assert!(!find(&m, &doc(&["Hello", "World"])).is_empty());
    assert!(!find(&m, &doc(&["Hello"])).is_empty());
}

#[test]
fn issue1450_patterns_ending_in_star() {
    let m = matcher(&[("TSTEND", vec![vec![Tok::text("a"), Tok::text("b").zero_or_more()]])]);
    for (s, last, count) in [
        ("a", (0, 1), 1),
        ("a b", (0, 2), 2),
        ("a c", (0, 1), 1),
        ("a b c", (0, 2), 2),
        ("a b b c", (0, 3), 3),
        ("a b b", (0, 3), 3),
    ] {
        let got = spans(&m, &split(s));
        assert_eq!(got.last(), Some(&last), "{s}");
        assert_eq!(got.len(), count, "{s}");
    }
}

#[test]
fn issue1945_two_overlapping_matches() {
    let m = matcher(&[("MWE", vec![vec![Tok::text("a"), Tok::text("a")]])]);
    assert_eq!(spans(&m, &doc(&["a", "a", "a"])), vec![(0, 2), (1, 3)]);
}

#[test]
fn issue1971_labels_are_the_rules_own() {
    let d = doc(&["Hello", "John", "Doe", "!"]).column("optional", &["False"; 4]);
    let p = vec![
        Tok::text("Doe"),
        Tok::text("!").optional(),
        Tok::any().eq(Attr::Column("optional"), "True").optional(),
        Tok::text("!").optional(),
    ];
    let m = matcher(&[("TEST", vec![p])]);
    assert!(found(&m, &d).iter().all(|(l, ..)| l == "TEST"));
    assert_eq!(spans(&m, &d), vec![(2, 3), (2, 4)]);
}

#[test]
fn issue1971_2_two_patterns_one_label() {
    let d = doc(&["EUR", "10", "is", "10", "EUR"]);
    let p1 = vec![Tok::text("EUR").is_in(Attr::Lower, &["eur"]), Tok::any().flag(Flag::LikeNum, true)];
    let p2 = vec![Tok::any().flag(Flag::LikeNum, true), Tok::text("EUR")];
    assert_eq!(find(&matcher(&[("TEST1", vec![p1, p2])]), &d).len(), 2);
}

#[test]
fn issue1971_3_several_columns() {
    let d = doc(&["hello", "world"]).column("a", &["1", "1"]).column("b", &["2", "2"]);
    let m = matcher(&[
        ("A", vec![vec![Tok::any().eq(Attr::Column("a"), "1")]]),
        ("B", vec![vec![Tok::any().eq(Attr::Column("b"), "2")]]),
    ]);
    let mut got = found(&m, &d);
    got.sort();
    let want: Vec<(String, u32, u32)> =
        [("A", 0, 1), ("A", 1, 2), ("B", 0, 1), ("B", 1, 2)].iter().map(|&(l, s, e)| (l.into(), s, e)).collect();
    assert_eq!(got, want);
}

#[test]
fn issue1971_4_two_column_values_on_one_token() {
    let d = doc(&["this", "is", "text"]).column("ext_a", &["str_a"; 3]).column("ext_b", &["str_b"; 3]);
    let t = || Tok::any().eq(Attr::Column("ext_a"), "str_a").eq(Attr::Column("ext_b"), "str_b");
    let m = matcher(&[("TEST", vec![vec![t(), t(), t()]])]);
    assert_eq!(spans(&m, &d), vec![(0, 3)]);
}

#[test]
fn issue2464_successive_optionals() {
    let m = matcher(&[("4", vec![vec![Tok::any().optional(), Tok::any().optional()]])]);
    assert_eq!(spans(&m, &doc(&["a", "b"])), vec![(0, 1), (0, 2), (1, 2)]);
}

#[test]
fn issue2569_plus_matches_every_run() {
    // "It is May 15, 1993." as spaCy's English tokenizer splits it.
    let d = doc(&["It", "is", "May", "15", ",", "1993", "."]).ents(&[(2, 6, "DATE")]);
    let m = matcher(&[("RULE", vec![vec![Tok::any().eq(Attr::Entity, "DATE").one_or_more()]])]);
    let got = spans(&m, &d);
    assert_eq!(got.len(), 10);
    assert_eq!(got.iter().max_by_key(|(s, e)| e - s), Some(&(2, 6)));
}

#[test]
fn issue2671_labels_with_optional_tokens() {
    let p = vec![Tok::lower("high"), Tok::any().flag(Flag::Punct, true).optional(), Tok::lower("adrenaline")];
    let m = matcher(&[("test_pattern", vec![p])]);
    // "This is a high-adrenaline situation." and the same with a space.
    let d1 = doc(&["This", "is", "a", "high", "-", "adrenaline", "situation", "."]);
    let d2 = doc(&["This", "is", "a", "high", "adrenaline", "situation", "."]);
    assert_eq!(found(&m, &d1), vec![("test_pattern".into(), 3, 6)]);
    assert_eq!(found(&m, &d2), vec![("test_pattern".into(), 3, 5)]);
}

#[test]
fn issue3009_quantifiers_before_literals() {
    let d = doc(&["also", "has", "to", "do", "with"]).tags(&["RB", "VBZ", "TO", "VB", "IN"]);
    let ascii_word = || Tok::any().flag(Flag::Ascii, true).flag(Flag::Punct, false);
    let patterns = [
        vec![Tok::text("has"), Tok::lower("to"), Tok::lower("do"), Tok::pos("IN")],
        vec![Tok::text("has"), ascii_word().zero_or_more(), Tok::lower("to"), Tok::lower("do"), Tok::pos("IN")],
        vec![Tok::text("has"), ascii_word().optional(), Tok::lower("to"), Tok::lower("do"), Tok::pos("IN")],
    ];
    let mut b = Matcher::builder();
    for (i, p) in patterns.into_iter().enumerate() {
        b.add(&i.to_string(), vec![p], None).unwrap();
    }
    let m = b.build().unwrap();
    assert_eq!(found(&m, &d).len(), 3);
}

#[test]
fn issue3328_sets_of_values() {
    let d = doc(&["Hello", ",", "how", "are", "you", "doing", "?"]);
    let m = matcher(&[(
        "TEST",
        vec![
            vec![Tok::any().is_in(Attr::Lower, &["hello", "how"])],
            vec![Tok::any().is_in(Attr::Lower, &["you", "doing"])],
        ],
    )]);
    let texts: Vec<&str> = find(&m, &d).iter().map(|x| d.words(x.start, x.end)).collect();
    assert_eq!(texts, vec!["Hello", "how", "you", "doing"]);
}

#[test]
fn issue3839_labels_with_wildcards() {
    let d = doc(&["terrific", "group", "of", "people"]);
    for p in [
        vec![Tok::lower("terrific"), Tok::any().optional(), Tok::lower("group")],
        vec![Tok::lower("terrific"), Tok::any().optional(), Tok::any().optional(), Tok::lower("group")],
    ] {
        assert_eq!(found(&matcher(&[("PATTERN", vec![p])]), &d)[0].0, "PATTERN");
    }
}

#[test]
fn issue3879_no_false_match_through_optionals() {
    let d = doc(&["This", "is", "a", "test", "."]);
    let m = matcher(&[("TEST", vec![vec![Tok::text("This").optional(), Tok::any().optional(), Tok::text("test")]])]);
    assert_eq!(spans(&m, &d), vec![(2, 4), (3, 4)]);
}

#[test]
fn issue3951_combinations_of_optionals() {
    let d = doc(&["Hello", "my", "new", "world"]);
    let p = vec![Tok::lower("hello"), Tok::lower("this").optional(), Tok::any().optional(), Tok::lower("world")];
    assert!(find(&matcher(&[("TEST", vec![p])]), &d).is_empty());
}

#[test]
fn issue4120_final_optional_tokens() {
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().optional()]])]);
    assert_eq!(find(&m, &doc(&["a"])).len(), 1);
    assert_eq!(find(&m, &doc(&["a", "b", "c"])).len(), 2);
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().optional(), Tok::text("b")]])]);
    assert_eq!(find(&m, &doc(&["a", "b", "b", "c"])).len(), 2);
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().optional(), Tok::text("b").optional()]])]);
    assert_eq!(find(&m, &doc(&["a", "b", "b", "c"])).len(), 3);
}

/// "(BBAAAAAB)." split into one token per character, and the five
/// patterns of the greedy tests.
fn greedy_doc() -> common::Doc {
    chars("(BBAAAAAB).")
}

const GREEDY: [&str; 5] = ["A A*", "A* A", "A A", "B A* B", "B* A* B"];

#[test]
fn greedy_first_is_regular_expression_search() {
    // What Python's re.finditer gives for AA*, A*A, AA, BA*B and B*A*B
    // over the same characters.
    let want: [&[(u32, u32)]; 5] = [&[(3, 8)], &[(3, 8)], &[(3, 5), (5, 7)], &[(1, 3)], &[(1, 9)]];
    for (p, w) in GREEDY.iter().zip(want) {
        assert_eq!(spans(&greedy(p, vec![ops(p)], Greedy::First), &greedy_doc()), w, "{p}");
    }
}

#[test]
fn greedy_longest() {
    let want: [&[(u32, u32)]; 5] = [&[(3, 8)], &[(3, 8)], &[(3, 5), (5, 7)], &[(2, 9)], &[(1, 9)]];
    let texts = ["AAAAA", "AAAAA", "AA", "BAAAAAB", "BBAAAAAB"];
    for ((p, w), t) in GREEDY.iter().zip(want).zip(texts) {
        let d = greedy_doc();
        let got = spans(&greedy(p, vec![ops(p)], Greedy::Longest), &d);
        assert_eq!(got, w, "{p}");
        for (s, e) in got {
            assert_eq!(d.words(s, e).replace(' ', ""), t);
        }
    }
}

#[test]
fn greedy_longest_prefers_the_first_of_equals() {
    let m = greedy("RULE", vec![ops("C C")], Greedy::Longest);
    assert_eq!(spans(&m, &chars("CCC")), vec![(0, 2)]);
}

#[test]
fn operator_combinations() {
    let cases = [
        ("aaab", "a a a b", true),
        ("aaab", "a+ b", true),
        ("aaab", "a+ a+ b", true),
        ("aaab", "a+ a+ a b", true),
        ("aaab", "a+ a+ a+ b", true),
        ("aaab", "a+ a a b", true),
        ("aaab", "a+ a a", true),
        ("aaab", "a+", true),
        ("aaa", "a+ b", false),
        ("aaa", "a+ a+ b", false),
        ("aaa", "a+ a+ a+ b", false),
        ("aaa", "a+ a b", false),
        ("aaa", "a+ a a b", false),
        ("aaab", "a+ a a", true),
        ("aaab", "a+", true),
        ("aaab", "a+ a b", true),
    ];
    for (s, p, want) in cases {
        let m = matcher(&[("PATTERN", vec![ops(p)])]);
        assert_eq!(!find(&m, &chars(s)).is_empty(), want, "{s} {p}");
    }
}

#[test]
fn matcher_end_zero_plus() {
    let m = matcher(&[("TSTEND", vec![vec![Tok::text("a"), Tok::text("b").zero_or_more()]])]);
    for (s, n) in [("a", 1), ("a b", 2), ("a c", 1), ("a b c", 2), ("a b b c", 3), ("a b b", 3)] {
        assert_eq!(find(&m, &split(s)).len(), n, "{s}");
    }
}

#[test]
fn sets_return_correct_tokens() {
    let m = matcher(&[(
        "TEST",
        vec![
            vec![Tok::any().is_in(Attr::Lower, &["zero"])],
            vec![Tok::any().is_in(Attr::Lower, &["one"])],
            vec![Tok::any().is_in(Attr::Lower, &["two"])],
        ],
    )]);
    let d = split("zero one two three");
    let texts: Vec<&str> = find(&m, &d).iter().map(|x| d.words(x.start, x.end)).collect();
    assert_eq!(texts, vec!["zero", "one", "two"]);
}

#[test]
fn remove_a_rule() {
    let d = doc(&["This", "is", "a", "test", "case", "."]);
    let mut b = Matcher::builder();
    b.add("Rule", vec![vec![Tok::text("test"), Tok::any().optional()]], None).unwrap();
    assert!(b.contains("Rule"));
    assert!(b.remove("Rule"));
    assert!(!b.remove("Rule"));
    assert!(find(&b.build().unwrap(), &d).is_empty());
    let mut b = Matcher::builder();
    b.add("Rule", vec![vec![Tok::text("test"), Tok::any().optional()]], None).unwrap();
    assert_eq!(find(&b.build().unwrap(), &d).len(), 2);
}

/// Alignments of `test_matcher_with_alignments_greedy_longest`. Where the
/// alignment spaCy gives is not the greedy backtracking one, the expected
/// value here is the greedy one and spaCy's is in the comment.
#[test]
fn alignments_greedy_longest() {
    let cases: &[(&str, &str, &[u16])] = &[
        ("aaab", "a* b", &[0, 0, 0, 1]),
        ("baab", "b a* b", &[0, 1, 1, 2]),
        ("aaab", "a a a b", &[0, 1, 2, 3]),
        ("aaab", "a+ b", &[0, 0, 0, 1]),
        ("aaba", "a+ b a+", &[0, 0, 1, 2]),
        ("aabaa", "a+ b a+", &[0, 0, 1, 2, 2]),
        ("aaba", "a+ b a*", &[0, 0, 1, 2]),
        ("aaaa", "a*", &[0, 0, 0, 0]),
        ("baab", "b a* b b*", &[0, 1, 1, 2]),
        ("aabb", "a* b* a*", &[0, 0, 1, 1]),
        ("aaab", "a+ a+ a b", &[0, 1, 2, 3]),
        ("aaab", "a+ a+ a+ b", &[0, 1, 2, 3]),
        ("aaab", "a+ a a b", &[0, 1, 2, 3]),
        ("aaab", "a+ a a", &[0, 1, 2]),
        // spaCy: [0, 1, 2]. The greedy `a+` takes two tokens, `a` the
        // third, and `a?` none.
        ("aaab", "a+ a a?", &[0, 0, 1]),
        ("aaaa", "a a a a a?", &[0, 1, 2, 3]),
        ("aaab", "a+ a b", &[0, 0, 1, 2]),
        ("aaab", "a+ a+ b", &[0, 0, 1, 2]),
        ("aaab", "a{2,} b", &[0, 0, 0, 1]),
        ("aaab", "a{,3} b", &[0, 0, 0, 1]),
        ("aaab", "a{2} b", &[0, 0, 1]),
        ("aaab", "a{2,3} b", &[0, 0, 0, 1]),
    ];
    for &(s, p, want) in cases {
        let m = greedy("PATTERN", vec![ops(p)], Greedy::Longest);
        let d = chars(s);
        let first = find(&m, &d)[0];
        assert_eq!(m.align(&d.seq(), &first).unwrap().as_deref(), Some(want), "{s} {p}");
    }
}

/// `test_matcher_with_alignments_non_greedy`: every match's alignment is
/// one spaCy gives for a match of that pattern, except the two cases noted.
#[test]
fn alignments_non_greedy() {
    let cases: &[(&str, &str, &[&[u16]])] = &[
        ("aaab", "a* b", &[&[0, 1], &[0, 0, 1], &[0, 0, 0, 1], &[1]]),
        ("baab", "b a* b", &[&[0, 1, 1, 2]]),
        ("aaab", "a a a b", &[&[0, 1, 2, 3]]),
        ("aaab", "a+ b", &[&[0, 1], &[0, 0, 1], &[0, 0, 0, 1]]),
        ("aaba", "a+ b a+", &[&[0, 1, 2], &[0, 0, 1, 2]]),
        ("aabaa", "a+ b a+", &[&[0, 1, 2], &[0, 0, 1, 2], &[0, 0, 1, 2, 2], &[0, 1, 2, 2]]),
        ("aaba", "a+ b a*", &[&[0, 1], &[0, 0, 1], &[0, 0, 1, 2], &[0, 1, 2]]),
        ("aaaa", "a*", &[&[0], &[0, 0], &[0, 0, 0], &[0, 0, 0, 0]]),
        ("baab", "b a* b b*", &[&[0, 1, 1, 2]]),
        // spaCy aligns a run of `a` alone to the last `a*` ([2], [2, 2]);
        // the greedy choice is the first ([0], [0, 0]).
        ("aabb", "a* b* a*", &[&[1], &[0], &[0, 0], &[1, 1], &[0, 1], &[0, 0, 1], &[0, 0, 1, 1], &[0, 1, 1]]),
        ("aaab", "a+ a+ a b", &[&[0, 1, 2, 3]]),
        ("aaab", "a+ a+ a+ b", &[&[0, 1, 2, 3]]),
        ("aaab", "a+ a a b", &[&[0, 1, 2, 3]]),
        ("aaab", "a+ a a", &[&[0, 1, 2]]),
        // spaCy: [[0, 1], [0, 1, 2]]; the greedy alignment of `aaa` is
        // [0, 0, 1].
        ("aaab", "a+ a a?", &[&[0, 1], &[0, 0, 1]]),
        ("aaaa", "a a a a a?", &[&[0, 1, 2, 3]]),
        ("aaab", "a+ a b", &[&[0, 1, 2], &[0, 0, 1, 2]]),
        ("aaab", "a+ a+ b", &[&[0, 1, 2], &[0, 0, 1, 2]]),
        ("aaab", "a{2,} b", &[&[0, 0, 1], &[0, 0, 0, 1]]),
        ("aaab", "a{3} b", &[&[0, 0, 0, 1]]),
        ("aaab", "a{2} b", &[&[0, 0, 1]]),
        ("aaab", "a{2,3} b", &[&[0, 0, 1], &[0, 0, 0, 1]]),
    ];
    for &(s, p, allowed) in cases {
        let m = matcher(&[("PATTERN", vec![ops(p)])]);
        let d = chars(s);
        for x in find(&m, &d) {
            let a = m.align(&d.seq(), &x).unwrap().unwrap();
            assert_eq!(a.len() as u32, x.end - x.start);
            assert!(allowed.contains(&a.as_slice()), "{s} {p} {x:?} {a:?}");
        }
    }
}

// test_matcher_api.py

fn api_matcher() -> Matcher {
    matcher(&[
        ("JS", vec![vec![Tok::text("JavaScript")]]),
        ("GoogleNow", vec![vec![Tok::text("Google"), Tok::text("Now")]]),
        ("Java", vec![vec![Tok::lower("java")]]),
    ])
}

fn labelled(rows: &[(&str, u32, u32)]) -> Vec<(String, u32, u32)> {
    rows.iter().map(|&(l, s, e)| (l.into(), s, e)).collect()
}

#[test]
fn matches_at_start_end_middle_and_several() {
    let m = api_matcher();
    assert!(find(&m, &doc(&["I", "like", "cheese", "."])).is_empty());
    assert_eq!(found(&m, &doc(&["JavaScript", "is", "good"])), labelled(&[("JS", 0, 1)]));
    assert_eq!(found(&m, &doc(&["I", "like", "java"])), labelled(&[("Java", 2, 3)]));
    assert_eq!(found(&m, &doc(&["I", "like", "Google", "Now", "best"])), labelled(&[("GoogleNow", 2, 4)]));
    assert_eq!(
        found(&m, &doc(&["I", "like", "Google", "Now", "and", "java", "best"])),
        labelled(&[("GoogleNow", 2, 4), ("Java", 5, 6)])
    );
    let d = split("JavaScript is good but Java is better");
    assert_eq!(found(&m, &d), labelled(&[("JS", 0, 1), ("Java", 4, 5)]));
}

#[test]
fn add_two_patterns() {
    let m = matcher(&[("NEW_API", vec![vec![Tok::text("a")], vec![Tok::text("a"), Tok::text("b")]])]);
    assert_eq!(find(&m, &doc(&["a", "b"])).len(), 2);
}

#[test]
fn fuzzy() {
    let d = doc(&["They", "like", "Goggle", "Now", "and", "Jav", "but", "not", "JvvaScrpt"]);
    let google = || vec![Tok::any().fuzzy(Attr::Text, "Google", None), Tok::text("Now")];
    let java = || vec![Tok::any().fuzzy(Attr::Lower, "java", None)];
    let js = || vec![Tok::any().fuzzy(Attr::Text, "JavaScript", None)];
    assert_eq!(spans(&matcher(&[("GoogleNow", vec![google()])]), &d), vec![(2, 4)]);
    assert_eq!(spans(&matcher(&[("Java", vec![java()])]), &d), vec![(5, 6)]);
    assert_eq!(
        spans(&matcher(&[("JS", vec![js()]), ("GoogleNow", vec![google()]), ("Java", vec![java()])]), &d),
        vec![(2, 4), (5, 6), (8, 9)]
    );
    let m = matcher(&[
        ("A", vec![vec![Tok::any().fuzzy(Attr::Text, "Javascripts", None)]]),
        ("B", vec![vec![Tok::any().fuzzy(Attr::Text, "Javascripts", Some(5))]]),
    ]);
    assert_eq!(spans(&m, &d), vec![(8, 9)]);
}

#[test]
fn fuzzy_sets_with_greedy() {
    let d = doc(&["They", "like", "Goggle", "Noo"]);
    for max in [None, Some(2)] {
        let in_set = Tok::any().fuzzy_in(Attr::Text, &["Google", "Now"], max).one_or_more();
        let not_in = Tok::any().fuzzy_not_in(Attr::Text, &["Google", "Now"], max).one_or_more();
        for g in [Greedy::First, Greedy::Longest] {
            let m = greedy("GoogleNow", vec![vec![in_set.clone()]], g);
            assert_eq!(spans(&m, &d), vec![(2, 4)]);
            let m = greedy("GoogleNow", vec![vec![not_in.clone()]], g);
            assert_eq!(spans(&m, &d), vec![(0, 2)]);
        }
    }
    for max in [None, Some(1)] {
        let t = Tok::any().fuzzy_in(Attr::Text, &["Google", "Now"], max).not_in(Attr::Text, &["Goggle"]).one_or_more();
        assert_eq!(spans(&greedy("GoogleNow", vec![vec![t]], Greedy::Longest), &d), vec![(3, 4)]);
    }
}

#[test]
fn fuzzy_with_a_fixed_bound() {
    let words: Vec<String> = (0..10).map(|i| format!("GoogleNow{}", "a".repeat(i))).collect();
    let d = doc(&words.iter().map(String::as_str).collect::<Vec<_>>());
    for n in 1..10 {
        let m = matcher(&[("GoogleNow", vec![vec![Tok::any().fuzzy(Attr::Text, "GoogleNow", Some(n))]])]);
        assert_eq!(find(&m, &d).len(), n as usize + 1);
    }
    let d = doc(&["GoogleNow", "GoogleNuw", "GoogleNuew", "GoogleNoweee", "GiggleNuw3", "gouggle5New"]);
    for n in 1..6 {
        let m = matcher(&[("GoogleNow", vec![vec![Tok::any().fuzzy(Attr::Text, "GoogleNow", Some(n))]])]);
        assert_eq!(find(&m, &d).len(), n as usize + 1);
    }
}

#[test]
fn any_token() {
    let d = doc(&["a", "b", "c"]);
    assert_eq!(spans(&matcher(&[("A.C", vec![vec![Tok::text("a"), Tok::any(), Tok::text("c")]])]), &d), vec![(0, 3)]);
    assert_eq!(spans(&matcher(&[("A.", vec![vec![Tok::text("a"), Tok::any()]])]), &d), vec![(0, 2)]);
    let p = vec![Tok::text("a"), Tok::any().flag(Flag::Alpha, true).one_or_more(), Tok::text("c")];
    assert_eq!(spans(&matcher(&[("A.C", vec![p])]), &d), vec![(0, 3)]);
    let m = matcher(&[("TEST", vec![vec![Tok::text("test"), Tok::any().zero_or_more()]])]);
    let d = doc(&["test", "hello", "world"]);
    let texts: Vec<&str> = find(&m, &d).iter().map(|x| d.words(x.start, x.end)).collect();
    assert_eq!(texts, vec!["test", "test hello", "test hello world"]);
}

#[test]
fn match_zero() {
    let p1 = vec![
        Tok::text("\""),
        Tok::any().flag(Flag::Punct, true).negated(),
        Tok::any().flag(Flag::Punct, true).negated(),
        Tok::text("\""),
    ];
    let p2 = vec![
        Tok::text("\""),
        Tok::any().flag(Flag::Punct, true),
        Tok::any().flag(Flag::Punct, true),
        Tok::any().flag(Flag::Punct, true),
        Tok::text("\""),
    ];
    let m = matcher(&[("Quote", vec![p1.clone()])]);
    assert_eq!(find(&m, &split("He said , \" some words \" ...")).len(), 1);
    assert_eq!(find(&m, &split("He said , \" some three words \" ...")).len(), 0);
    let m = matcher(&[("Quote", vec![p1, p2])]);
    assert_eq!(find(&m, &split("He said , \" some three words \" ...")).len(), 0);
    let p = vec![Tok::text("\""), Tok::any().flag(Flag::Punct, false).zero_or_more(), Tok::text("\"")];
    assert_eq!(find(&matcher(&[("Quote", vec![p])]), &split("He said , \" some words \" ...")).len(), 1);
}

#[test]
fn match_one_plus() {
    let d = doc(&["Philippe", "Philippe"]);
    assert_eq!(find(&matcher(&[("BasicPhilippe", vec![vec![Tok::text("Philippe")]])]), &d).len(), 2);
    let p = vec![Tok::text("Philippe"), Tok::text("Philippe").one_or_more()];
    let mut b = Matcher::builder();
    for (l, ps) in [
        ("JS", vec![vec![Tok::text("JavaScript")]]),
        ("GoogleNow", vec![vec![Tok::text("Google"), Tok::text("Now")]]),
        ("Java", vec![vec![Tok::lower("java")]]),
        ("KleenePhilippe", vec![p]),
    ] {
        b.add(l, ps, None).unwrap();
    }
    assert_eq!(find(&b.build().unwrap(), &d).len(), 1);
}

#[test]
fn column_values() {
    let fruit = |w: &[&str]| -> Vec<&'static str> {
        w.iter().map(|w| if ["apple", "banana"].contains(w) { "True" } else { "False" }).collect()
    };
    let m = matcher(&[("HAVING_FRUIT", vec![vec![Tok::text("an"), Tok::any().eq(Attr::Column("is_fruit"), "True")]])]);
    let w = ["an", "apple"];
    assert_eq!(find(&m, &doc(&w).column("is_fruit", &fruit(&w))).len(), 1);
    let w = ["an", "aardvark"];
    assert_eq!(find(&m, &doc(&w).column("is_fruit", &fruit(&w))).len(), 0);
    // test_matcher_extension_set_membership
    let m = matcher(&[("REVERSED", vec![vec![Tok::any().is_in(Attr::Column("reversed"), &["eyb", "ih"])]])]);
    let d = doc(&["hi", "bye", "hello"]).column("reversed", &["ih", "eyb", "olleh"]);
    assert_eq!(find(&m, &d).len(), 2);
    let d = doc(&["aardvark"]).column("reversed", &["kravdraa"]);
    assert_eq!(find(&m, &d).len(), 0);
}

#[test]
fn set_values_and_operators() {
    let m = matcher(&[("A_OR_AN", vec![vec![Tok::any().is_in(Attr::Text, &["an", "a"])]])]);
    assert_eq!(find(&m, &doc(&["an", "a", "apple"])).len(), 2);
    assert_eq!(find(&m, &doc(&["aardvark"])).len(), 0);
    let m = matcher(&[(
        "DET_HOUSE",
        vec![vec![Tok::any().is_in(Attr::Text, &["a", "the"]).optional(), Tok::text("house")]],
    )]);
    assert_eq!(find(&m, &doc(&["In", "a", "house"])).len(), 2);
    assert_eq!(find(&m, &doc(&["my", "house"])).len(), 1);
    // IS_SUBSET acts as IN for attributes other than MORPH; an empty set
    // matches nothing.
    let d = doc(&["a", "b", "c"]).tags(&["A", "", ""]);
    assert_eq!(find(&matcher(&[("M", vec![vec![Tok::any().is_in(Attr::Pos, &["A", "B"])]])]), &d).len(), 1);
    assert_eq!(find(&matcher(&[("M", vec![vec![Tok::any().is_in(Attr::Pos, &[])]])]), &d).len(), 0);
}

#[test]
fn regular_expressions() {
    let m = matcher(&[("A_OR_AN", vec![vec![Tok::any().regex(Attr::Text, "(?:a|an)")]])]);
    assert_eq!(find(&m, &doc(&["an", "a", "hi"])).len(), 2);
    assert_eq!(find(&m, &doc(&["bye"])).len(), 0);
    let m = matcher(&[("A_OR_AN", vec![vec![Tok::any().regex_in(Attr::Text, &["(?:a)", "(?:an)"])]])]);
    assert_eq!(find(&m, &doc(&["an", "a", "hi"])).len(), 2);
    assert_eq!(find(&m, &doc(&["bye"])).len(), 0);
    let m = matcher(&[("A_OR_AN", vec![vec![Tok::any().regex_not_in(Attr::Text, &["(?:a)", "(?:an)"])]])]);
    assert_eq!(find(&m, &doc(&["an", "a", "hi"])).len(), 1);
    assert_eq!(find(&m, &doc(&["bye"])).len(), 1);
    let m = matcher(&[("NON_ALPHA", vec![vec![Tok::any().regex(Attr::Shape, "^[^x]+$")]])]);
    assert_eq!(find(&m, &doc(&["99", "problems", "!"])).len(), 2);
    assert_eq!(find(&m, &doc(&["bye"])).len(), 0);
}

#[test]
fn compare_length() {
    let cases: [(Cmp, &[&str]); 6] = [
        (Cmp::Eq, &["a", "aaa"]),
        (Cmp::Ne, &["aa"]),
        (Cmp::Ge, &["a"]),
        (Cmp::Le, &["aaa"]),
        (Cmp::Gt, &["a", "aa"]),
        (Cmp::Lt, &["aa", "aaa"]),
    ];
    for (cmp, bad) in cases {
        let m = matcher(&[("LENGTH_COMPARE", vec![vec![Tok::any().length(cmp, 2)]])]);
        assert_eq!(find(&m, &doc(&["a", "aa", "aaa"])).len(), 3 - bad.len(), "{cmp:?}");
        assert_eq!(find(&m, &doc(bad)).len(), 0, "{cmp:?}");
    }
}

#[test]
fn missing_layers_are_reported() {
    for (attr, d) in [(Attr::Pos, doc(&["Test"])), (Attr::Lemma, doc(&["Test"])), (Attr::Entity, doc(&["Test"]))] {
        let m = matcher(&[("TEST", vec![vec![Tok::any().eq(attr, "a")]])]);
        assert_eq!(m.find(&d.seq()).err(), Some(MatchError::Missing(attr)));
    }
    let m = matcher(&[("TEST", vec![vec![Tok::any().eq(Attr::Pos, "a")]])]);
    assert!(m.find(&doc(&["Test"]).tags(&["TAG"]).seq()).is_ok());
    let m = matcher(&[("TEST", vec![vec![Tok::text("a")]])]);
    assert!(m.find(&doc(&["Test"]).seq()).is_ok());
    let m = matcher(&[("TEST", vec![vec![Tok::any().flag(Flag::Stop, true)]])]);
    assert_eq!(m.find(&doc(&["the"]).seq()).err(), Some(MatchError::MissingFlag(Flag::Stop)));
}

#[test]
fn token_attribute_flags() {
    let cases = [
        (Flag::Alpha, "a"),
        (Flag::Ascii, "a"),
        (Flag::Digit, "1"),
        (Flag::Lower, "a"),
        (Flag::Upper, "A"),
        (Flag::Title, "Aaaa"),
        (Flag::Punct, "."),
        (Flag::Space, "\n"),
        (Flag::Bracket, "["),
        (Flag::Quote, "\""),
        (Flag::LikeNum, "1"),
    ];
    for (flag, text) in cases {
        let m = matcher(&[("Rule", vec![vec![Tok::any().flag(flag, true)]])]);
        assert_eq!(find(&m, &doc(&[text])).len(), 1, "{flag:?}");
    }
    let m = matcher(&[("Rule", vec![vec![Tok::any().flag(Flag::Stop, true)]])]);
    assert_eq!(find(&m, &doc(&["the"]).stop(&[true])).len(), 1);
    // spaCy's SPACY: a token followed by whitespace. The last token of a
    // Doc built from words has a space after it in spaCy; here the text
    // ends there, so the test uses two words.
    let m = matcher(&[("Rule", vec![vec![Tok::any().flag(Flag::SpaceAfter, true)]])]);
    assert_eq!(spans(&m, &doc(&["the", "end"])), vec![(0, 1)]);
}

#[test]
fn no_zero_length_matches() {
    let d = doc(&["a", "b"]).tags(&["A", "B"]);
    let m = matcher(&[("TEST", vec![vec![Tok::pos("C").optional()]])]);
    assert!(find(&m, &d).is_empty());
    // test_matcher_remove_zero_operator: `!` alone tests nothing, so it
    // can never match, and is refused.
    let mut b = Matcher::builder();
    assert!(b.add("Rule", vec![vec![Tok::any().op(Op::Not)]], None).is_err());
}

#[test]
fn min_max_operators() {
    let d = doc(&["foo", "bar", "foo", "foo", "bar", "foo", "foo", "foo", "bar", "bar"]);
    for (op, n) in [
        (Op::Repeat { min: 3, max: Some(3) }, 1),
        (Op::Repeat { min: 2, max: None }, 4),
        (Op::Repeat { min: 0, max: Some(2) }, 9),
        (Op::Repeat { min: 2, max: Some(3) }, 4),
    ] {
        let m = matcher(&[("TEST", vec![vec![Tok::text("foo").op(op)]])]);
        assert_eq!(find(&m, &d).len(), n, "{op:?}");
    }
}

#[test]
fn long_matches_align_without_deep_recursion() {
    // One token per step of the walk: a recursive walk would overflow the
    // stack here.
    let mut words = vec!["a"; 200_000];
    words.push("b");
    let d = doc(&words);
    let mut b = Matcher::builder();
    b.add("R", vec![vec![Tok::text("a").one_or_more(), Tok::text("b")]], Some(Greedy::Longest)).unwrap();
    let m = b.build().unwrap();
    // Built directly: finding it runs the pattern from every start to the
    // end of the document, which is slow in a debug build.
    let x = Match { start: 0, end: 200_001, label: m.labels().tag("R").unwrap() };
    let a = m.align(&d.seq(), &x).unwrap().unwrap();
    assert_eq!(a.len(), 200_001);
    assert!(a[..200_000].iter().all(|&x| x == 0) && a[200_000] == 1);
}

#[test]
fn the_greedy_filter_keeps_many_matches_quickly() {
    // Every token a match: the filter keeps them all without comparing
    // each with every kept one.
    let words = vec!["a"; 400_000];
    let d = doc(&words);
    let mut b = Matcher::builder();
    b.add("R", vec![vec![Tok::text("a")]], Some(Greedy::First)).unwrap();
    let t = std::time::Instant::now();
    assert_eq!(b.build().unwrap().find(&d.seq()).unwrap().len(), 400_000);
    assert!(t.elapsed().as_secs() < 20, "{:?}", t.elapsed());
}

#[test]
fn repeats_too_long_to_write_out_are_refused() {
    let mut b = Matcher::builder();
    assert_eq!(b.add("R", vec![vec![Tok::text("a").repeat(0, Some(5_000_000))]], None), Err(PatternError::TooLong));
    assert_eq!(b.add("R", vec![vec![Tok::text("a").repeat(u32::MAX, None)]], None), Err(PatternError::TooLong));
    assert!(b.add("R", vec![vec![Tok::text("a").repeat(0, Some(1000))]], None).is_ok());
}
