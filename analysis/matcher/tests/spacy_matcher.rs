//! spaCy's `Matcher` tests (spaCy, MIT: `spacy/tests/matcher/
//! test_matcher_logic.py` and `test_matcher_api.py`), ported with the same
//! expected matches. Each test names the one it ports. spaCy's dict
//! patterns are written with `Tok`: `{"ORTH": x}` is `Tok::text(x)`,
//! `{"LOWER": x}` is `Tok::lower(x)`, `{"TAG": x}` and `{"POS": x}` are
//! `Tok::attr(Attr::Pos, x)`, a custom extension attribute is
//! `Tok::text_where`, and `"OP"` is `!`, `.opt()`, `.star()`,
//! `.plus()` or `.repeat()`. Documents built with spaCy's tokenizer are
//! built here from the tokens it makes.

mod common;

use common::Doc;
use turbo_analysis::IxRange;
use turbo_matcher::{Attr, Cmp, Flag, Greedy, Iob, MatchError, Matcher, PatternError, Tok};

fn matcher(rules: &[(&str, Vec<Vec<Tok>>)]) -> Matcher {
    let mut m = Matcher::new();
    for (label, patterns) in rules {
        for p in patterns {
            m.add(label, p).unwrap();
        }
    }
    m
}

fn spans(found: &[(String, usize, usize)]) -> Vec<(usize, usize)> {
    found.iter().map(|m| (m.1, m.2)).collect()
}

/// A pattern from spaCy's test notation: `a`, `a+`, `a*`, `a?`, `a{2,3}`.
fn notation(pattern: &str) -> Vec<Tok> {
    pattern
        .split(' ')
        .map(|part| {
            let tok = Tok::text(&part[..1]);
            match &part[1..] {
                "" => tok,
                "+" => tok.plus(),
                "*" => tok.star(),
                "?" => tok.opt(),
                range => {
                    let inner = range.trim_start_matches('{').trim_end_matches('}');
                    let (min, max) = match inner.split_once(',') {
                        None => {
                            let n = inner.parse().unwrap();
                            (n, Some(n))
                        }
                        Some((a, b)) => (a.parse().unwrap_or(0), b.parse().ok()),
                    };
                    tok.repeat(min, max)
                }
            }
        })
        .collect()
}

const CELTICS: &str = "how many points did lebron james score against the boston celtics last night";

#[test]
fn issue118() {
    for patterns in [
        vec![vec![Tok::lower("celtics")], vec![Tok::lower("boston"), Tok::lower("celtics")]],
        vec![vec![Tok::lower("boston"), Tok::lower("celtics")], vec![Tok::lower("celtics")]],
    ] {
        let m = matcher(&[("BostonCeltics", patterns)]);
        assert_eq!(spans(&Doc::split(CELTICS).run(&m)), [(9, 11), (10, 11)]);
    }
}

#[test]
fn issue118_prefix_reorder() {
    for patterns in [
        vec![vec![Tok::lower("boston")], vec![Tok::lower("boston"), Tok::lower("celtics")]],
        vec![vec![Tok::lower("boston"), Tok::lower("celtics")], vec![Tok::lower("boston")]],
    ] {
        let m = matcher(&[("BostonCeltics", patterns)]);
        assert_eq!(spans(&Doc::split(CELTICS).run(&m)), [(9, 10), (9, 11)]);
    }
}

#[test]
fn issue242() {
    let doc = Doc::split("There are different food safety standards in different countries .");
    let m = matcher(&[(
        "FOOD",
        vec![vec![Tok::lower("food"), Tok::lower("safety")], vec![Tok::lower("safety"), Tok::lower("standards")]],
    )]);
    assert_eq!(spans(&doc.run(&m)), [(3, 5), (4, 6)]);
}

#[test]
fn issue587() {
    let doc = Doc::split("a b ; c");
    let mut m = Matcher::new();
    m.add("TEST1", &[Tok::text("a"), Tok::text("b")]).unwrap();
    assert_eq!(doc.run(&m).len(), 1);
    m.add("TEST2", &[Tok::text("a"), Tok::text("b"), Tok::flag(Flag::Punct, true), Tok::text("c")]).unwrap();
    assert_eq!(doc.run(&m).len(), 2);
    m.add("TEST3", &[Tok::text("a"), Tok::text("b"), Tok::flag(Flag::Punct, true), Tok::text("d")]).unwrap();
    assert_eq!(doc.run(&m).len(), 2);
}

#[test]
fn issue588() {
    assert_eq!(Matcher::new().add("TEST", &[]), Err(PatternError::Empty));
}

#[test]
fn issue590() {
    let doc = Doc::words(&["n", "=", "1", ";", "a", ":", "5", "%"]);
    let alpha = || Tok::flag(Flag::Alpha, true);
    let num = || Tok::flag(Flag::LikeNum, true);
    let m = matcher(&[(
        "ab",
        vec![vec![alpha(), Tok::text(":"), num(), Tok::text("%")], vec![alpha(), Tok::text("="), num()]],
    )]);
    assert_eq!(doc.run(&m).len(), 2);
}

#[test]
fn issue850() {
    let m = matcher(&[("FarAway", vec![vec![Tok::lower("bob"), Tok::any().star(), Tok::lower("frank")]])]);
    assert_eq!(spans(&Doc::words(&["bob", "and", "and", "frank"]).run(&m)), [(0, 4)]);
}

#[test]
fn issue850_basic() {
    let m = matcher(&[("FarAway", vec![vec![Tok::lower("bob"), Tok::lower("and").star(), Tok::lower("frank")]])]);
    assert_eq!(spans(&Doc::words(&["bob", "and", "and", "frank"]).run(&m)), [(0, 4)]);
}

#[test]
fn issue1434() {
    let m = matcher(&[("MyMatcher", vec![vec![Tok::text("Hello"), Tok::flag(Flag::Alpha, true).opt()]])]);
    assert!(!Doc::words(&["Hello", "World"]).run(&m).is_empty());
    assert!(!Doc::words(&["Hello"]).run(&m).is_empty());
}

#[test]
fn issue1450() {
    let m = matcher(&[("TSTEND", vec![vec![Tok::text("a"), Tok::text("b").star()]])]);
    for (string, start, end) in
        [("a", 0, 1), ("a b", 0, 2), ("a c", 0, 1), ("a b c", 0, 2), ("a b b c", 0, 3), ("a b b", 0, 3)]
    {
        let found = Doc::split(string).run(&m);
        assert_eq!(spans(&found).last(), Some(&(start, end)), "{string}");
    }
}

#[test]
fn issue1945() {
    let m = matcher(&[("MWE", vec![vec![Tok::text("a"), Tok::text("a")]])]);
    assert_eq!(spans(&Doc::words(&["a", "a", "a"]).run(&m)), [(0, 2), (1, 3)]);
}

#[test]
fn issue1971() {
    // `{"_": {"optional": True}, "OP": "?"}` with the extension's default
    // False: a test no token passes.
    let optional = || Tok::text_where(|_| false).opt();
    let pattern = vec![Tok::text("Doe"), Tok::text("!").opt(), optional(), Tok::text("!").opt()];
    let m = matcher(&[("TEST", vec![pattern])]);
    let found = Doc::words(&["Hello", "John", "Doe", "!"]).run(&m);
    assert!(!found.is_empty());
    assert!(found.iter().all(|f| f.0 == "TEST"));
}

#[test]
fn issue_1971_2() {
    let num = || Tok::flag(Flag::LikeNum, true);
    let pattern1 = vec![Tok::text("EUR").and(Tok::attr_in(Attr::Lower, ["eur"])), num()];
    let pattern2 = vec![num(), Tok::text("EUR")];
    let m = matcher(&[("TEST1", vec![pattern1, pattern2])]);
    assert_eq!(Doc::words(&["EUR", "10", "is", "10", "EUR"]).run(&m).len(), 2);
}

#[test]
fn issue_1971_3() {
    // Extensions `a` (default 1) and `b` (default 2) tested for their
    // defaults: every token passes.
    let m = matcher(&[("A", vec![vec![Tok::text_where(|_| true)]]), ("B", vec![vec![Tok::text_where(|_| true)]])]);
    let mut found = Doc::words(&["hello", "world"]).run(&m);
    found.sort();
    let want = [("A", 0, 1), ("A", 1, 2), ("B", 0, 1), ("B", 1, 2)].map(|(l, s, e)| (l.to_string(), s, e));
    assert_eq!(found, want);
}

#[test]
fn issue_1971_4() {
    let ext = || Tok::text_where(|_| true).and(Tok::text_where(|_| true));
    let m = matcher(&[("TEST", vec![vec![ext(), ext(), ext()]])]);
    let found = Doc::words(&["this", "is", "text"]).run(&m);
    assert_eq!(found, [("TEST".to_string(), 0, 3)]);
}

#[test]
fn issue2464() {
    let m = matcher(&[("4", vec![vec![Tok::any().opt(), Tok::any().opt()]])]);
    assert_eq!(Doc::words(&["a", "b"]).run(&m).len(), 3);
}

#[test]
fn issue2569() {
    // spaCy's tokenizer splits "It is May 15, 1993." into these tokens.
    let doc = Doc::words(&["It", "is", "May", "15", ",", "1993", "."]).with_ents(&[(2, 6, "DATE")]);
    let m = matcher(&[("RULE", vec![vec![Tok::attr(Attr::Entity, "DATE").plus()]])]);
    let mut found = m.find(&doc.input()).unwrap();
    assert_eq!(found.len(), 10);
    found.sort_by_key(|f| std::cmp::Reverse(f.tokens.len()));
    assert_eq!(found[0].tokens.len(), 4);
    // spaCy prints "May 15, 1993", keeping the source's spacing; these
    // words were joined with a space.
    assert_eq!(doc.text_of(found[0].tokens), "May 15 , 1993");
}

#[test]
fn issue2671() {
    let pattern = vec![Tok::lower("high"), Tok::flag(Flag::Punct, true).opt(), Tok::lower("adrenaline")];
    let m = matcher(&[("test_pattern", vec![pattern])]);
    for words in [
        ["This", "is", "a", "high", "-", "adrenaline", "situation", "."].as_slice(),
        ["This", "is", "a", "high", "adrenaline", "situation", "."].as_slice(),
    ] {
        let found = Doc::words(words).run(&m);
        assert_eq!(found.len(), 1);
        assert!(found.iter().all(|f| f.0 == "test_pattern"));
    }
}

#[test]
fn issue3009() {
    let ascii_word = || Tok::flag(Flag::Ascii, true).and(Tok::flag(Flag::Punct, false));
    let tail = || vec![Tok::lower("to"), Tok::lower("do"), Tok::attr(Attr::Pos, "IN")];
    let patterns = [
        [vec![Tok::text("has")], tail()].concat(),
        [vec![Tok::text("has"), ascii_word().star()], tail()].concat(),
        [vec![Tok::text("has"), ascii_word().opt()], tail()].concat(),
    ];
    let doc = Doc::words(&["also", "has", "to", "do", "with"]).with_tags(&["RB", "VBZ", "TO", "VB", "IN"]);
    let mut m = Matcher::new();
    for (i, p) in patterns.iter().enumerate() {
        m.add(&i.to_string(), p).unwrap();
        assert!(!doc.run(&m).is_empty());
    }
}

#[test]
fn issue3328() {
    let doc = Doc::words(&["Hello", ",", "how", "are", "you", "doing", "?"]);
    let m = matcher(&[(
        "TEST",
        vec![vec![Tok::attr_in(Attr::Lower, ["hello", "how"])], vec![Tok::attr_in(Attr::Lower, ["you", "doing"])]],
    )]);
    assert_eq!(doc.texts(&m), ["Hello", "how", "you", "doing"]);
}

#[test]
fn issue3839() {
    let doc = Doc::words(&["terrific", "group", "of", "people"]);
    let m = matcher(&[("PATTERN", vec![vec![Tok::lower("terrific"), Tok::any().opt(), Tok::lower("group")]])]);
    assert_eq!(doc.run(&m)[0].0, "PATTERN");
    let pattern2 = vec![Tok::lower("terrific"), Tok::any().opt(), Tok::any().opt(), Tok::lower("group")];
    let m = matcher(&[("PATTERN", vec![pattern2])]);
    assert_eq!(doc.run(&m)[0].0, "PATTERN");
}

#[test]
fn issue3879() {
    let doc = Doc::words(&["This", "is", "a", "test", "."]);
    let m = matcher(&[("TEST", vec![vec![Tok::text("This").opt(), Tok::any().opt(), Tok::text("test")]])]);
    assert_eq!(doc.run(&m).len(), 2);
}

#[test]
fn issue3951() {
    let pattern = vec![Tok::lower("hello"), Tok::lower("this").opt(), Tok::any().opt(), Tok::lower("world")];
    let m = matcher(&[("TEST", vec![pattern])]);
    assert!(Doc::words(&["Hello", "my", "new", "world"]).run(&m).is_empty());
}

#[test]
fn issue4120() {
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().opt()]])]);
    assert_eq!(Doc::words(&["a"]).run(&m).len(), 1);
    assert_eq!(Doc::words(&["a", "b", "c"]).run(&m).len(), 2);
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().opt(), Tok::text("b")]])]);
    assert_eq!(Doc::words(&["a", "b", "b", "c"]).run(&m).len(), 2);
    let m = matcher(&[("TEST", vec![vec![Tok::text("a"), Tok::any().opt(), Tok::text("b").opt()]])]);
    assert_eq!(Doc::words(&["a", "b", "b", "c"]).run(&m).len(), 3);
}

/// The document of the greedy tests: `" ".join("(BBAAAAAB).")`.
fn greedy_doc() -> Doc {
    Doc::split("( B B A A A A A B ) .")
}

fn greedy_patterns() -> [Vec<Tok>; 5] {
    let t = Tok::text;
    [
        vec![t("A"), t("A").star()],
        vec![t("A").star(), t("A")],
        vec![t("A"), t("A")],
        vec![t("B"), t("A").star(), t("B")],
        vec![t("B").star(), t("A").star(), t("B")],
    ]
}

#[test]
fn greedy_matching_first() {
    // What Python's `re.finditer` finds for "AA*", "A*A", "AA", "BA*B" and
    // "B*A*B" in "(BBAAAAAB).", as the spaCy test compares.
    let re = [vec!["A A A A A"], vec!["A A A A A"], vec!["A A", "A A"], vec!["B B"], vec!["B B A A A A A B"]];
    let doc = greedy_doc();
    for (pattern, want) in greedy_patterns().iter().zip(re) {
        let mut m = Matcher::new();
        m.add("RULE", pattern).unwrap();
        m.set_greedy(m.label("RULE").unwrap(), Greedy::First);
        assert_eq!(doc.texts(&m), want);
    }
}

#[test]
fn greedy_matching_longest() {
    let longest = ["A A A A A", "A A A A A", "A A", "B A A A A A B", "B B A A A A A B"];
    let doc = greedy_doc();
    for (pattern, want) in greedy_patterns().iter().zip(longest) {
        let mut m = Matcher::new();
        m.add("RULE", pattern).unwrap();
        m.set_greedy(m.label("RULE").unwrap(), Greedy::Longest);
        let texts = doc.texts(&m);
        assert!(!texts.is_empty());
        assert!(texts.iter().all(|t| t == want), "{texts:?}");
    }
}

#[test]
fn greedy_matching_longest_first() {
    let doc = Doc::split("C C C");
    let mut m = Matcher::new();
    let label = {
        let p = m.add("RULE", &[Tok::text("C"), Tok::text("C")]).unwrap();
        m.pattern_label(p)
    };
    m.set_greedy(label, Greedy::Longest);
    assert_eq!(spans(&doc.run(&m)), [(0, 2)]);
}

#[test]
fn match_consuming() {
    // As many matches as `re.finditer` finds.
    let counts = [1, 1, 2, 1, 1];
    let doc = greedy_doc();
    for (pattern, want) in greedy_patterns().iter().zip(counts) {
        let mut m = Matcher::new();
        let p = m.add("RULE", pattern).unwrap();
        m.set_greedy(m.pattern_label(p), Greedy::First);
        assert_eq!(doc.run(&m).len(), want);
    }
}

#[test]
fn operator_combos() {
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
    for (string, pattern, found) in cases {
        let doc = Doc::words(&string.split("").filter(|s| !s.is_empty()).collect::<Vec<_>>());
        let m = matcher(&[("PATTERN", vec![notation(pattern)])]);
        assert_eq!(!doc.run(&m).is_empty(), found, "{string} {pattern}");
    }
}

#[test]
fn matcher_end_zero_plus() {
    let m = matcher(&[("TSTEND", vec![vec![Tok::text("a"), Tok::text("b").star()]])]);
    for (string, n) in [("a", 1), ("a b", 2), ("a c", 1), ("a b c", 2), ("a b b c", 3), ("a b b", 3)] {
        assert_eq!(Doc::split(string).run(&m).len(), n, "{string}");
    }
}

#[test]
fn matcher_sets_return_correct_tokens() {
    let m = matcher(&[(
        "TEST",
        vec![
            vec![Tok::attr_in(Attr::Lower, ["zero"])],
            vec![Tok::attr_in(Attr::Lower, ["one"])],
            vec![Tok::attr_in(Attr::Lower, ["two"])],
        ],
    )]);
    assert_eq!(Doc::split("zero one two three").texts(&m), ["zero", "one", "two"]);
}

#[test]
fn matcher_remove() {
    let doc = Doc::split("This is a test case .");
    let mut m = Matcher::new();
    assert_eq!(m.len(), 0);
    m.add("Rule", &[Tok::text("test"), Tok::any().opt()]).unwrap();
    assert!(m.contains("Rule"));
    assert_eq!(doc.run(&m).len(), 2);
    m.remove("Rule").unwrap();
    assert!(doc.run(&m).is_empty());
    assert_eq!(m.remove("Rule"), Err(MatchError::NoLabel("Rule".into())));
}

// test_matcher_api.py

fn api_matcher() -> Matcher {
    matcher(&[
        ("JS", vec![vec![Tok::text("JavaScript")]]),
        ("GoogleNow", vec![vec![Tok::text("Google"), Tok::text("Now")]]),
        ("Java", vec![vec![Tok::lower("java")]]),
    ])
}

fn owned(v: &[(&str, usize, usize)]) -> Vec<(String, usize, usize)> {
    v.iter().map(|&(l, s, e)| (l.to_string(), s, e)).collect()
}

#[test]
fn matcher_from_api_docs() {
    let mut m = Matcher::new();
    assert_eq!(m.len(), 0);
    m.add("Rule", &[Tok::text("test")]).unwrap();
    assert_eq!(m.len(), 1);
    m.remove("Rule").unwrap();
    assert!(!m.contains("Rule"));
    m.add("Rule", &[Tok::text("test")]).unwrap();
    assert!(m.contains("Rule"));
}

#[test]
fn matcher_empty_patterns() {
    // spaCy warns; there is nothing to match.
    assert!(Doc::split("This is quite something").run(&Matcher::new()).is_empty());
}

#[test]
fn matcher_len_contains() {
    let mut m = api_matcher();
    assert_eq!(m.len(), 3);
    m.add("TEST", &[Tok::text("test")]).unwrap();
    assert!(m.contains("TEST"));
    assert!(!m.contains("TEST2"));
}

#[test]
fn matcher_add_new_api() {
    let m = matcher(&[("NEW_API", vec![vec![Tok::text("a")], vec![Tok::text("a"), Tok::text("b")]])]);
    assert_eq!(Doc::split("a b").run(&m).len(), 2);
}

#[test]
fn matcher_no_match() {
    assert!(Doc::split("I like cheese .").run(&api_matcher()).is_empty());
}

#[test]
fn matcher_match_start_end_middle_multi() {
    let m = api_matcher();
    assert_eq!(Doc::split("JavaScript is good").run(&m), owned(&[("JS", 0, 1)]));
    assert_eq!(Doc::split("I like java").run(&m), owned(&[("Java", 2, 3)]));
    assert_eq!(Doc::split("I like Google Now best").run(&m), owned(&[("GoogleNow", 2, 4)]));
    assert_eq!(Doc::split("I like Google Now and java best").run(&m), owned(&[("GoogleNow", 2, 4), ("Java", 5, 6)]));
}

#[test]
fn matcher_empty_dict() {
    let doc = Doc::split("a b c");
    let m = matcher(&[("A.C", vec![vec![Tok::text("a"), Tok::any(), Tok::text("c")]])]);
    assert_eq!(spans(&doc.run(&m)), [(0, 3)]);
    let m = matcher(&[("A.", vec![vec![Tok::text("a"), Tok::any()]])]);
    assert_eq!(spans(&doc.run(&m))[0], (0, 2));
}

#[test]
fn matcher_operator_shadow() {
    let pattern = vec![Tok::text("a"), Tok::flag(Flag::Alpha, true).plus(), Tok::text("c")];
    let m = matcher(&[("A.C", vec![pattern])]);
    assert_eq!(spans(&Doc::split("a b c").run(&m)), [(0, 3)]);
}

#[test]
fn matcher_match_zero() {
    let words1 = Doc::split("He said , \" some words \" ...");
    let words2 = Doc::split("He said , \" some three words \" ...");
    let punct = || Tok::flag(Flag::Punct, true);
    let pattern1 = vec![Tok::text("\""), !punct(), !punct(), Tok::text("\"")];
    let pattern2 = vec![Tok::text("\""), punct(), punct(), punct(), Tok::text("\"")];
    let mut m = api_matcher();
    m.add("Quote", &pattern1).unwrap();
    assert_eq!(words1.run(&m).len(), 1);
    assert_eq!(words2.run(&m).len(), 0);
    m.add("Quote", &pattern2).unwrap();
    assert_eq!(words2.run(&m).len(), 0);
}

#[test]
fn matcher_match_zero_plus() {
    let pattern = vec![Tok::text("\""), Tok::flag(Flag::Punct, false).star(), Tok::text("\"")];
    let m = matcher(&[("Quote", vec![pattern])]);
    assert_eq!(Doc::split("He said , \" some words \" ...").run(&m).len(), 1);
}

#[test]
fn matcher_match_one_plus() {
    let doc = Doc::split("Philippe Philippe");
    let control = matcher(&[("BasicPhilippe", vec![vec![Tok::text("Philippe")]])]);
    assert_eq!(doc.run(&control).len(), 2);
    let mut m = api_matcher();
    m.add("KleenePhilippe", &[Tok::text("Philippe"), Tok::text("Philippe").plus()]).unwrap();
    assert_eq!(doc.run(&m).len(), 1);
}

#[test]
fn matcher_any_token_operator() {
    let m = matcher(&[("TEST", vec![vec![Tok::text("test"), Tok::any().star()]])]);
    assert_eq!(Doc::split("test hello world").texts(&m), ["test", "test hello", "test hello world"]);
}

#[test]
fn matcher_extension_attribute() {
    let is_fruit = || Tok::text_where(|t| t == "apple" || t == "banana");
    let m = matcher(&[("HAVING_FRUIT", vec![vec![Tok::text("an"), is_fruit()]])]);
    assert_eq!(Doc::split("an apple").run(&m).len(), 1);
    assert_eq!(Doc::split("an aardvark").run(&m).len(), 0);
}

#[test]
fn matcher_set_value() {
    let m = matcher(&[("A_OR_AN", vec![vec![Tok::attr_in(Attr::Text, ["an", "a"])]])]);
    assert_eq!(Doc::split("an a apple").run(&m).len(), 2);
    assert_eq!(Doc::split("aardvark").run(&m).len(), 0);
}

#[test]
fn matcher_set_value_operator() {
    let m = matcher(&[("DET_HOUSE", vec![vec![Tok::attr_in(Attr::Text, ["a", "the"]).opt(), Tok::text("house")]])]);
    assert_eq!(Doc::split("In a house").run(&m).len(), 2);
    assert_eq!(Doc::split("my house").run(&m).len(), 1);
}

#[test]
fn matcher_compare_length() {
    for (cmp, bad) in [
        (Cmp::Eq, vec!["a", "aaa"]),
        (Cmp::Ne, vec!["aa"]),
        (Cmp::Ge, vec!["a"]),
        (Cmp::Le, vec!["aaa"]),
        (Cmp::Gt, vec!["a", "aa"]),
        (Cmp::Lt, vec!["aa", "aaa"]),
    ] {
        let m = matcher(&[("LENGTH_COMPARE", vec![vec![Tok::length(cmp, 2)]])]);
        assert_eq!(Doc::split("a aa aaa").run(&m).len(), 3 - bad.len(), "{cmp:?}");
        assert_eq!(Doc::words(&bad).run(&m).len(), 0, "{cmp:?}");
    }
}

#[test]
fn matcher_extension_set_membership() {
    let reversed_in = || Tok::text_where(|t| ["eyb", "ih"].contains(&t.chars().rev().collect::<String>().as_str()));
    let m = matcher(&[("REVERSED", vec![vec![reversed_in()]])]);
    assert_eq!(Doc::split("hi bye hello").run(&m).len(), 2);
    assert_eq!(Doc::split("aardvark").run(&m).len(), 0);
}

#[test]
fn attr_pipeline_checks() {
    let plain = Doc::words(&["Test"]);
    let tagged = Doc::words(&["Test"]).with_tags(&["TAG"]).with_lemmas(&["LEMMA"]);
    for (attr, layer) in [(Attr::Pos, "opennlp:pos"), (Attr::Lemma, "opennlp:lemmas")] {
        let m = matcher(&[("TEST", vec![vec![Tok::attr(attr, "a")]])]);
        assert!(m.find(&tagged.input()).is_ok());
        assert_eq!(m.find(&plain.input()), Err(MatchError::Missing(layer)));
    }
    for attr in [Attr::Text, Attr::Lower] {
        let m = matcher(&[("TEST", vec![vec![Tok::attr(attr, "a")]])]);
        assert!(m.find(&tagged.input()).is_ok() && m.find(&plain.input()).is_ok());
    }
}

#[test]
fn matcher_schema_token_attributes() {
    for (flag, text) in [
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
        (Flag::LeftPunct, "``"),
        (Flag::RightPunct, "''"),
        (Flag::LikeNum, "1"),
    ] {
        let m = matcher(&[("Rule", vec![vec![Tok::flag(flag, true)]])]);
        assert_eq!(m.len(), 1);
        assert_eq!(Doc::words(&[text]).run(&m).len(), 1, "{flag:?}");
    }
    // `IS_STOP` with "the" a stopword, and `SPACY`: spaCy's `Doc(words=)`
    // puts a space after every word.
    let m = matcher(&[("Rule", vec![vec![Tok::flag(Flag::Stop, true)]])]);
    assert_eq!(Doc::words(&["the"]).with_stop(&[true]).run(&m).len(), 1);
    let m = matcher(&[("Rule", vec![vec![Tok::flag(Flag::SpaceAfter, true)]])]);
    assert_eq!(Doc::words_then(&["the"], " ").run(&m).len(), 1);
    assert_eq!(Doc::words(&["the"]).run(&m).len(), 0);
}

#[test]
fn matcher_span() {
    let m = api_matcher();
    let doc = Doc::split("JavaScript is good but Java is better");
    assert_eq!(doc.run(&m).len(), 2);
    assert_eq!(m.find_in(&doc.input(), IxRange::new(0, 3)).unwrap().len(), 1);
    assert_eq!(m.find_in(&doc.input(), IxRange::new(4, 7)).unwrap().len(), 1);
}

#[test]
fn matcher_as_spans() {
    let m = api_matcher();
    let doc = Doc::split("JavaScript is good but Java is better");
    let found = m.find(&doc.input()).unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!((doc.text_of(found[0].tokens), m.label_name(found[0].label)), ("JavaScript", "JS"));
    assert_eq!((doc.text_of(found[1].tokens), m.label_name(found[1].label)), ("Java", "Java"));
    let found = m.find_in(&doc.input(), IxRange::new(1, 7)).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!((doc.text_of(found[0].tokens), m.label_name(found[0].label)), ("Java", "Java"));
}

#[test]
fn matcher_remove_zero_operator() {
    let mut m = matcher(&[("Rule", vec![vec![!Tok::any()]])]);
    assert!(Doc::split("This is a test .").run(&m).is_empty());
    assert!(m.contains("Rule"));
    m.remove("Rule").unwrap();
    assert!(!m.contains("Rule"));
}

#[test]
fn matcher_no_zero_length() {
    let doc = Doc::words(&["a", "b"]).with_tags(&["A", "B"]);
    let m = matcher(&[("TEST", vec![vec![Tok::attr(Attr::Pos, "C").opt()]])]);
    assert!(doc.run(&m).is_empty());
}

#[test]
fn matcher_ent_iob_key() {
    let m = matcher(&[("Rule", vec![vec![Tok::iob(Iob::Inside)]])]);
    let doc1 = Doc::split("I visited New York and California").with_ents(&[(2, 4, "GPE"), (5, 6, "GPE")]);
    let doc2 = Doc::split("I visited my friend Alicia").with_ents(&[(4, 5, "PERSON")]);
    assert_eq!(doc1.texts(&m), ["York"]);
    assert!(doc2.texts(&m).is_empty());
    let m = matcher(&[("Rule", vec![vec![Tok::iob(Iob::Inside).plus()]])]);
    let doc = Doc::split("I visited my friend Anna Maria Esperanza").with_ents(&[(4, 7, "PERSON")]);
    assert_eq!(doc.texts(&m), ["Maria", "Maria Esperanza", "Esperanza"]);
}

#[test]
fn matcher_min_max_operator() {
    let doc = Doc::split("foo bar foo foo bar foo foo foo bar bar");
    let count = |min, max| {
        let m = matcher(&[("TEST", vec![vec![Tok::text("foo").repeat(min, max)]])]);
        doc.run(&m).len()
    };
    assert_eq!(count(3, Some(3)), 1);
    assert_eq!(count(2, None), 4);
    assert_eq!(count(0, Some(2)), 9);
    assert_eq!(count(2, Some(3)), 4);
}

#[test]
fn bad_repeats_are_refused() {
    let mut m = Matcher::new();
    assert_eq!(m.add("R", &[Tok::text("a").repeat(3, Some(2))]), Err(PatternError::Repeat { min: 3, max: 2 }));
    assert_eq!(m.add("R", &[Tok::text("a").repeat(0, Some(0))]), Err(PatternError::Repeat { min: 0, max: 0 }));
    assert!(matches!(m.add("R", &[Tok::text("a").repeat(0, Some(40_000))]), Err(PatternError::TooLarge { .. })));
    assert!(m.is_empty());
}
