//! `test_matcher_logic.py`: spaCy's regression issues, greedy matching, and
//! operators.

use super::*;

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
