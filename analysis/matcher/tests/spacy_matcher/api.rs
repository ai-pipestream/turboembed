//! spaCy's tests of the `Matcher` API, and repeats that are refused.

use super::*;

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
    for (attr, layer) in [(Attr::Pos, "turbo:pos"), (Attr::Lemma, "turbo:lemmas")] {
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
