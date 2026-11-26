//! Match alignments: which pattern token each matched token is for.

use super::*;

/// A document of one token per character of `s`.
fn chars(s: &str) -> Doc {
    let words: Vec<String> = s.chars().map(String::from).collect();
    Doc::words(&words.iter().map(String::as_str).collect::<Vec<_>>())
}

/// `test_matcher_with_alignments_greedy_longest`. Where the alignment
/// spaCy gives is not the greedy backtracking one, the expected value here
/// is the greedy one and spaCy's is in the comment.
#[test]
fn alignments_greedy_longest() {
    let cases: &[(&str, &str, &[u32])] = &[
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
        let mut m = matcher(&[("PATTERN", vec![notation(p)])]);
        m.set_greedy(m.label("PATTERN").unwrap(), Greedy::Longest);
        let d = chars(s);
        let input = d.input();
        let first = m.find(&input).unwrap()[0];
        assert_eq!(m.align(&input, &first).unwrap().as_deref(), Some(want), "{s} {p}");
    }
}

/// `test_matcher_with_alignments_non_greedy`: every match's alignment is
/// one spaCy gives for a match of that pattern, except the two cases noted.
#[test]
fn alignments_non_greedy() {
    let cases: &[(&str, &str, &[&[u32]])] = &[
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
        let m = matcher(&[("PATTERN", vec![notation(p)])]);
        let d = chars(s);
        let input = d.input();
        for x in m.find(&input).unwrap() {
            let a = m.align(&input, &x).unwrap().unwrap();
            assert_eq!(a.len(), x.tokens.len());
            assert!(allowed.contains(&a.as_slice()), "{s} {p} {x:?} {a:?}");
        }
    }
}

#[test]
fn alignments_of_negated_and_repeated_tokens_and_hand_made_matches() {
    let d = Doc::split("x a b c y");
    let m = matcher(&[("R", vec![vec![Tok::text("x"), (!Tok::text("y")).repeat(1, Some(3)), Tok::text("y").opt()]])]);
    let input = d.input();
    let found = m.find(&input).unwrap();
    let last = *found.last().unwrap();
    assert_eq!(d.text_of(last.tokens), "x a b c y");
    assert_eq!(m.align(&input, &last).unwrap(), Some(vec![0, 1, 1, 1, 2]));
    // A match the pattern cannot take, made by hand.
    let wrong = turbo_matcher::Match { tokens: IxRange::new(1, 3), ..last };
    assert_eq!(m.align(&input, &wrong).unwrap(), None);
    let past = turbo_matcher::Match { tokens: IxRange::new(4, 9), ..last };
    assert!(m.align(&input, &past).is_err());
}

#[test]
fn long_matches_align_without_deep_recursion() {
    // A recursive walk would take one stack frame per token here.
    let mut words = vec!["a"; 200_000];
    words.push("b");
    let d = Doc::words(&words);
    let m = matcher(&[("R", vec![vec![Tok::text("a").plus(), Tok::text("b")]])]);
    // Made directly: finding it runs the pattern from every start to the
    // end of the document, which is slow in a debug build.
    let x = turbo_matcher::Match {
        tokens: IxRange::new(0, 200_001),
        label: m.label("R").unwrap(),
        pattern: turbo_matcher::PatternId(0),
    };
    let a = m.align(&d.input(), &x).unwrap().unwrap();
    assert_eq!(a.len(), 200_001);
    assert!(a[..200_000].iter().all(|&x| x == 0) && a[200_000] == 1);
}
