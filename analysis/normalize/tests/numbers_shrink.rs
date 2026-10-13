//! `Numbers` and `Shrink`: the expectations of OpenNLP's
//! NumberCharSequenceNormalizerCharacterizationTest and
//! ShrinkCharSequenceNormalizerCharacterizationTest, and the one-pass
//! shrink against a three-pass reference.

mod scanners_support;

use std::borrow::Cow;

use scanners_support::{Rng, placeholders, run};
use turbo_normalize::{Numbers, Shrink, apply};
use turbo_normalize_layers::PlaceholderKind::Number;

fn numbers(input: &str) -> String {
    run(&Numbers, input)
}

fn shrink(input: &str) -> String {
    run(&Shrink, input)
}

#[test]
fn ascii_digit_runs_collapse_to_a_single_space() {
    for (input, expected) in [
        ("", ""),
        ("abc", "abc"),
        ("123", " "),
        ("absc 123,0123 abcd", "absc  ,  abcd"),
        ("a1b22c333d", "a b c d"),
        (" 12 34 ", "     "),
        ("0", " "),
        ("12.34", " . "),
        ("-5", "- "),
        ("12ab34", " ab "),
        ("x0123456789y", "x y"),
        ("1", " "),
        ("a1", "a "),
        ("1a", " a"),
        ("  123  ", "     "),
        ("phone: +1-800-555-0199", "phone: + - - - "),
    ] {
        assert_eq!(numbers(input), expected, "{input:?}");
    }
}

#[test]
fn non_ascii_digits_are_not_touched() {
    for (input, expected) in [
        ("\u{661}\u{662}\u{663}", "\u{661}\u{662}\u{663}"),
        ("\u{9E9}", "\u{9E9}"),
        ("\u{2460}", "\u{2460}"),
        ("\u{1D7CF}", "\u{1D7CF}"),
        ("\u{1F600}1\u{1F600}", "\u{1F600} \u{1F600}"),
        ("1\u{1D7CF}2", " \u{1D7CF} "),
    ] {
        assert_eq!(numbers(input), expected, "{input:?}");
    }
}

#[test]
fn digit_free_input_is_borrowed() {
    assert!(matches!(apply(&Numbers, "no digits in here at all"), Cow::Borrowed(_)));
}

#[test]
fn number_placeholders_span_each_run() {
    let (out, found) = placeholders(Numbers, "call 555-0199 at 9");
    assert_eq!(out, "call  -  at  ");
    let texts: Vec<&str> = found
        .iter()
        .map(|(s, k)| {
            assert_eq!(*k, Number);
            s.as_str()
        })
        .collect();
    assert_eq!(texts, ["555", "0199", "9"]);
}

/// The digit-run rule against a direct reading of `\d+` -> " ".
#[test]
fn numbers_match_the_digit_run_rule_on_random_inputs() {
    let pool =
        ["a", "Z", " ", "0", "1", "23", "007", ".", ",", "-", "+", "\u{661}", "\u{2460}", "\u{1D7CF}", "\u{1F600}"];
    let mut rng = Rng::new(42);
    for _ in 0..5000 {
        let input = rng.input(&pool);
        let mut expected = String::new();
        let mut in_run = false;
        for c in input.chars() {
            if c.is_ascii_digit() {
                if !in_run {
                    expected.push(' ');
                }
                in_run = true;
            } else {
                expected.push(c);
                in_run = false;
            }
        }
        assert_eq!(numbers(&input), expected, "{input:?}");
    }
}

/// OpenNLP's three shrink passes, ported one to one over chars (a Rust
/// string has no lone surrogates, so chars are the code points).
fn three_pass(text: &str) -> String {
    let space = |c: char| matches!(c, '\t'..='\r' | ' ');
    let t: Vec<char> = text.chars().collect();
    let mut first = Vec::new();
    let mut i = 0;
    while i < t.len() {
        if space(t[i]) {
            let mut end = i + 1;
            while end < t.len() && space(t[end]) {
                end += 1;
            }
            if end - i >= 2 {
                first.push(' ');
            } else {
                first.push(t[i]);
            }
            i = end;
        } else {
            first.push(t[i]);
            i += 1;
        }
    }
    let mut second = Vec::new();
    let mut i = 0;
    while i < first.len() {
        let c = first[i];
        if !matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}') {
            let mut end = i + 1;
            while end < first.len() && first[end].eq_ignore_ascii_case(&c) {
                end += 1;
            }
            if end - i >= 3 {
                second.extend([c, c]);
                i = end;
                continue;
            }
        }
        second.push(c);
        i += 1;
    }
    let s: String = second.into_iter().collect();
    s.trim_matches(|c: char| c <= ' ').to_string()
}

#[test]
fn whitespace_runs_of_two_or_more_collapse_to_one_space() {
    for (input, expected) in [
        ("", ""),
        ("   ", ""),
        ("a text    extra space", "a text extra space"),
        ("a  b", "a b"),
        ("a \t b", "a b"),
        ("a\tb", "a\tb"),
        ("a\t\tb", "a b"),
        ("\tab", "ab"),
        ("ab\t", "ab"),
        ("a\n\n\nb", "a b"),
        ("a\nb", "a\nb"),
        ("a\r\n\r\nb", "a b"),
        (" \t\n ", ""),
        ("x X x X", "x X x X"),
    ] {
        assert_eq!(shrink(input), expected, "{input:?}");
    }
}

#[test]
fn repeated_character_runs_shrink_to_two_ascii_case_insensitively() {
    for (input, expected) in [
        ("Helllllloooooo", "Helloo"),
        ("Hello", "Hello"),
        ("HHello", "HHello"),
        ("aaab", "aab"),
        ("aaaa", "aa"),
        ("aAAb", "aab"),
        ("aAaB", "aaB"),
        ("aAa", "aa"),
        ("aA", "aA"),
        ("heyyy", "heyy"),
        ("cooool!!!!", "cool!!"),
        ("....", ".."),
        ("----", "--"),
        ("aa", "aa"),
        ("SSSs", "SS"),
        ("111222333", "112233"),
        ("ababab", "ababab"),
        ("xxxXXX", "xx"),
        ("aaabbb ccc", "aabb cc"),
        ("aaa\n\n\nbbb", "aa bb"),
        ("a   bbb  c", "a bb c"),
    ] {
        assert_eq!(shrink(input), expected, "{input:?}");
    }
}

#[test]
fn edges_are_trimmed_like_string_trim() {
    assert_eq!(shrink("  aaa  "), "aa");
    assert_eq!(shrink("a"), "a");
    assert_eq!(shrink("a  "), "a");
    assert_eq!(shrink("\u{1}a\u{1}"), "a");
    assert_eq!(shrink("\u{1}\u{1}\u{1}a\u{1}\u{1}\u{1}"), "a");
}

#[test]
fn line_terminators_never_start_a_repeat_run() {
    assert_eq!(shrink("\u{85}\u{85}\u{85}"), "\u{85}\u{85}\u{85}");
    assert_eq!(shrink("\u{2028}\u{2028}\u{2028}"), "\u{2028}\u{2028}\u{2028}");
    assert_eq!(shrink("\u{2029}\u{2029}\u{2029}"), "\u{2029}\u{2029}\u{2029}");
    assert_eq!(shrink("\u{A0}\u{A0}\u{A0}"), "\u{A0}\u{A0}");
}

#[test]
fn non_ascii_repeats_compare_by_exact_code_point() {
    assert_eq!(shrink("\u{E9}\u{E9}\u{E9}\u{E9}"), "\u{E9}\u{E9}");
    assert_eq!(shrink("\u{C9}\u{C9}\u{E9}\u{E9}"), "\u{C9}\u{C9}\u{E9}\u{E9}");
    assert_eq!(shrink("\u{1F600}\u{1F600}\u{1F600}"), "\u{1F600}\u{1F600}");
    assert_eq!(shrink("\u{1F600}\u{1F600}\u{1F600}\u{1F600}"), "\u{1F600}\u{1F600}");
}

#[test]
fn a_shrunk_run_keeps_its_first_character_aligned_to_itself() {
    let (out, a) = turbo_normalize::normalize(&Shrink, "Hellllo");
    assert_eq!(out, "Hello");
    // The first "l" stays where it was; the second stands for the rest.
    assert_eq!(a.span_to_parent(turbo_analysis::Span::new(2, 3)), turbo_analysis::Span::new(2, 3));
    assert_eq!(a.span_to_parent(turbo_analysis::Span::new(3, 4)), turbo_analysis::Span::new(3, 6));
}

#[test]
fn shrink_without_changes_is_borrowed() {
    assert!(matches!(apply(&Shrink, "single spaced words only"), Cow::Borrowed(_)));
}

/// The one-pass shrink against OpenNLP's three passes, on the pool of
/// ShrinkCharSequenceNormalizerCharacterizationTest (without lone
/// surrogates).
#[test]
fn one_pass_matches_three_passes_on_random_inputs() {
    let pool = [
        "a",
        "aa",
        "aaa",
        "A",
        "b",
        "Z",
        "!",
        "!!",
        ".",
        "-",
        " ",
        "  ",
        "\t",
        "\n",
        "\r",
        "\u{1}",
        "\u{85}",
        "\u{2028}",
        "\u{A0}",
        "\u{E9}",
        "\u{C9}",
        "\u{1F600}",
    ];
    let mut rng = Rng::new(42);
    for _ in 0..50_000 {
        let input = rng.input(&pool);
        assert_eq!(shrink(&input), three_pass(&input), "{input:?}");
    }
}
