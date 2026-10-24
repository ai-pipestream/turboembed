use super::{Rng, char_starts, froms, golden_texts, old_java_regex, parse_rounds, read_data, rounds};
use crate::java_regex::{Engine, Flags, JavaRegex, Scratch};

/// Compares [`JavaRegex`] with the translation it replaced: the same
/// compile result and limits, and on each text the same match from each
/// offset `froms` gives, with each engine. Returns whether it compiled.
fn check_java(pattern: &str, flags: Flags, texts: &[String], rng: &mut Rng, scratch: &mut Scratch) -> bool {
    let old_flags = old_java_regex::Flags { case_insensitive: flags.case_insensitive, dotall: flags.dotall };
    let old = old_java_regex::JavaRegex::compile_with(pattern, old_flags);
    let new = JavaRegex::compile_with(pattern, flags);
    let (old, new) = match (old, new) {
        (Ok(old), Ok(new)) => (old, new),
        (Err(old), Err(new)) => {
            assert_eq!(new, old, "{pattern:?} {flags:?}");
            return false;
        }
        (old, new) => panic!("{pattern:?} {flags:?}: was {old:?}, is {new:?}"),
    };
    let (a, b) = (old.limits(), new.limits());
    assert_eq!(
        (a.word_boundary, a.dollar, a.mid_pair),
        (b.word_boundary, b.dollar, b.mid_pair),
        "limits of {pattern:?} {flags:?}"
    );
    for text in texts {
        // The old translation cannot serve as an oracle for the newly
        // supported mark boundaries and final terminators; JVM goldens do.
        if !old.supports(&old_java_regex::TextFacts::of(text)) {
            continue;
        }
        let starts = char_starts(text);
        for from in froms(rng, starts.len() - 1) {
            let from = starts[from];
            let want = old.regex().find_at(text, from).map(|m| (m.start(), m.end()));
            for engine in [Engine::Backtrack, Engine::Pike, Engine::Auto] {
                let got = new.find_at_with(text, from, scratch, engine);
                assert_eq!(got, want, "{pattern:?} {flags:?} {engine:?} from {from} in {text:?}");
            }
        }
    }
    true
}

/// Characters for generated patterns and texts: ASCII letters in both
/// cases, the ones whose Unicode case folds reach ASCII (U+212A, U+017F),
/// line terminators, a nonspacing mark and a character outside the BMP.
const TEXT_CHARS: &[char] = &[
    'a',
    'a',
    'b',
    'b',
    'c',
    'A',
    'B',
    'k',
    'K',
    's',
    'x',
    ' ',
    ' ',
    '-',
    '.',
    '_',
    '0',
    '1',
    '9',
    '\n',
    '\r',
    '\t',
    '\u{B}',
    '\u{E9}',
    '\u{C9}',
    '\u{301}',
    '\u{1F600}',
    '\u{85}',
    '\u{2028}',
    '\u{212A}',
    '\u{17F}',
    '\u{A0}',
];

fn random_text(rng: &mut Rng) -> String {
    let n = rng.below(24);
    (0..n).map(|_| *rng.pick(TEXT_CHARS)).collect()
}

fn literal(rng: &mut Rng) -> String {
    let c = *rng.pick(&[
        "a",
        "b",
        "c",
        "A",
        "k",
        "K",
        "x",
        " ",
        "-",
        "_",
        "0",
        "\u{E9}",
        "\u{C9}",
        "\u{1F600}",
        "\u{212A}",
        "\u{17F}",
        "\\.",
        "\\-",
        "\\\\",
        "\\t",
        "\\n",
        "\\x41",
        "\\x{1F600}",
        "\\u00e9",
        "\\uD83D\\uDE00",
        "\\0141",
        "\\cJ",
        "\\e",
        "\\Qa.b\\E",
        "\\Q\\E",
        "\\Q]\\E",
        "}",
        "]",
    ]);
    c.to_string()
}

fn class(rng: &mut Rng) -> String {
    let mut s = String::from("[");
    if rng.one_in(3) {
        s.push('^');
    }
    for _ in 0..1 + rng.below(3) {
        s.push_str(rng.pick(&[
            "a",
            "b-k",
            "A-Z",
            "Z-a",
            "0-9",
            "\u{E9}",
            "\u{C0}-\u{FF}",
            "\\w",
            "\\W",
            "\\d",
            "\\s",
            "\\S",
            "\\h",
            "\\v",
            "\\v-",
            "-",
            ".",
            "|",
            "\\]",
            "\\x{1F600}",
            "\u{10000}-\u{10FFFF}",
            "\\x00-\\uFFFF",
            "\\n",
            "k",
            "s",
        ]));
    }
    s.push(']');
    s
}

fn atom(rng: &mut Rng, depth: usize) -> String {
    match rng.below(if depth < 3 { 9 } else { 6 }) {
        0 | 1 => literal(rng),
        2 => (*rng.pick(&[".", "\\d", "\\D", "\\w", "\\W", "\\s", "\\S", "\\h", "\\H", "\\v", "\\V"])).to_string(),
        3 => class(rng),
        4 => (*rng.pick(&["^", "$", "\\b", "\\B", "\\A", "\\z", "\\Z"])).to_string(),
        5 => (*rng.pick(&["(?i)", "(?s)", "(?-i)", "(?is)", "(?i-s)"])).to_string(),
        _ => {
            let open = *rng.pick(&["(", "(?:", "(?i:", "(?s:", "(?-i:", "(?<n>"]);
            format!("{open}{})", pattern(rng, depth + 1))
        }
    }
}

fn quantifier(rng: &mut Rng) -> String {
    let q = match rng.below(10) {
        0 => "?".to_string(),
        1 => "*".to_string(),
        2 => "+".to_string(),
        3 => format!("{{{}}}", rng.below(4)),
        4 => format!("{{{},}}", rng.below(3)),
        5 => {
            let n = rng.below(3);
            format!("{{{n},{}}}", n + rng.below(3))
        }
        _ => return String::new(),
    };
    if rng.one_in(3) { q + "?" } else { q }
}

fn pattern(rng: &mut Rng, depth: usize) -> String {
    let mut s = String::new();
    for _ in 0..1 + rng.below(4) {
        s.push_str(&atom(rng, depth));
        s.push_str(&quantifier(rng));
    }
    if rng.one_in(4) {
        s.push('|');
        s.push_str(&pattern(rng, depth + 1));
    }
    s
}

/// Characters strung together at random, mostly ones the parser treats
/// specially, for the refusals.
fn soup(rng: &mut Rng) -> String {
    const META: &[char] = &[
        'a', 'b', '(', ')', '[', ']', '{', '}', '|', '*', '+', '?', '^', '$', '\\', '.', '-', ',', '0', '1', '3', ':',
        '<', '>', '=', '!', '&', 'd', 'w', 's', 'b', 'B', 'p', 'Q', 'E', 'i', 'k', 'x', 'u', 'c', 'v', '\u{E9}',
    ];
    let n = 1 + rng.below(10);
    (0..n).map(|_| *rng.pick(META)).collect()
}

fn flags(rng: &mut Rng) -> Flags {
    Flags { case_insensitive: rng.one_in(3), dotall: rng.one_in(5) }
}

#[test]
fn golden_patterns_match_the_regex_translation() {
    let mut rng = Rng(0x0395);
    let short: Vec<String> = golden_texts().into_iter().filter(|t| t.chars().count() <= 80).take(300).collect();
    let mut scratch = Scratch::default();
    let mut compiled = 0;
    // The patterns of the `regex-user` golden: an expectation, the flags
    // and the pattern on each line, separated by tabs.
    let patterns = read_data("fixtures/namefind/patterns.txt");
    for line in patterns.lines() {
        let fields: Vec<&str> = line.splitn(3, '\t').collect();
        let (f, pattern) = (fields[1], fields[2]);
        let flags = Flags { case_insensitive: f.contains('i'), dotall: f.contains('s') };
        let mut texts = short.clone();
        texts.extend((0..20).map(|_| random_text(&mut rng)));
        compiled += usize::from(check_java(pattern, flags, &texts, &mut rng, &mut scratch));
    }
    assert!(compiled >= 65, "{compiled}");
}

#[test]
fn generated_patterns_match_the_regex_translation() {
    let mut rng = Rng(0xD1FF_0395);
    let golden: Vec<String> = golden_texts().into_iter().filter(|t| t.chars().count() <= 60).collect();
    let mut scratch = Scratch::default();
    let (mut compiled, mut total) = (0, 0);
    for _ in 0..2500 * rounds() {
        let pattern = if rng.one_in(5) { soup(&mut rng) } else { pattern(&mut rng, 0) };
        let mut texts: Vec<String> = (0..6).map(|_| random_text(&mut rng)).collect();
        texts.push(rng.pick(&golden).clone());
        compiled += usize::from(check_java(&pattern, flags(&mut rng), &texts, &mut rng, &mut scratch));
        total += 1;
    }
    // Enough of them compile for the matching to be tested.
    assert!(compiled * 3 > total, "{compiled} of {total}");
}

#[test]
fn engines_agree_on_long_texts() {
    // Long enough that the backtracker's table is in use from the start
    // of each search, and that a search fails at many starts.
    let mut rng = Rng(0x10_0395);
    let mut scratch = Scratch::default();
    for pattern in ["(a|aa)*c", ".*x", "(a|ab)(c|bcd)(d*)", "a{2,5}?b|a", "(?i)[a-c]+?\\b", "(a|b|ab)*?c$", "\\B.\\b"] {
        let texts: Vec<String> = (0..4)
            .map(|_| (0..300 + rng.below(300)).map(|_| *rng.pick(&['a', 'a', 'b', 'c', 'd', ' '])).collect())
            .collect();
        assert!(check_java(pattern, Flags::NONE, &texts, &mut rng, &mut scratch));
    }
}

#[test]
fn rounds_clamps_zero_and_non_numeric_values() {
    assert_eq!(parse_rounds(None), 1);
    assert_eq!(parse_rounds(Some("")), 1);
    assert_eq!(parse_rounds(Some("0")), 1);
    assert_eq!(parse_rounds(Some("-1")), 1);
    assert_eq!(parse_rounds(Some("3")), 3);
}
