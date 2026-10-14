//! Differential tests of the matchers against the `regex` crate
//! translations they replaced ([`old_java_regex`] and [`old_builtin`]),
//! which the golden files check against Java. Both sides must report the
//! same match, at the same offsets, from every start offset tried, on the
//! golden sentences and on generated patterns and texts.
//!
//! The texts are seeded, so a failure repeats. `TURBO_TEST_PARITY_ROUNDS`
//! (default 1) multiplies the number of generated cases.

mod old_builtin;
mod old_java_regex;

use std::path::Path;

use crate::builtin::{self, BuiltIn};
use crate::java_regex::{Engine, Flags, JavaRegex, Scratch};

/// SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    /// A count from `around`, or one off either side of it.
    fn near(&mut self, around: &[usize]) -> usize {
        let n = *self.pick(around);
        match self.below(3) {
            0 => n.saturating_sub(1),
            1 => n,
            _ => n + 1,
        }
    }
}

fn rounds() -> usize {
    std::env::var("TURBO_TEST_PARITY_ROUNDS").ok().and_then(|r| r.parse().ok()).unwrap_or(1)
}

/// A string escaped by the golden generator: `\uXXXX` per UTF-16 unit
/// outside printable ASCII, `\\`, `\t`, `\n` and `\r`.
fn unescape(s: &str) -> String {
    let mut units = Vec::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            units.push(c as u16);
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(0x5C),
            Some('t') => units.push(0x09),
            Some('n') => units.push(0x0A),
            Some('r') => units.push(0x0D),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {s}"),
        }
    }
    String::from_utf16(&units).unwrap()
}

fn read_golden(relative: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("golden").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// The golden sentences, their tokens joined by spaces as
/// `RegexNameFinder.find` joins them.
fn golden_texts() -> Vec<String> {
    let mut texts: Vec<String> = read_golden("inputs/sentences.txt").lines().map(String::from).collect();
    let generated = read_golden("expected/sentences.txt");
    for line in generated.lines().skip(3) {
        let (_, tokens) = line.split_once('\t').unwrap();
        let tokens: Vec<String> = tokens.split('\t').map(unescape).collect();
        texts.push(tokens.join(" "));
    }
    texts
}

/// The byte offset of each character start, and of the end.
fn char_starts(text: &str) -> Vec<usize> {
    text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect()
}

/// The start offsets to search from: all of them in a short text, a
/// sample in a long one, always with the first and the last.
fn froms(rng: &mut Rng, chars: usize) -> Vec<usize> {
    if chars <= 160 {
        return (0..=chars).collect();
    }
    let mut out = vec![0, chars];
    out.extend((0..24).map(|_| rng.below(chars + 1)));
    out
}

// Built-in patterns.

/// Compares [`builtin::search`] with the stand-in search it replaced, in
/// character offsets (the stand-in has one character per character of
/// the text, in fewer bytes), from each offset `froms` gives, and over a
/// whole `Matcher.find` loop.
fn check_builtin(b: BuiltIn, text: &str, rng: &mut Rng) {
    let old = old_builtin::BuiltIn::ALL[b as usize];
    assert_eq!(old.java_name(), b.java_name());
    let mut stand_in = String::new();
    old_builtin::push_stand_in(&mut stand_in, text);
    let real = char_starts(text);
    let fake = char_starts(&stand_in);
    assert_eq!(real.len(), fake.len());
    let to_chars = |starts: &[usize], m: Option<(usize, usize)>| {
        m.map(|(s, e)| {
            let index =
                |at: usize| starts.binary_search(&at).unwrap_or_else(|_| panic!("{at} is not a character start"));
            (index(s), index(e))
        })
    };
    let compare = |from: usize| {
        let got = to_chars(&real, builtin::search(b, text, real[from]));
        let want = to_chars(&fake, old_builtin::search(old, &stand_in, fake[from]));
        assert_eq!(got, want, "{} from character {from} in {text:?}", b.java_name());
        got
    };
    for from in froms(rng, real.len() - 1) {
        compare(from);
    }
    // The loop `RegexNameFinder.find` runs.
    let mut from = 0;
    while let Some((start, end)) = compare(from) {
        from = if end > start { end } else { end + 1 };
        if from >= real.len() {
            break;
        }
    }
}

fn label(rng: &mut Rng, len: usize, alphabet: &[u8]) -> String {
    (0..len).map(|_| *rng.pick(alphabet) as char).collect()
}

const HOST: &[u8] = b"abcxyzABZ0189-_";
const DIGITS: &[u8] = b"0123456789";
const EMAIL_ATOM: &[u8] = b"abzAZ09!#$%&'*+/=?^_`{|}~-";

fn digits(rng: &mut Rng, n: usize) -> String {
    label(rng, n, DIGITS)
}

fn phone(rng: &mut Rng) -> String {
    let mut groups = [3, 3, 4].map(|n| if rng.one_in(4) { rng.near(&[n]) } else { n });
    if rng.one_in(4) {
        groups[0] = 0;
    }
    let (a, b, c) = (digits(rng, groups[0]), digits(rng, groups[1]), digits(rng, groups[2]));
    match rng.below(4) {
        0 => format!("({a}) {b}-{c}"),
        1 => format!("({a}){b}-{c}"),
        2 => format!("{a}-{b}-{c}"),
        _ => format!("{b}-{c}"),
    }
}

fn email(rng: &mut Rng) -> String {
    let mut s = String::new();
    if rng.one_in(4) {
        // A quoted local part, with escapes and characters it refuses.
        s.push('"');
        let n = rng.near(&[1, 255]);
        // Mostly characters it accepts, so that the count limit is reached.
        let noisy = rng.one_in(3);
        for _ in 0..n {
            match if noisy { rng.below(12) } else { rng.below(2) * 3 } {
                0 => s.push_str("\\\""),
                1 => s.push_str("\\ "),
                2 => s.push(*rng.pick(&[' ', '\t', '\u{E9}', '\\'])),
                _ => s.push(*rng.pick(&['a', 'Z', '0', '.', '@', '[', '~', '\u{1}'])),
            }
        }
        s.push('"');
    } else {
        let segments = rng.near(&[1, 2, 11]);
        for i in 0..segments {
            if i > 0 {
                s.push_str(if rng.one_in(10) { ".." } else { "." });
            }
            let len = rng.near(&[1, 3, 64]).max(1);
            s.push_str(&label(rng, len, EMAIL_ATOM));
        }
    }
    s.push('@');
    if rng.one_in(5) {
        let octets = [0, 1, 9, 10, 99, 100, 199, 200, 249, 250, 255, 256, 299, 300, 1000];
        let parts: Vec<String> = (0..rng.near(&[4]))
            .map(|_| {
                let o = rng.pick(&octets).to_string();
                if rng.one_in(6) { format!("0{o}") } else { o }
            })
            .collect();
        s.push('[');
        s.push_str(&parts.join("."));
        if !rng.one_in(8) {
            s.push(']');
        }
        return s;
    }
    let labels = rng.near(&[1, 2, 3, 21]).max(1);
    for i in 0..labels {
        if i > 0 {
            s.push('.');
        }
        let len = rng.near(&[1, 2, 62, 63]).max(1);
        let mut l = label(rng, len, b"abzAZ09-");
        if !rng.one_in(4) {
            // Labels must start and end with a letter or digit.
            l.replace_range(0..1, "a");
            let last = l.len() - 1;
            l.replace_range(last.., "z");
        }
        s.push_str(&l);
    }
    s
}

/// Atoms of a path, query or fragment part: about 255 of them when
/// `long`, a few otherwise. `noisy` mixes in atoms the pattern refuses;
/// without, the count limits are reached.
fn url_atoms(rng: &mut Rng, query: bool, noisy: bool, long: bool) -> String {
    let n = if long { rng.near(&[255]) } else { rng.near(&[0, 2]) };
    let mut s = String::new();
    for _ in 0..n {
        match if noisy { rng.below(10) } else { [0, 4, 5, 6][rng.below(4)] } {
            0 => s.push_str("%a9"),
            1 => s.push_str(if rng.one_in(2) { "%G1" } else { "%F" }),
            2 if query => s.push(*rng.pick(&['*', ':'])),
            3 => s.push(*rng.pick(&['?', '&', '#', '/', '@', '\u{E9}', '\u{301}'])),
            _ => s.push(*rng.pick(&['a', 'Z', '0', '_', '-', '~', '!', '$', '+', '|', '.', ',', '='])),
        }
    }
    s
}

fn url(rng: &mut Rng) -> String {
    let mut s = String::new();
    let noisy = rng.one_in(3);
    // Long parts, or many of them, but not both.
    let long = rng.one_in(4);
    let many = |rng: &mut Rng| if long { rng.near(&[0, 1]) } else { rng.near(&[0, 1, 50]) };
    if rng.one_in(3) {
        s.push(*rng.pick(&['x', '.', '_', '\u{E9}']));
    }
    s.push_str(rng.pick(&["http://", "HTTPS://", "ftp://", "fTpS://", "~/", "/", "www.", "WwW.", "htp://"]));
    if rng.one_in(5) {
        let (u, p) = (rng.near(&[1, 63]), rng.near(&[1, 63]));
        s.push_str(&format!("{}:{}@", label(rng, u, b"aZ0_"), label(rng, p, b"aZ0_")));
    }
    let labels = rng.near(&[1, 2, 20]).max(1);
    for _ in 0..labels {
        let len = rng.near(&[1, 5, 63]).max(1);
        s.push_str(&label(rng, len, HOST));
        s.push('.');
    }
    let tld = *rng.pick(&["com", "org", "info", "museum", "travel", "co", "c", "comx", "INFO", "x1", "de"]);
    s.push_str(tld);
    if rng.one_in(4) {
        let n = rng.near(&[0, 5]);
        s.push(':');
        s.push_str(&digits(rng, n));
    }
    let segments = many(rng);
    for _ in 0..segments {
        s.push('/');
        s.push_str(&url_atoms(rng, false, noisy, long));
    }
    if rng.one_in(3) {
        s.push('?');
        s.push_str(&url_atoms(rng, true, noisy, long));
        for _ in 0..many(rng) {
            s.push('&');
            s.push_str(&url_atoms(rng, true, noisy, long));
        }
    }
    if rng.one_in(4) {
        s.push('#');
        s.push_str(&url_atoms(rng, true, noisy, long));
    }
    if rng.one_in(3) {
        s.push(*rng.pick(&['a', '.', '-', '/', '\u{E9}', '\u{301}', '_']));
    }
    s
}

fn spaces(rng: &mut Rng) -> String {
    let n = rng.below(3);
    (0..n).map(|_| *rng.pick(&[' ', ' ', '\t', '\n', '\u{B}'])).collect()
}

fn mgrs(rng: &mut Rng) -> String {
    let a = rng.near(&[1, 2]);
    let (b, c) = (rng.near(&[1, 5]), rng.near(&[0, 1, 5]));
    let letters = rng.near(&[2]);
    format!(
        "{}{}{}{}{}{}{}{}",
        digits(rng, a),
        label(rng, 1, b"aZ5"),
        spaces(rng),
        label(rng, letters, b"aZbY"),
        spaces(rng),
        digits(rng, b),
        spaces(rng),
        digits(rng, c)
    )
}

fn dms_half(rng: &mut Rng, hemispheres: &[&str]) -> String {
    let mut s = String::new();
    s.push_str(rng.pick(&["", "", "-", "+", "|"]));
    let n = rng.near(&[1, 3]);
    s.push_str(&digits(rng, n));
    s.push_str(rng.pick(&["d", "D", "\u{B0}", " ", "|", "x", "\t"]));
    if !rng.one_in(3) {
        s.push_str(&spaces(rng));
        let n = rng.near(&[1, 2]);
        s.push_str(&digits(rng, n));
        s.push_str(rng.pick(&["'", "\u{2019}", "|", " ", "\"", "x"]));
    }
    if !rng.one_in(3) {
        s.push_str(&spaces(rng));
        let n = rng.near(&[1, 2]);
        s.push_str(&digits(rng, n));
        s.push_str(rng.pick(&["\"", "\u{201D}", "|", "'", "x"]));
    }
    s.push_str(&spaces(rng));
    s.push_str(rng.pick(hemispheres));
    s
}

fn dms(rng: &mut Rng) -> String {
    let lat = dms_half(rng, &["N", "n", "S", "s", "|", "", "E", "x"]);
    let sep = *rng.pick(&["", " ", ",", ", ", " ,", "  ", ",,"]);
    let lon = dms_half(rng, &["E", "e", "W", "w", "|", "", "N", "x"]);
    format!("{lat}{sep}{lon}")
}

fn noise(rng: &mut Rng) -> String {
    const CHARS: &[char] = &[
        'a',
        'Z',
        '0',
        '9',
        '.',
        '@',
        '-',
        '_',
        '/',
        ':',
        '"',
        '\\',
        '(',
        ')',
        '[',
        ']',
        '%',
        '|',
        ',',
        '\'',
        '\u{B0}',
        '\u{2019}',
        '\u{201D}',
        '\u{E9}',
        '\u{301}',
        '\u{1F600}',
        '\u{212A}',
        '\u{17F}',
        '\u{0}',
    ];
    let n = 1 + rng.below(8);
    (0..n).map(|_| *rng.pick(CHARS)).collect()
}

fn builtin_text(rng: &mut Rng) -> String {
    let mut s = String::new();
    for i in 0..1 + rng.below(4) {
        if i > 0 || rng.one_in(4) {
            s.push_str(rng.pick(&[" ", " ", "", "\t", ",", "\n", "\u{E9}", "."]));
        }
        let part = match rng.below(6) {
            0 => phone(rng),
            1 => email(rng),
            2 => url(rng),
            3 => mgrs(rng),
            4 => dms(rng),
            _ => noise(rng),
        };
        s.push_str(&part);
    }
    s
}

#[test]
fn builtins_match_the_regex_translation_on_golden_sentences() {
    let mut rng = Rng(395);
    for text in golden_texts() {
        for b in BuiltIn::ALL {
            check_builtin(b, &text, &mut rng);
        }
    }
}

#[test]
fn builtins_match_the_regex_translation_on_generated_texts() {
    let mut rng = Rng(0x5EED_0395);
    for _ in 0..800 * rounds() {
        let text = builtin_text(&mut rng);
        for b in BuiltIn::ALL {
            check_builtin(b, &text, &mut rng);
        }
    }
}

// User patterns.

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
    let expected = read_golden("expected/regex-user.txt");
    for line in expected.lines().filter(|l| l.starts_with("pattern\t")) {
        let fields: Vec<&str> = line.split('\t').collect();
        let (f, pattern) = (fields[1], fields[2]);
        let flags = Flags { case_insensitive: f.contains('i'), dotall: f.contains('s') };
        let mut texts = short.clone();
        texts.extend((0..20).map(|_| random_text(&mut rng)));
        compiled += usize::from(check_java(&unescape(pattern), flags, &texts, &mut rng, &mut scratch));
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
