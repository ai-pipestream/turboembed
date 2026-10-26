use super::{Rng, char_starts, froms, golden_texts, rounds};
use crate::builtin::{self, BuiltIn};

/// Every match is a valid, ordered character range, from every start offset
/// and over a whole find loop. JVM goldens check the expected match text.
fn check_builtin(b: BuiltIn, text: &str, rng: &mut Rng) {
    let starts = char_starts(text);
    let check = |from: usize| {
        let got = builtin::search(b, text, starts[from]);
        if let Some((start, end)) = got {
            assert!(start >= starts[from] && start <= end && end <= text.len(), "{} in {text:?}", b.java_name());
            assert!(starts.binary_search(&start).is_ok() && starts.binary_search(&end).is_ok());
        }
        got
    };
    for from in froms(rng, starts.len() - 1) {
        check(from);
    }
    walk_matches(b, text, |_, _| {});
}

fn walk_matches(b: BuiltIn, text: &str, mut visit: impl FnMut(usize, usize)) {
    let (mut from, mut previous_end) = (0, 0);
    while let Some((start, end)) = builtin::search(b, text, from) {
        assert!(start >= from && start <= end && end <= text.len(), "{} in {text:?}", b.java_name());
        assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        assert!(start >= previous_end, "overlapping {} matches in {text:?}", b.java_name());
        previous_end = end;
        visit(start, end);
        let Some(next) = next_match_start(text, start, end) else { break };
        from = next;
    }
}

fn next_match_start(text: &str, start: usize, end: usize) -> Option<usize> {
    if end > start { Some(end) } else { text[end..].chars().next().map(|c| end + c.len_utf8()) }
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
fn builtins_return_valid_matches_on_golden_sentences() {
    let mut rng = Rng(395);
    for text in golden_texts() {
        for b in BuiltIn::ALL {
            check_builtin(b, &text, &mut rng);
        }
    }
}

#[test]
fn builtins_return_valid_matches_on_generated_texts() {
    let mut rng = Rng(0x5EED_0395);
    for _ in 0..800 * rounds() {
        let text = builtin_text(&mut rng);
        for b in BuiltIn::ALL {
            check_builtin(b, &text, &mut rng);
        }
    }
}

#[test]
fn match_iteration_keeps_every_match_after_multibyte_text() {
    for (text, expected) in [
        ("😀😀😀😀 123-4567 234-5678", vec![(17, 25), (26, 34)]),
        ("é 123-4567 234-5678", vec![(3, 11), (12, 20)]),
        ("123-4567", vec![(0, 8)]),
    ] {
        let mut got = Vec::new();
        walk_matches(BuiltIn::UsaPhoneNum, text, |a, b| got.push((a, b)));
        assert_eq!(got, expected, "{text:?}");
    }
}

#[test]
fn empty_match_advances_by_a_unicode_scalar_or_stops_at_eof() {
    let text = "é😀";
    assert_eq!(next_match_start(text, 0, 0), Some(2));
    assert_eq!(next_match_start(text, 2, 2), Some(6));
    assert_eq!(next_match_start(text, 6, 6), None);
    assert_eq!(next_match_start(text, 0, 2), Some(2));
}
