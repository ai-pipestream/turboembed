use super::{Rng, char_starts, froms, golden_texts, old_builtin, rounds};
use crate::builtin::{self, BuiltIn};

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
