//! GLiNER2's word splitter (`WhitespaceTokenSplitter`), without a regular
//! expression engine. Its pattern, tried in this order at each position:
//!
//! 1. a URL: `https?://` or `www.` and the non-space characters after it;
//! 2. an e-mail address: `[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}`;
//! 3. a handle: `@[a-z0-9_]+`;
//! 4. a word: word characters, joined by single hyphens or underscores,
//!    where a word character is a letter, a number, a mark, `_`, or a
//!    zero-width joiner or non-joiner;
//! 5. any one character that is not whitespace.
//!
//! The pattern matches ASCII letters without regard to case. Each word is
//! lower-cased, and keeps its byte range in the text.
//!
//! Python's `\w` and `\s` follow `str.isalnum` and `str.isspace`; here a
//! letter, number or mark is its Unicode general category, and whitespace
//! is Unicode White_Space plus the four separators U+001C to U+001F that
//! Python also counts.

use turbo_unicode::category::general_category;
use turbo_unicode::whitespace::is_python_space;

/// One word: its lower-cased text and its byte range in the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Word {
    pub text: String,
    pub start: usize,
    pub end: usize,
}

fn is_word(c: char) -> bool {
    let g = general_category(c);
    g.is_letter() || g.is_number() || g.is_mark() || c == '_' || c == '\u{200c}' || c == '\u{200d}'
}

/// The length in bytes of the longest prefix of `s` whose characters all
/// satisfy `f`.
fn run(s: &str, f: impl Fn(char) -> bool) -> usize {
    s.char_indices().find(|&(_, c)| !f(c)).map_or(s.len(), |(i, _)| i)
}

fn starts_with_ascii_ci(s: &str, prefix: &str) -> bool {
    s.len() >= prefix.len() && s.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

fn url(s: &str) -> Option<usize> {
    let head = ["https://", "http://", "www."].into_iter().find(|p| starts_with_ascii_ci(s, p))?;
    let rest = run(&s[head.len()..], |c| !is_python_space(c));
    (rest > 0).then_some(head.len() + rest)
}

/// An e-mail address at the start of `s`: `Ok` with its length, or `Err`
/// with the length of the local-part run, since every later start inside
/// that run reaches the same `@` and fails the same way.
fn email(s: &str) -> Result<usize, usize> {
    let local = run(s, |c| c.is_ascii_alphanumeric() || "._%+-".contains(c));
    if local == 0 || !s[local..].starts_with('@') {
        return Err(local);
    }
    let at = local + 1;
    let domain = &s[at..];
    let all = run(domain, |c| c.is_ascii_alphanumeric() || c == '.' || c == '-');
    // The domain is as long as it can be with a dot after it and two or
    // more letters after the dot, those letters as many as there are. One
    // pass from the right: `letters` is the run of letters after `k`.
    let bytes = domain.as_bytes();
    let mut letters = 0;
    for k in (1..all).rev() {
        if bytes[k] == b'.' && letters >= 2 {
            return Ok(at + k + 1 + letters);
        }
        letters = if bytes[k].is_ascii_alphabetic() { letters + 1 } else { 0 };
    }
    Err(local)
}

fn handle(s: &str) -> Option<usize> {
    let rest = s.strip_prefix('@')?;
    let n = run(rest, |c| c.is_ascii_alphanumeric() || c == '_');
    (n > 0).then_some(1 + n)
}

fn word(s: &str) -> Option<usize> {
    let mut end = run(s, is_word);
    if end == 0 {
        return None;
    }
    while let Some(c) = s[end..].chars().next().filter(|&c| c == '-' || c == '_') {
        let next = run(&s[end + c.len_utf8()..], is_word);
        if next == 0 {
            break;
        }
        end += c.len_utf8() + next;
    }
    Some(end)
}

/// The words of `text`, in order.
pub fn split(text: &str) -> Vec<Word> {
    let mut out = Vec::new();
    let mut at = 0;
    // No e-mail address starts before this: a failed local-part run, so the
    // scan stays linear in the text.
    let mut no_mail = 0;
    while at < text.len() {
        let rest = &text[at..];
        let c = rest.chars().next().expect("not at the end");
        if is_python_space(c) {
            at += c.len_utf8();
            continue;
        }
        let mail = || {
            if at < no_mail {
                return None;
            }
            email(rest).map_err(|local| no_mail = at + local).ok()
        };
        let len = url(rest).or_else(mail).or_else(|| handle(rest)).or_else(|| word(rest)).unwrap_or(c.len_utf8());
        out.push(Word { text: rest[..len].to_lowercase(), start: at, end: at + len });
        at += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<String> {
        split(s).into_iter().map(|w| w.text).collect()
    }

    #[test]
    fn words_punctuation_and_offsets() {
        let w = split("Steve Jobs founded Apple, in 1976.");
        let got: Vec<(&str, usize, usize)> = w.iter().map(|w| (w.text.as_str(), w.start, w.end)).collect();
        assert_eq!(
            got,
            [
                ("steve", 0, 5),
                ("jobs", 6, 10),
                ("founded", 11, 18),
                ("apple", 19, 24),
                (",", 24, 25),
                ("in", 26, 28),
                ("1976", 29, 33),
                (".", 33, 34)
            ]
        );
    }

    #[test]
    fn urls_mail_handles_and_joined_words() {
        assert_eq!(texts("see https://a.b/c?d=1 now"), ["see", "https://a.b/c?d=1", "now"]);
        assert_eq!(texts("WWW.Example.com."), ["www.example.com."]);
        assert_eq!(texts("mail Ann.B@Ex-ample.co.UK, ok"), ["mail", "ann.b@ex-ample.co.uk", ",", "ok"]);
        // No letters after the last dot: the address ends at the dot before.
        assert_eq!(texts("a@b.com.9"), ["a@b.com", ".", "9"]);
        assert_eq!(texts("@user_1 says"), ["@user_1", "says"]);
        assert_eq!(texts("state-of-the-art x--y _z"), ["state-of-the-art", "x", "-", "-", "y", "_z"]);
        // Marks stay inside their word; ZWJ joins.
        assert_eq!(texts("cafe\u{301} a\u{200d}b"), ["cafe\u{301}", "a\u{200d}b"]);
        assert_eq!(texts("\u{1c}x\u{a0}y"), ["x", "y"]);
        // A failed local part does not hide an address after it.
        assert_eq!(texts("a.b c@d.ef"), ["a", ".", "b", "c@d.ef"]);
    }

    /// Inputs that rescanned the rest of the text at every position: a
    /// local part with no domain, an `@` with no dot after it, and a long
    /// run of local-part characters. At this length a quadratic scan takes
    /// minutes; a linear one, milliseconds.
    #[test]
    fn failed_addresses_scan_the_text_once() {
        const N: usize = 400_000;
        let at_no_dot = format!("a@{}", "a".repeat(N));
        assert_eq!(texts(&at_no_dot), ["a", &format!("@{}", "a".repeat(N))]);
        let dots = "a.".repeat(N / 2);
        assert_eq!(split(&dots).len(), N);
        let plus = format!("{}@", "a+".repeat(N / 2));
        assert_eq!(split(&plus).len(), N + 1);
    }
}
