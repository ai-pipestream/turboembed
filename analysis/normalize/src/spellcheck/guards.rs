//! The token guards of OpenNLP's `SpellCheckingCharSequenceNormalizer`:
//! the punctuation peeled off a token, and the tests that keep numbers,
//! URLs and e-mail addresses from being corrected.

use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::java;

/// A letter (`Character.isLetter`) or a number of any kind (Nd, Nl, No).
fn is_letter_or_number(c: char) -> bool {
    java::is_letter(c) || general_category(c).is_number()
}

/// The core of a token: the byte range left when the leading and the
/// trailing runs of code points that are neither letters nor numbers are
/// peeled off. A token with neither gives an empty core at its end.
pub(crate) fn core(token: &str) -> (usize, usize) {
    let start = token.find(is_letter_or_number).unwrap_or(token.len());
    let end = token[start..]
        .rfind(is_letter_or_number)
        .map_or(start, |i| start + i + token[start + i..].chars().next().map_or(0, char::len_utf8));
    (start, end)
}

/// OpenNLP's number test: an optional sign, then ASCII digits, full stops
/// and commas with at least one digit, then an optional percent sign.
pub(crate) fn is_number_like(core: &str) -> bool {
    let b = core.as_bytes();
    let start = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let end = if b.len() > start && b[b.len() - 1] == b'%' { b.len() - 1 } else { b.len() };
    if start >= end {
        return false;
    }
    let body = &b[start..end];
    body.iter().all(|&c| c.is_ascii_digit() || c == b'.' || c == b',') && body.iter().any(u8::is_ascii_digit)
}

const URL_PREFIXES: [&str; 3] = ["http://", "https://", "www."];
const TOP_LEVEL_DOMAINS: [&str; 6] = ["com", "org", "net", "edu", "gov", "io"];

/// OpenNLP's URL and e-mail test: a URL prefix with something after it,
/// a whole e-mail address whose domain has a dot with something after it,
/// or a dot after the first character followed by one of six top-level
/// domains that ends a word.
pub(crate) fn is_url_like(core: &str) -> bool {
    has_url_prefix(core) || is_dotted_mail_address(core) || has_top_level_domain(core)
}

fn has_url_prefix(core: &str) -> bool {
    URL_PREFIXES.iter().any(|p| core.len() > p.len() && core.starts_with(p))
}

/// `[-+_.0-9A-Za-z]+@[-0-9A-Za-z][-.0-9A-Za-z]+` over the whole core (the
/// e-mail scan of OpenNLP's `MailAddressScan` anchored at both ends), with
/// a dot in the domain that is not its last character.
fn is_dotted_mail_address(core: &str) -> bool {
    let b = core.as_bytes();
    let local = b.iter().take_while(|&&c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'+' | b'_' | b'.')).count();
    if local == 0 || b.get(local) != Some(&b'@') {
        return false;
    }
    let domain = &b[local + 1..];
    let domain_ok = domain.len() >= 2
        && (domain[0].is_ascii_alphanumeric() || domain[0] == b'-')
        && domain.iter().all(|&c| c.is_ascii_alphanumeric() || c == b'-' || c == b'.');
    domain_ok && domain[..domain.len() - 1].contains(&b'.')
}

fn has_top_level_domain(core: &str) -> bool {
    core.char_indices().skip(1).filter(|&(_, c)| c == '.').any(|(dot, _)| {
        let after = &core[dot + 1..];
        TOP_LEVEL_DOMAINS.iter().any(|tld| after.starts_with(tld) && ends_word(&after[tld.len()..]))
    })
}

/// Whether a word in ASCII letters ends where `rest` starts: `rest` is
/// empty or starts with something other than an ASCII letter or digit, an
/// underscore or a nonspacing mark.
fn ends_word(rest: &str) -> bool {
    match rest.chars().next() {
        None => true,
        Some(c) => !(c.is_ascii_alphanumeric() || c == '_' || general_category(c) == GeneralCategory::Mn),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peel(t: &str) -> &str {
        let (s, e) = core(t);
        &t[s..e]
    }

    #[test]
    fn core_peels_non_letters_and_non_numbers() {
        assert_eq!(peel("\"wrold\","), "wrold");
        assert_eq!(peel("(quikc)"), "quikc");
        assert_eq!(peel("\u{AB}quikc\u{BB}"), "quikc");
        // A combining mark is not a letter, so it peels off as a suffix.
        assert_eq!(peel("quikc\u{301}"), "quikc");
        assert_eq!(peel("!!!"), "");
        assert_eq!(core("!!!"), (3, 3));
        assert_eq!(peel("½x²"), "½x²");
        assert_eq!(peel("teh-wrold."), "teh-wrold");
    }

    #[test]
    fn number_like_shapes() {
        for s in ["5%%", "+%", "1,2a", "+", "", "%", "..,,"] {
            assert!(!is_number_like(s), "{s:?}");
        }
        for s in ["+3,14%", "5%", "-42%", "1.2.3", "3.14", "12345"] {
            assert!(is_number_like(s), "{s:?}");
        }
    }

    #[test]
    fn url_like_shapes() {
        for s in [
            "http://x",
            "https://example.org/a",
            "www.a",
            "www.example",
            "user@example.com",
            "a.b+c@d-e.f",
            "user@sub.example.co",
            "quikc.com",
            "a.io",
            "a.com/path",
            "a.com-x",
            "a.com\u{E9}",
            "a.edu\u{1F600}",
            "x.gov.uk",
            "a.net.",
            "a.comx.org",
        ] {
            assert!(is_url_like(s), "{s:?}");
        }
        for s in [
            "",
            "http://",
            "https://",
            "www.",
            "HTTP://x",
            "WWW.a",
            "ftp://x",
            "user@localhost",
            "user@example.",
            "@example.de",
            "user@.example.de",
            "user@exa_mple.de",
            "user@example!",
            "user@@example.de",
            ".com",
            "a.co",
            "a.comx",
            "a.com1",
            "a.com_",
            "a.com\u{301}",
            "a.COM",
            "acom",
            "quick",
        ] {
            assert!(!is_url_like(s), "{s:?}");
        }
    }

    /// xorshift64, for the generated tokens.
    struct Rng(u64);

    impl Rng {
        fn below(&mut self, n: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 % n as u64) as usize
        }

        fn token(&mut self, alphabet: &[&str]) -> String {
            (0..self.below(7)).map(|_| alphabet[self.below(alphabet.len())]).collect()
        }
    }

    /// OpenNLP's former number pattern `[+-]?[\d.,]*\d[\d.,]*%?`, matched
    /// by trying each choice of the optional sign and percent sign.
    fn number_pattern(s: &str) -> bool {
        let body_ok = |b: &str| {
            b.chars().all(|c| c.is_ascii_digit() || c == '.' || c == ',') && b.chars().any(|c| c.is_ascii_digit())
        };
        let signs: &[&str] = if s.starts_with(['+', '-']) { &[&s[1..], s] } else { &[s] };
        signs.iter().any(|t| body_ok(t) || t.strip_suffix('%').is_some_and(body_ok))
    }

    #[test]
    fn number_test_agrees_with_the_former_pattern() {
        let alphabet = ["+", "-", "%", ".", ",", "0", "5", "9", "a"];
        let mut rng = Rng(42);
        for _ in 0..20_000 {
            let t = rng.token(&alphabet);
            assert_eq!(is_number_like(&t), number_pattern(&t), "{t:?}");
        }
    }

    /// OpenNLP's reference pattern for the URL guard,
    /// `(?:https?://|www\.)\S+|[-+_.0-9A-Za-z]+@[-0-9A-Za-z]+\.[-.0-9A-Za-z]+|\S+\.(?:com|org|net|edu|gov|io)\b\S*`,
    /// matched over the whole token (which has no whitespace) by trying
    /// every split each alternative allows.
    fn url_pattern(s: &str) -> bool {
        let prefixed = ["http://", "https://", "www."].iter().any(|p| s.len() > p.len() && s.starts_with(p));
        let local = |c: char| c.is_ascii_alphanumeric() || "-+_.".contains(c);
        let label = |c: char| c.is_ascii_alphanumeric() || c == '-';
        let domain = |c: char| label(c) || c == '.';
        let mail = s.char_indices().filter(|&(_, c)| c == '@').any(|(at, _)| {
            let (l, d) = (&s[..at], &s[at + 1..]);
            !l.is_empty()
                && l.chars().all(local)
                && d.char_indices().filter(|&(_, c)| c == '.').any(|(dot, _)| {
                    let (a, b) = (&d[..dot], &d[dot + 1..]);
                    !a.is_empty() && a.chars().all(label) && !b.is_empty() && b.chars().all(domain)
                })
        });
        // `\b` after an ASCII letter: the next character is not a word
        // character, which for Java is an ASCII letter or digit, `_`, or a
        // nonspacing mark.
        let word = |c: char| c.is_ascii_alphanumeric() || c == '_' || general_category(c) == GeneralCategory::Mn;
        let domain_suffix = s.char_indices().filter(|&(i, c)| i > 0 && c == '.').any(|(dot, _)| {
            ["com", "org", "net", "edu", "gov", "io"]
                .iter()
                .any(|t| s[dot + 1..].starts_with(t) && s[dot + 1 + t.len()..].chars().next().is_none_or(|c| !word(c)))
        });
        prefixed || mail || domain_suffix
    }

    #[test]
    fn url_test_agrees_with_the_reference_pattern() {
        let alphabet = [
            "http://",
            "https://",
            "www.",
            "com",
            "io",
            "co",
            "m",
            "o",
            "@",
            ".",
            "-",
            "+",
            "_",
            "1",
            "a",
            "\u{E9}",
            "\u{301}",
            "/",
            "\u{1F600}",
        ];
        let mut rng = Rng(42);
        for _ in 0..50_000 {
            let t = rng.token(&alphabet);
            assert_eq!(is_url_like(&t), url_pattern(&t), "{t:?}");
        }
    }
}
