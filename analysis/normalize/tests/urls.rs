//! `Urls` and `BoundedUrls`: the expectations of OpenNLP's
//! UrlCharSequenceNormalizerCharacterizationTest and
//! BoundedUrlCharSequenceNormalizerTest, the one-pass scan against a
//! two-pass reference, the placeholder spans, and adversarial URLs.

mod scanners_support;

use scanners_support::{Rng, placeholders, run};
use turbo_normalize::{BoundedUrls, Urls, apply};
use turbo_normalize_layers::PlaceholderKind::{self, Email, Url};

fn urls(input: &str) -> String {
    run(&Urls, input)
}

fn bounded(input: &str) -> String {
    run(&BoundedUrls, input)
}

/// OpenNLP's two passes, ported one to one: every URL becomes a space,
/// then every address in the result does.
mod reference {
    fn is_body(c: u8) -> bool {
        c.is_ascii_alphanumeric() || b"-_.?&~;+=/#".contains(&c)
    }
    fn is_local(c: u8) -> bool {
        c.is_ascii_alphanumeric() || b"-+_.".contains(&c)
    }

    fn url_end(t: &[u8], i: usize) -> Option<usize> {
        let body = if t[i..].starts_with(b"https://") {
            i + 8
        } else if t[i..].starts_with(b"http://") {
            i + 7
        } else {
            return None;
        };
        if body >= t.len() || !is_body(t[body]) {
            return None;
        }
        let mut end = body + 1;
        while end < t.len() && is_body(t[end]) {
            end += 1;
        }
        Some(end)
    }

    fn mail_end(t: &[u8], i: usize) -> Option<usize> {
        if (i > 0 && is_local(t[i - 1])) || !is_local(t[i]) {
            return None;
        }
        let mut at = i + 1;
        while at < t.len() && is_local(t[at]) {
            at += 1;
        }
        if at >= t.len() || t[at] != b'@' {
            return None;
        }
        let d = at + 1;
        if d >= t.len() || !(t[d].is_ascii_alphanumeric() || t[d] == b'-') {
            return None;
        }
        let mut end = d + 1;
        while end < t.len() && (t[end].is_ascii_alphanumeric() || t[end] == b'-' || t[end] == b'.') {
            end += 1;
        }
        (end - d >= 2).then_some(end)
    }

    fn pass(text: &str, matcher: fn(&[u8], usize) -> Option<usize>) -> String {
        let t = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        while i < t.len() {
            match matcher(t, i) {
                Some(end) => {
                    out.push(b' ');
                    i = end;
                }
                None => {
                    out.push(t[i]);
                    i += 1;
                }
            }
        }
        String::from_utf8(out).unwrap()
    }

    pub fn two_pass(text: &str) -> String {
        pass(&pass(text, url_end), mail_end)
    }
}

#[test]
fn urls_become_a_single_space() {
    assert_eq!(urls(""), "");
    assert_eq!(urls("asdf http://asdf.com/dfa/cxs 2nnfdf"), "asdf   2nnfdf");
    assert_eq!(urls("asdf http://asdf.com/dfa/cxs 2nnfdf http://asdf.com/dfa/cxs"), "asdf   2nnfdf  ");
    assert_eq!(urls("http://example.com"), " ");
    assert_eq!(urls("visit http://example.com now"), "visit   now");
    assert_eq!(urls("https://a.b/c?d=e&f=g#h+i~j;k=l_m-n"), " ");
    assert_eq!(urls("https://x"), " ");
    assert_eq!(urls("http://EXAMPLE.com"), " ");
    assert_eq!(urls("xhttp://y"), "x ");
    assert_eq!(urls("ahttp://x"), "a ");
}

#[test]
fn url_recognition_rejects_at_the_exact_boundary() {
    assert_eq!(urls("http:// x"), "http:// x");
    assert_eq!(urls("http://"), "http://");
    assert_eq!(urls("HTTP://X.COM"), "HTTP://X.COM");
    assert_eq!(urls("httpss://x"), "httpss://x");
    assert_eq!(urls("ftp://x"), "ftp://x");
    assert_eq!(urls("www.example.com"), "www.example.com");
    assert_eq!(urls("http://x:8080/y"), " :8080/y");
    assert_eq!(urls("http://a%20b"), " %20b");
    assert_eq!(urls("http://http://x"), " ://x");
    assert_eq!(
        urls("http://\u{43F}\u{440}\u{438}\u{43C}\u{435}\u{440}"),
        "http://\u{43F}\u{440}\u{438}\u{43C}\u{435}\u{440}"
    );
    assert_eq!(
        urls("http://x.\u{43F}\u{440}\u{438}\u{43C}\u{435}\u{440}"),
        " \u{43F}\u{440}\u{438}\u{43C}\u{435}\u{440}"
    );
    assert_eq!(urls("http://user@example.com"), " @example.com");
}

#[test]
fn mail_addresses_become_a_single_space() {
    assert_eq!(urls("asdf asd.fdfa@hasdk23.com.br 2nnfdf"), "asdf   2nnfdf");
    assert_eq!(urls("asdf asd.fdfa@hasdk23.com.br 2nnfdf asd.fdfa@hasdk23.com.br"), "asdf   2nnfdf  ");
    assert_eq!(urls("asdf asd+fdfa@hasdk23.com.br 2nnfdf"), "asdf   2nnfdf");
    assert_eq!(urls("asdf asd.fdfa@hasdk23.com_br 2nnfdf"), "asdf  _br 2nnfdf");
    assert_eq!(urls(".user@x.com"), " ");
    assert_eq!(urls("-a+b_c.d@e-f.g2 end"), "  end");
    assert_eq!(urls("a@b.c"), " ");
    assert_eq!(urls("1@2.3"), " ");
    assert_eq!(urls("a@-b"), " ");
    assert_eq!(urls("x@y..z"), " ");
    assert_eq!(urls("x@y-."), " ");
    assert_eq!(urls("mailto:user@example.com"), "mailto: ");
    assert_eq!(urls("a@b.co,c@d.eu"), " , ");
    assert_eq!(urls("a b@c.de f"), "a   f");
    assert_eq!(urls("see http://x.y, then a@b.cd!"), "see  , then  !");
}

#[test]
fn mail_recognition_rejects_at_the_exact_boundary() {
    assert_eq!(urls("a@b c"), "a@b c");
    assert_eq!(urls("a@bc d"), "  d");
    assert_eq!(urls("a@b."), " ");
    assert_eq!(urls("user@example.com."), " ");
    assert_eq!(urls("Contact a@b.com."), "Contact  ");
    assert_eq!(urls("a@.com"), "a@.com");
    assert_eq!(urls("x@.y"), "x@.y");
    assert_eq!(urls("user@@example.com"), "user@@example.com");
    assert_eq!(urls("@example.com"), "@example.com");
    assert_eq!(urls("a@b@cd.com"), "a@ ");
}

#[test]
fn the_character_before_an_address_is_read_one_character_back() {
    assert_eq!(urls("\u{1F600}a@b.co"), "\u{1F600} ");
    assert_eq!(urls("\u{E9}a@b.co"), "\u{E9} ");
}

#[test]
fn no_match_input_is_returned_unchanged() {
    let plain = "no links in this sentence at all";
    assert!(matches!(apply(&Urls, plain), std::borrow::Cow::Borrowed(_)));
    assert!(matches!(apply(&BoundedUrls, plain), std::borrow::Cow::Borrowed(_)));
}

const WEIRD_URLS: &[&str] = &[
    "http://user:pass@example.com/path",
    "https://example.com:8080/x",
    "http://example.com/foo%20bar",
    "http://[2001:db8::1]/x",
    "http://example.com\\path\\file",
    "https://xn--nxasmq6b.example/x",
    "http://\u{e9}xample.com/x",
    "HtTp://Example.com",
    "//example.com/protocol-relative",
    "http://example.com.",
    "http://example.com?",
    "http://a..b//c#frag?q=1;2+3~4",
    "mailto:someone@example.com",
    "a@b@cd.com",
    "user@[1.2.3.4]",
    "user@%41.com",
    "user@example.com:25",
    "data:text/html,x",
    "javascript:alert(1)",
    "ftp://example.com/file",
    "http://example.com/a b http://second.example/c",
    "http:///triple-slash",
    "http//missing-colon.example",
    "https://.leading.dot",
];

#[test]
fn weird_urls_match_the_two_pass_reference() {
    for &input in WEIRD_URLS {
        assert_eq!(urls(input), reference::two_pass(input), "{input:?}");
    }
    // A few of them by hand, so the reference is checked too.
    assert_eq!(urls("http://user:pass@example.com/path"), " : /path");
    assert_eq!(urls("http://[2001:db8::1]/x"), "http://[2001:db8::1]/x");
    assert_eq!(urls("user@example.com:25"), " :25");
    assert_eq!(urls("http:///triple-slash"), " ");
    assert_eq!(urls("https://.leading.dot"), " ");
}

/// The one-pass scan against OpenNLP's two passes, on the pool of
/// UrlCharSequenceNormalizerCharacterizationTest (without its lone
/// surrogates, which a Rust string cannot hold) and on pieces chosen to
/// make URLs and addresses touch.
#[test]
fn one_pass_matches_two_passes_on_random_inputs() {
    let opennlp_pool = [
        "a",
        "b",
        "Z",
        "1",
        " ",
        ".",
        ",",
        "-",
        "+",
        "_",
        "@",
        ":",
        "/",
        "%",
        "~",
        ";",
        "=",
        "&",
        "?",
        "#",
        "http",
        "https",
        "ttp",
        "://",
        "http://",
        "s",
        "x.com",
        "co",
        ".com",
        "a@b",
        "@b.co",
        "\u{E9}",
        "\u{1F600}",
    ];
    let touching = ["a@", "b", "bc", "h", "http://", "https://", "x", ".", "-", "@", "c.d", "ttp://", "s://", "q"];
    let mut rng = Rng::new(42);
    for pool in [&opennlp_pool[..], &touching[..]] {
        for _ in 0..20_000 {
            let input = rng.input(pool);
            assert_eq!(urls(&input), reference::two_pass(&input), "{input:?}");
        }
    }
}

#[test]
fn an_address_cut_by_a_following_url_matches_the_two_passes() {
    // The first pass turns the URL into a space, which ends the domain.
    assert_eq!(urls("a@bchttp://x"), "  ");
    assert_eq!(urls("a@bhttp://x"), "a@b ");
    assert_eq!(urls("a@bhttp://x"), reference::two_pass("a@bhttp://x"));
}

#[test]
fn url_placeholders_span_the_replaced_text() {
    let (out, found) = placeholders(Urls, "mail a.b@c.de or see http://x.y/z?q=1 now");
    assert_eq!(out, "mail   or see   now");
    assert_eq!(found, [("a.b@c.de".to_string(), Email), ("http://x.y/z?q=1".to_string(), Url)]);
}

/// Placeholders as (source text, kind).
type Expected = [(&'static str, PlaceholderKind)];

/// An adversarial input and what each URL normalizer makes of it.
type Case = (&'static str, &'static str, &'static Expected, &'static str, &'static Expected);

fn rows(found: &[(&str, PlaceholderKind)]) -> Vec<(String, PlaceholderKind)> {
    found.iter().map(|&(s, k)| (s.to_string(), k)).collect()
}

/// The adversarial cases: what each normalizer replaces, and the
/// placeholder spans, which map back to exactly the URL text.
#[test]
fn adversarial_urls() {
    // (input, Urls output, Urls placeholders, BoundedUrls output, BoundedUrls placeholders)
    let cases: &[Case] = &[
        ("see http://x.y/z.", "see  ", &[("http://x.y/z.", Url)], "see  .", &[("http://x.y/z", Url)]),
        ("(http://x.y/a_(b))", "( (b))", &[("http://x.y/a_", Url)], "( )", &[("http://x.y/a_(b)", Url)]),
        ("http://x.y/a, b", " , b", &[("http://x.y/a", Url)], " , b", &[("http://x.y/a", Url)]),
        (
            "http://a.b/c http://d.e/f",
            "   ",
            &[("http://a.b/c", Url), ("http://d.e/f", Url)],
            "   ",
            &[("http://a.b/c", Url), ("http://d.e/f", Url)],
        ),
        (
            "http://a.b/c me@d.ef",
            "   ",
            &[("http://a.b/c", Url), ("me@d.ef", Email)],
            "   ",
            &[("http://a.b/c", Url), ("me@d.ef", Email)],
        ),
        (
            "http://m\u{FC}nchen.de/x",
            " \u{FC}nchen.de/x",
            &[("http://m", Url)],
            " ",
            &[("http://m\u{FC}nchen.de/x", Url)],
        ),
        ("http://", "http://", &[], "http://", &[]),
    ];
    for &(input, plain_out, plain_rows, bounded_out, bounded_rows) in cases {
        let (out, found) = placeholders(Urls, input);
        assert_eq!((out.as_str(), found), (plain_out, rows(plain_rows)), "Urls on {input:?}");
        let (out, found) = placeholders(BoundedUrls, input);
        assert_eq!((out.as_str(), found), (bounded_out, rows(bounded_rows)), "BoundedUrls on {input:?}");
    }
}

#[test]
fn bounded_removes_whole_http_urls() {
    let cases = [
        ("http://f:21/", " "),
        ("http://f:21/ b", "  b"),
        ("http://example.com/%20foo", " "),
        ("http://user:pass@foo.example.com:8080/x?y#z", " "),
        ("http://[::1]/", " "),
        ("http://[1080::8:800:200C:417A]/index.html", " "),
        ("http://example.com/foo\"bar", " \"bar"),
        ("http://example.com/a'b(c){d}!e", " "),
        ("http://example.com/a`b", " "),
        ("http://example.com/foo<bar", " <bar"),
        ("http://a.b.c/", " "),
        ("HTTP://EXAMPLE.COM/", " "),
        ("Https://example.com/", " "),
        ("https://xn--mnchen-3ya.de/pfad", " "),
        ("http://m\u{FC}nchen.de/stra\u{DF}e", " "),
        ("http://\u{4F8B}\u{3048}.\u{30C6}\u{30B9}\u{30C8}/\u{30D1}\u{30B9}", " "),
        ("http://example.com/\u{1F600}/x", " "),
        ("http://example.com/\u{1D400}", " "),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

#[test]
fn bounded_keeps_other_schemes() {
    for input in [
        "git+https://github.com/foo/bar",
        "blob:https://example.com:443/",
        "telnet://user:pass@foobar.com:23/",
        "redis://user:pass@host:6379/0",
        "rsync://user:pass@host/module",
        "ftp://ftp.example.com/pub",
        "ws://example.com/socket",
        "sc://user:pass@x/",
        "javascript://example.com/",
    ] {
        assert_eq!(bounded(input), input);
    }
}

#[test]
fn bounded_needs_a_left_boundary_before_the_scheme() {
    let cases = [
        ("xhttp://example.com/", "xhttp://example.com/"),
        ("1http://example.com/", "1http://example.com/"),
        ("_http://example.com/", "_http://example.com/"),
        ("a.http://example.com/", "a.http://example.com/"),
        ("a-http://example.com/", "a-http://example.com/"),
        ("\u{E4}http://example.com/", "\u{E4}http://example.com/"),
        ("(http://example.com/a)", "( )"),
        ("[http://example.com/a]", "[ ]"),
        ("<http://example.com/a>", "< >"),
        ("\"http://example.com/a\"", "\" \""),
        ("\u{AB}http://example.com/a\u{BB}", "\u{AB} \u{BB}"),
        ("see:http://example.com/a", "see:http://example.com/a"),
        ("see,http://example.com/a", "see, "),
        ("see/http://example.com/a", "see/ "),
        ("see http://example.com/a", "see  "),
        ("\u{201C}http://example.com/a\u{201D}", "\u{201C} \u{201D}"),
        ("\u{201E}http://example.com/a\u{201C}", "\u{201E} \u{201C}"),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

#[test]
fn bounded_gives_back_trailing_punctuation() {
    let cases = [
        ("see http://example.com/a.", "see  ."),
        ("see http://example.com/a, then", "see  , then"),
        ("http://example.com/a?", " ?"),
        ("http://example.com/a!?", " !?"),
        ("http://example.com/a:", " :"),
        ("http://example.com/a;", " ;"),
        ("(http://example.com/a).", "( )."),
        ("http://example.com/a(b)", " "),
        ("http://example.com/a(b).", " ."),
        ("http://example.com/a(b))", " )"),
        ("http://[::1]", " "),
        ("http://example.com/a]", " ]"),
        ("http://example.com/a?b=1#c", " "),
        ("http://example.com/a?b=1&c=2;d", " "),
        ("http://example.com/..", " .."),
        ("http://example.com/a/", " "),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

#[test]
fn bounded_ends_the_body_at_delimiters() {
    let cases = [
        ("http://example.com/a b", "  b"),
        ("http://example.com/a\tb", " \tb"),
        ("http://example.com/a\nb", " \nb"),
        ("http://example.com/a\u{A0}b", " \u{A0}b"),
        ("http://example.com/a\u{3000}b", " \u{3000}b"),
        ("http://example.com/a\u{2028}b", " \u{2028}b"),
        ("http://example.com/a\u{85}b", " \u{85}b"),
        ("http://example.com/a\u{1}b", " \u{1}b"),
        ("http://example.com/a\u{7F}b", " \u{7F}b"),
        ("http://example.com/a\u{200B}b", " "),
        ("http://example.com/a>b", " >b"),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

#[test]
fn bounded_keeps_text_without_a_url() {
    for input in [
        "http://",
        "https://",
        "http://?",
        "http://.",
        "http:// x",
        "http:/x",
        "http:foo",
        "http//x",
        "://x",
        "http",
        "https:",
        "",
    ] {
        assert_eq!(bounded(input), input);
    }
}

#[test]
fn bounded_handles_urls_at_any_position() {
    let cases = [
        ("http://a.b/ text", "  text"),
        ("text http://a.b/", "text  "),
        ("http://a.b/", " "),
        ("http://a.b/ and http://c.d/", "  and  "),
        ("http://a.b/ http://c.d/", "   "),
        ("x http://a.b/ y https://c.d/e z", "x   y   z"),
        ("http://a.b/?q=http://c.d/", " "),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

#[test]
fn bounded_removes_mail_addresses_outside_urls() {
    let cases = [
        ("asdf asd.fdfa@hasdk23.com.br 2nnfdf", "asdf   2nnfdf"),
        ("asdf asd+fdfa@hasdk23.com.br 2nnfdf", "asdf   2nnfdf"),
        ("asdf asd.fdfa@hasdk23.com_br 2nnfdf", "asdf  _br 2nnfdf"),
        ("mail me@example.com or http://example.com/x", "mail   or  "),
        ("http://user:pass@example.com/x me@example.com", "   "),
        ("a@b", "a@b"),
        ("a@bc", " "),
        ("@example.com", "@example.com"),
    ];
    for (input, expected) in cases {
        assert_eq!(bounded(input), expected, "{input:?}");
    }
}

/// A deliberate difference from OpenNLP: internationalized addresses are
/// replaced whole. OpenNLP leaves the first two unchanged and replaces
/// only the ASCII tail of the third.
#[test]
fn bounded_replaces_internationalized_addresses() {
    let (out, found) = placeholders(BoundedUrls, "write to Jos\u{E9}@example.com today");
    assert_eq!(out, "write to   today");
    assert_eq!(found, rows(&[("Jos\u{E9}@example.com", Email)]));
    let address = "\u{7528}\u{6237}@\u{4F8B}\u{5B50}.\u{5E7F}\u{544A}";
    assert_eq!(bounded(&format!("{address} now")), "  now");
    assert_eq!(bounded("\u{E9}a@b.co"), " ");
    // Urls keeps OpenNLP's ASCII scan.
    assert_eq!(urls("write to Jos\u{E9}@example.com today"), "write to Jos\u{E9}@example.com today");
    assert_eq!(urls("\u{E9}a@b.co"), "\u{E9} ");
    // A letter before the address still joins it to a word.
    assert_eq!(bounded("x\u{E9}a@b.co"), " ");
    assert_eq!(bounded("a\u{301}@bc"), " ");
}

/// Long runs of closing brackets cost linear work: the brackets are
/// counted once and given back one at a time.
#[test]
fn bounded_long_bracket_suffix() {
    for suffix in [")", "]", "}", ")]}", ").]!}"] {
        let tail = suffix.repeat(1000);
        assert_eq!(bounded(&format!("http://example.org/{tail}")), format!(" {tail}"));
    }
    let tail = ")]}".repeat(1000);
    assert_eq!(bounded(&format!("http://example.org/(a)[b]{{c}}{tail}")), format!(" {tail}"));
}
