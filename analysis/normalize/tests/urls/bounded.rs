//! `BoundedUrls`: OpenNLP's `BoundedUrlCharSequenceNormalizerTest`.

use super::*;

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
