//! `EmailSpanTest`: whole addresses in context, and no part of a longer
//! domain or local part.

mod support;

use support::*;
use turbo_pii::rewrite::{Pseudonymizer, mask_all};
use turbo_pii::{Extractor, Pack, PiiDetector, PiiKind};

const ADDRESS: &str = "User+tag@Example.COM";
const NORMALIZED: &str = "User+tag@example.com";

fn emails() -> PiiDetector {
    only(Extractor::Cursor, &[PiiKind::Email])
}

#[track_caller]
fn assert_whole(prefix: &str, suffix: &str) {
    let text = format!("{prefix}{ADDRESS}{suffix}");
    let start = prefix.len() as u32;
    assert_found(&emails(), &text, &[(start, start + ADDRESS.len() as u32, "email", NORMALIZED)]);
    let pii = layer(&emails(), &text);
    assert_eq!(mask_all(&text, &pii, '*').unwrap(), format!("{prefix}{}{suffix}", "*".repeat(ADDRESS.len())));
    assert_eq!(Pseudonymizer::new("", "").rewrite(&text, &pii).unwrap().text(), format!("{prefix}EMAIL-1{suffix}"));
}

#[test]
fn a_partial_domain_is_refused() {
    for separator in ["", ".", "-", "..", "--", ".-", "-."] {
        for c in ["é", "\u{661}", "\u{10400}", "\u{1D7D8}"] {
            assert_none(&emails(), &format!("Contact: {ADDRESS}{separator}{c}"));
        }
    }
}

#[test]
fn trailing_punctuation_stays_out() {
    for suffix in ["", ".", "-", "..", "--", ".-", "-.", ". é", "- é", ". \u{661}", ".😀", "-😀", ", next", "; next"]
    {
        assert_whole("Contact: ", suffix);
    }
}

#[test]
fn offsets_in_context() {
    for (prefix, suffix) in [
        ("<", ">"),
        ("(", ")"),
        ("\"", "\""),
        ("'", "'"),
        ("email=", ""),
        ("mailto:", ""),
        ("/contacts/", ""),
        ("😀 ", "!"),
        ("e\u{301} ", "."),
        ("\t", "\r\n"),
    ] {
        assert_whole(prefix, suffix);
    }
}

#[test]
fn a_partial_local_part_is_refused() {
    for c in ["é", "\u{661}", "\u{10400}", "\u{1D7D8}"] {
        assert_none(&emails(), &format!("{c}{ADDRESS}"));
    }
}

#[test]
fn the_scan_goes_on_after_a_refused_domain() {
    let text = "Skip user@example.com.é; contact User+tag@Example.COM.";
    let start = text.find(ADDRESS).unwrap() as u32;
    assert_found(&emails(), text, &[(start, start + ADDRESS.len() as u32, "email", NORMALIZED)]);
    let pii = layer(&pack(Pack::Contact), text);
    assert_eq!(
        mask_all(text, &pii, '*').unwrap(),
        format!("Skip user@example.com.é; contact {}.", "*".repeat(ADDRESS.len()))
    );
    assert_eq!(
        Pseudonymizer::new("", "").rewrite(text, &pii).unwrap().text(),
        "Skip user@example.com.é; contact EMAIL-1."
    );
}
