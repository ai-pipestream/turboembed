//! #968 external partition: per-kind precision and recall on whole
//! third-party documents (RFC texts and Wikipedia wikitext), gold in
//! `tests/data/human-eval/labels-external.tsv` (newlines in covered
//! text escaped as `\n`, for the display-wrapped RFC 7519 JWTs).
//! Kept apart from the mixed set in `precision_recall.rs` so the two
//! partitions never pool. Provenance, licences and annotation method
//! are recorded in that directory's `README.md`.

mod support;

use support::{Counts, load_labelled, measure, ratio, ratio_text};

/// (id, goldens-escaped, file, pinned SHA-256 of the document text).
const EXTERNAL_DOCS: [(&str, bool, &str, &str); 10] = [
    ("rfc7519", false, "rfc7519.txt", "fecd930e9ccf2276b95c0017c6c4ff5d09352e4bc3c7629946447894e0f97248"),
    ("rfc5737", false, "rfc5737.txt", "9d16217614a74a9b064eff900e5cd07a525793cf8227dca1267100e880f0c410"),
    ("rfc3849", false, "rfc3849.txt", "b0741fc609be565b356c62d2f9b9ef30469f3fe7bde386c2690e173751bb1941"),
    ("rfc7042", false, "rfc7042.txt", "8ad7b042253d1204cef2a5eddd6682072a5d4f9edc3b749df15df7bccdeca1df"),
    (
        "wikipedia-iban",
        false,
        "wikipedia-iban.wikitext",
        "bdb116257bb8d25897a11b61de009167c673856ce0458183d83d336d73d8e7b1",
    ),
    (
        "wikipedia-social-security-number",
        false,
        "wikipedia-social-security-number.wikitext",
        "91728bbdc40693f8f721bf57e1d128d72812a30c18a48b887e2a515c33687de5",
    ),
    (
        "wikipedia-social-insurance-number",
        false,
        "wikipedia-social-insurance-number.wikitext",
        "fe5d23a52f0a7c7b73dc2bbe6a10aceaf7498cef6d9d42574127ce8cfecc6988",
    ),
    (
        "wikipedia-nhs-number",
        false,
        "wikipedia-nhs-number.wikitext",
        "0112ee5ee1c94040591ececad7135a4028f5ee3a6ec1b0d4fe9699a42f42cad5",
    ),
    (
        "wikipedia-imei",
        false,
        "wikipedia-imei.wikitext",
        "bb82b401cc09ee2fca976b0f49dafa1aaea2080f769f524f9c9eb12d9d9aceb7",
    ),
    (
        "wikipedia-routing-transit-number",
        false,
        "wikipedia-routing-transit-number.wikitext",
        "75e90f864f2d7896f6ac5ab10a77d6f9392de4e78266d7fc26d2bf6ae41b67af",
    ),
];

/// Only kinds with gold in these documents are pinned here; mentions
/// of any other kind count into the total false positives.
#[test]
fn precision_and_recall_on_external_prose() {
    let set = load_labelled(&EXTERNAL_DOCS, "labels-external.tsv", true);
    let by_kind = measure(&set);
    eprintln!("\nexternal prose evaluation set (#968):");
    let want_by_kind: [(&str, usize, usize, usize); 12] = [
        ("aba-routing", 4, 0, 4),
        ("ca-sin", 0, 0, 2),
        ("email", 17, 1, 1),
        ("iban", 4, 0, 0),
        ("imei", 0, 0, 2),
        ("ipv4", 8, 3, 0),
        ("ipv6", 3, 0, 0),
        ("jwt", 0, 0, 2),
        ("mac", 29, 0, 0),
        ("phone", 5, 14, 0),
        ("uk-nhs", 1, 1, 3),
        ("us-ssn", 3, 0, 0),
    ];
    let mut total = Counts::default();
    for (kind, c) in &by_kind {
        let (p, r) = (ratio_text(c.tp, c.tp + c.fp), ratio_text(c.tp, c.tp + c.fn_));
        eprintln!("{kind:<16} {:>5} {:>5} {:>5} {p:>9} {r:>7}", c.tp, c.fp, c.fn_);
        total.tp += c.tp;
        total.fp += c.fp;
        total.fn_ += c.fn_;
    }
    eprintln!(
        "{} documents, {} bytes; all: tp {} fp {} fn {}, precision {:.4}, recall {:.4}",
        set.len(),
        set.iter().map(|l| l.text.len()).sum::<usize>(),
        total.tp,
        total.fp,
        total.fn_,
        ratio(total.tp, total.tp + total.fp),
        ratio(total.tp, total.tp + total.fn_)
    );
    for (kind, tp, fp, fn_) in want_by_kind {
        let c = by_kind.get(kind).unwrap();
        assert_eq!((c.tp, c.fp, c.fn_), (tp, fp, fn_), "{kind}");
    }
    assert_eq!((total.tp, total.fp, total.fn_), (74, 27, 14));
}

#[test]
fn wrapped_rfc_address_keeps_the_complete_source_span() {
    let mut set = load_labelled(&EXTERNAL_DOCS, "labels-external.tsv", true);
    let index = EXTERNAL_DOCS.iter().position(|&(id, ..)| id == "rfc3849").unwrap();
    let doc = set.remove(index);
    assert_eq!(&doc.text[7658..7679], "ietf-\n   ipr@ietf.org");
    assert!(doc.labels.contains(&(7658, 7679, "email")));
    assert!(!doc.labels.contains(&(7667, 7679, "email")));
    let counts = measure(&[doc]);
    let email = &counts["email"];
    assert_eq!((email.tp, email.fp, email.fn_), (3, 1, 1));
}

#[test]
fn zero_support_is_undefined() {
    assert!(ratio(0, 0).is_nan());
    assert_eq!(ratio_text(0, 0), "N/A");
    assert_eq!(ratio_text(0, 2), "0.000");
    assert_eq!(ratio_text(2, 2), "1.000");
    assert_eq!(ratio(0, 2), 0.0);
    assert_eq!(ratio(2, 2), 1.0);
}
