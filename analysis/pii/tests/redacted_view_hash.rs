//! The redacted view's config hash binds every parameter that changes what
//! it writes, so an incremental run never reuses a view written under a
//! policy that redacts differently.

use turbo_analysis::{Analyzer, Annotator, Document};
use turbo_pii::PiiDetector;
use turbo_pii::PiiKind;
use turbo_pii::rewrite::{
    Abbreviator, FormatRedactor, HmacTokenizer, MapReplacer, MaskPolicy, Pseudonymizer, REDACTED, RedactedView,
    RewritePolicy, RewriteStrategy, Sha256Tokenizer, StaticReplacer, Truncate,
};

fn hash(policy: RewritePolicy) -> u64 {
    RedactedView::new(policy).config_hash()
}

fn of(s: impl Into<RewriteStrategy>) -> u64 {
    hash(RewritePolicy::new(s))
}

fn redacted(policy: RewritePolicy, text: &str) -> String {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(RedactedView::new(policy).bound()).build().unwrap();
    let mut docs = vec![Document::new(text)];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs[0].get_key(&REDACTED).unwrap().text().to_owned()
}

#[test]
fn masks_that_keep_different_characters_hash_apart() {
    let whole = RewritePolicy::new(MaskPolicy::new('*').unwrap());
    let first4 = RewritePolicy::new(MaskPolicy::first_n(4, '*').unwrap());
    let text = "mail jane@example.com now";
    assert_ne!(redacted(whole.clone(), text), redacted(first4.clone(), text));
    assert_ne!(hash(whole), hash(first4));
    // The mask character alone: '*' and '#' write different views.
    let star = RewritePolicy::new(MaskPolicy::new('*').unwrap());
    let hash_mark = RewritePolicy::new(MaskPolicy::new('#').unwrap());
    assert_ne!(redacted(star.clone(), text), redacted(hash_mark.clone(), text));
    assert_ne!(hash(star), hash(hash_mark));
}

#[test]
fn static_replacements_hash_apart() {
    assert_ne!(of(StaticReplacer::new("FIRST")), of(StaticReplacer::new("SECOND")));
    assert_ne!(
        of(StaticReplacer::new("X").with(PiiKind::Email, "A")),
        of(StaticReplacer::new("X").with(PiiKind::Email, "B"))
    );
    let per_kind =
        |rep: &str| hash(RewritePolicy::new(Pseudonymizer::default()).with(PiiKind::Card, StaticReplacer::new(rep)));
    assert_ne!(per_kind("[CARD]"), per_kind("[C]"));
    assert_eq!(per_kind("[CARD]"), per_kind("[CARD]"));
}

#[test]
fn every_strategy_parameter_is_bound() {
    let pairs = [
        (of(Pseudonymizer::new("[", "]")), of(Pseudonymizer::new("<", ">"))),
        (
            of(HmacTokenizer::new(b"0123456789abcdef0123456789abcdef").unwrap()),
            of(HmacTokenizer::new(b"fedcba9876543210fedcba9876543210").unwrap()),
        ),
        (of(FormatRedactor::new("[%t]").unwrap()), of(FormatRedactor::new("<%t>").unwrap())),
        (of(Sha256Tokenizer::default()), of(Sha256Tokenizer::default().with_prefix("h:"))),
        (of(Truncate::keep_start(2)), of(Truncate::keep_end(2))),
        (of(Abbreviator::new()), of(Abbreviator::with_punctuation("-", ""))),
        (of(MapReplacer::new([("jane", "A")], true).unwrap()), of(MapReplacer::new([("jane", "B")], true).unwrap())),
    ];
    for (i, (a, b)) in pairs.iter().enumerate() {
        assert_ne!(a, b, "pair {i}");
    }
    let rev = |n| hash(RewritePolicy::new(Pseudonymizer::default()).with_revision(n));
    assert_ne!(rev(1), rev(2));
}
