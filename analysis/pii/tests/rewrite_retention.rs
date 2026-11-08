//! Integration tests for partial character retention (`Truncate` and
//! `MaskPolicy`) and initials abbreviation (`Abbreviator`), split from
//! `rewrite_strategies.rs`.

mod support;

use support::*;
use turbo_analysis::{Annotator, Document, Span, View};
use turbo_pii::rewrite::source_view;
use turbo_pii::rewrite::{Abbreviator, KeepSide, MaskPolicy, MaskedView, Truncate, mask};
use turbo_pii::{PiiDetector, PiiKind};

const PERSON_KIND: PiiKind = test_kind(0);

fn cursor() -> PiiDetector {
    PiiDetector::cursor()
}

#[test]
fn truncate_keeps_start_or_end_chars_and_masks_all_when_mention_is_too_short() {
    let s4 = Truncate::keep_start(4);
    let e4 = Truncate::keep_end(4);
    assert_eq!(s4.side(), KeepSide::Start);
    assert_eq!(e4.side(), KeepSide::End);
    assert_eq!(Truncate::last_n(4), e4);
    assert_eq!(s4.count(), 4);
    assert_eq!(s4.mask(), '*');

    assert_eq!(s4.apply("jane@example.com"), "jane************");
    assert_eq!(e4.apply("4111111111111111"), "************1111");
    assert_eq!(s4.with_mask('#').keeping_format().apply("4111-2222-3333-4444"), "4111-####-####-####");
    assert_eq!(e4.keeping_format().apply("4111-2222-3333-4444"), "****-****-****-4444");

    // Short mentions (<= N chars) are masked completely so short secrets never stay in the clear.
    for short in ["a", "ab", "abc", "abcd", "Zoë", "😀𝕒b"] {
        let len = short.chars().count();
        assert!(len <= 4);
        assert_eq!(s4.apply(short), "*".repeat(len), "keep_start(4) on {short:?}");
        assert_eq!(e4.apply(short), "*".repeat(len), "keep_end(4) on {short:?}");
    }
    assert_eq!(e4.keeping_format().apply("12-34"), "**-**");
    assert_eq!(Truncate::keep_start(0).apply("secret"), "******");

    // Multi-byte Unicode characters inside and around mentions keep valid UTF-8 boundaries.
    assert_eq!(Truncate::keep_start(2).apply("ZoëKrüger"), "Zo*******");
    assert_eq!(Truncate::keep_end(3).apply("ZoëKrüger"), "******ger");
    assert_eq!(Truncate::keep_start(2).apply("😀𝕒𝕓𝕔"), "😀𝕒**");
    assert_eq!(Truncate::keep_end(2).apply("😀𝕒𝕓𝕔"), "**𝕓𝕔");

    // MaskPolicy::first_n, last_n, keeping_leading, MaskedView, and MaskPolicy::rewrite.
    let first4 = MaskPolicy::first_n(4, '*').unwrap().keeping_format();
    let last4 = MaskPolicy::last_n(4, '*').unwrap().keeping_format();
    let both2 = MaskPolicy::new('*').unwrap().keeping_format().keeping_leading(2).keeping_trailing(2);
    assert_eq!(first4.leading(), 4);
    assert_eq!(last4.trailing(), 4);
    assert_eq!(first4.masked("4111-2222-3333-4444"), "4111-****-****-****");
    assert_eq!(last4.masked("4111-2222-3333-4444"), "****-****-****-4444");
    assert_eq!(both2.masked("4111-2222-3333-4444"), "41**-****-****-**44");

    let text = "Für Zoë: card 4111-1111-1111-1111 😀";
    let pii = layer(&cursor(), text);
    let doc = Document::new(text);
    let mv = MaskedView::all(first4).view(doc.get::<View>().unwrap(), &pii).unwrap();
    assert_eq!(mv.text(), mask(text, pii.mentions().map(|m| (m.span, first4))).unwrap());
    assert_eq!(mv.text(), "Für Zoë: card 4111-****-****-**** 😀");
    let mr = first4.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(mr.text(), "Für Zoë: card 4111-****-****-**** 😀");
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &mr);

    // MaskedView::config_hash does not collide for large leading/trailing values (65536, 1 << 48).
    let base_hash = MaskedView::all(MaskPolicy::new('*').unwrap()).config_hash();
    let lead_64k = MaskedView::all(MaskPolicy::new('*').unwrap().keeping_leading(65536)).config_hash();
    let trail_2p48 = MaskedView::all(MaskPolicy::new('*').unwrap().keeping_trailing(1usize << 48)).config_hash();
    assert_ne!(lead_64k, base_hash);
    assert_ne!(trail_2p48, base_hash);
    assert_ne!(lead_64k, trail_2p48);
}

#[test]
fn abbreviator_reduces_multi_word_hyphenated_and_unicode_names_to_initials() {
    let dotted = Abbreviator::new();
    let compact = Abbreviator::compact();
    let spaced = Abbreviator::with_punctuation(". ", ".");
    let hyph_dotted = Abbreviator::dotted().preserving_hyphens();
    let hyph_compact = Abbreviator::compact().preserving_hyphens();

    assert_eq!(dotted.abbreviate("John Fitzgerald Kennedy"), "J.F.K.");
    assert_eq!(compact.abbreviate("John Fitzgerald Kennedy"), "JFK");
    assert_eq!(spaced.abbreviate("John Fitzgerald Kennedy"), "J. F. K.");

    assert_eq!(dotted.abbreviate("Jean-Luc Picard"), "J.L.P.");
    assert_eq!(compact.abbreviate("Jean-Luc Picard"), "JLP");
    assert_eq!(hyph_dotted.abbreviate("Jean-Luc Picard"), "J.-L.P.");
    assert_eq!(hyph_compact.abbreviate("Jean-Luc Picard"), "J-LP");

    assert_eq!(hyph_dotted.abbreviate("åsa-britt östergren"), "Å.-B.Ö.");
    assert_eq!(dotted.abbreviate("élodie d'aubigné"), "É.D.A.");
    assert_eq!(dotted.clone().preserving_case().abbreviate("élodie d'aubigné"), "é.d.a.");
    assert_eq!(dotted.abbreviate("---"), "*");

    let text = "Guests: John Fitzgerald Kennedy, Jean-Luc Picard, and Åsa-Britt Östergren.";
    let pii = manual(&[
        (8, 31, PERSON_KIND, "John Fitzgerald Kennedy"),
        (33, 48, PERSON_KIND, "Jean-Luc Picard"),
        (54, 75, PERSON_KIND, "Åsa-Britt Östergren"),
    ]);
    let r = hyph_dotted.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "Guests: J.F.K., J.-L.P., and Å.-B.Ö..");
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);
}
