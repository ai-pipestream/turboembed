//! #1884: HIPAA Safe Harbor ZIP generalization (`ZipSafeHarbor`). The
//! bundled table against the published restricted-prefix list, the ZIP
//! shapes kept and refused, a caller table, and the strategy as a rewrite
//! and inside a per-kind policy. Values are made up.

mod support;

use support::*;
use turbo_analysis::Span;
use turbo_pii::rewrite::{
    HHS_2000_RESTRICTED_PREFIXES, MaskPolicy, RewritePolicy, RewriteStrategy, ZipSafeHarbor, ZipTableError, source_view,
};
use turbo_pii::{Pack, PiiKind, PiiStrategyKind};

/// The three-digit ZIP areas of 20,000 or fewer people from the 2000
/// Census, as HHS lists them in "Guidance Regarding Methods for
/// De-identification of Protected Health Information in Accordance with
/// the HIPAA Privacy Rule", written out here independently of the crate.
const PUBLISHED: [&str; 17] = [
    "036", "059", "063", "102", "203", "556", "692", "790", "821", "823", "830", "831", "878", "879", "884", "890",
    "893",
];

#[test]
fn the_bundled_table_is_the_published_restricted_prefix_list() {
    assert_eq!(HHS_2000_RESTRICTED_PREFIXES, PUBLISHED);
    let zip = ZipSafeHarbor::hhs_2000();
    assert!(zip.is_hhs_2000());
    assert_eq!(zip.restricted().collect::<Vec<_>>(), PUBLISHED);
    assert_eq!(ZipSafeHarbor::default(), zip);
    for p in PUBLISHED {
        assert_eq!(zip.apply(&format!("{p}17")), "000**", "{p}");
        assert_eq!(zip.apply(&format!("{p}17-0042")), "000**-****", "{p}");
    }
    // Every other three-digit area keeps its digits, the neighbours of each
    // restricted prefix included.
    for n in 0..1000u32 {
        let p = format!("{n:03}");
        let restricted = PUBLISHED.contains(&p.as_str());
        assert_eq!(zip.is_restricted(p.as_bytes().try_into().unwrap()), restricted, "{p}");
        let want = if restricted { "000**".to_owned() } else { format!("{p}**") };
        assert_eq!(zip.apply(&format!("{p}55")), want);
    }
}

#[test]
fn a_zip_keeps_its_area_and_masks_the_rest() {
    let zip = ZipSafeHarbor::hhs_2000();
    assert_eq!(zip.apply("62704"), "627**");
    assert_eq!(zip.apply("62704-1234"), "627**-****");
    assert_eq!(zip.apply("00501"), "005**");
    assert_eq!(zip.apply("03601"), "000**");
    assert_eq!(zip.apply("89301-1234"), "000**-****");
    assert_eq!(zip.clone().with_mask('#').apply("62704-1234"), "627##-####");
}

#[test]
fn what_is_not_a_zip_is_masked_whole() {
    let zip = ZipSafeHarbor::hhs_2000();
    for not in ["6270", "627041", "62704 1234", "62704-123", "62704-12345", "6270A", "A2704", "62704_1234", ""] {
        assert_eq!(zip.apply(not), "*".repeat(not.chars().count()), "{not:?}");
    }
    // Non-ASCII digits are not a ZIP code; one mask per character.
    assert_eq!(zip.apply("\u{0666}\u{0662}\u{0667}\u{0660}\u{0664}"), "*****");
    assert_eq!(zip.apply("jane@example.com"), "****************");
}

#[test]
fn a_caller_table_replaces_the_bundled_one() {
    let zip = ZipSafeHarbor::with_restricted(["627", "036", "627"]).unwrap();
    assert!(!zip.is_hhs_2000());
    assert_eq!(zip.restricted().collect::<Vec<_>>(), ["036", "627"], "sorted, duplicates folded");
    assert_eq!(zip.apply("62704"), "000**");
    assert_eq!(zip.apply("05901"), "059**", "not in the caller table");
    assert_eq!(ZipSafeHarbor::with_restricted(PUBLISHED).unwrap(), ZipSafeHarbor::hhs_2000());

    assert_eq!(ZipSafeHarbor::with_restricted(Vec::<&str>::new()), Err(ZipTableError::Empty));
    for (bad, at) in [(vec!["12"], 0), (vec!["123", "1a3"], 1), (vec!["1234"], 0), (vec!["\u{0661}23"], 0)] {
        assert_eq!(ZipSafeHarbor::with_restricted(&bad), Err(ZipTableError::NotThreeDigits(at)), "{bad:?}");
    }
    let shown = ZipTableError::NotThreeDigits(1).to_string();
    assert!(!shown.contains("1a3"), "the error names the position, not the entry: {shown}");
}

#[test]
fn the_rewrite_generalizes_detected_zips_and_records_its_strategy() {
    let text = "Springfield, IL 62704-1234; Hanover, NH 03601.";
    let pii = layer(&pack(Pack::UsIdentity), text);
    let zips: Vec<Span> = pii.mentions().filter(|m| m.kind == PiiKind::UsZip).map(|m| m.span).collect();
    assert_eq!(zips.len(), 2, "both ZIP codes are found");
    let r = ZipSafeHarbor::hhs_2000().rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "Springfield, IL 627**-****; Hanover, NH 000**.");
    let sources: Vec<Span> = pii.mentions().map(|m| m.span).collect();
    assert_offset_and_span_maps(text, &sources, &r);
    assert!(r.edits().edits().all(|e| e.strategy == PiiStrategyKind::ZipSafeHarbor));
    assert_eq!(r.edits().edits().count(), 2);
}

#[test]
fn a_policy_sends_us_zip_here_and_other_kinds_elsewhere() {
    let text = "IL 62704, call (555) 123-4567.";
    let pii = layer(&wide(), text);
    assert!(pii.mentions().any(|m| m.kind == PiiKind::UsZip));
    assert!(pii.mentions().any(|m| m.kind == PiiKind::Phone));
    let policy = RewritePolicy::new(MaskPolicy::new('#').unwrap()).with(PiiKind::UsZip, ZipSafeHarbor::hhs_2000());
    assert_eq!(policy.strategy(PiiKind::UsZip).kind(), PiiStrategyKind::ZipSafeHarbor);
    let r = policy.rewrite(&source_view(text), &pii).unwrap();
    assert_eq!(r.text(), "IL 627**, call ##############.");
    let strategies: Vec<_> = r.edits().edits().map(|e| e.strategy).collect();
    assert_eq!(strategies, [PiiStrategyKind::ZipSafeHarbor, PiiStrategyKind::Mask]);
}

#[test]
fn overlapping_mentions_are_masked_whole() {
    let text = "62704-1234";
    let pii = manual(&[(0, 10, PiiKind::UsZip, "62704-1234"), (6, 10, PiiKind::Phone, "1234")]);
    let zip = ZipSafeHarbor::hhs_2000();
    assert_eq!(zip.rewrite(&source_view(text), &pii).unwrap().text(), "**********");
    let policy = RewritePolicy::new(RewriteStrategy::from(zip.clone()));
    assert_eq!(policy.rewrite(&source_view(text), &pii).unwrap().text(), "**********");
    // A ZIP member of another winner's cluster keeps none of its digits.
    let phone_wins = manual(&[(0, 10, PiiKind::Phone, "62704-1234"), (0, 5, PiiKind::UsZip, "62704")]);
    let keep_end = turbo_pii::rewrite::Truncate::keep_end(4);
    let p = RewritePolicy::new(keep_end).with(PiiKind::UsZip, zip.clone());
    assert_eq!(p.rewrite(&source_view(text), &phone_wins).unwrap().text(), "******1234");
    assert!(zip.clone().refusing_overlaps().rewrite(&source_view(text), &pii).is_err());
}

/// A five-digit mention of another kind (a caller's account number) is not
/// a ZIP code: the strategy masks it whole, alone and as the default of a
/// policy, so it never keeps its first three digits.
#[test]
fn a_zip_shaped_mention_of_another_kind_is_masked_whole() {
    let account = PiiKind::Custom(turbo_pii::CustomKindId::new(0));
    let text = "acct 62704 zip 62704";
    let pii = manual(&[(5, 10, account, "62704"), (15, 20, PiiKind::UsZip, "62704")]);
    let want = "acct ***** zip 627**";
    assert_eq!(ZipSafeHarbor::hhs_2000().rewrite(&source_view(text), &pii).unwrap().text(), want);
    let policy = RewritePolicy::new(ZipSafeHarbor::hhs_2000());
    assert_eq!(policy.rewrite(&source_view(text), &pii).unwrap().text(), want);
}
