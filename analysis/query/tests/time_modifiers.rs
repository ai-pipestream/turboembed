//! Time phrase cache shortcut (#1051): relational and boundary
//! modifiers around a cached trigger phrase must reach the
//! full time finder, not be stripped for a bare-phrase hit.

use turbo_analysis_layers::Date;
use turbo_query::QueryAnalysisProfile;

fn temporal(q: &str) -> Vec<(String, Option<i64>, Option<i64>)> {
    let profile = QueryAnalysisProfile::with_reference_date(Date::new(2026, 7, 14).unwrap());
    let a = profile.analyze(q).unwrap();
    a.temporal_filters().map(|(f, t)| (f.surface.clone(), t.start_epoch_seconds, t.end_epoch_seconds)).collect()
}

#[test]
fn relational_modifiers_survive_the_phrase_cache() {
    let bare = temporal("flights tomorrow");
    let before = temporal("flights before tomorrow");
    assert_eq!(before.len(), 1, "{before:?}");
    assert_eq!(before[0].0, "before tomorrow");
    assert_ne!((before[0].1, before[0].2), (bare[0].1, bare[0].2), "before must change the range");
    assert_eq!(before[0].1, None, "before is open at the start: {before:?}");
    assert_eq!(before[0].2, bare[0].1, "before ends where tomorrow starts");

    let since = temporal("logs since last week");
    assert_eq!(since.len(), 1, "{since:?}");
    assert_eq!(since[0].0, "since last week");
    assert_eq!(since[0].2, None, "since is open at the end: {since:?}");

    let after = temporal("news after yesterday");
    assert_eq!(after.len(), 1, "{after:?}");
    assert_eq!(after[0].0, "after yesterday");
    assert_eq!(after[0].2, None, "after is open at the end: {after:?}");
}

#[test]
fn boundary_modifiers_survive_the_phrase_cache() {
    let bare = temporal("reports next month");
    let early = temporal("reports early next month");
    assert_eq!(early.len(), 1, "{early:?}");
    assert_eq!(early[0].0, "early next month");
    let (bs, be) = (bare[0].1.unwrap(), bare[0].2.unwrap());
    let (es, ee) = (early[0].1.unwrap(), early[0].2.unwrap());
    assert_eq!(es, bs);
    assert!(ee < be, "early narrows the month: {early:?} vs {bare:?}");

    let end = temporal("reports end of this month");
    assert_eq!(end.len(), 1, "{end:?}");
    assert_eq!(end[0].0, "end of this month");
}

#[test]
fn unmodified_phrases_still_hit() {
    for q in ["flights tomorrow", "news yesterday", "reports last week", "trips next month"] {
        let t = temporal(q);
        assert_eq!(t.len(), 1, "{q}: {t:?}");
        assert!(t[0].1.is_some() && t[0].2.is_some(), "{q}: {t:?}");
    }
}
