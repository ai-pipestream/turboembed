use turbo_analysis::{Annotator, Layer, ViewId};
use turbo_analysis_layers::{DateTime, TimeKind};
use turbo_grammar::TimeAnnotator;
use turbo_grammar::time::zone::Region;

#[test]
fn zones_supply_context_for_bare_clock_ranges_in_all_languages() {
    let reference = DateTime::parse("2018-11-29T12:00:00");
    for (a, text, matched) in [
        (TimeAnnotator::english(), "😀 Meet between 9-6 PST.", "between 9-6 PST"),
        (TimeAnnotator::spanish(), "😀 Reunión entre 9-6 PST.", "entre 9-6 PST"),
        (TimeAnnotator::dutch(), "😀 Vergadering tussen 9-6 PT.", "tussen 9-6 PT"),
    ] {
        let times = a.with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, reference);
        assert_eq!(times.check(), Ok(()));
        let got: Vec<_> = times.spans.iter().map(|(i, s)| (&text[s.range()], times.get(i))).collect();
        assert_eq!(got.len(), 1, "{text}: {got:?}");
        let (span, m) = got[0];
        assert_eq!(span, matched);
        let doc = turbo_analysis::Document::new(text);
        let table = doc.offsets(ViewId::SOURCE);
        assert_eq!(table.span_from_utf16(text, table.span_to_utf16_typed(text, m.span)).unwrap(), m.span);
        assert_eq!(table.span_from_code_point(text, table.span_to_code_point_typed(text, m.span)).unwrap(), m.span);
        assert_eq!(m.kind, TimeKind::TimeRange);
        assert_eq!(m.timex, "(T09-08:00,T18-08:00,PT9H)");
        assert_eq!((m.start, m.end, m.duration_ms), (Some(9 * 3600), Some(18 * 3600), Some(9 * 3600 * 1000)));
        assert!(m.ambiguous);
    }
}

#[test]
fn pt_uses_standard_time_with_region_and_is_unresolved_without_one() {
    use turbo_grammar::time::zone::{self, Zone};
    assert_eq!(zone::at("pt", 0, Region::NorthAmerica).unwrap().zone, Zone::Fixed(-480));
    assert_eq!(zone::at("pt", 0, Region::Unspecified).unwrap().zone, Zone::Unresolved);
    assert!(zone::at("pt_units", 0, Region::NorthAmerica).is_none());
    assert!(zone::at("ptabc", 0, Region::NorthAmerica).is_none());
}

#[test]
fn corrected_time_algorithm_has_new_provenance() {
    assert_ne!(TimeAnnotator::VERSION, "4");
}

#[test]
fn absent_or_invalid_zones_do_not_enable_latent_ranges() {
    for ctor in [TimeAnnotator::english, TimeAnnotator::spanish, TimeAnnotator::dutch] {
        for text in ["between 9-6", "entre 9-6", "tussen 9-6", "tussen 9-6 UTC+99", "$9-6 PST"] {
            let a = ctor().find(text, ViewId::SOURCE, None);
            let b = ctor().with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, None);
            assert_eq!(a.len(), b.len(), "{text}");
            for (i, _) in a.spans.iter() {
                assert_eq!(a.get(i), b.get(i), "{text}");
            }
        }
    }
    let t = TimeAnnotator::dutch().with_time_zones(Region::Unspecified).find("tussen 9-6 pt", ViewId::SOURCE, None);
    assert_eq!(t.len(), 1);
    let m = t.get(t.spans.iter().next().unwrap().0);
    assert_eq!((m.timex, m.start, m.end), ("", None, None));
}

#[test]
fn generic_names_use_standard_offsets_and_explicit_daylight_names_differ() {
    use turbo_grammar::time::zone::{self, Zone};
    for (text, minutes) in [
        ("Mountain Time", -420),
        ("Pacific Time", -480),
        ("Eastern Time", -300),
        ("Mountain Standard Time", -420),
        ("Mountain Daylight Time", -360),
        ("MDT", -360),
        ("MST", -420),
    ] {
        let z = zone::at(text, 0, Region::Unspecified).unwrap();
        assert_eq!(z.end, text.len());
        assert_eq!(z.zone, Zone::Fixed(minutes));
    }
    let text = "2018-11-28 at 1pm mountain timezone";
    let t = TimeAnnotator::english().with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, None);
    assert_eq!(t.len(), 1);
    let m = t.get(t.spans.iter().next().unwrap().0);
    assert_eq!(m.timex, "2018-11-28T13-07:00");
    let utc = m.start.unwrap() - i64::from(-420) * 60;
    assert_eq!(utc, DateTime::parse("2018-11-28T20:00:00").unwrap().local_seconds());
}

#[test]
fn zones_do_not_promote_font_points_pages_or_currency_ranges() {
    for ctor in [TimeAnnotator::english, TimeAnnotator::spanish, TimeAnnotator::dutch] {
        for text in [
            "Use 10-12 pt font.",
            "Set the heading to 14-16 pt.",
            "Scored 3-4 pt in the game.",
            "9-6 pt",
            "Pages 9-6 EST.",
            "USD 9-6 PST",
            "EUR 9-6 PST",
            "€ 9-6 PST",
        ] {
            let baseline = ctor().find(text, ViewId::SOURCE, None);
            let zoned = ctor().with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, None);
            assert_eq!(baseline.len(), zoned.len(), "{text}");
            for (i, _) in baseline.spans.iter() {
                assert_eq!(baseline.get(i), zoned.get(i), "{text}");
            }
        }
    }
}

#[test]
fn explicit_clock_notation_keeps_zone_attachment() {
    for ctor in [TimeAnnotator::english, TimeAnnotator::spanish, TimeAnnotator::dutch] {
        let text = "09:00-18:00 PST";
        let t = ctor().with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, None);
        assert_eq!(t.len(), 1);
        let m = t.get(t.spans.iter().next().unwrap().0);
        assert_eq!(&text[m.span.range()], text);
        assert!(m.timex.contains("-08:00"), "{}", m.timex);
    }
}
