//! The upstream zone cases run even though their fixture marks .NET unsupported.
//! Policy/fixture differences are asserted explicitly, not counted as conformance.
use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::{DateTime, TimeKind};
use turbo_grammar::TimeAnnotator;
use turbo_grammar::time::zone::{self, Region, Zone};

fn cases(lang: &str) -> serde_json::Value {
    let file = format!("{}/tests/data/recognizers-text/{lang}/DateTime/DateTimeModel.json", env!("CARGO_MANIFEST_DIR"));
    let source = std::fs::read_to_string(file).unwrap();
    serde_json::from_str(source.trim_start_matches('\u{feff}')).unwrap()
}

#[test]
fn upstream_clock_ranges_have_exact_source_spans_and_preferred_local_readings() {
    for (lang, a, index, start, end, timex) in [
        ("English", TimeAnnotator::english(), 349, 10, 31, "(T06:30-08:00,T09-08:00,PT2H30M)"),
        ("Spanish", TimeAnnotator::spanish(), 392, 11, 29, "(T06:30-08:00,T09-08:00,PT2H30M)"),
        ("Dutch", TimeAnnotator::dutch(), 294, 3, 24, "(T06:30-08:00,T09-08:00,PT2H30M)"),
        ("Dutch", TimeAnnotator::dutch(), 293, 104, 117, "(T09-08:00,T18-08:00,PT9H)"),
    ] {
        let cases = cases(lang);
        let case = &cases[index];
        let text = case["Input"].as_str().unwrap();
        let r = DateTime::parse(case["Context"]["ReferenceDateTime"].as_str().unwrap());
        let t = a.with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, r);
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.len(), if index == 293 { 4 } else { 1 });
        let m = t.spans.iter().map(|(i, _)| t.get(i)).find(|m| m.kind == TimeKind::TimeRange).unwrap();
        assert_eq!((m.span.start, m.span.end, m.timex), (start, end, timex), "{lang} {index}");
        assert!(m.ambiguous);
        let (s, e) = if index == 293 { (9 * 3600, 18 * 3600) } else { (6 * 3600 + 30 * 60, 9 * 3600) };
        assert_eq!((m.start, m.end, m.duration_ms), (Some(s), Some(e), Some((e - s) * 1000)));
        if lang == "Spanish" {
            // The fixture writes 'past' instead of input 'pst' and gives
            // shifted UTF-16 boundaries. Preserve actual source bytes.
            assert_eq!(case["Results"][0]["Text"], "entre 6:30 a 9 past");
            assert_eq!(&text[m.span.range()], "entre 6:30 a 9 pst");
        }
        if index == 293 {
            assert_eq!(case["Results"][3]["Resolution"]["values"][0]["utcOffsetMins"], "-420");
        }
    }
}

#[test]
fn mountain_fixture_summer_offsets_are_intentional_policy_divergences() {
    for (lang, a, index, span, timex) in [
        ("English", TimeAnnotator::english(), 517, "1pm mountain timezone", "T13-07:00"),
        ("Dutch", TimeAnnotator::dutch(), 494, "om 13.00 Mountain tijdzone", "T13:00-07:00"),
    ] {
        let cases = cases(lang);
        let case = &cases[index];
        let text = case["Input"].as_str().unwrap();
        assert_eq!(case["Results"][0]["Resolution"]["values"][0]["utcOffsetMins"], "-360");
        let r = DateTime::parse(case["Context"]["ReferenceDateTime"].as_str().unwrap());
        let t = a.with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, r);
        assert_eq!(t.len(), 1);
        let m = t.get(t.spans.iter().next().unwrap().0);
        assert_eq!((&text[m.span.range()], m.timex, m.start), (span, timex, Some(13 * 3600)));
        if lang == "Dutch" {
            assert_eq!(case["Results"][0]["Start"], 22);
            assert_eq!(m.span.start, 19); // Native Dutch extraction retains 'om'.
        }
    }
}

#[test]
fn standalone_fixture_zones_are_readable_but_not_times_mentions() {
    for (lang, a, index, start) in [
        ("English", TimeAnnotator::english(), 516, 11),
        ("Spanish", TimeAnnotator::spanish(), 556, 12),
        ("Dutch", TimeAnnotator::dutch(), 493, 13),
    ] {
        let cases = cases(lang);
        let case = &cases[index];
        let text = case["Input"].as_str().unwrap();
        let z = zone::at(text, start, Region::NorthAmerica).unwrap();
        assert_eq!((z.start, z.end, z.zone), (start, text.len(), Zone::Fixed(-480)));
        assert_eq!(case["Results"][0]["Resolution"]["values"][0]["utcOffsetMins"], "-480");
        assert!(a.with_time_zones(Region::NorthAmerica).find(text, ViewId::SOURCE, None).is_empty());
    }
}
