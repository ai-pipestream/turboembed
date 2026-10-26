use turbo_analysis::{Annotator, Layer, ViewId};
use turbo_analysis_layers::DateTime;
use turbo_grammar::time::zone::{self, Region, Zone};
use turbo_grammar::{DUTCH_TIMES, ENGLISH_TIMES, SPANISH_TIMES, TimeAnnotator};

fn reference(s: &str) -> DateTime {
    DateTime::parse(s).unwrap()
}
#[test]
fn real_time_expressions_keep_the_zone_in_aligned_timex() {
    for (grammar, text, want) in [
        (&ENGLISH_TIMES as &dyn turbo_grammar::Grammar, "Meet at 3pm EST.", "T15-05:00"),
        (&ENGLISH_TIMES, "Meet at 14:00 UTC+2.", "T14:00+02:00"),
        (&SPANISH_TIMES, "Nos vemos a las 14:00 UTC+2.", "T14:00+02:00"),
        (&DUTCH_TIMES, "We spreken af om 14:00 UTC+2.", "T14:00+02:00"),
        (&SPANISH_TIMES, "Nos vemos a las 14:00 hora de Madrid.", "T14:00+02:00"),
        (&DUTCH_TIMES, "We spreken af om 14:00 Amsterdamse tijd.", "T14:00+02:00"),
    ] {
        let a = TimeAnnotator::new(grammar).with_time_zones(Region::Unspecified);
        let t = a.find(text, ViewId::SOURCE, Some(reference("2024-07-01T12:00")));
        assert_eq!(t.check(), Ok(()));
        assert_eq!(t.len(), 1, "{text}");
        let (i, span) = t.spans.iter().next().unwrap();
        assert_eq!(t.get(i).timex, want, "{text}");
        assert!(!text[span.range()].ends_with('.'));
        assert!(
            text[span.range()].contains("EST")
                || text[span.range()].contains("UTC+2")
                || text[span.range()].contains("Madrid")
                || text[span.range()].contains("Amsterdamse")
        );
    }
}
#[test]
fn offsets_and_regions_are_validated_without_partial_numeric_matches() {
    for (s, n) in [("UTC", 0), ("GMT-03:30", -210), ("UTC+0545", 345), ("UTC+14:00", 840), ("PST", -480)] {
        let found = zone::at(s, 0, Region::Unspecified).unwrap();
        assert_eq!(found.end, s.len());
        assert_eq!(found.zone, Zone::Fixed(n));
    }
    for s in [
        "UTC+15",
        "UTC+14:01",
        "UTC+2:70",
        "UTC+",
        "UTC+2:3",
        "UTC+023",
        "UTC+2abc",
        "UTC+2.5",
        "UTC+2+3",
        "UTC_name",
        "ESTuary",
    ] {
        assert!(zone::at(s, 0, Region::Unspecified).is_none(), "{s}");
    }
    assert_eq!(zone::at("CST", 0, Region::Unspecified).unwrap().zone, Zone::Unresolved);
    assert_eq!(zone::at("CST", 0, Region::China).unwrap().zone, Zone::Fixed(480));
    assert_eq!(zone::at("CST", 0, Region::NorthAmerica).unwrap().zone, Zone::Fixed(-360));
    assert_eq!(zone::at("IST", 0, Region::India).unwrap().zone, Zone::Fixed(330));
    let plain = TimeAnnotator::english();
    assert_ne!(plain.config_hash(), TimeAnnotator::english().with_time_zones(Region::Unspecified).config_hash());
    assert_ne!(
        TimeAnnotator::english().with_time_zones(Region::China).config_hash(),
        TimeAnnotator::english().with_time_zones(Region::NorthAmerica).config_hash()
    );
}
#[test]
fn madrid_uses_the_expressed_date_and_refuses_dst_gaps_and_folds() {
    for (s, offset) in [
        ("2024-01-01T14:00", 60),
        ("2024-07-01T14:00", 120),
        ("2024-03-31T01:59:59", 60),
        ("2024-03-31T03:00", 120),
        ("2024-10-27T01:59:59", 120),
        ("2024-10-27T03:00", 60),
    ] {
        assert_eq!(zone::offset_for(Zone::CentralEurope, Some(reference(s))), Some(offset));
    }
    for s in ["2024-03-31T02:30", "2024-10-27T02:30", "1990-07-01T14:00"] {
        assert_eq!(zone::offset_for(Zone::CentralEurope, Some(reference(s))), None);
    }
    assert_eq!(zone::offset_for(Zone::CentralEurope, None), None);
    let a = TimeAnnotator::spanish().with_time_zones(Region::Unspecified);
    let t = a.find("a las 02:30 hora de Madrid", ViewId::SOURCE, Some(reference("2024-03-31T12:00")));
    let m = t.get(t.spans.iter().next().unwrap().0);
    assert_eq!(m.timex, "");
    assert!(m.start.is_none());
    let t = a.find("a las 14:00 hora de Madrid", ViewId::SOURCE, None);
    assert!(t.get(t.spans.iter().next().unwrap().0).timex.is_empty());
}

#[test]
fn upstream_zone_offsets_are_counted_separately_from_disabled_cases() {
    let mut agree = 0;
    let mut disagree = Vec::new();
    let mut unsupported = Vec::new();
    for language in ["English", "Spanish", "Dutch"] {
        let file = format!(
            "{}/tests/data/recognizers-text/{language}/DateTime/DateTimeModel.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(file).unwrap();
        let cases: serde_json::Value = serde_json::from_str(text.strip_prefix('\u{feff}').unwrap_or(&text)).unwrap();
        for (i, case) in cases.as_array().unwrap().iter().enumerate() {
            let offsets: Vec<_> = case["Results"]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|r| r["Resolution"]["values"].as_array().into_iter().flatten())
                .filter_map(|v| v["utcOffsetMins"].as_str())
                .collect();
            if offsets.is_empty() {
                continue;
            }
            assert!(case["NotSupported"].as_str().unwrap().contains("dotnet"));
            let input = case["Input"].as_str().unwrap();
            let zone = input
                .char_indices()
                .filter(|&(at, _)| {
                    at == 0
                        || input[..at]
                            .chars()
                            .last()
                            .is_some_and(|c| turbo_unicode::whitespace::WhitespaceMode::Unicode.is_whitespace(c as u32))
                })
                .find_map(|(at, _)| zone::at(input, at, Region::NorthAmerica));
            match zone.and_then(|z| zone::offset_for(z.zone, None)) {
                Some(n) if offsets.iter().all(|s| s.parse::<i16>().unwrap() == n) => agree += 1,
                Some(_) => disagree.push((language, i)),
                None => unsupported.push((language, i)),
            }
        }
    }
    assert_eq!(agree, 6);
    // These fixtures assign summer offsets to generic mountain/PT in
    // November, unlike the documented standard-time policy here.
    assert_eq!(disagree, [("English", 517), ("Dutch", 293), ("Dutch", 494)]);
    assert!(unsupported.is_empty(), "{unsupported:?}");
}

#[test]
fn unresolved_calendar_date_range_ignores_trailing_zone_and_formats_without_offset() {
    use turbo_grammar::time::timex::{Timex, TimexDate, TimexPoint, TimexTime, TimexYear};
    use turbo_grammar::time::zone::Offsets;

    let a = TimeAnnotator::english().with_time_zones(Region::Unspecified);
    let text = "2/29-3/1/2019 UTC";
    let t = a.find(text, ViewId::SOURCE, Some(reference("2019-01-01T12:00")));
    assert_eq!(t.check(), Ok(()));
    assert_eq!(t.len(), 1);
    let (idx, span) = t.spans.iter().next().unwrap();
    assert_eq!(&text[span.range()], "2/29-3/1/2019");
    let m = t.get(idx);
    assert_eq!(m.timex, "(2019-02-29,2019-03-01,PXD)");
    assert_eq!((m.start, m.end), (None, None));

    let s = TimexPoint {
        date: TimexDate::Ymd { year: TimexYear::Year(2019), month: Some(2), day: Some(29) },
        time: TimexTime::None,
    };
    let e = TimexPoint {
        date: TimexDate::Ymd { year: TimexYear::Year(2019), month: Some(3), day: Some(1) },
        time: TimexTime::None,
    };
    let mut out = String::new();
    zone::write_timex(&mut out, Timex::UnresolvedRange(s, e), Offsets { start: -300, end: -300 }).unwrap();
    assert_eq!(out, "(2019-02-29,2019-03-01,PXD)");
}
