use turbo_analysis::{Layer, ViewId};
use turbo_analysis_layers::DateTime;
use turbo_grammar::time::zone::Region;
use turbo_grammar::{DUTCH_TIMES, ENGLISH_TIMES, Grammar, SPANISH_TIMES, TimeAnnotator};

#[test]
fn nonclock_readings_keep_their_spans_and_values_with_zones_enabled() {
    for (grammar, texts) in [
        (
            &ENGLISH_TIMES as &dyn Grammar,
            &[
                "this morning CST",
                "this morning EST",
                "tomorrow morning IST",
                "this afternoon UTC+2",
                "tonight London time",
                "tomorrow EST",
                "3 hours EST",
                "every Monday CST",
            ][..],
        ),
        (
            &SPANISH_TIMES,
            &["por la mañana hora de Madrid", "esta mañana CST", "mañana por la tarde UTC+2", "una hora EST"][..],
        ),
        (&DUTCH_TIMES, &["vanmorgen Amsterdamse tijd", "deze ochtend UTC+2", "vanavond CST", "morgen IST"][..]),
    ] {
        for text in texts {
            for reference in [Some(DateTime::parse("2024-07-01T12:00").unwrap()), None] {
                let plain = TimeAnnotator::new(grammar).find(text, ViewId::SOURCE, reference);
                assert!(!plain.is_empty(), "{text}");
                for region in [Region::Unspecified, Region::NorthAmerica, Region::China, Region::India, Region::Ireland]
                {
                    let zoned =
                        TimeAnnotator::new(grammar).with_time_zones(region).find(text, ViewId::SOURCE, reference);
                    assert_eq!(zoned.check(), Ok(()));
                    assert_eq!(zoned.len(), plain.len(), "{text}: {region:?}");
                    for (i, _) in plain.spans.iter() {
                        assert_eq!(zoned.get(i), plain.get(i), "{text}: {region:?}, reference {reference:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn explicit_clock_zones_keep_their_opt_in_region_and_reference_rules() {
    let july = Some(DateTime::parse("2024-07-01T12:00").unwrap());
    for (text, region, reference, expected) in [
        ("3pm EST", Region::Unspecified, None, "T15-05:00"),
        ("3pm CST", Region::Unspecified, july, ""),
        ("3pm CST", Region::China, None, "T15+08:00"),
        ("3pm CST", Region::NorthAmerica, None, "T15-06:00"),
        ("3pm IST", Region::India, None, "T15+05:30"),
        ("3pm London time", Region::Unspecified, None, ""),
        ("3pm London time", Region::Unspecified, july, "T15+01:00"),
        ("tomorrow at 3pm EST", Region::Unspecified, None, ""),
        ("from 2pm to 4pm EST", Region::Unspecified, None, "(T14-05:00,T16-05:00,PT2H)"),
    ] {
        let zoned = TimeAnnotator::english().with_time_zones(region).find(text, ViewId::SOURCE, reference);
        assert_eq!(zoned.check(), Ok(()));
        assert_eq!(zoned.len(), 1, "{text}");
        let (i, span) = zoned.spans.iter().next().unwrap();
        assert_eq!(span.range(), 0..text.len(), "{text}");
        assert_eq!(zoned.get(i).timex, expected, "{text}");
        let plain = TimeAnnotator::english().find(text, ViewId::SOURCE, reference);
        assert_eq!(plain.len(), 1, "{text}");
        assert!(plain.get(i).span.end < span.end, "zone must remain opt-in: {text}");
    }
}

#[test]
fn a_separate_date_mention_does_not_change_the_city_clock_reference() {
    let text = "1:30am London time on 2024-10-27";
    let zoned = TimeAnnotator::english().with_time_zones(Region::Unspecified).find(
        text,
        ViewId::SOURCE,
        Some(DateTime::parse("2024-07-01T12:00").unwrap()),
    );
    let got: Vec<_> = zoned.spans.iter().map(|(i, s)| (&text[s.range()], zoned.get(i).timex)).collect();
    assert_eq!(got, [("1:30am London time", "T01:30+01:00"), ("2024-10-27", "2024-10-27")]);
}
