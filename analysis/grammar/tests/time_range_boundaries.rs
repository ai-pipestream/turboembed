//! Running text and calendar boundaries beyond the curated spec cases.
use turbo_analysis_layers::DateTime;
use turbo_grammar::time::spec::{DateSpec, Edge, Point, Range, YearRef};
use turbo_grammar::time::{Instant, Resolution, Resolved, TimeValue, resolve};
use turbo_grammar::{DUTCH_TIMES, ENGLISH_TIMES, Grammar, Output, Parser, SPANISH_TIMES};

const NOW: &str = "2026-11-07T16:12:00";

fn mentions(grammar: &'static dyn Grammar, text: &str, now: &str) -> Vec<(String, Resolved)> {
    let mut parser = Parser::new(grammar);
    parser.parse(text);
    let mut found = Vec::new();
    parser.select(text, Output::Time, &mut found);
    found
        .iter()
        .map(|m| {
            let value = parser.times().get(&m.value).unwrap();
            (text[m.start as usize..m.end as usize].to_owned(), resolve(&value, DateTime::parse(now).unwrap()).unwrap())
        })
        .collect()
}

fn bounds(r: &Resolved) -> Vec<(String, String)> {
    r.alternatives
        .iter()
        .map(|a| match a.value {
            Resolution::Range { start: Some(s), end: Some(e) } => (s.to_string(), e.to_string()),
            _ => panic!("expected a resolved bounded range: {a:?}"),
        })
        .collect()
}

#[test]
fn leap_day_at_either_endpoint_pairs_matching_occurrences_in_all_three_languages() {
    for (grammar, adjacent, following, cross_year) in [
        (
            &ENGLISH_TIMES as &'static dyn Grammar,
            "from February 28 to February 29",
            "from February 29 to March 1",
            "from May 1 to February 29",
        ),
        (
            &SPANISH_TIMES as &'static dyn Grammar,
            "del 28 de febrero al 29 de febrero",
            "del 29 de febrero al 1 de marzo",
            "del 1 de mayo al 29 de febrero",
        ),
        (
            &DUTCH_TIMES as &'static dyn Grammar,
            "van 28 februari tot 29 februari",
            "van 29 februari tot 1 maart",
            "van 1 mei tot 29 februari",
        ),
    ] {
        for (text, want) in [
            (adjacent, [("2024-02-28", "2024-02-29"), ("2028-02-28", "2028-02-29")]),
            (following, [("2024-02-29", "2024-03-01"), ("2028-02-29", "2028-03-01")]),
            (cross_year, [("2023-05-01", "2024-02-29"), ("2027-05-01", "2028-02-29")]),
        ] {
            let got = mentions(grammar, text, NOW);
            assert_eq!(got.len(), 1, "{text}: {got:?}");
            assert!(got[0].0.contains("29"), "{text}: {:?}", got[0]);
            let r = &got[0].1;
            assert_eq!(bounds(r), want.map(|(s, e)| (s.into(), e.into())), "{text}");
            assert_eq!(r.preferred, 1, "{text}");
            for a in &r.alternatives {
                let expected = if text == cross_year { "P304D" } else { "P1D" };
                assert!(a.timex.to_string().ends_with(&format!(",{expected})")), "{text}: {}", a.timex);
            }
        }
    }
}

#[test]
fn leap_pairing_honors_reference_year_and_inclusive_edges() {
    let text = "from February 28 to February 29";
    for (now, years) in [
        ("2024-02-28T12:00:00", [2020, 2024]),
        ("2024-03-01T00:00:00", [2024, 2028]),
        ("2100-01-15T00:00:00", [2096, 2104]),
    ] {
        let got = mentions(&ENGLISH_TIMES, text, now);
        assert_eq!(got.len(), 1, "{now}: {got:?}");
        assert_eq!(bounds(&got[0].1), years.map(|y| (format!("{y}-02-28"), format!("{y}-02-29"))));
    }
    let value = TimeValue::range(Range::Between {
        start: Point::date(DateSpec::Cal { year: YearRef::Any, month: 2, day: 28, weekday: 0 }),
        start_edge: Edge::Start,
        end: Point::date(DateSpec::Cal { year: YearRef::Any, month: 2, day: 29, weekday: 0 }),
        end_edge: Edge::End,
    });
    let r = resolve(&value, DateTime::parse(NOW).unwrap()).unwrap();
    assert_eq!(bounds(&r), [("2024-02-28".into(), "2024-03-01".into()), ("2028-02-28".into(), "2028-03-01".into())]);
    assert!(r.alternatives.iter().all(|a| a.timex.to_string().ends_with(",P2D)")));
    let got = mentions(&ENGLISH_TIMES, "from February 29 to February 28", NOW);
    assert_eq!(got.len(), 1);
    assert_eq!(
        bounds(&got[0].1),
        [("2024-02-29".into(), "2025-02-28".into()), ("2028-02-29".into(), "2029-02-28".into())]
    );
    assert!(got[0].1.alternatives.iter().all(|a| a.timex.to_string().ends_with(",P365D)")));
}

#[test]
fn to_date_rejects_objects_noun_modifiers_and_intransitive_infinitives() {
    for text in [
        "She wants to date him.",
        "I need to date this letter.",
        "Is it safe to date coworkers?",
        "He plans to date her next week.",
        "They went to date night on Friday.",
        "We march to date night.",
        "She wants to date.",
        "I wanted to date.",
        "They planned to date.",
        "We are trying to date.",
        "I would like to date.",
        "She hopes to date.",
        "He is hoping to date.",
        "They prefer to date.",
        "They preferred to date.",
        "They tried to date.",
        "He used to date.",
        "We have to date.",
        "To date is difficult.",
        "My dentist uses the most up to date equipment.",
        "dates which are greater than or equals to date 1",
    ] {
        let got = mentions(&ENGLISH_TIMES, text, NOW);
        assert!(got.iter().all(|(s, _)| !turbo_unicode::java::lowercase_cow(s).contains("to date")), "{text}: {got:?}");
    }
    // Suppressing the verb phrase must not suppress an independent time.
    for (text, wanted) in
        [("He plans to date her next week.", "next week"), ("They went to date night on Friday.", "Friday")]
    {
        let got = mentions(&ENGLISH_TIMES, text, NOW);
        assert!(got.iter().any(|(s, _)| s.contains(wanted)), "{text}: {got:?}");
    }
}

#[test]
fn to_date_preserves_retrospective_adjuncts_and_month_to_present_ranges() {
    for text in [
        "Sales to date exceed expectations.",
        "Results to date are encouraging.",
        "The evidence to date suggests caution.",
        "The best performance to date.",
        "Revenue to date has increased.",
        "Costs to date remain low.",
        "Our experience to date shows the limits.",
        "How many Azure clusters docked to date?",
        "To date, no errors were found.",
    ] {
        let got = mentions(&ENGLISH_TIMES, text, NOW);
        let (_, r) = got
            .iter()
            .find(|(s, _)| turbo_unicode::java::lowercase_cow(s) == "to date")
            .unwrap_or_else(|| panic!("{text}: {got:?}"));
        assert_eq!(r.alternatives.len(), 1);
        assert!(matches!(r.alternatives[0].value, Resolution::Range { start: None, end: Some(Instant::Date(_)) }));
        assert_eq!(r.alternatives[0].timex.to_string(), "PRESENT_REF");
    }
    let got = mentions(&ENGLISH_TIMES, "How many Azure clusters docked from March to date?", NOW);
    assert!(got.iter().any(|(s, _)| s == "from March to date"), "{got:?}");
}
