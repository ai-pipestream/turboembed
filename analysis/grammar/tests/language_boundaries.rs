//! Public language rules keep calendar and numeric limits after the module split.
use turbo_analysis::{Annotator, ViewId};
use turbo_analysis_layers::{Date, DateTime};
use turbo_grammar::time::{Resolution, resolve};
use turbo_grammar::{
    ENGLISH, ENGLISH_TIMES, ENGLISH_UNITS, Grammar, GrammarAnnotator, Output, Parser, SPANISH, SPANISH_TIMES,
    TimeAnnotator, Value,
};

fn full(text: &str, grammar: &'static dyn Grammar, output: Output) -> Option<Value> {
    let mut parser = Parser::new(grammar);
    parser.parse(text);
    let mut mentions = Vec::new();
    parser.select(text, output, &mut mentions);
    mentions.into_iter().find(|m| m.start == 0 && m.end as usize == text.len()).map(|m| m.value)
}

#[test]
fn fractions_and_ordinals_do_not_overflow_or_divide_by_zero() {
    for grammar in [&ENGLISH as &dyn Grammar, &SPANISH] {
        assert!(full("1/0", grammar, Output::Number).is_none());
        assert!(full("999999999999999999999999999999999999th", grammar, Output::Ordinal).is_none());
        assert!(full("1e2147483648", grammar, Output::Number).is_none());
        assert!(full("1e-2147483648", grammar, Output::Number).is_none());
    }
}

#[test]
fn rates_do_not_accept_a_zero_time_denominator() {
    assert!(full("5 meters per 0 seconds", &ENGLISH_UNITS, Output::Quantity).is_none());
    assert!(full("5 meters per second", &ENGLISH_UNITS, Output::Quantity).is_some());
    assert!(full("minus five years old", &ENGLISH_UNITS, Output::Quantity).is_none());
    assert!(full("five years old", &ENGLISH_UNITS, Output::Quantity).is_some());
}

#[test]
fn explicit_calendar_dates_never_resolve_to_another_day() {
    let reference = DateTime::midnight(Date::from_epoch_day(0).unwrap());
    for grammar in [&ENGLISH_TIMES as &dyn Grammar, &SPANISH_TIMES] {
        for text in ["2023-02-29", "2024-02-29", "2024-13-01", "2024-01-32", "2024-00-01", "2024-01-00"] {
            let mut parser = Parser::new(grammar);
            parser.parse(text);
            let mut mentions = Vec::new();
            parser.select(text, Output::Time, &mut mentions);
            let resolved: Vec<_> = mentions
                .iter()
                .filter(|m| m.start == 0 && m.end as usize == text.len())
                .filter_map(|m| parser.times().get(&m.value))
                .filter_map(|v| resolve(&v, reference))
                .flat_map(|r| r.alternatives)
                .filter_map(|a| match a.value {
                    Resolution::Point(p) => Some(p.to_string()),
                    _ => None,
                })
                .collect();
            if text == "2024-02-29" {
                assert_eq!(resolved, ["2024-02-29"], "{}: {text}", grammar.name());
            } else {
                assert!(resolved.is_empty(), "{}: {text}: {resolved:?}", grammar.name());
            }
        }
    }
}

#[test]
fn date_ranges_and_annotator_versions() {
    let reference = DateTime::midnight(Date::from_epoch_day(0).unwrap());
    let annotator = TimeAnnotator::english();
    for text in ["from 2024 to 2020", "from December 2024 to January 2020"] {
        let times = annotator.find(text, ViewId::SOURCE, Some(reference));
        for (&start, &end) in times.start.iter().zip(&times.end) {
            if let (Some(start), Some(end)) = (start, end) {
                assert!(start <= end, "{text}");
            }
        }
    }
    assert_eq!(GrammarAnnotator::VERSION, "3");
    assert_eq!(TimeAnnotator::VERSION, "5");
}
