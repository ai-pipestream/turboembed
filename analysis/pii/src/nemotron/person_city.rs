//! A bare city name that the trained PII head reads as a person's name
//! (#2312): `Paris won the race`, `we saw Madison`. The rules of
//! `scan::city_person` catch a person word next to the name; with the
//! head loaded, its person spans settle the rest. Only a bare reading
//! gives way: a name before a state or after a city label
//! (`Paris, TX`, `City: Paris`) stays a city whatever the head says.
//!
//! The head's own `city` candidates answer to the same person rules
//! (#2388): one on a name with a person word around it (`we met
//! Madison`, `Salem, who`) is turned down, as the bare reading was.
//!
//! A name the text uses for a person elsewhere is a person's where it
//! stands alone too (#2398): once the person rules read `Paris's mother`
//! in a text, the head's `city` on `Paris won the race` in the same text
//! stands only where a place word stands before the name or the head is
//! sure of the city ([`CITY_OF_PERSON_NAME_MIN`]).

use turbo_analysis::Span;
use turbo_analysis_layers::{CustomKinds, PiiDetectorRef, PiiKind};

use super::ModelCandidate;
use crate::hits::Hits;
use crate::scan::city::anchored_at;
use crate::scan::city_bare::{after_place_word, person_context};
use crate::text::name_char;

/// The head's kinds for a person's name, as `Nemotron-PII` labels them.
const PERSON_KINDS: [&str; 2] = ["first-name", "last-name"];

/// The spans of the bare city candidates of the `city` detector that lie
/// inside a person span the head admits (its probability at or above
/// `threshold`, when one is set), and that no other rule found on the
/// same span. With no person span, no city is looked at.
pub(super) fn bare_cities_in_persons(
    text: &str,
    hits: &Hits,
    city: PiiDetectorRef,
    candidates: &[ModelCandidate],
    model_kinds: &CustomKinds,
    threshold: Option<f32>,
) -> Vec<Span> {
    let persons: Vec<Span> = candidates
        .iter()
        .filter(|mc| {
            mc.probability.is_finite()
                && threshold.is_none_or(|min| mc.probability.clamp(0.0, 1.0) >= min)
                && model_kinds.name(mc.kind).is_some_and(|name| PERSON_KINDS.contains(&name))
        })
        .map(|mc| mc.span)
        .collect();
    if persons.is_empty() {
        return Vec::new();
    }
    hits.admitted_spans(city)
        .filter(|c| {
            persons.iter().any(|p| p.start <= c.start && c.end <= p.end)
                && !hits.other_at(city, *c)
                && !anchored_at(text, c.start as usize, c.end as usize)
        })
        .collect()
}

/// Whether a `city` candidate of the head on `span` stands on a name the
/// person rules of the bare reading refuse (`we met Madison`, `Dayton
/// knew she`), and no state after it or city label before it anchors it.
pub(super) fn city_reads_as_person(text: &str, span: Span) -> bool {
    let (start, end) = (span.start as usize, span.end as usize);
    person_context(text, start, end) && !anchored_at(text, start, end)
}

/// The least probability at which a bare `city` of the head stands on a
/// name the text uses for a person elsewhere, with no place word before
/// it. On the person-name memo the trained head gave the city uses of
/// that kind 0.90 and 0.99 (`Salem is where`, `saw Columbus from the
/// plane`) and three of the four people standing alone 0.72 to 0.81
/// (docs/pii.md).
pub const CITY_OF_PERSON_NAME_MIN: f32 = 0.85;

/// The most names of one text whose other uses are read: each costs a
/// pass over the text, so a text of many unsure names keeps the cities
/// past this many.
const MAX_NAMES: usize = 32;

/// The spans of the head's admitted `city` candidates that give way to
/// a person use of the same name elsewhere in `text` (#2398): a bare
/// name under [`CITY_OF_PERSON_NAME_MIN`], with no place word before
/// it, that the person rules read as a person's at another place, where
/// no rule but the city scanner's bare reading found anything.
pub(super) fn unsure_cities_of_person_names(
    text: &str,
    hits: &Hits,
    city: PiiDetectorRef,
    candidates: &[ModelCandidate],
    rules: &[(Span, PiiKind, String)],
    threshold: Option<f32>,
) -> Vec<Span> {
    // Each name read so far and whether the text uses it for a person.
    let mut names: Vec<(&str, bool)> = Vec::new();
    candidates
        .iter()
        .filter(|mc| {
            let (start, end) = (mc.span.start as usize, mc.span.end as usize);
            let unsure = mc.kind == PiiKind::City
                && mc.probability.is_finite()
                && mc.probability < CITY_OF_PERSON_NAME_MIN
                && threshold.is_none_or(|min| mc.probability.clamp(0.0, 1.0) >= min);
            let Some(name) = text.get(start..end).filter(|name| unsure && !name.is_empty()) else {
                return false;
            };
            let other_rule = |&(rs, rk, _): &(Span, PiiKind, String)| {
                mc.span.start < rs.end && rs.start < mc.span.end && (rs, rk) != (mc.span, PiiKind::City)
            };
            if rules.iter().any(other_rule)
                || hits.other_at(city, mc.span)
                || anchored_at(text, start, end)
                || after_place_word(text, start)
            {
                return false;
            }
            if let Some(&(_, person)) = names.iter().find(|(n, _)| *n == name) {
                return person;
            }
            if names.len() == MAX_NAMES {
                return false;
            }
            let person = person_use(text, name);
            names.push((name, person));
            person
        })
        .map(|mc| mc.span)
        .collect()
}

/// Whether `name` stands at some place of `text` as a word of its own
/// with person context around it and nothing that anchors it as a city.
/// The person rules never hold on a name standing alone, so the use is
/// another than the one asked about.
fn person_use(text: &str, name: &str) -> bool {
    text.match_indices(name).any(|(at, name)| {
        let to = at + name.len();
        !text[..at].chars().next_back().is_some_and(name_char)
            && !text[to..].chars().next().is_some_and(name_char)
            && person_context(text, at, to)
            && !anchored_at(text, at, to)
    })
}
