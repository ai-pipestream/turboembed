//! A bare city name that the trained PII head reads as a person's name
//! (#2312): `Paris won the race`, `we saw Madison`. The rules of
//! `scan::city_person` catch a person word next to the name; with the
//! head loaded, its person spans settle the rest. Only a bare reading
//! gives way: a name before a state or after a city label
//! (`Paris, TX`, `City: Paris`) stays a city whatever the head says.

use turbo_analysis::Span;
use turbo_analysis_layers::{CustomKinds, PiiDetectorRef};

use super::ModelCandidate;
use crate::hits::Hits;
use crate::scan::city::anchored_at;

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
