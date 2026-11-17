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

use turbo_analysis::Span;
use turbo_analysis_layers::{CustomKinds, PiiDetectorRef};

use super::ModelCandidate;
use crate::hits::Hits;
use crate::scan::city::anchored_at;
use crate::scan::city_bare::person_context;

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
