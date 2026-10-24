//! Geocoders, resolutions, the `turbo:locations` layer and the geocode
//! annotator (`opennlp.tools.geo.Geocoder`, `GeoResolution`,
//! `GeocodeAnnotator`, `opennlp.geo.PopulationPriorGeocoder` and
//! `SpatialCoherenceGeocoder`).

mod annotator;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::fmt;

use turbo_analysis::Span;
use turbo_jvm::chars::compare_to;

use crate::entry::{GazetteerEntry, GeoPoint};
use crate::gazetteer::Gazetteer;
use crate::ranking::feature_rank;
use crate::store::Entry;

pub use self::annotator::{DEFAULT_LOCATION_TYPE, GeocodeAnnotator, Locations};
pub(crate) use self::annotator::{hash_location_types, normalize_location_types};

/// Admission before geocoding. This is a candidate filter, not a place-name
/// classifier. Population separation scores do not measure mention validity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MentionPolicy {
    /// Resolve every entity with an accepted location type (OpenNLP parity).
    #[default]
    All,
    /// Reject a single ASCII letter, including surrounding whitespace.
    /// Useful for dictionary candidates: bundled aliases `S` and `C` can
    /// otherwise turn possessives or scientific symbols into cities.
    /// Non-ASCII names and multi-character names remain eligible. Legitimate
    /// single-letter names also need the `All` policy or a separate pipeline.
    ExcludeSingleAsciiLetters,
}

impl MentionPolicy {
    pub(super) fn admits(self, text: &str) -> bool {
        let text = text.trim();
        self == Self::All || !(text.len() == 1 && text.as_bytes()[0].is_ascii_alphabetic())
    }
}

const SINGLE_CANDIDATE_CONFIDENCE: f64 = 0.9;
const TIE_CONFIDENCE: f64 = 0.5;
const POPULATION_CONFIDENCE_SPAN: f64 = 0.4;
const EARTH_RADIUS_KM: f64 = 6371.0;

/// Why geocoding or configuring a geocode annotator failed.
#[derive(Clone, Debug, PartialEq)]
pub enum GeocodeError {
    /// A resolution mention span was empty.
    EmptyMention(Span),
    /// A resolution confidence was outside `[0.0, 1.0]` or `NaN`.
    InvalidConfidence(f64),
    /// A mention span fell outside the input text or off a UTF-8 character boundary.
    MentionOutOfBounds { span: Span, text_len: usize },
    /// The configured location entity type set was empty or contained a blank label.
    InvalidLocationTypes(&'static str),
    /// An underlying gazetteer or I/O failure occurred.
    Io(String),
}

impl fmt::Display for GeocodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GeocodeError::EmptyMention(span) => write!(f, "mention must not be empty, got: {span:?}"),
            GeocodeError::InvalidConfidence(c) => write!(f, "confidence must be in [0, 1], got: {c}"),
            GeocodeError::MentionOutOfBounds { span, text_len } => {
                write!(f, "mention span {span:?} is outside the text of length {text_len}")
            }
            GeocodeError::InvalidLocationTypes(msg) => f.write_str(msg),
            GeocodeError::Io(msg) => write!(f, "geocoding failed: {msg}"),
        }
    }
}

impl std::error::Error for GeocodeError {}

/// The resolution of one location mention to a [`GazetteerEntry`]
/// (`opennlp.tools.geo.GeoResolution`).
#[derive(Clone, Debug, PartialEq)]
pub struct GeoResolution {
    pub(super) mention: Span,
    pub(super) entry: GazetteerEntry,
    pub(super) confidence: f64,
}

impl GeoResolution {
    /// Creates a checked resolution: `mention` must not be empty and
    /// `confidence` must be in `[0.0, 1.0]`.
    pub fn new(mention: Span, entry: GazetteerEntry, confidence: f64) -> Result<GeoResolution, GeocodeError> {
        if mention.is_empty() {
            return Err(GeocodeError::EmptyMention(mention));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(GeocodeError::InvalidConfidence(confidence));
        }
        Ok(GeoResolution { mention, entry, confidence })
    }

    /// The span of the resolved mention in the document text.
    pub fn mention(&self) -> Span {
        self.mention
    }

    /// The resolved gazetteer record.
    pub fn entry(&self) -> &GazetteerEntry {
        &self.entry
    }

    /// The resolution confidence in `[0.0, 1.0]`.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Consumes the resolution and returns its parts.
    pub fn into_parts(self) -> (Span, GazetteerEntry, f64) {
        (self.mention, self.entry, self.confidence)
    }
}

/// Compares two [`Entry`] views by `CandidateRanking.BY_PRIOR`: population
/// descending, feature class rank (`CITY` < `ADMIN` < `POI` < other), source
/// in UTF-16 order, and record id in UTF-16 order.
pub fn by_prior_view(a: &Entry<'_>, b: &Entry<'_>) -> Ordering {
    b.population()
        .cmp(&a.population())
        .then_with(|| feature_rank(a.feature_class()).cmp(&feature_rank(b.feature_class())))
        .then_with(|| compare_to(a.source(), b.source()).cmp(&0))
        .then_with(|| compare_to(a.record_id(), b.record_id()).cmp(&0))
}

/// Compares two owned [`GazetteerEntry`] records by `CandidateRanking.BY_PRIOR`.
pub fn by_prior_entry(a: &GazetteerEntry, b: &GazetteerEntry) -> Ordering {
    b.population()
        .cmp(&a.population())
        .then_with(|| feature_rank(a.feature_class()).cmp(&feature_rank(b.feature_class())))
        .then_with(|| compare_to(a.source(), b.source()).cmp(&0))
        .then_with(|| compare_to(a.record_id(), b.record_id()).cmp(&0))
}

fn validate_mentions(text: &str, location_mentions: &[Span]) -> Result<(), GeocodeError> {
    let text_len = text.len();
    for &span in location_mentions {
        let start = span.start as usize;
        let end = span.end as usize;
        if start > end || end > text_len || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            return Err(GeocodeError::MentionOutOfBounds { span, text_len });
        }
    }
    Ok(())
}

/// Resolves location mentions in a document text to gazetteer records
/// (`opennlp.tools.geo.Geocoder`).
pub trait Geocoder {
    /// Resolves each resolvable mention in `location_mentions` against `text`.
    /// Unresolvable mentions are omitted; every returned [`GeoResolution::mention`]
    /// equals one of the input spans.
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError>;

    /// Deterministic 64-bit configuration and gazetteer fingerprint for
    /// [`Annotator::config_hash`](turbo_analysis::Annotator::config_hash).
    fn fingerprint(&self) -> u64 {
        0
    }
}

impl<G: Geocoder + ?Sized> Geocoder for &G {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

impl<G: Geocoder + ?Sized> Geocoder for Box<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

impl<G: Geocoder + ?Sized> Geocoder for std::sync::Arc<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
    }

    fn fingerprint(&self) -> u64 {
        (**self).fingerprint()
    }
}

/// Disambiguates each location mention independently by population prior
/// (`opennlp.geo.PopulationPriorGeocoder`).
pub struct PopulationPriorGeocoder<G> {
    gazetteer: G,
}

impl<G: Gazetteer> PopulationPriorGeocoder<G> {
    /// Wraps `gazetteer`.
    pub fn new(gazetteer: G) -> PopulationPriorGeocoder<G> {
        PopulationPriorGeocoder { gazetteer }
    }

    /// Reference to the underlying gazetteer.
    pub fn gazetteer(&self) -> &G {
        &self.gazetteer
    }
}

fn score_population_separation(first: f64, second: f64) -> f64 {
    let sum = first + second;
    if sum == 0.0 {
        return TIE_CONFIDENCE;
    }
    let separation = (first - second) / sum;
    let raw = TIE_CONFIDENCE + POPULATION_CONFIDENCE_SPAN * separation;
    raw.clamp(0.0, 1.0)
}

impl<G: Gazetteer> Geocoder for PopulationPriorGeocoder<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        validate_mentions(text, location_mentions)?;
        if location_mentions.is_empty() {
            return Ok(Vec::new());
        }
        let mut resolutions = Vec::with_capacity(location_mentions.len());
        for &mention in location_mentions {
            let surface = &text[mention.start as usize..mention.end as usize];
            let mut candidates = self.gazetteer.lookup(surface);
            if candidates.is_empty() {
                continue;
            }
            if candidates.len() == 1 {
                resolutions.push(GeoResolution::new(mention, candidates[0].to_entry(), SINGLE_CANDIDATE_CONFIDENCE)?);
                continue;
            }
            candidates.sort_by(by_prior_view);
            let confidence =
                score_population_separation(candidates[0].population() as f64, candidates[1].population() as f64);
            resolutions.push(GeoResolution::new(mention, candidates[0].to_entry(), confidence)?);
        }
        Ok(resolutions)
    }

    fn fingerprint(&self) -> u64 {
        const TAG: u64 = 0x504f_505f_5052_494f;
        let h = (0xcbf2_9ce4_8422_2325 ^ TAG).wrapping_mul(0x0100_0000_01b3);
        (h ^ self.gazetteer.fingerprint()).wrapping_mul(0x0100_0000_01b3)
    }
}

/// Disambiguates location mentions in two passes: provisional population-prior
/// picks in pass 1, then minimum mean haversine distance to all other resolved
/// mentions in pass 2 (`opennlp.geo.SpatialCoherenceGeocoder`).
pub struct SpatialCoherenceGeocoder<G> {
    gazetteer: G,
}

impl<G: Gazetteer> SpatialCoherenceGeocoder<G> {
    /// Wraps `gazetteer`.
    pub fn new(gazetteer: G) -> SpatialCoherenceGeocoder<G> {
        SpatialCoherenceGeocoder { gazetteer }
    }

    /// Reference to the underlying gazetteer.
    pub fn gazetteer(&self) -> &G {
        &self.gazetteer
    }
}

fn haversine_km(a: GeoPoint, b: GeoPoint) -> f64 {
    let lat1 = a.latitude().to_radians();
    let lat2 = b.latitude().to_radians();
    let d_lat = (b.latitude() - a.latitude()).to_radians();
    let d_lon = (b.longitude() - a.longitude()).to_radians();
    let sin_lat = (d_lat / 2.0).sin();
    let sin_lon = (d_lon / 2.0).sin();
    let h = (sin_lat * sin_lat + lat1.cos() * lat2.cos() * sin_lon * sin_lon).min(1.0);
    2.0 * EARTH_RADIUS_KM * h.sqrt().asin()
}

fn mean_distance_km(point: GeoPoint, others: &[GeoPoint]) -> f64 {
    let mut sum = 0.0;
    for &other in others {
        sum += haversine_km(point, other);
    }
    sum / (others.len() as f64)
}

impl<G: Gazetteer> Geocoder for SpatialCoherenceGeocoder<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        validate_mentions(text, location_mentions)?;
        let n = location_mentions.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let mut candidates_by_mention: Vec<Vec<Entry<'_>>> = Vec::with_capacity(n);
        let mut provisional_picks: Vec<Option<GeoPoint>> = Vec::with_capacity(n);
        for &mention in location_mentions {
            let surface = &text[mention.start as usize..mention.end as usize];
            let mut ranked = self.gazetteer.lookup(surface);
            if ranked.is_empty() {
                provisional_picks.push(None);
                candidates_by_mention.push(ranked);
            } else {
                ranked.sort_by(by_prior_view);
                provisional_picks.push(Some(ranked[0].location()));
                candidates_by_mention.push(ranked);
            }
        }

        let mut resolutions = Vec::with_capacity(n);
        let mut context = Vec::with_capacity(n);
        for (i, candidates) in candidates_by_mention.iter().enumerate() {
            if candidates.is_empty() {
                continue;
            }
            let mention = location_mentions[i];
            if candidates.len() == 1 {
                resolutions.push(GeoResolution::new(mention, candidates[0].to_entry(), SINGLE_CANDIDATE_CONFIDENCE)?);
                continue;
            }

            context.clear();
            for (j, &pick) in provisional_picks.iter().enumerate() {
                if j != i
                    && let Some(pt) = pick
                {
                    context.push(pt);
                }
            }

            if context.is_empty() {
                let first_pop = candidates[0].population() as f64;
                let second_pop = candidates[1].population() as f64;
                let separation = if first_pop <= 0.0 { 0.0 } else { 1.0 - (second_pop / first_pop) };
                let confidence = TIE_CONFIDENCE + POPULATION_CONFIDENCE_SPAN * separation;
                resolutions.push(GeoResolution::new(mention, candidates[0].to_entry(), confidence)?);
                continue;
            }

            let mut best_index = 0usize;
            let mut best_distance = f64::INFINITY;
            let mut second_distance = f64::INFINITY;
            for (c, candidate) in candidates.iter().enumerate() {
                let distance = mean_distance_km(candidate.location(), &context);
                if distance < best_distance {
                    second_distance = best_distance;
                    best_distance = distance;
                    best_index = c;
                } else if distance < second_distance {
                    second_distance = distance;
                }
            }
            let separation = if second_distance == 0.0 { 0.0 } else { 1.0 - (best_distance / second_distance) };
            let confidence = TIE_CONFIDENCE + POPULATION_CONFIDENCE_SPAN * separation;
            resolutions.push(GeoResolution::new(mention, candidates[best_index].to_entry(), confidence)?);
        }
        Ok(resolutions)
    }

    fn fingerprint(&self) -> u64 {
        const TAG: u64 = 0x5350_4154_5f43_4f48;
        let h = (0xcbf2_9ce4_8422_2325 ^ TAG).wrapping_mul(0x0100_0000_01b3);
        (h ^ self.gazetteer.fingerprint()).wrapping_mul(0x0100_0000_01b3)
    }
}
