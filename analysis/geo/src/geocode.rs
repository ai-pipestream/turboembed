//! Geocoders, resolutions, the `turbo:locations` layer and the geocode
//! annotator (`opennlp.tools.geo.Geocoder`, `GeoResolution`,
//! `GeocodeAnnotator`, `opennlp.geo.PopulationPriorGeocoder` and
//! `SpatialCoherenceGeocoder`).

use std::cmp::Ordering;
use std::collections::HashSet;
use std::fmt;

use turbo_analysis::{Annotator, Error, Ix, Layer, Overlap, Refs, Span, Spans, View, ViewId};
use turbo_analysis_layers::Entities;
use turbo_jvm::chars::compare_to;
use turbo_unicode::java::to_lowercase;

use crate::entry::{GazetteerEntry, GeoPoint};
use crate::gazetteer::Gazetteer;
use crate::ranking::feature_rank;
use crate::store::Entry;
use crate::text::is_blank;

/// The conventional entity type label for location mentions.
pub const DEFAULT_LOCATION_TYPE: &str = "location";

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
    MentionOutOfBounds {
        span: Span,
        text_len: usize,
    },
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
    mention: Span,
    entry: GazetteerEntry,
    confidence: f64,
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
}

impl<G: Geocoder + ?Sized> Geocoder for &G {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
    }
}

impl<G: Geocoder + ?Sized> Geocoder for Box<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
    }
}

impl<G: Geocoder + ?Sized> Geocoder for std::sync::Arc<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        (**self).resolve(text, location_mentions)
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
                resolutions.push(GeoResolution::new(
                    mention,
                    candidates[0].to_entry(),
                    SINGLE_CANDIDATE_CONFIDENCE,
                )?);
                continue;
            }
            candidates.sort_by(by_prior_view);
            let confidence = score_population_separation(
                candidates[0].population() as f64,
                candidates[1].population() as f64,
            );
            resolutions.push(GeoResolution::new(mention, candidates[0].to_entry(), confidence)?);
        }
        Ok(resolutions)
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
                resolutions.push(GeoResolution::new(
                    mention,
                    candidates[0].to_entry(),
                    SINGLE_CANDIDATE_CONFIDENCE,
                )?);
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
}

/// Resolved location mentions over a text view (`turbo:locations`), matching
/// OpenNLP's `GeocodeAnnotator.LOCATIONS`.
///
/// Multiple candidate resolutions for a single mention share the mention's
/// span in ranked order (`Overlap::Any`).
#[derive(Clone, Debug)]
pub struct Locations {
    pub spans: Spans<Locations>,
    pub entry: Vec<GazetteerEntry>,
    pub confidence: Vec<f64>,
}

impl Locations {
    /// An empty locations layer over `view`.
    pub fn new(view: ViewId) -> Locations {
        Locations { spans: Spans::new(view), entry: Vec::new(), confidence: Vec::new() }
    }

    /// Appends one [`GeoResolution`] to the layer.
    pub fn push(&mut self, resolution: GeoResolution) -> Ix<Locations> {
        let (span, entry, confidence) = resolution.into_parts();
        self.entry.push(entry);
        self.confidence.push(confidence);
        self.spans.push(span)
    }

    /// Returns resolution `i` as a [`GeoResolution`].
    pub fn get(&self, i: Ix<Locations>) -> GeoResolution {
        let idx = i.get();
        GeoResolution {
            mention: self.spans.get(i),
            entry: self.entry[idx].clone(),
            confidence: self.confidence[idx],
        }
    }

    /// Borrowed entry and confidence for resolution `i`.
    pub fn resolution(&self, i: Ix<Locations>) -> (Span, &GazetteerEntry, f64) {
        let idx = i.get();
        (self.spans.get(i), &self.entry[idx], self.confidence[idx])
    }
}

impl Layer for Locations {
    const ID: &'static str = "turbo:locations";
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.entry.len() != n || self.confidence.len() != n {
            return Err("an entry and a confidence for every location span".into());
        }
        self.spans.check(Self::OVERLAP)?;
        for (i, span) in self.spans.iter() {
            let idx = i.get();
            if span.is_empty() {
                return Err(format!("location span {idx} is empty at {}", span.start));
            }
            let c = self.confidence[idx];
            if !(0.0..=1.0).contains(&c) {
                return Err(format!("location {idx} has confidence {c} outside [0, 1]"));
            }
        }
        Ok(())
    }
}

pub(crate) fn normalize_location_types<'a>(
    types: impl IntoIterator<Item = &'a str>,
) -> Result<HashSet<String>, GeocodeError> {
    let mut normalized = HashSet::new();
    for raw in types {
        if is_blank(raw) {
            return Err(GeocodeError::InvalidLocationTypes(
                "locationTypes must not contain a null or blank element",
            ));
        }
        let mut lower = String::new();
        to_lowercase(raw, &mut lower);
        normalized.insert(lower);
    }
    if normalized.is_empty() {
        return Err(GeocodeError::InvalidLocationTypes("locationTypes must not be empty"));
    }
    Ok(normalized)
}

/// Resolves location entities from `opennlp:entities` into `turbo:locations`
/// (`opennlp.tools.geo.GeocodeAnnotator`).
pub struct GeocodeAnnotator<G> {
    geocoder: G,
    location_types: HashSet<String>,
}

impl<G: Geocoder> GeocodeAnnotator<G> {
    /// Creates an annotator that resolves entities typed `"location"`
    /// (case-insensitively).
    pub fn new(geocoder: G) -> GeocodeAnnotator<G> {
        let mut location_types = HashSet::new();
        location_types.insert(DEFAULT_LOCATION_TYPE.to_string());
        GeocodeAnnotator { geocoder, location_types }
    }

    /// Creates an annotator that resolves entities whose type label matches
    /// any element of `location_types` case-insensitively (`Locale.ROOT`).
    pub fn with_location_types<'a>(
        geocoder: G,
        location_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<GeocodeAnnotator<G>, GeocodeError> {
        Ok(GeocodeAnnotator { geocoder, location_types: normalize_location_types(location_types)? })
    }
}

impl<G: Geocoder + Send + Sync + 'static> Annotator for GeocodeAnnotator<G> {
    const ID: &'static str = "turbo-geo:geocode";
    type Reads = (View, Entities);
    type Writes = (Locations,);

    fn annotate(&self, (view, entities): Refs<'_, Self>) -> Result<(Locations,), Error> {
        if entities.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "entities are in a different view".into(),
            });
        }
        let mut mentions = Vec::new();
        let mut lower_buf = String::new();
        for (ix, span) in entities.spans.iter() {
            lower_buf.clear();
            to_lowercase(entities.kinds.label(entities.kind[ix.get()]), &mut lower_buf);
            if self.location_types.contains(&lower_buf) {
                mentions.push(span);
            }
        }

        let mut out = Locations::new(view.id());
        if mentions.is_empty() {
            return Ok((out,));
        }

        let allowed: HashSet<Span> = mentions.iter().copied().collect();
        let resolutions = self.geocoder.resolve(view.text(), &mentions).map_err(|e| Error::Annotator {
            annotator: Self::ID,
            detail: e.to_string(),
        })?;
        for resolution in resolutions {
            if !allowed.contains(&resolution.mention()) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "geocoder returned a mention outside locationMentions: {:?}",
                        resolution.mention()
                    ),
                });
            }
            out.push(resolution);
        }
        Ok((out,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundled::BundledGazetteer;
    use crate::entry::{EntryParts, FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY};
    use crate::in_memory::InMemoryGazetteer;
    use crate::user::UserGazetteer;
    use turbo_analysis::{Analyzer, Document, IxRange, Key, Provenance, Tag, TagSet};
    use turbo_analysis_layers::Tokens;

    const ROWS: &[&str] = &[
        "1\tSpringfield\t\t39.79968\t-89.64399\tUS\tCITY\t750000\t\tIllinois",
        "2\tSpringfield\t\t37.18095\t-93.29229\tUS\tCITY\t450000\t\tMissouri",
        "3\tSpringfield\t\t42.10148\t-72.58982\tUS\tCITY\t680000\t\tMassachusetts",
        "4\tSolo\t\t-7.55611\t110.83167\tID\tCITY\t100000\t\tCentral Java",
        "5\tTietown\t\t10.0\t10.0\t\tADMIN\t5000",
        "6\tTietown\t\t11.0\t11.0\t\tCITY\t5000",
        "7\tGhostville\t\t20.0\t20.0\t\tCITY\t0",
        "8\tGhostville\t\t21.0\t21.0\t\tCITY\t0",
    ];

    fn fixture_gazetteer() -> UserGazetteer {
        let mut data = String::new();
        for row in ROWS {
            data.push_str(row);
            data.push('\n');
        }
        UserGazetteer::from_bytes(data.as_bytes(), "naturalearth").unwrap()
    }

    fn mention_of(text: &str, mention: &str) -> Span {
        let start = text.find(mention).unwrap();
        Span::new(start as u32, (start + mention.len()) as u32)
    }

    fn place(id: &str, name: &str, lat: f64, lon: f64, country: Option<&str>, population: i64) -> GazetteerEntry {
        GazetteerEntry::new(EntryParts {
            source: "test".into(),
            record_id: id.into(),
            name: name.into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(lat, lon).unwrap(),
            bounding_box: None,
            country_code: country.map(Into::into),
            containment: Vec::new(),
            population,
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes: Vec::new(),
        })
        .unwrap()
    }

    struct ReversedGazetteer<G>(G);

    impl<G: Gazetteer> Gazetteer for ReversedGazetteer<G> {
        fn lookup(&self, name: &str) -> Vec<Entry<'_>> {
            let mut v = self.0.lookup(name);
            v.reverse();
            v
        }

        fn by_id(&self, source: &str, record_id: &str) -> Option<Entry<'_>> {
            self.0.by_id(source, record_id)
        }

        fn by_region(&self, code: crate::entry::RegionCode) -> Option<Entry<'_>> {
            self.0.by_region(code)
        }

        fn sources(&self) -> Vec<&str> {
            self.0.sources()
        }
    }

    #[test]
    fn population_prior_resolves_ambiguity_and_scores_confidence() {
        let g = PopulationPriorGeocoder::new(fixture_gazetteer());

        let text = "She grew up in Springfield.";
        let mention = mention_of(text, "Springfield");
        let res = g.resolve(text, &[mention]).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].entry().record_id(), "1");
        assert_eq!(res[0].mention(), mention);
        let expected = 0.5 + 0.4 * (70_000.0 / 1_430_000.0);
        assert_eq!(res[0].confidence(), expected);

        let text2 = "From Springfield to Atlantis.";
        let s = mention_of(text2, "Springfield");
        let a = mention_of(text2, "Atlantis");
        let res2 = g.resolve(text2, &[s, a]).unwrap();
        assert_eq!(res2.len(), 1);
        assert_eq!(res2[0].mention(), s);

        let text3 = "Solo, then Springfield, then Tietown.";
        let solo = mention_of(text3, "Solo");
        let sp = mention_of(text3, "Springfield");
        let tie = mention_of(text3, "Tietown");
        let res3 = g.resolve(text3, &[solo, sp, tie]).unwrap();
        assert_eq!(res3.len(), 3);
        assert_eq!(res3[0].mention(), solo);
        assert_eq!(res3[0].confidence(), 0.9);
        assert_eq!(res3[1].mention(), sp);
        assert_eq!(res3[2].mention(), tie);
        assert_eq!(res3[2].entry().record_id(), "6");
        assert_eq!(res3[2].confidence(), 0.5);

        let text4 = "Ghostville again.";
        let res4 = g.resolve(text4, &[mention_of(text4, "Ghostville")]).unwrap();
        assert_eq!(res4[0].confidence(), 0.5);
        assert_eq!(res4[0].entry().record_id(), "7");

        assert!(g.resolve("Springfield", &[]).unwrap().is_empty());
        assert!(g.resolve("Springfield", &[Span::new(3, 3)]).unwrap().is_empty());
        assert!(g.resolve("abc", &[Span::new(0, 4)]).is_err());
    }

    #[test]
    fn population_prior_re_ranks_reversed_gazetteer_and_handles_max_populations() {
        let small = place("small", "Springfield", 0.0, 0.0, None, 1000);
        let large = place("large", "Springfield", 1.0, 1.0, None, 2000);
        let rev = ReversedGazetteer(InMemoryGazetteer::from_entries(&[small, large]).unwrap());
        let res = PopulationPriorGeocoder::new(rev).resolve("Springfield", &[Span::new(0, 11)]).unwrap();
        assert_eq!(res[0].entry().record_id(), "large");

        let adv1 = place("adv-1", "Overflowville", 0.0, 0.0, None, i64::MAX);
        let adv2 = place("adv-2", "Overflowville", 0.0, 0.0, None, i64::MAX);
        let tie = PopulationPriorGeocoder::new(InMemoryGazetteer::from_entries(&[adv1, adv2]).unwrap())
            .resolve("Overflowville", &[Span::new(0, 13)])
            .unwrap();
        assert_eq!(tie[0].confidence(), 0.5);

        let adv_max = place("adv-max", "Overflowville", 0.0, 0.0, None, i64::MAX);
        let adv_one = place("adv-one", "Overflowville", 0.0, 0.0, None, 1);
        let dom = PopulationPriorGeocoder::new(InMemoryGazetteer::from_entries(&[adv_max, adv_one]).unwrap())
            .resolve("Overflowville", &[Span::new(0, 13)])
            .unwrap();
        assert!(dom[0].confidence() > 0.5 && dom[0].confidence() <= 1.0);
        assert_eq!(dom[0].entry().record_id(), "adv-max");

        let admin = GazetteerEntry::new(EntryParts {
            source: "ext".into(),
            record_id: "admin-1".into(),
            name: "Twinsburg".into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(41.3, -81.4).unwrap(),
            bounding_box: None,
            country_code: Some("US".into()),
            containment: Vec::new(),
            population: 5000,
            feature_class: Some(FEATURE_CLASS_ADMIN.into()),
            attributes: Vec::new(),
        })
        .unwrap();
        let city = GazetteerEntry::new(EntryParts {
            source: "ext".into(),
            record_id: "city-1".into(),
            name: "Twinsburg".into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(41.4, -81.5).unwrap(),
            bounding_box: None,
            country_code: Some("US".into()),
            containment: Vec::new(),
            population: 5000,
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes: Vec::new(),
        })
        .unwrap();
        let rev_twins = ReversedGazetteer(InMemoryGazetteer::from_entries(&[admin, city]).unwrap());
        let res_twins = PopulationPriorGeocoder::new(rev_twins).resolve("Twinsburg", &[Span::new(0, 9)]).unwrap();
        assert_eq!(res_twins[0].entry().record_id(), "city-1");
        assert_eq!(res_twins[0].confidence(), 0.5);
    }

    #[test]
    fn population_prior_resolves_against_bundled_table() {
        let text = "The flight goes from Paris to Springfield.";
        let paris = mention_of(text, "Paris");
        let springfield = mention_of(text, "Springfield");
        let g = PopulationPriorGeocoder::new(BundledGazetteer::get());
        let res = g.resolve(text, &[paris, springfield]).unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].entry().country_code(), Some("FR"));
        assert_eq!(res[1].entry().country_code(), Some("US"));
        assert_eq!(res[1].entry().containment(), &["Massachusetts".to_string()]);
    }

    #[test]
    fn spatial_coherence_moves_ambiguous_mention_toward_context() {
        let gaz = InMemoryGazetteer::from_entries(&[
            place("paris-fr", "Paris", 48.85, 2.35, Some("FR"), 2_100_000),
            place("paris-tx", "Paris", 33.66, -95.56, Some("US"), 25_000),
            place("dallas", "Dallas", 32.78, -96.80, Some("US"), 1_300_000),
            place("houston", "Houston", 29.76, -95.36, Some("US"), 2_300_000),
            place("london", "London", 51.51, -0.13, Some("GB"), 8_800_000),
        ])
        .unwrap();
        let geocoder = SpatialCoherenceGeocoder::new(&gaz);

        let t1 = "the tour hits Dallas, Paris and Houston";
        let r1 = geocoder
            .resolve(t1, &[mention_of(t1, "Dallas"), mention_of(t1, "Paris"), mention_of(t1, "Houston")])
            .unwrap();
        assert_eq!(r1.len(), 3);
        assert_eq!(r1[1].entry().record_id(), "paris-tx");
        assert!(r1[1].confidence() > 0.5);

        let t2 = "a week in Paris";
        let r2 = geocoder.resolve(t2, &[mention_of(t2, "Paris")]).unwrap();
        assert_eq!(r2.len(), 1);
        assert_eq!(r2[0].entry().record_id(), "paris-fr");

        let t3 = "flights between London and Paris";
        let r3 = geocoder.resolve(t3, &[mention_of(t3, "London"), mention_of(t3, "Paris")]).unwrap();
        assert_eq!(r3[1].entry().record_id(), "paris-fr");

        let t4 = "Dallas and Atlantis";
        let r4 = geocoder.resolve(t4, &[mention_of(t4, "Dallas"), mention_of(t4, "Atlantis")]).unwrap();
        assert_eq!(r4.len(), 1);
        assert_eq!(r4[0].entry().record_id(), "dallas");
        assert_eq!(r4[0].confidence(), 0.9);

        assert!(geocoder.resolve("x", &[Span::new(0, 5)]).is_err());

        // Antipodal context produces a finite distance.
        let antipodal_gaz = InMemoryGazetteer::from_entries(&[
            place("first", "Choice", -61.23628525861706, -132.98189410882517, Some("US"), 2),
            place("second", "Choice", -61.23628525861706, -132.98189410882517, Some("US"), 1),
            place("context", "Context", 61.2362852585404, 47.01810589109053, Some("AU"), 1),
        ])
        .unwrap();
        let antipodal = SpatialCoherenceGeocoder::new(antipodal_gaz);
        let t5 = "Choice Context";
        let r5 = antipodal.resolve(t5, &[mention_of(t5, "Choice"), mention_of(t5, "Context")]).unwrap();
        assert_eq!(r5[0].entry().record_id(), "first");
    }

    #[test]
    fn geocode_annotator_filters_location_entities_and_validates_returned_spans() {
        let kinds = TagSet::new("ner", &["PLACE", "place", "person"]);
        let text = "from Paris to Lyon with Alice";
        let mut doc = Document::new(text);
        doc.add(
            &Key::DEFAULT,
            Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None },
            Provenance::by("test"),
        )
        .unwrap();
        let mut entities = Entities {
            spans: Spans::new(ViewId::SOURCE),
            tokens: vec![IxRange::new(0, 0), IxRange::new(0, 0), IxRange::new(0, 0)],
            kind: vec![Tag(0), Tag(1), Tag(2)],
            kinds,
            score: vec![1.0, 1.0, 1.0],
        };
        entities.spans.push(mention_of(text, "Paris"));
        entities.spans.push(mention_of(text, "Lyon"));
        entities.spans.push(mention_of(text, "Alice"));
        doc.add(&Key::DEFAULT, entities, Provenance::by("test")).unwrap();

        let gaz = InMemoryGazetteer::from_entries(&[
            place("paris", "Paris", 48.85, 2.35, Some("FR"), 2_100_000),
            place("lyon", "Lyon", 45.76, 4.83, Some("FR"), 500_000),
        ])
        .unwrap();
        let annotator =
            GeocodeAnnotator::with_location_types(PopulationPriorGeocoder::new(gaz), ["Place"]).unwrap();
        let analyzer = Analyzer::builder().add(annotator).given(Key::<Entities>::DEFAULT).build().unwrap();
        analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
        let locs = doc.get::<Locations>().unwrap();
        assert_eq!(locs.len(), 2);
        assert_eq!(locs.get(Ix::new(0)).entry().name(), "Paris");
        assert_eq!(locs.get(Ix::new(1)).entry().name(), "Lyon");

        assert!(GeocodeAnnotator::with_location_types(PopulationPriorGeocoder::new(BundledGazetteer::get()), []).is_err());
        assert!(
            GeocodeAnnotator::with_location_types(
                PopulationPriorGeocoder::new(BundledGazetteer::get()),
                ["\u{00A0}"]
            )
            .is_err()
        );
    }
}
