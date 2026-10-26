//! The `turbo:locations` layer and `GeocodeAnnotator`
//! (`opennlp.tools.geo.GeocodeAnnotator`).

use std::collections::HashSet;

use turbo_analysis::{
    Annotator, Document, Error, Ix, Layer, Overlap, ProjectableLayer, Refs, Span, Spans, View, ViewId,
};
use turbo_analysis_layers::Entities;
use turbo_unicode::java::to_lowercase;

use crate::entry::GazetteerEntry;
use crate::text::is_blank;

use super::{GeoResolution, GeocodeError, Geocoder, MentionPolicy};

/// The conventional entity type label for location mentions.
pub const DEFAULT_LOCATION_TYPE: &str = "location";

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
        GeoResolution { mention: self.spans.get(i), entry: self.entry[idx].clone(), confidence: self.confidence[idx] }
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
        if !self.spans.is_start_ordered() {
            return Err("location spans must be in non-decreasing start order".into());
        }
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

impl ProjectableLayer for Locations {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Locations {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            entry: self.entry.clone(),
            confidence: self.confidence.clone(),
        })
    }
}

pub(crate) fn normalize_location_types<'a>(
    types: impl IntoIterator<Item = &'a str>,
) -> Result<HashSet<String>, GeocodeError> {
    let mut normalized = HashSet::new();
    for raw in types {
        if is_blank(raw) {
            return Err(GeocodeError::InvalidLocationTypes("locationTypes must not contain a null or blank element"));
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

pub(crate) fn hash_location_types(seed: u64, location_types: &HashSet<String>) -> u64 {
    let mut labels: Vec<_> = location_types.iter().collect();
    labels.sort_unstable();
    let mut hash = turbo_analysis::Fnv1a::with_seed(seed);
    for label in labels {
        hash.bytes(label.as_bytes()).u8(0xff);
    }
    hash.finish()
}

/// Resolves location entities from `opennlp:entities` into `turbo:locations`
/// (`opennlp.tools.geo.GeocodeAnnotator`).
pub struct GeocodeAnnotator<G> {
    geocoder: G,
    location_types: HashSet<String>,
    mention_policy: MentionPolicy,
}

impl<G: Geocoder> GeocodeAnnotator<G> {
    /// Creates an annotator that resolves entities typed `"location"`
    /// (case-insensitively).
    pub fn new(geocoder: G) -> GeocodeAnnotator<G> {
        let mut location_types = HashSet::new();
        location_types.insert(DEFAULT_LOCATION_TYPE.to_string());
        GeocodeAnnotator { geocoder, location_types, mention_policy: MentionPolicy::All }
    }

    /// Creates an annotator that resolves entities whose type label matches
    /// any element of `location_types` case-insensitively (`Locale.ROOT`).
    pub fn with_location_types<'a>(
        geocoder: G,
        location_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<GeocodeAnnotator<G>, GeocodeError> {
        Ok(GeocodeAnnotator {
            geocoder,
            location_types: normalize_location_types(location_types)?,
            mention_policy: MentionPolicy::All,
        })
    }

    /// Applies admission before the geocoder sees the mentions, so rejected
    /// candidates cannot affect a spatial geocoder's context either.
    pub fn with_mention_policy(mut self, policy: MentionPolicy) -> Self {
        self.mention_policy = policy;
        self
    }
}

impl<G: Geocoder + Send + Sync + 'static> Annotator for GeocodeAnnotator<G> {
    const ID: &'static str = "turbo-geo:geocode";
    type Reads = (View, Entities);
    type Writes = (Locations,);

    fn annotate(&self, (view, entities): Refs<'_, Self>) -> Result<(Locations,), Error> {
        if entities.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "entities are in a different view".into() });
        }
        entities.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        let mut mentions = Vec::new();
        let mut lower_buf = String::new();
        for (ix, span) in entities.spans.iter() {
            if span.is_empty() {
                continue;
            }
            lower_buf.clear();
            to_lowercase(entities.kinds.label(entities.kind[ix.get()]), &mut lower_buf);
            if self.location_types.contains(&lower_buf) {
                let surface = view.text().get(span.range()).ok_or_else(|| Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("mention span {span:?} is outside the text or off a UTF-8 boundary"),
                })?;
                if self.mention_policy.admits(surface) {
                    mentions.push(span);
                }
            }
        }

        let mut out = Locations::new(view.id());
        if mentions.is_empty() {
            return Ok((out,));
        }

        let allowed: HashSet<Span> = mentions.iter().copied().collect();
        let resolutions = self
            .geocoder
            .resolve(view.text(), &mentions)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        for resolution in resolutions {
            if !allowed.contains(&resolution.mention()) {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!("geocoder returned a mention outside locationMentions: {:?}", resolution.mention()),
                });
            }
            out.push(resolution);
        }
        out.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        let policy = match self.mention_policy {
            MentionPolicy::All => 0,
            MentionPolicy::ExcludeSingleAsciiLetters => 1,
        };
        let seed = turbo_analysis::Fnv1a::new().fold_u64(policy).fold_u64(self.geocoder.fingerprint()).finish();
        hash_location_types(seed, &self.location_types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{EntryParts, GeoPoint};
    use turbo_analysis::{Ix, Key, Provenance, ViewBuilder};

    #[test]
    fn projects_locations_across_views() {
        let mut doc = Document::new(">>> Paris");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(v0.text());
        b.replace(4, "");
        let (t1, a1) = b.finish();
        doc.add(&Key::<View>::named("clean"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();

        let entry = GazetteerEntry::new(EntryParts {
            source: "geonames".into(),
            record_id: "2988507".into(),
            name: "Paris".into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(48.85341, 2.3488).unwrap(),
            bounding_box: None,
            country_code: Some("FR".into()),
            containment: Vec::new(),
            population: 2_138_551,
            feature_class: Some("CITY".into()),
            attributes: Vec::new(),
        })
        .unwrap();
        let res = GeoResolution::new(Span::new(0, 5), entry, 0.95).unwrap();
        let mut locs = Locations::new(v1_id);
        locs.push(res);

        let projected = doc.project_layer(&locs, ViewId::SOURCE).unwrap();
        assert_eq!(projected.spans.view(), ViewId::SOURCE);
        assert_eq!(projected.spans.get(Ix::new(0)), Span::new(4, 9));
        assert_eq!(projected.get(Ix::new(0)).mention(), Span::new(4, 9));
    }
}
