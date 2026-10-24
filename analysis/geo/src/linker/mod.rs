//! Geographic entity linker (`opennlp.addons.geoentitylinker` parity and
//! typed graph linking): combines fuzzy string matching, Geohash spatial
//! binning, country context proximity, province/containment proximity, and
//! population priors to rank [`Locations`] candidates and link
//! `Ix<Entities>` to `Ix<Locations>` via [`Edges<Entities, Locations>`].

mod fuzzy;
mod spatial;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::sync::Arc;

use turbo_analysis::{Annotator, Edge, Edges, Error, Ix, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::Entities;
use turbo_unicode::java::to_lowercase;

use crate::entry::GazetteerEntry;
use crate::fold::Folder;
use crate::gazetteer::Gazetteer;
use crate::geocode::{
    DEFAULT_LOCATION_TYPE, GeoResolution, GeocodeError, Geocoder, Locations, MentionPolicy, by_prior_entry,
    hash_location_types, normalize_location_types,
};

pub use self::fuzzy::{FuzzyMetric, FuzzyStringScorer, jaro_similarity, jaro_winkler, normalized_levenshtein};
pub use self::spatial::{
    CountryProximityScorer, CountrySignal, GeoHashBinningScorer, ProvinceProximityScorer, ProvinceSignal,
    encode_geohash, shared_geohash_prefix_len,
};

/// Directed edges connecting entity mentions (`Ix<Entities>`) to their
/// ranked candidate resolutions (`Ix<Locations>`).
pub type EntityLocationLinks = Edges<Entities, Locations>;

/// Edge tag label for the highest-scoring (primary) resolution of an entity.
pub const PRIMARY_LINK_LABEL: &str = "resolved";
/// Edge tag label for a secondary candidate resolution of an entity.
pub const CANDIDATE_LINK_LABEL: &str = "candidate";

const DEFAULT_MAX_CANDIDATES: usize = 5;
const DEFAULT_RETRIEVAL_LIMIT: usize = 25;

/// Relative weights for the five normalized `[0.0, 1.0]` component scorers
/// combined by [`GeoEntityLinker`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkerWeights {
    pub fuzzy: f64,
    pub geohash: f64,
    pub country: f64,
    pub province: f64,
    pub population: f64,
}

impl Default for LinkerWeights {
    fn default() -> Self {
        LinkerWeights { fuzzy: 0.30, geohash: 0.20, country: 0.20, province: 0.15, population: 0.15 }
    }
}

impl LinkerWeights {
    /// Creates checked weights: every weight must be finite and non-negative,
    /// and at least one weight must be strictly positive.
    pub fn new(
        fuzzy: f64,
        geohash: f64,
        country: f64,
        province: f64,
        population: f64,
    ) -> Result<LinkerWeights, GeocodeError> {
        let w = LinkerWeights { fuzzy, geohash, country, province, population };
        w.validate()?;
        Ok(w)
    }

    /// Validates that every weight is finite and non-negative and that at
    /// least one weight is strictly positive.
    pub fn validate(&self) -> Result<(), GeocodeError> {
        let all = [self.fuzzy, self.geohash, self.country, self.province, self.population];
        for &w in &all {
            if !w.is_finite() || w < 0.0 {
                return Err(GeocodeError::InvalidConfidence(w));
            }
        }
        let max_w = all.into_iter().fold(0.0f64, f64::max);
        if max_w <= 0.0 {
            return Err(GeocodeError::InvalidConfidence(0.0));
        }
        Ok(())
    }

    /// Combines component scores in `[0.0, 1.0]` into a normalized score in `[0.0, 1.0]`.
    ///
    /// Scales weights by their maximum before summing so extreme finite
    /// weights (such as `f64::MAX`) never overflow to `INFINITY` / `NaN`.
    #[inline]
    pub fn combine(&self, fuzzy: f64, geohash: f64, country: f64, province: f64, population: f64) -> f64 {
        if self.validate().is_err() {
            return 0.0;
        }
        let max_w =
            [self.fuzzy, self.geohash, self.country, self.province, self.population].into_iter().fold(0.0f64, f64::max);
        let wf = self.fuzzy / max_w;
        let wg = self.geohash / max_w;
        let wc = self.country / max_w;
        let wp = self.province / max_w;
        let wpop = self.population / max_w;
        let sum = wf + wg + wc + wp + wpop;
        let num = wf * fuzzy.clamp(0.0, 1.0)
            + wg * geohash.clamp(0.0, 1.0)
            + wc * country.clamp(0.0, 1.0)
            + wp * province.clamp(0.0, 1.0)
            + wpop * population.clamp(0.0, 1.0);
        (num / sum).clamp(0.0, 1.0)
    }
}

/// One scored candidate location produced by [`GeoEntityLinker`] for a mention.
#[derive(Clone, Debug, PartialEq)]
pub struct LinkedCandidate {
    pub entry: GazetteerEntry,
    pub score: f64,
    pub fuzzy_score: f64,
    pub geohash_score: f64,
    pub country_score: f64,
    pub province_score: f64,
    pub population_score: f64,
}

/// Multi-scorer geographic entity linker (`GeoEntityLinker`).
///
/// Combines [`FuzzyStringScorer`], [`GeoHashBinningScorer`],
/// [`CountryProximityScorer`], [`ProvinceProximityScorer`], and log-population
/// normalization to rank candidate [`GazetteerEntry`] records per mention and
/// emit both [`Locations`] and [`Edges<Entities, Locations>`].
pub struct GeoEntityLinker<G> {
    gazetteer: G,
    fuzzy: FuzzyStringScorer,
    geohash: GeoHashBinningScorer,
    country: CountryProximityScorer,
    province: ProvinceProximityScorer,
    weights: LinkerWeights,
    max_candidates_per_mention: usize,
    location_types: HashSet<String>,
    mention_policy: MentionPolicy,
}

impl<G: Gazetteer> GeoEntityLinker<G> {
    /// Creates a linker wrapping `gazetteer` with default scorers and weights.
    pub fn new(gazetteer: G) -> GeoEntityLinker<G> {
        let mut location_types = HashSet::new();
        location_types.insert(DEFAULT_LOCATION_TYPE.to_string());
        GeoEntityLinker {
            gazetteer,
            fuzzy: FuzzyStringScorer::default(),
            geohash: GeoHashBinningScorer::default(),
            country: CountryProximityScorer::default(),
            province: ProvinceProximityScorer::default(),
            weights: LinkerWeights::default(),
            max_candidates_per_mention: DEFAULT_MAX_CANDIDATES,
            location_types,
            mention_policy: MentionPolicy::All,
        }
    }

    pub fn gazetteer(&self) -> &G {
        &self.gazetteer
    }

    pub fn with_fuzzy_scorer(mut self, fuzzy: FuzzyStringScorer) -> Self {
        self.fuzzy = fuzzy;
        self
    }

    pub fn with_geohash_scorer(mut self, geohash: GeoHashBinningScorer) -> Self {
        self.geohash = geohash;
        self
    }

    pub fn with_country_scorer(mut self, country: CountryProximityScorer) -> Self {
        self.country = country;
        self
    }

    pub fn with_province_scorer(mut self, province: ProvinceProximityScorer) -> Self {
        self.province = province;
        self
    }

    pub fn with_weights(mut self, weights: LinkerWeights) -> Self {
        self.weights = weights;
        self
    }

    pub fn with_max_candidates(mut self, max_candidates: usize) -> Self {
        self.max_candidates_per_mention = max_candidates.max(1);
        self
    }

    pub fn with_mention_policy(mut self, policy: MentionPolicy) -> Self {
        self.mention_policy = policy;
        self
    }

    pub fn with_location_types<'a>(
        mut self,
        location_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, GeocodeError> {
        self.location_types = normalize_location_types(location_types)?;
        Ok(self)
    }

    /// Scores and ranks candidates for each mention span in `mentions` over `text`.
    pub fn link_mentions(&self, text: &str, mentions: &[Span]) -> Result<Vec<Vec<LinkedCandidate>>, GeocodeError> {
        self.weights.validate()?;
        let text_len = text.len();
        for &span in mentions {
            let s = span.start as usize;
            let e = span.end as usize;
            if s > e || e > text_len || !text.is_char_boundary(s) || !text.is_char_boundary(e) {
                return Err(GeocodeError::MentionOutOfBounds { span, text_len });
            }
        }
        if mentions.is_empty() {
            return Ok(Vec::new());
        }

        let retrieval_limit = self.max_candidates_per_mention.max(DEFAULT_RETRIEVAL_LIMIT);
        let min_sim = self.fuzzy.min_similarity();
        let mut raw_by_mention: Vec<Vec<(GazetteerEntry, f64)>> = Vec::with_capacity(mentions.len());
        let mut hashes_by_mention: Vec<Vec<String>> = Vec::with_capacity(mentions.len());
        let text_country_signals = self.country.extract_text_signals(text);
        let mut cand_country_signals: Vec<(usize, CountrySignal)> = Vec::new();
        let mut cand_province_signals: Vec<(usize, ProvinceSignal)> = Vec::new();
        let mut folder = Folder::new();
        let mut mention_keys: Vec<String> = Vec::with_capacity(mentions.len());

        for (m_idx, &span) in mentions.iter().enumerate() {
            let surface = &text[span.range()];
            mention_keys.push(folder.fold(surface));
            let hits = self.gazetteer.lookup_fuzzy(surface, &self.fuzzy, min_sim, retrieval_limit);
            let mut entries = Vec::with_capacity(hits.len());
            let mut hashes = Vec::with_capacity(hits.len());
            for (idx, (entry_view, sim)) in hits.into_iter().enumerate() {
                let owned = entry_view.to_entry();
                hashes.push(encode_geohash(owned.location(), self.geohash.precision()));
                let rank_weight = 1.0 / (1.0 + idx as f64);
                if let Some(code) = owned.region_code() {
                    cand_country_signals
                        .push((m_idx, CountrySignal { span, country: code, weight: 0.5 * sim * rank_weight }));
                }
                for admin in owned.containment() {
                    let folded_name = folder.fold(admin);
                    if !folded_name.is_empty() {
                        cand_province_signals
                            .push((m_idx, ProvinceSignal { span, folded_name, weight: 0.5 * sim * rank_weight }));
                    }
                }
                entries.push((owned, sim));
            }
            raw_by_mention.push(entries);
            hashes_by_mention.push(hashes);
        }

        let mut out = Vec::with_capacity(mentions.len());
        for (m_idx, (&span, candidates)) in mentions.iter().zip(raw_by_mention).enumerate() {
            if candidates.is_empty() {
                out.push(Vec::new());
                continue;
            }
            let max_pop_ln =
                candidates.iter().map(|(e, _)| ((e.population().max(0) as f64) + 1.0).ln()).fold(0.0f64, f64::max);

            let own_key = &mention_keys[m_idx];
            let filtered_hashes: Vec<Vec<String>> = hashes_by_mention
                .iter()
                .enumerate()
                .map(|(j, h)| if j == m_idx || &mention_keys[j] != own_key { h.clone() } else { Vec::new() })
                .collect();
            let mut country_signals = text_country_signals.clone();
            country_signals
                .extend(cand_country_signals.iter().filter(|(j, _)| &mention_keys[*j] != own_key).map(|(_, s)| *s));
            let province_signals: Vec<ProvinceSignal> = cand_province_signals
                .iter()
                .filter(|(j, _)| &mention_keys[*j] != own_key)
                .map(|(_, s)| s.clone())
                .collect();

            let mut scored: Vec<LinkedCandidate> = Vec::with_capacity(candidates.len());
            for (c_idx, (entry, fuzzy_score)) in candidates.into_iter().enumerate() {
                let geohash_score =
                    self.geohash.score_in_context(m_idx, &hashes_by_mention[m_idx][c_idx], &filtered_hashes);
                let country_score = self.country.score(span, entry.region_code(), &country_signals);
                let folded_containment = ProvinceProximityScorer::fold_containment(
                    &mut folder,
                    entry.containment().iter().map(String::as_str),
                );
                let province_score = self.province.score(span, &folded_containment, &province_signals, text);
                let population_score = if max_pop_ln <= 0.0 {
                    0.5
                } else {
                    (((entry.population().max(0) as f64) + 1.0).ln() / max_pop_ln).clamp(0.0, 1.0)
                };
                let score =
                    self.weights.combine(fuzzy_score, geohash_score, country_score, province_score, population_score);
                scored.push(LinkedCandidate {
                    entry,
                    score,
                    fuzzy_score,
                    geohash_score,
                    country_score,
                    province_score,
                    population_score,
                });
            }
            scored.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| by_prior_entry(&a.entry, &b.entry)));
            scored.truncate(self.max_candidates_per_mention);
            out.push(scored);
        }
        Ok(out)
    }
}

impl<G: Gazetteer> Geocoder for GeoEntityLinker<G> {
    fn resolve(&self, text: &str, location_mentions: &[Span]) -> Result<Vec<GeoResolution>, GeocodeError> {
        let linked = self.link_mentions(text, location_mentions)?;
        let mut resolutions = Vec::with_capacity(location_mentions.len());
        for (&span, mut candidates) in location_mentions.iter().zip(linked) {
            if !span.is_empty()
                && let Some(best) = candidates.drain(..).next()
            {
                resolutions.push(GeoResolution::new(span, best.entry, best.score)?);
            }
        }
        Ok(resolutions)
    }

    fn fingerprint(&self) -> u64 {
        const TAG: u64 = 0x4745_4f5f_4c49_4e4b;
        self.fingerprint_with(TAG)
    }
}

impl<G: Gazetteer> GeoEntityLinker<G> {
    fn fingerprint_with(&self, tag: u64) -> u64 {
        let metric_disc = match self.fuzzy.metric() {
            FuzzyMetric::Combined => 0u64,
            FuzzyMetric::NormalizedLevenshtein => 1u64,
            FuzzyMetric::JaroWinkler => 2u64,
        };
        let policy_disc = match self.mention_policy {
            MentionPolicy::All => 0u64,
            MentionPolicy::ExcludeSingleAsciiLetters => 1u64,
        };
        let mut h = turbo_analysis::Fnv1a::new();
        h.fold_u64(tag);
        for bits in [
            self.gazetteer.fingerprint(),
            self.weights.fuzzy.to_bits(),
            self.weights.geohash.to_bits(),
            self.weights.country.to_bits(),
            self.weights.province.to_bits(),
            self.weights.population.to_bits(),
            metric_disc,
            self.fuzzy.min_similarity().to_bits(),
            self.geohash.precision() as u64,
            self.country.decay_bytes().to_bits(),
            self.province.decay_bytes().to_bits(),
            self.max_candidates_per_mention as u64,
            policy_disc,
        ] {
            h.fold_u64(bits);
        }
        hash_location_types(h.finish(), &self.location_types)
    }
}

impl<G: Gazetteer + Send + Sync + 'static> Annotator for GeoEntityLinker<G> {
    const ID: &'static str = "turbo-geo:entity-linker";
    type Reads = (View, Entities);
    type Writes = (Locations, EntityLocationLinks);

    fn annotate(&self, (view, entities): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        use turbo_analysis::Layer as _;
        self.weights.validate().map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        if entities.spans.view() != view.id() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "entities are in a different view".into() });
        }
        entities.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        let mut admitted_entities: Vec<(Ix<Entities>, Span)> = Vec::new();
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
                    admitted_entities.push((ix, span));
                }
            }
        }

        let kinds: Arc<TagSet> = TagSet::new("turbo-geo:entity-link", &[PRIMARY_LINK_LABEL, CANDIDATE_LINK_LABEL]);
        let mut locations = Locations::new(view.id());
        if admitted_entities.is_empty() {
            return Ok((locations, Edges::new(kinds, [])));
        }

        let spans: Vec<Span> = admitted_entities.iter().map(|&(_, s)| s).collect();
        let linked = self
            .link_mentions(view.text(), &spans)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;

        let mut edges: Vec<Edge<Entities, Locations>> = Vec::new();
        for ((ent_ix, span), candidates) in admitted_entities.into_iter().zip(linked) {
            for (rank, cand) in candidates.into_iter().enumerate() {
                let res = GeoResolution::new(span, cand.entry, cand.score)
                    .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
                let loc_ix = locations.push(res);
                let tag = if rank == 0 { Tag(0) } else { Tag(1) };
                edges.push(Edge::new(ent_ix, loc_ix, tag, cand.score));
            }
        }
        locations.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
        Ok((locations, Edges::new(kinds, edges)))
    }

    fn config_hash(&self) -> u64 {
        self.fingerprint_with(0)
    }
}
