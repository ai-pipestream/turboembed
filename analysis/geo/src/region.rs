//! Document-level region ballot annotator (`opennlp.tools.geo.DocumentRegionAnnotator`).
//!
//! Reads `(View, Entities, Locations)` and writes `(Regions,)`, combining:
//! 1. The first [`Locations`] resolution per location-typed entity mention,
//!    weighted by its resolution confidence when its entry carries a country
//!    code.
//! 2. English country display-name matching (`0.95` weight) for location-typed
//!    entity mentions that the geocoder left unresolved or resolved to a
//!    country-less entry.
//! 3. Country flag emoji pairs (`U+1F1E6..=U+1F1FF` decoding to an assigned
//!    ISO 3166-1 alpha-2 country code, scanned left-to-right non-overlapping)
//!    across the document text (`0.95` weight).

pub(crate) mod tables;
#[cfg(test)]
mod tests;

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{Entities, MarginError, RegionCode, RegionVote, Regions};
use turbo_jvm::hash_map::{JavaHash, JavaHashMap, string_hash};
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::java::to_lowercase;
use turbo_unicode::normalize::{Form, normalize};

use self::tables::{is_assigned_iso_country, lookup_english_country};
use crate::geocode::{DEFAULT_LOCATION_TYPE, GeocodeError, Locations, hash_location_types, normalize_location_types};

const COUNTRY_NAME_CONFIDENCE: f64 = 0.95;
const FLAG_EMOJI_CONFIDENCE: f64 = 0.95;
const REGIONAL_INDICATOR_A: u32 = 0x1F1E6;
const REGIONAL_INDICATOR_Z: u32 = 0x1F1FF;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// Reusable two-buffer scratch space for normalizing and accent-stripping
/// candidate country surface strings without per-entity heap allocations.
#[derive(Default)]
struct CountryNameScratch {
    buf_a: String,
    buf_b: String,
}

impl CountryNameScratch {
    fn country_code_for_name(&mut self, mention: &str) -> Option<RegionCode> {
        country_code_for_name(mention, &mut self.buf_a, &mut self.buf_b)
    }
}

/// The one country-name normalization both region votes and
/// spatial scoring use: NFKC, curly apostrophes read as `'`,
/// whitespace collapsed to one space, lowercased, trimmed of
/// whitespace at the ends (table keys keep meaningful
/// punctuation, as in `myanmar (burma)`); when that names no country,
/// the same with non-spacing marks stripped.
pub(crate) fn country_code_for_name(raw: &str, buf_a: &mut String, buf_b: &mut String) -> Option<RegionCode> {
    buf_a.clear();
    normalize(Form::Nfkc, raw, buf_a);
    buf_b.clear();
    let mut prev_ws = false;
    for ch in buf_a.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                buf_b.push(' ');
                prev_ws = true;
            }
        } else {
            buf_b.push(if ch == '\u{2019}' { '\'' } else { ch });
            prev_ws = false;
        }
    }
    buf_a.clear();
    to_lowercase(buf_b, buf_a);
    let normalized = buf_a.trim_matches(|c: char| c.is_whitespace());
    if normalized.is_empty() {
        return None;
    }
    if let Some(code) = lookup_english_country(normalized) {
        return Some(code);
    }
    buf_b.clear();
    normalize(Form::Nfd, normalized, buf_b);
    buf_a.clear();
    buf_a.extend(buf_b.chars().filter(|&c| general_category(c) != GeneralCategory::Mn));
    lookup_english_country(buf_a)
}

fn is_regional_indicator(c: char) -> bool {
    let cp = c as u32;
    (REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RegionKey(RegionCode);

impl JavaHash for RegionKey {
    #[inline]
    fn java_hash(&self) -> i32 {
        string_hash(self.0.as_str())
    }

    fn tree_order(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.cmp(&other.0))
    }
}

fn add_vote(weights: &mut JavaHashMap<RegionKey, f64>, country: RegionCode, weight: f64) {
    weights.merge(RegionKey(country), weight, |sum, w| *sum += w);
}

fn add_flag_emoji_votes(text: &str, weights: &mut JavaHashMap<RegionKey, f64>) {
    let mut chars = text.chars().peekable();
    while let Some(first) = chars.next() {
        if !is_regional_indicator(first) {
            continue;
        }
        let Some(&second) = chars.peek() else {
            break;
        };
        if !is_regional_indicator(second) {
            continue;
        }
        let _ = chars.next();
        let bytes =
            [b'A' + (first as u32 - REGIONAL_INDICATOR_A) as u8, b'A' + (second as u32 - REGIONAL_INDICATOR_A) as u8];
        if is_assigned_iso_country(bytes)
            && let Some(code) = RegionCode::new(bytes)
        {
            add_vote(weights, code, FLAG_EMOJI_CONFIDENCE);
        }
    }
}

/// Aggregates resolved location mentions, English country-name mentions and
/// country flag emojis into the document-scoped `turbo:regions` ballot
/// (`opennlp.tools.geo.DocumentRegionAnnotator`).
pub struct DocumentRegionAnnotator {
    location_types: HashSet<String>,
}

impl Default for DocumentRegionAnnotator {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentRegionAnnotator {
    /// Creates a region annotator that considers entities typed `"location"`
    /// (case-insensitively).
    pub fn new() -> DocumentRegionAnnotator {
        let mut location_types = HashSet::new();
        location_types.insert(DEFAULT_LOCATION_TYPE.to_string());
        DocumentRegionAnnotator { location_types }
    }

    /// Creates a region annotator that considers entities whose type label
    /// matches any element of `location_types` case-insensitively (`Locale.ROOT`).
    pub fn with_location_types<'a>(
        location_types: impl IntoIterator<Item = &'a str>,
    ) -> Result<DocumentRegionAnnotator, GeocodeError> {
        Ok(DocumentRegionAnnotator { location_types: normalize_location_types(location_types)? })
    }

    /// Returns the winning vote when the top row of `ballot` leads the runner-up
    /// (or `0.0` when `ballot` has one row) by at least `min_margin`
    /// (`DocumentRegionAnnotator.winner`).
    pub fn winner(ballot: &Regions, min_margin: f64) -> Result<Option<RegionVote>, MarginError> {
        ballot.winner(min_margin)
    }
}

impl Annotator for DocumentRegionAnnotator {
    const ID: &'static str = "turbo-geo:document-region";
    type Reads = (View, Entities, Locations);
    type Writes = (Regions,);

    fn annotate(&self, (view, entities, locations): Refs<'_, Self>) -> Result<(Regions,), Error> {
        if entities.spans.view() != view.id() || locations.spans.view() != view.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: "entities or locations are in a different view".into(),
            });
        }

        let mut first_by_span: HashMap<Span, usize> = HashMap::new();
        for (ix, span) in locations.spans.iter() {
            first_by_span.entry(span).or_insert(ix.get());
        }

        // Keyed by JavaHashMap<RegionKey, f64> so weights.merge(...) and the
        // entrySet() iteration that sums `total` follow Java's HashMap<String, Double>
        // order bit for bit.
        let mut weights: JavaHashMap<RegionKey, f64> = JavaHashMap::new();
        let mut lower_buf = String::new();
        let mut name_scratch = CountryNameScratch::default();
        let text = view.text();

        for (ix, span) in entities.spans.iter() {
            lower_buf.clear();
            to_lowercase(entities.kinds.label(entities.kind[ix.get()]), &mut lower_buf);
            if !self.location_types.contains(&lower_buf) {
                continue;
            }
            if let Some(&loc_idx) = first_by_span.get(&span)
                && let Some(country) = locations.entry[loc_idx].region_code()
            {
                add_vote(&mut weights, country, locations.confidence[loc_idx]);
                continue;
            }
            let surface = text.get(span.range()).ok_or_else(|| Error::Annotator {
                annotator: Self::ID,
                detail: format!("mention span {span:?} is outside the text or off a UTF-8 boundary"),
            })?;
            if let Some(country) = name_scratch.country_code_for_name(surface) {
                add_vote(&mut weights, country, COUNTRY_NAME_CONFIDENCE);
            }
        }

        add_flag_emoji_votes(text, &mut weights);

        let mut total = 0.0;
        let mut positive: Vec<(RegionCode, f64)> = Vec::with_capacity(weights.len());
        for (&RegionKey(country), &w) in weights.iter() {
            if w > 0.0 {
                total += w;
                positive.push((country, w));
            }
        }
        let mut ballot = Regions::new();
        if total <= 0.0 {
            return Ok((ballot,));
        }

        let mut rows: Vec<RegionVote> =
            positive.into_iter().filter_map(|(country, w)| RegionVote::new(country, w / total).ok()).collect();
        rows.sort_by(|a, b| b.share.total_cmp(&a.share).then_with(|| a.country_code.cmp(&b.country_code)));
        for row in rows {
            ballot.push(row);
        }
        Ok((ballot,))
    }

    fn config_hash(&self) -> u64 {
        hash_location_types(FNV_OFFSET_BASIS, &self.location_types)
    }
}

#[cfg(test)]
mod country_key_tests {
    #[test]
    fn whitespace_and_case_variants_resolve_the_same() {
        let (mut a, mut b) = (String::new(), String::new());
        let us = super::country_code_for_name("United States", &mut a, &mut b);
        assert!(us.is_some());
        assert_eq!(super::country_code_for_name("United\nStates", &mut a, &mut b), us);
        let ci = super::country_code_for_name("C\u{00F4}te d\u{2019}Ivoire", &mut a, &mut b);
        assert!(ci.is_some());
        assert_eq!(super::country_code_for_name("C\u{00D4}TE D'IVOIRE", &mut a, &mut b), ci);
    }

    #[test]
    fn parenthesized_country_names_keep_their_closing_paren() {
        let (mut a, mut b) = (String::new(), String::new());
        let mm = super::country_code_for_name("Myanmar (Burma)", &mut a, &mut b);
        assert_eq!(mm.map(|c| c.as_str().to_string()), Some("MM".to_string()));
    }
}
