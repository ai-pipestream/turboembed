//! Spatial cluster density (`GeoHashBinningScorer`) and geographic context
//! proximity scorers (`CountryProximityScorer`, `ProvinceProximityScorer`).

use turbo_analysis::Span;
use turbo_unicode::GeneralCategory;
use turbo_unicode::category::general_category;
use turbo_unicode::java::to_lowercase;
use turbo_unicode::normalize::{Form, normalize};

use crate::entry::{GeoPoint, RegionCode};
use crate::fold::Folder;
use crate::region::tables::{is_assigned_iso_country, lookup_english_country};

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
const MAX_GEOHASH_PRECISION: usize = 12;
const DEFAULT_GEOHASH_PRECISION: usize = 4;
const NEUTRAL_CONTEXT_SCORE: f64 = 0.5;
const EARTH_RADIUS_KM: f64 = 6_371.008_8;
const SPATIAL_KERNEL_SCALE_KM: f64 = 100.0;
const REGIONAL_INDICATOR_A: u32 = 0x1F1E6;
const REGIONAL_INDICATOR_Z: u32 = 0x1F1FF;

/// Encodes `point` into a standard base-32 Geohash string of length
/// `precision` (clamped to `1..=12`).
pub fn encode_geohash(point: GeoPoint, precision: usize) -> String {
    let prec = precision.clamp(1, MAX_GEOHASH_PRECISION);
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let lat = point.latitude();
    let lon = point.longitude();
    let mut is_lon = true;
    let mut bit = 0u8;
    let mut ch_idx = 0u8;
    let mut out = String::with_capacity(prec);

    while out.len() < prec {
        if is_lon {
            let mid = 0.5 * (lon_range.0 + lon_range.1);
            if lon >= mid {
                ch_idx = (ch_idx << 1) | 1;
                lon_range.0 = mid;
            } else {
                ch_idx <<= 1;
                lon_range.1 = mid;
            }
        } else {
            let mid = 0.5 * (lat_range.0 + lat_range.1);
            if lat >= mid {
                ch_idx = (ch_idx << 1) | 1;
                lat_range.0 = mid;
            } else {
                ch_idx <<= 1;
                lat_range.1 = mid;
            }
        }
        is_lon = !is_lon;
        bit += 1;
        if bit == 5 {
            out.push(GEOHASH_BASE32[ch_idx as usize] as char);
            bit = 0;
            ch_idx = 0;
        }
    }
    out
}

fn decode_geohash(hash: &str) -> Option<(f64, f64)> {
    if hash.is_empty() {
        return None;
    }
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;
    for b in hash.bytes() {
        let idx = GEOHASH_BASE32.iter().position(|&c| c == b.to_ascii_lowercase())? as u8;
        for shift in (0..5).rev() {
            let bit = (idx >> shift) & 1;
            if is_lon {
                let mid = 0.5 * (lon_range.0 + lon_range.1);
                if bit == 1 {
                    lon_range.0 = mid;
                } else {
                    lon_range.1 = mid;
                }
            } else {
                let mid = 0.5 * (lat_range.0 + lat_range.1);
                if bit == 1 {
                    lat_range.0 = mid;
                } else {
                    lat_range.1 = mid;
                }
            }
            is_lon = !is_lon;
        }
    }
    Some((0.5 * (lat_range.0 + lat_range.1), 0.5 * (lon_range.0 + lon_range.1)))
}

fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r_lat1 = lat1.to_radians();
    let r_lat2 = lat2.to_radians();
    let d_lat = (lat2 - lat1).to_radians();
    let d_lon = (lon2 - lon1).to_radians();
    let sin_lat = (0.5 * d_lat).sin();
    let sin_lon = (0.5 * d_lon).sin();
    let a = (sin_lat * sin_lat + r_lat1.cos() * r_lat2.cos() * sin_lon * sin_lon).min(1.0);
    2.0 * EARTH_RADIUS_KM * a.sqrt().asin()
}

/// Number of leading Geohash characters shared by `a` and `b`.
#[inline]
pub fn shared_geohash_prefix_len(a: &str, b: &str) -> usize {
    a.bytes().zip(b.bytes()).take_while(|(x, y)| x == y).count()
}

/// Scores candidate locations by spatial cluster density across Geohash prefix
/// bins and great-circle proximity to other mention candidates in the document.
#[derive(Clone, Copy, Debug)]
pub struct GeoHashBinningScorer {
    precision: usize,
}

impl Default for GeoHashBinningScorer {
    fn default() -> Self {
        GeoHashBinningScorer { precision: DEFAULT_GEOHASH_PRECISION }
    }
}

impl GeoHashBinningScorer {
    /// Creates a Geohash binning scorer with `precision` characters (`1..=12`).
    pub fn new(precision: usize) -> GeoHashBinningScorer {
        GeoHashBinningScorer { precision: precision.clamp(1, MAX_GEOHASH_PRECISION) }
    }

    pub fn precision(&self) -> usize {
        self.precision
    }

    /// Scores `candidate_hash` for mention `mention_idx` against the Geohash
    /// strings of all mentions' candidate sets (`all_hashes_by_mention`),
    /// returning a value in `[0.0, 1.0]`.
    pub fn score_in_context(
        &self,
        mention_idx: usize,
        candidate_hash: &str,
        all_hashes_by_mention: &[Vec<String>],
    ) -> f64 {
        let mut other_mentions = 0usize;
        let mut support_sum = 0.0f64;
        let denom = self.precision.max(1) as f64;
        let cand_coord = decode_geohash(candidate_hash);
        let own_hashes = all_hashes_by_mention.get(mention_idx).map(Vec::as_slice);

        for (j, other_hashes) in all_hashes_by_mention.iter().enumerate() {
            if j == mention_idx || other_hashes.is_empty() {
                continue;
            }
            if other_hashes.len() > 1 && own_hashes.is_some_and(|own| own == other_hashes.as_slice()) {
                continue;
            }
            other_mentions += 1;
            let best_support = other_hashes
                .iter()
                .map(|h| {
                    let prefix_support =
                        (shared_geohash_prefix_len(candidate_hash, h).min(self.precision) as f64) / denom;
                    let dist_support = if let (Some((lat1, lon1)), Some((lat2, lon2))) = (cand_coord, decode_geohash(h))
                    {
                        let km = haversine_km(lat1, lon1, lat2, lon2);
                        1.0 / (1.0 + (km / SPATIAL_KERNEL_SCALE_KM).powi(2))
                    } else {
                        0.0
                    };
                    prefix_support.max(dist_support)
                })
                .fold(0.0f64, f64::max);
            support_sum += best_support;
        }
        if other_mentions == 0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        (support_sum / other_mentions as f64).clamp(0.0, 1.0)
    }
}

/// A country signal anchored at a byte span in the document text.
#[derive(Clone, Copy, Debug)]
pub struct CountrySignal {
    pub span: Span,
    pub country: RegionCode,
    pub weight: f64,
}

#[inline]
pub(crate) fn spans_overlap(a: Span, b: Span) -> bool {
    a.start < b.end && b.start < a.end
}

#[inline]
fn span_distance_bytes(a: Span, b: Span) -> u32 {
    b.start.saturating_sub(a.end).max(a.start.saturating_sub(b.end))
}

fn normalize_country_token(raw: &str) -> Option<RegionCode> {
    let mut nfkc = String::new();
    normalize(Form::Nfkc, raw, &mut nfkc);
    let mut collapsed = String::with_capacity(nfkc.len());
    let mut prev_ws = false;
    for ch in nfkc.chars() {
        if ch.is_whitespace() {
            if !prev_ws {
                collapsed.push(' ');
                prev_ws = true;
            }
        } else {
            collapsed.push(if ch == '\u{2019}' { '\'' } else { ch });
            prev_ws = false;
        }
    }
    let mut lower = String::new();
    to_lowercase(&collapsed, &mut lower);
    let trimmed = lower.trim_matches(|c: char| !c.is_alphanumeric());
    if trimmed.is_empty() {
        return None;
    }
    if let Some(code) = lookup_english_country(trimmed) {
        return Some(code);
    }
    let mut nfd = String::new();
    normalize(Form::Nfd, trimmed, &mut nfd);
    let stripped: String = nfd.chars().filter(|&c| general_category(c) != GeneralCategory::Mn).collect();
    lookup_english_country(&stripped)
}

/// Scores candidate locations by proximity to country evidence in the document.
#[derive(Clone, Copy, Debug)]
pub struct CountryProximityScorer {
    decay_bytes: f64,
}

impl Default for CountryProximityScorer {
    fn default() -> Self {
        CountryProximityScorer { decay_bytes: 250.0 }
    }
}

impl CountryProximityScorer {
    /// Creates a country proximity scorer whose inverse-distance weight halves
    /// every `decay_bytes` UTF-8 bytes from the mention span.
    pub fn new(decay_bytes: f64) -> CountryProximityScorer {
        let d = if decay_bytes.is_finite() { decay_bytes.max(1.0) } else { 250.0 };
        CountryProximityScorer { decay_bytes: d }
    }

    pub fn decay_bytes(&self) -> f64 {
        self.decay_bytes
    }

    /// Extracts country signals from the document `text` (1-to-3 word phrases
    /// matching English country display names and regional-indicator flag emoji pairs).
    pub fn extract_text_signals(&self, text: &str) -> Vec<CountrySignal> {
        let mut signals = Vec::new();

        let mut chars = text.char_indices().peekable();
        while let Some((idx, first)) = chars.next() {
            let cp1 = first as u32;
            if !(REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp1) {
                continue;
            }
            if let Some(&(_, second)) = chars.peek() {
                let cp2 = second as u32;
                if (REGIONAL_INDICATOR_A..=REGIONAL_INDICATOR_Z).contains(&cp2) {
                    let (idx2, _) = chars.next().expect("peeked");
                    let bytes = [b'A' + (cp1 - REGIONAL_INDICATOR_A) as u8, b'A' + (cp2 - REGIONAL_INDICATOR_A) as u8];
                    if is_assigned_iso_country(bytes)
                        && let Some(country) = RegionCode::new(bytes)
                    {
                        signals.push(CountrySignal {
                            span: Span::new(idx as u32, (idx2 + second.len_utf8()) as u32),
                            country,
                            weight: 1.0,
                        });
                    }
                }
            }
        }

        let mut words: Vec<(usize, usize)> = Vec::new();
        let mut in_word = false;
        let mut word_start = 0usize;
        for (i, ch) in text.char_indices() {
            if ch.is_alphanumeric() || ch == '&' || ch == '\'' || ch == '\u{2019}' || ch == '-' {
                if !in_word {
                    in_word = true;
                    word_start = i;
                }
            } else if in_word {
                words.push((word_start, i));
                in_word = false;
            }
        }
        if in_word {
            words.push((word_start, text.len()));
        }

        let mut i = 0usize;
        while i < words.len() {
            let mut matched = false;
            for window in (1..=3.min(words.len() - i)).rev() {
                let start = words[i].0;
                let end = words[i + window - 1].1;
                if let Some(country) = normalize_country_token(&text[start..end]) {
                    signals.push(CountrySignal { span: Span::new(start as u32, end as u32), country, weight: 1.0 });
                    i += window;
                    matched = true;
                    break;
                }
            }
            if !matched {
                i += 1;
            }
        }
        signals
    }

    /// Scores `candidate_country` at `mention_span` against `signals` (skipping
    /// any signal that equals or overlaps `mention_span`), returning a value in `[0.0, 1.0]`.
    pub fn score(&self, mention_span: Span, candidate_country: Option<RegionCode>, signals: &[CountrySignal]) -> f64 {
        let Some(target) = candidate_country else {
            return NEUTRAL_CONTEXT_SCORE;
        };
        let mut matched_weight = 0.0f64;
        let mut total_weight = 0.0f64;
        for sig in signals {
            if sig.span == mention_span || spans_overlap(sig.span, mention_span) {
                continue;
            }
            let dist = span_distance_bytes(mention_span, sig.span) as f64;
            let w = sig.weight / (1.0 + dist / self.decay_bytes);
            total_weight += w;
            if sig.country == target {
                matched_weight += w;
            }
        }
        if total_weight <= 0.0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let ratio = matched_weight / total_weight;
        let strength = total_weight.min(1.0);
        (NEUTRAL_CONTEXT_SCORE + strength * (ratio - NEUTRAL_CONTEXT_SCORE)).clamp(0.0, 1.0)
    }
}

/// An administrative / province context signal anchored at a byte span in the
/// document text.
#[derive(Clone, Debug)]
pub struct ProvinceSignal {
    pub span: Span,
    pub folded_name: String,
    pub weight: f64,
}

/// Scores candidate locations by proximity of their administrative
/// `containment` hierarchy to administrative signals in the document text and
/// co-occurring mentions.
#[derive(Clone, Copy, Debug)]
pub struct ProvinceProximityScorer {
    decay_bytes: f64,
}

impl Default for ProvinceProximityScorer {
    fn default() -> Self {
        ProvinceProximityScorer { decay_bytes: 250.0 }
    }
}

impl ProvinceProximityScorer {
    /// Creates a province proximity scorer with distance decay `decay_bytes`.
    pub fn new(decay_bytes: f64) -> ProvinceProximityScorer {
        let d = if decay_bytes.is_finite() { decay_bytes.max(1.0) } else { 250.0 };
        ProvinceProximityScorer { decay_bytes: d }
    }

    pub fn decay_bytes(&self) -> f64 {
        self.decay_bytes
    }

    /// Scores a candidate's folded `containment` names at `mention_span`
    /// against `signals` and the document text `doc_text`, skipping any signal
    /// or text occurrence that overlaps `mention_span` and weighting direct
    /// text occurrences by inverse byte-distance decay.
    pub fn score(
        &self,
        mention_span: Span,
        candidate_containment_folded: &[String],
        signals: &[ProvinceSignal],
        doc_text: &str,
    ) -> f64 {
        if candidate_containment_folded.is_empty() {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let mut matched_weight = 0.0f64;
        let mut total_weight = 0.0f64;

        for sig in signals {
            if sig.span == mention_span || spans_overlap(sig.span, mention_span) || sig.folded_name.is_empty() {
                continue;
            }
            let dist = span_distance_bytes(mention_span, sig.span) as f64;
            let w = sig.weight / (1.0 + dist / self.decay_bytes);
            total_weight += w;
            if candidate_containment_folded.iter().any(|c| c == &sig.folded_name) {
                matched_weight += w;
            }
        }

        for match_span in find_folded_phrase_spans(doc_text, candidate_containment_folded) {
            if match_span == mention_span || spans_overlap(match_span, mention_span) {
                continue;
            }
            let dist = span_distance_bytes(mention_span, match_span) as f64;
            let w = 1.0 / (1.0 + dist / self.decay_bytes);
            matched_weight += w;
            total_weight += w;
        }

        if total_weight <= 0.0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        let ratio = matched_weight / total_weight;
        let strength = total_weight.min(1.0);
        (NEUTRAL_CONTEXT_SCORE + strength * (ratio - NEUTRAL_CONTEXT_SCORE)).clamp(0.0, 1.0)
    }

    /// Folds a list of raw containment strings using `folder`.
    pub fn fold_containment<'a>(folder: &mut Folder, raw: impl IntoIterator<Item = &'a str>) -> Vec<String> {
        raw.into_iter().map(|s| folder.fold(s)).filter(|s| !s.is_empty()).collect()
    }
}

fn find_folded_phrase_spans(doc_text: &str, phrases: &[String]) -> Vec<Span> {
    let mut folder = Folder::new();
    let mut words: Vec<(Span, String)> = Vec::new();
    let mut in_word = false;
    let mut start = 0usize;
    for (i, ch) in doc_text.char_indices() {
        if ch.is_alphanumeric() {
            if !in_word {
                in_word = true;
                start = i;
            }
        } else if in_word {
            let folded = folder.fold(&doc_text[start..i]);
            if !folded.is_empty() {
                words.push((Span::new(start as u32, i as u32), folded));
            }
            in_word = false;
        }
    }
    if in_word {
        let folded = folder.fold(&doc_text[start..]);
        if !folded.is_empty() {
            words.push((Span::new(start as u32, doc_text.len() as u32), folded));
        }
    }

    let mut out = Vec::new();
    let mut buf = String::new();
    for phrase in phrases {
        if phrase.is_empty() {
            continue;
        }
        let n_words = phrase.split(' ').count();
        if n_words == 0 || n_words > words.len() {
            continue;
        }
        for i in 0..=words.len() - n_words {
            buf.clear();
            for (k, (_, w)) in words[i..i + n_words].iter().enumerate() {
                if k > 0 {
                    buf.push(' ');
                }
                buf.push_str(w);
            }
            if &buf == phrase {
                out.push(Span::new(words[i].0.start, words[i + n_words - 1].0.end));
            }
        }
    }
    out
}
