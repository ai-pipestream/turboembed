//! Geohash encoding and decoding and the spatial cluster density scorer
//! (`GeoHashBinningScorer`).

use crate::entry::GeoPoint;

const GEOHASH_BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
const MAX_GEOHASH_PRECISION: usize = 12;
const DEFAULT_GEOHASH_PRECISION: usize = 4;
const NEUTRAL_CONTEXT_SCORE: f64 = 0.5;
const EARTH_RADIUS_KM: f64 = 6_371.008_8;
const SPATIAL_KERNEL_SCALE_KM: f64 = 100.0;
const NOT_BASE32: u8 = u8::MAX;

/// The base-32 digit of every byte, either case, or [`NOT_BASE32`].
static GEOHASH_DIGITS: [u8; 256] = {
    let mut table = [NOT_BASE32; 256];
    let mut i = 0;
    while i < GEOHASH_BASE32.len() {
        let c = GEOHASH_BASE32[i];
        table[c as usize] = i as u8;
        table[c.to_ascii_uppercase() as usize] = i as u8;
        i += 1;
    }
    table
};

#[cfg(test)]
thread_local! {
    /// How many times [`decode_geohash`] ran on this thread.
    pub(super) static DECODES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

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

/// The centre `(latitude, longitude)` of `hash`'s cell, or `None` for an
/// empty hash or one with a byte outside the base-32 alphabet (either case).
pub(super) fn decode_geohash(hash: &str) -> Option<(f64, f64)> {
    #[cfg(test)]
    DECODES.with(|n| n.set(n.get() + 1));
    if hash.is_empty() {
        return None;
    }
    let mut lat_range = (-90.0f64, 90.0f64);
    let mut lon_range = (-180.0f64, 180.0f64);
    let mut is_lon = true;
    for b in hash.bytes() {
        let idx = GEOHASH_DIGITS[b as usize];
        if idx == NOT_BASE32 {
            return None;
        }
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
            let others = other_hashes.iter().map(|h| (h.as_str(), decode_geohash(h)));
            support_sum += self.best_support(candidate_hash, cand_coord, others);
        }
        self.mean_support(other_mentions, support_sum)
    }

    /// The best support one other mention's candidates (each a hash and its
    /// decoded centre) give `candidate_hash`, whose centre is `cand_coord`.
    pub(super) fn best_support<'h>(
        &self,
        candidate_hash: &str,
        cand_coord: Option<(f64, f64)>,
        others: impl Iterator<Item = (&'h str, Option<(f64, f64)>)>,
    ) -> f64 {
        let denom = self.precision.max(1) as f64;
        others
            .map(|(h, coord)| {
                let prefix_support = (shared_geohash_prefix_len(candidate_hash, h).min(self.precision) as f64) / denom;
                let dist_support = if let (Some((lat1, lon1)), Some((lat2, lon2))) = (cand_coord, coord) {
                    let km = haversine_km(lat1, lon1, lat2, lon2);
                    1.0 / (1.0 + (km / SPATIAL_KERNEL_SCALE_KM).powi(2))
                } else {
                    0.0
                };
                prefix_support.max(dist_support)
            })
            .fold(0.0f64, f64::max)
    }

    /// The score from the summed best supports of `other_mentions` mentions.
    pub(super) fn mean_support(&self, other_mentions: usize, support_sum: f64) -> f64 {
        if other_mentions == 0 {
            return NEUTRAL_CONTEXT_SCORE;
        }
        (support_sum / other_mentions as f64).clamp(0.0, 1.0)
    }
}
