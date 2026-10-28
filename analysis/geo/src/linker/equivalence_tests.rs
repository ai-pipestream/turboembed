//! `link_mentions` against a private copy of the linker as it was before the
//! per-document context (#1498): every score bit, order and tie-break the
//! same on random documents with many mentions sharing Geohashes, plus a
//! count of Geohash decodes.

use std::collections::HashSet;

use turbo_analysis::Span;

use super::geohash::DECODES;
use super::*;
use crate::entry::{EntryParts, FEATURE_CLASS_CITY, GeoPoint, RegionCode};
use crate::in_memory::InMemoryGazetteer;

/// The decode the linker used before: an alphabet scan per byte.
fn old_decode(hash: &str) -> Option<(f64, f64)> {
    const BASE32: &[u8; 32] = b"0123456789bcdefghjkmnpqrstuvwxyz";
    if hash.is_empty() {
        return None;
    }
    let (mut lat, mut lon, mut is_lon) = ((-90.0f64, 90.0f64), (-180.0f64, 180.0f64), true);
    for b in hash.bytes() {
        let idx = BASE32.iter().position(|&c| c == b.to_ascii_lowercase())? as u8;
        for shift in (0..5).rev() {
            let range = if is_lon { &mut lon } else { &mut lat };
            let mid = 0.5 * (range.0 + range.1);
            if (idx >> shift) & 1 == 1 {
                range.0 = mid;
            } else {
                range.1 = mid;
            }
            is_lon = !is_lon;
        }
    }
    Some((0.5 * (lat.0 + lat.1), 0.5 * (lon.0 + lon.1)))
}

fn old_haversine(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let (r_lat1, r_lat2) = (lat1.to_radians(), lat2.to_radians());
    let sin_lat = (0.5 * (lat2 - lat1).to_radians()).sin();
    let sin_lon = (0.5 * (lon2 - lon1).to_radians()).sin();
    let a = (sin_lat * sin_lat + r_lat1.cos() * r_lat2.cos() * sin_lon * sin_lon).min(1.0);
    2.0 * 6_371.008_8 * a.sqrt().asin()
}

fn old_geohash_score(precision: usize, m: usize, cand: &str, all: &[Vec<String>]) -> f64 {
    let (mut other_mentions, mut support_sum) = (0usize, 0.0f64);
    let denom = precision.max(1) as f64;
    let cand_coord = old_decode(cand);
    for (j, other) in all.iter().enumerate() {
        if j == m || other.is_empty() || (other.len() > 1 && all[m] == *other) {
            continue;
        }
        other_mentions += 1;
        let best = other
            .iter()
            .map(|h| {
                let prefix = (shared_geohash_prefix_len(cand, h).min(precision) as f64) / denom;
                let dist = if let (Some((a, b)), Some((c, d))) = (cand_coord, old_decode(h)) {
                    1.0 / (1.0 + (old_haversine(a, b, c, d) / 100.0).powi(2))
                } else {
                    0.0
                };
                prefix.max(dist)
            })
            .fold(0.0f64, f64::max);
        support_sum += best;
    }
    if other_mentions == 0 { 0.5 } else { (support_sum / other_mentions as f64).clamp(0.0, 1.0) }
}

fn old_phrase_spans(text: &str, phrases: &[String]) -> Vec<Span> {
    let mut folder = Folder::new();
    let mut words: Vec<(Span, String)> = Vec::new();
    let mut start = None;
    for (i, ch) in text.char_indices().chain([(text.len(), ' ')]) {
        match (ch.is_alphanumeric() && i < text.len(), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let folded = folder.fold(&text[s..i]);
                if !folded.is_empty() {
                    words.push((Span::new(s as u32, i as u32), folded));
                }
                start = None;
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    for phrase in phrases.iter().filter(|p| !p.is_empty()) {
        let n = phrase.split(' ').count();
        if n > words.len() {
            continue;
        }
        for i in 0..=words.len() - n {
            let joined: Vec<&str> = words[i..i + n].iter().map(|(_, w)| w.as_str()).collect();
            if joined.join(" ") == *phrase {
                out.push(Span::new(words[i].0.start, words[i + n - 1].0.end));
            }
        }
    }
    out
}

fn decay_weight(weight: f64, decay: f64, mention: Span, at: Span) -> Option<f64> {
    if at == mention || spatial::spans_overlap(at, mention) {
        return None;
    }
    let dist = at.start.saturating_sub(mention.end).max(mention.start.saturating_sub(at.end)) as f64;
    Some(weight / (1.0 + dist / decay))
}

fn blend(matched: f64, total: f64) -> f64 {
    if total <= 0.0 { 0.5 } else { (0.5 + total.min(1.0) * (matched / total - 0.5)).clamp(0.0, 1.0) }
}

fn old_country(decay: f64, span: Span, country: Option<RegionCode>, signals: &[CountrySignal]) -> f64 {
    let Some(target) = country else { return 0.5 };
    let (mut matched, mut total) = (0.0f64, 0.0f64);
    for sig in signals {
        if let Some(w) = decay_weight(sig.weight, decay, span, sig.span) {
            total += w;
            if sig.country == target {
                matched += w;
            }
        }
    }
    blend(matched, total)
}

fn old_province(decay: f64, span: Span, folded: &[String], signals: &[ProvinceSignal], text: &str) -> f64 {
    if folded.is_empty() {
        return 0.5;
    }
    let (mut matched, mut total) = (0.0f64, 0.0f64);
    for sig in signals.iter().filter(|s| !s.folded_name.is_empty()) {
        if let Some(w) = decay_weight(sig.weight, decay, span, sig.span) {
            total += w;
            if folded.contains(&sig.folded_name) {
                matched += w;
            }
        }
    }
    for at in old_phrase_spans(text, folded) {
        if let Some(w) = decay_weight(1.0, decay, span, at) {
            matched += w;
            total += w;
        }
    }
    blend(matched, total)
}

/// `link_mentions` before #1498: per-mention copies of the context.
fn old_link<G: Gazetteer>(l: &GeoEntityLinker<G>, text: &str, mentions: &[Span]) -> Vec<Vec<LinkedCandidate>> {
    let min_sim = l.fuzzy.min_similarity();
    let limit = l.max_candidates_per_mention.max(DEFAULT_RETRIEVAL_LIMIT);
    let (mut raw, mut hashes, mut keys) = (Vec::new(), Vec::<Vec<String>>::new(), Vec::new());
    let text_signals = l.country.extract_text_signals(text);
    let (mut cand_country, mut cand_province) = (Vec::new(), Vec::new());
    let mut folder = Folder::new();
    for (m, &span) in mentions.iter().enumerate() {
        let surface = &text[span.range()];
        keys.push(folder.fold(surface));
        let (mut entries, mut hs) = (Vec::new(), Vec::new());
        for (idx, (view, sim)) in l.gazetteer.lookup_fuzzy(surface, &l.fuzzy, min_sim, limit).into_iter().enumerate() {
            let owned = view.to_entry();
            hs.push(encode_geohash(owned.location(), l.geohash.precision()));
            let weight = 0.5 * sim * (1.0 / (1.0 + idx as f64));
            if let Some(country) = owned.region_code() {
                cand_country.push((m, CountrySignal { span, country, weight }));
            }
            for admin in owned.containment() {
                let folded_name = folder.fold(admin);
                if !folded_name.is_empty() {
                    cand_province.push((m, ProvinceSignal { span, folded_name, weight }));
                }
            }
            entries.push((owned, sim));
        }
        raw.push(entries);
        hashes.push(hs);
    }
    let mut out = Vec::new();
    for (m, (&span, candidates)) in mentions.iter().zip(raw).enumerate() {
        let max_pop_ln =
            candidates.iter().map(|(e, _)| ((e.population().max(0) as f64) + 1.0).ln()).fold(0.0f64, f64::max);
        let filtered: Vec<Vec<String>> = hashes
            .iter()
            .enumerate()
            .map(|(j, h)| if j == m || keys[j] != keys[m] { h.clone() } else { vec![] })
            .collect();
        let mut countries = text_signals.clone();
        countries.extend(cand_country.iter().filter(|(j, _)| keys[*j] != keys[m]).map(|(_, s)| *s));
        let provinces: Vec<ProvinceSignal> =
            cand_province.iter().filter(|(j, _)| keys[*j] != keys[m]).map(|(_, s)| s.clone()).collect();
        let mut scored = Vec::new();
        for (c, (entry, fuzzy_score)) in candidates.into_iter().enumerate() {
            let geohash_score = old_geohash_score(l.geohash.precision(), m, &hashes[m][c], &filtered);
            let country_score = old_country(l.country.decay_bytes(), span, entry.region_code(), &countries);
            let folded =
                ProvinceProximityScorer::fold_containment(&mut folder, entry.containment().iter().map(|s| &**s));
            let province_score = old_province(l.province.decay_bytes(), span, &folded, &provinces, text);
            let population_score = if max_pop_ln <= 0.0 {
                0.5
            } else {
                (((entry.population().max(0) as f64) + 1.0).ln() / max_pop_ln).clamp(0.0, 1.0)
            };
            let score = l.weights.combine(fuzzy_score, geohash_score, country_score, province_score, population_score);
            let (fz, gh, co, pr, po) = (fuzzy_score, geohash_score, country_score, province_score, population_score);
            scored.push(LinkedCandidate {
                entry,
                score,
                fuzzy_score: fz,
                geohash_score: gh,
                country_score: co,
                province_score: pr,
                population_score: po,
            });
        }
        scored.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| by_prior_entry(&a.entry, &b.entry)));
        scored.truncate(l.max_candidates_per_mention);
        out.push(scored);
    }
    out
}

struct Lcg(u64);

impl Lcg {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

const NAMES: [&str; 8] = ["Springfield", "Springfeld", "Paris", "Salem", "Salen", "Clinton", "New Salem", "Lyon"];
const ADMINS: [&str; 5] = ["Illinois", "Ile-de-France", "Texas", "Sangamon County", "Lamar County"];
const COUNTRIES: [(&str, &str); 4] = [("US", "United States"), ("FR", "France"), ("DE", "Germany"), ("CA", "Canada")];

fn random_gazetteer(rng: &mut Lcg) -> InMemoryGazetteer {
    let mut entries = Vec::new();
    for k in 0..40 {
        // Few clusters at coarse steps so many candidates share a Geohash.
        let (lat, lon) = ((rng.below(5) as f64) * 7.5 - 15.0, (rng.below(5) as f64) * 9.0 - 20.0);
        let containment = (0..rng.below(3)).map(|_| ADMINS[rng.below(ADMINS.len())].to_owned()).collect();
        let parts = EntryParts {
            source: "geonames".into(),
            record_id: format!("r{k}"),
            name: NAMES[rng.below(NAMES.len())].into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(lat, lon).unwrap(),
            bounding_box: None,
            country_code: (rng.below(5) > 0).then(|| COUNTRIES[rng.below(COUNTRIES.len())].0.into()),
            containment,
            population: [0, 10, 10, 5_000, 2_000_000][rng.below(5)],
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes: Vec::new(),
        };
        entries.push(GazetteerEntry::new(parts).unwrap());
    }
    InMemoryGazetteer::from_entries(&entries).unwrap()
}

fn random_document(rng: &mut Lcg, mentions: usize) -> (String, Vec<Span>) {
    let (mut text, mut spans) = (String::new(), Vec::new());
    for _ in 0..mentions {
        text.push_str(["Then ", "near ", "Ok. ", ""][rng.below(4)]);
        let start = text.len();
        text.push_str(NAMES[rng.below(NAMES.len())]);
        spans.push(Span::new(start as u32, text.len() as u32));
        match rng.below(4) {
            0 => text.push_str(&format!(", {} ", COUNTRIES[rng.below(COUNTRIES.len())].1)),
            1 => text.push_str(&format!(" in {} ", ADMINS[rng.below(ADMINS.len())])),
            _ => text.push(' '),
        }
    }
    (text, spans)
}

fn bits(linked: &[Vec<LinkedCandidate>]) -> Vec<Vec<(String, [u64; 6])>> {
    let row = |c: &LinkedCandidate| {
        let s = [c.score, c.fuzzy_score, c.geohash_score, c.country_score, c.province_score, c.population_score];
        (c.entry.record_id().to_owned(), s.map(f64::to_bits))
    };
    linked.iter().map(|cands| cands.iter().map(row).collect()).collect()
}

#[test]
fn context_scoring_matches_the_per_mention_copies_bit_for_bit() {
    let mut rng = Lcg(1498);
    for round in 0..24 {
        let gaz = random_gazetteer(&mut rng);
        let precision = [1, 2, 3, 4, 6][rng.below(5)];
        let linker = GeoEntityLinker::new(&gaz)
            .with_geohash_scorer(GeoHashBinningScorer::new(precision))
            .with_country_scorer(CountryProximityScorer::new([1.0, 40.0, 250.0][rng.below(3)]))
            .with_max_candidates([1, 3, 50][rng.below(3)]);
        let (text, spans) = random_document(&mut rng, [1, 2, 7, 40, 90][round % 5]);
        let got = linker.link_mentions(&text, &spans).unwrap();
        let want = old_link(&linker, &text, &spans);
        assert_eq!(bits(&got), bits(&want), "round {round}: {text}");
        assert_eq!(got, want);
    }
}

#[test]
fn each_distinct_geohash_is_decoded_once_per_document() {
    let mut rng = Lcg(7);
    let gaz = random_gazetteer(&mut rng);
    let linker = GeoEntityLinker::new(&gaz).with_max_candidates(1_000);
    let (text, spans) = random_document(&mut rng, 60);
    let before = DECODES.with(|n| n.get());
    let linked = linker.link_mentions(&text, &spans).unwrap();
    let decodes = DECODES.with(|n| n.get()) - before;
    let hashes: Vec<String> =
        linked.iter().flatten().map(|c| encode_geohash(c.entry.location(), linker.geohash.precision())).collect();
    let distinct: HashSet<&String> = hashes.iter().collect();
    assert!(hashes.len() > 10 * distinct.len(), "{} rows, {} distinct", hashes.len(), distinct.len());
    assert_eq!(decodes, distinct.len());

    // The old scoring decoded inside the pairwise loop.
    let before = DECODES.with(|n| n.get());
    let all = vec![hashes.clone(), hashes[..1].to_vec(), hashes[1..3].to_vec()];
    linker.geohash.score_in_context(0, &hashes[0], &all);
    assert_eq!(DECODES.with(|n| n.get()) - before, 4);
}

#[test]
fn decode_table_matches_the_alphabet_scan_for_every_byte() {
    for b in 0..=255u8 {
        for hash in [vec![b], vec![b'd', b], vec![b, b'P', b'z']] {
            let Ok(hash) = String::from_utf8(hash) else { continue };
            assert_eq!(geohash::decode_geohash(&hash), old_decode(&hash), "{hash:?}");
        }
    }
    assert_eq!(geohash::decode_geohash(""), None);
    assert_eq!(geohash::decode_geohash("dp\u{e9}"), None);
}
