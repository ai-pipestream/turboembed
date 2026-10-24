//! Disambiguation scoring components:
//! 1. Alias and popularity prior (`prior_score`)
//! 2. Entity-type compatibility (`type_compatibility`)
//! 3. Local and document context overlap + optional dense vector cosine (`context_overlap_score`)
//! 4. Document-level coherence and coreference cluster propagation

use std::collections::{HashMap, HashSet};

use crate::layer::CandidateLink;
use crate::providers::{KbProvider, RawCandidate};

/// Linear combination weights for the four disambiguation signals.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkerWeights {
    pub prior_weight: f64,
    pub type_weight: f64,
    pub context_weight: f64,
    pub coherence_weight: f64,
}

impl Default for LinkerWeights {
    fn default() -> Self {
        Self { prior_weight: 0.30, type_weight: 0.25, context_weight: 0.30, coherence_weight: 0.15 }
    }
}

impl LinkerWeights {
    /// Returns non-negative weights normalized to sum to `1.0`.
    pub fn normalized(&self) -> (f64, f64, f64, f64) {
        let p = if self.prior_weight.is_finite() { self.prior_weight.max(0.0) } else { 0.0 };
        let t = if self.type_weight.is_finite() { self.type_weight.max(0.0) } else { 0.0 };
        let x = if self.context_weight.is_finite() { self.context_weight.max(0.0) } else { 0.0 };
        let c = if self.coherence_weight.is_finite() { self.coherence_weight.max(0.0) } else { 0.0 };
        let max_w = p.max(t).max(x).max(c);
        if max_w <= 0.0 {
            return (0.25, 0.25, 0.25, 0.25);
        }
        let (ps, ts, xs, cs) = (p / max_w, t / max_w, x / max_w, c / max_w);
        let sum = ps + ts + xs + cs;
        (ps / sum, ts / sum, xs / sum, cs / sum)
    }

    /// Validates that all four weights are finite and non-negative.
    pub fn validate(&self) -> Result<(), String> {
        for (name, val) in [
            ("prior_weight", self.prior_weight),
            ("type_weight", self.type_weight),
            ("context_weight", self.context_weight),
            ("coherence_weight", self.coherence_weight),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(format!("invalid {name} {val}: must be finite and >= 0.0"));
            }
        }
        Ok(())
    }
}

/// Configuration for [`crate::EntityLinker`].
#[derive(Clone, Debug, PartialEq)]
pub struct LinkerConfig {
    /// Relative weights for prior, type compatibility, context overlap, and coherence.
    pub weights: LinkerWeights,
    /// Minimum combined score in `[0.0, 1.0]` required to link an entity; below this yields `NIL`.
    pub nil_threshold: f64,
    /// Minimum entity-type compatibility in `[0.0, 1.0]` required for a candidate to be eligible.
    pub min_type_compatibility: f64,
    /// Maximum number of ranked [`CandidateLink`]s retained per entity in [`crate::LinkedEntities`].
    pub top_k: usize,
    /// Maximum candidates requested from each [`KbProvider`] per mention.
    pub max_candidates_per_provider: usize,
    /// Minimum alias similarity in `[0.0, 1.0]` for fuzzy candidate generation (`1.0` disables fuzzy).
    pub fuzzy_threshold: f64,
    /// Whether to propagate candidates and evidence across coreference clusters.
    pub coref_propagation: bool,
    /// Token window radius around a mention when sentence boundaries are unavailable.
    pub context_window_tokens: usize,
}

impl LinkerConfig {
    /// Validates that all thresholds are finite in `[0.0, 1.0]` and weights are finite and non-negative.
    pub fn validate(&self) -> Result<(), String> {
        self.weights.validate()?;
        for (name, val) in [
            ("nil_threshold", self.nil_threshold),
            ("min_type_compatibility", self.min_type_compatibility),
            ("fuzzy_threshold", self.fuzzy_threshold),
        ] {
            if !val.is_finite() || !(0.0..=1.0).contains(&val) {
                return Err(format!("invalid {name} {val}: must be finite and in [0.0, 1.0]"));
            }
        }
        Ok(())
    }
}

impl Default for LinkerConfig {
    fn default() -> Self {
        Self {
            weights: LinkerWeights::default(),
            nil_threshold: 0.35,
            min_type_compatibility: 0.05,
            top_k: 5,
            max_candidates_per_provider: 16,
            fuzzy_threshold: 0.88,
            coref_propagation: true,
            context_window_tokens: 24,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CoarseType {
    Person,
    Org,
    Location,
    Facility,
    Product,
    Work,
    Event,
    Law,
    Norp,
    Concept,
}

fn classify_type(raw: &str) -> Option<CoarseType> {
    let norm = raw.trim().to_ascii_uppercase();
    if norm.is_empty() {
        return None;
    }
    let norm = norm.strip_prefix("B-").or_else(|| norm.strip_prefix("I-")).unwrap_or(&norm);
    match norm {
        "PERSON" | "PER" | "HUMAN" | "INDIVIDUAL" | "SCIENTIST" | "POLITICIAN" | "PRESIDENT" | "AUTHOR" | "ARTIST"
        | "ACTOR" | "ATHLETE" | "LEADER" | "MONARCH" => Some(CoarseType::Person),
        "ORG" | "ORGANIZATION" | "ORGANISATION" | "COMPANY" | "CORP" | "CORPORATION" | "INSTITUTION" | "AGENCY"
        | "UNIVERSITY" | "SCHOOL" | "TEAM" | "BAND" | "PARTY" | "BANK" | "AIRLINE" => Some(CoarseType::Org),
        "GPE" | "LOC" | "LOCATION" | "PLACE" | "CITY" | "TOWN" | "VILLAGE" | "ADMIN" | "STATE" | "PROVINCE"
        | "COUNTRY" | "NATION" | "REGION" | "CONTINENT" | "ISLAND" | "RIVER" | "MOUNTAIN" | "CAPITAL" => {
            Some(CoarseType::Location)
        }
        "FAC" | "FACILITY" | "BUILDING" | "AIRPORT" | "BRIDGE" | "HIGHWAY" | "STADIUM" | "MUSEUM" | "HOSPITAL"
        | "POI" | "MONUMENT" => Some(CoarseType::Facility),
        "PRODUCT" | "PROD" | "DEVICE" | "HARDWARE" | "SOFTWARE" | "VEHICLE" | "CAR" | "AUTOMOBILE" | "BRAND"
        | "FOOD" | "FRUIT" | "DRUG" | "WEAPON" => Some(CoarseType::Product),
        "WORK_OF_ART" | "WORK" | "BOOK" | "FILM" | "MOVIE" | "SONG" | "ALBUM" | "PAINTING" | "PLAY" | "PUBLICATION" => {
            Some(CoarseType::Work)
        }
        "EVENT" | "WAR" | "BATTLE" | "HURRICANE" | "FESTIVAL" | "TOURNAMENT" | "ELECTION" => Some(CoarseType::Event),
        "LAW" | "TREATY" | "ACT" | "CONSTITUTION" | "REGULATION" => Some(CoarseType::Law),
        "NORP" | "NATIONALITY" | "RELIGION" | "ETHNIC_GROUP" | "LANGUAGE" => Some(CoarseType::Norp),
        "CONCEPT" | "NOUN" | "VERB" | "ADJECTIVE" | "ADVERB" | "TERM" | "TOPIC" | "MISC" | "OTHER" | "UNKNOWN"
        | "ENTITY" => Some(CoarseType::Concept),
        _ => None,
    }
}

/// Computes entity-type compatibility in `[0.0, 1.0]` between a mention's NER tag
/// and a KB entity's type list.
pub fn type_compatibility(ner_kind: &str, kb_types: &[String]) -> f64 {
    let ner_trimmed = ner_kind.trim();
    if ner_trimmed.is_empty() || kb_types.is_empty() {
        return 0.75;
    }
    if kb_types.iter().any(|t| t.trim().eq_ignore_ascii_case(ner_trimmed)) {
        return 1.0;
    }
    let Some(ner_coarse) = classify_type(ner_trimmed) else {
        return 0.70;
    };
    if ner_coarse == CoarseType::Concept {
        return 0.75;
    }

    let mut best: f64 = 0.0;
    for kb_t in kb_types {
        let Some(kb_coarse) = classify_type(kb_t) else {
            best = best.max(0.50);
            continue;
        };
        let score = if ner_coarse == kb_coarse {
            1.0
        } else if kb_coarse == CoarseType::Concept {
            0.65
        } else {
            match (ner_coarse, kb_coarse) {
                (CoarseType::Location, CoarseType::Facility)
                | (CoarseType::Facility, CoarseType::Location)
                | (CoarseType::Org, CoarseType::Facility)
                | (CoarseType::Facility, CoarseType::Org)
                | (CoarseType::Org, CoarseType::Product)
                | (CoarseType::Product, CoarseType::Org)
                | (CoarseType::Location, CoarseType::Norp)
                | (CoarseType::Norp, CoarseType::Location)
                | (CoarseType::Org, CoarseType::Norp)
                | (CoarseType::Norp, CoarseType::Org)
                | (CoarseType::Work, CoarseType::Product)
                | (CoarseType::Product, CoarseType::Work) => 0.55,
                _ => 0.0,
            }
        };
        best = best.max(score);
    }
    best
}

/// Computes the combined alias and popularity prior score in `[0.0, 1.0]`.
pub fn prior_score(cand: &RawCandidate) -> f64 {
    let base = 0.60 * cand.alias_prior.clamp(0.0, 1.0) + 0.40 * cand.popularity_prior.clamp(0.0, 1.0);
    (base * cand.match_similarity.clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

/// Computes local and document context overlap (plus optional dense vector similarity)
/// in `[0.0, 1.0]`.
pub fn context_overlap_score(
    cand: &RawCandidate,
    local_terms: &HashSet<String>,
    doc_terms: &HashSet<String>,
    providers: &[&dyn KbProvider],
) -> f64 {
    let has_lexical_context = !local_terms.is_empty() || !doc_terms.is_empty();
    let lexical = if !has_lexical_context || cand.context_terms.is_empty() {
        0.45
    } else {
        let mut weighted_hits = 0.0;
        for term in &cand.context_terms {
            let idf = providers
                .iter()
                .find(|p| p.provider_id() == cand.provider_id)
                .map_or(1.0, |p| p.term_idf(term))
                .max(0.2);
            if local_terms.contains(term) {
                weighted_hits += 2.2 * idf;
            } else if doc_terms.contains(term) {
                weighted_hits += 1.0 * idf;
            }
        }
        if weighted_hits <= 0.0 {
            0.05
        } else {
            (0.15 + 0.85 * (weighted_hits / (weighted_hits + 2.2))).clamp(0.0, 1.0)
        }
    };

    match cand.vector_score {
        Some(vscore) if has_lexical_context => (0.55 * lexical + 0.45 * vscore.clamp(0.0, 1.0)).clamp(0.0, 1.0),
        Some(vscore) => vscore.clamp(0.0, 1.0),
        None => lexical,
    }
}

/// Propagates high-confidence candidate entities across coreference clusters so
/// shorter co-referring mentions (such as `"Lincoln"` co-referring with
/// `"Abraham Lincoln"`) inherit the cluster anchor's candidate and type evidence.
pub(crate) fn propagate_coref_candidates(
    clusters: &[Vec<usize>],
    mention_texts: &[String],
    raw_per_mention: &mut [Vec<RawCandidate>],
    type_scores: &mut [Vec<f64>],
    coref_boost: &mut [HashMap<(String, String), f64>],
) {
    for cluster in clusters {
        if cluster.len() < 2 {
            continue;
        }
        // Choose the cluster anchor: longest surface mention that has at least one type-compatible candidate
        let mut anchor_idx: Option<usize> = None;
        let mut best_key: (usize, u64) = (0, 0);
        for &m_idx in cluster {
            if m_idx >= raw_per_mention.len() || raw_per_mention[m_idx].is_empty() {
                continue;
            }
            let len = mention_texts[m_idx].split_whitespace().count() * 100 + mention_texts[m_idx].len();
            let top_prelim = raw_per_mention[m_idx]
                .iter()
                .zip(&type_scores[m_idx])
                .filter(|(_, ts)| **ts >= 0.5)
                .map(|(c, ts)| prior_score(c) + *ts)
                .fold(0.0f64, f64::max);
            if top_prelim <= 0.0 {
                continue;
            }
            let key = (len, top_prelim.to_bits());
            if anchor_idx.is_none() || key > best_key {
                best_key = key;
                anchor_idx = Some(m_idx);
            }
        }
        let Some(anchor) = anchor_idx else {
            continue;
        };
        let anchor_top = raw_per_mention[anchor]
            .iter()
            .zip(&type_scores[anchor])
            .max_by(|(ca, tsa), (cb, tsb)| (prior_score(ca) + *tsa).total_cmp(&(prior_score(cb) + *tsb)))
            .map(|(c, ts)| (c.clone(), *ts));
        let Some((anchor_cand, anchor_ts)) = anchor_top else {
            continue;
        };

        for &m_idx in cluster {
            if m_idx == anchor || m_idx >= raw_per_mention.len() {
                continue;
            }
            let key = (anchor_cand.provider_id.clone(), anchor_cand.kb_id.clone());
            coref_boost[m_idx].insert(key, 1.0);

            if let Some(pos) = raw_per_mention[m_idx]
                .iter()
                .position(|c| c.provider_id == anchor_cand.provider_id && c.kb_id == anchor_cand.kb_id)
            {
                type_scores[m_idx][pos] = type_scores[m_idx][pos].max(anchor_ts);
                raw_per_mention[m_idx][pos].alias_prior =
                    raw_per_mention[m_idx][pos].alias_prior.max(anchor_cand.alias_prior);
            } else {
                let mut inherited = anchor_cand.clone();
                inherited.matched_alias = mention_texts[m_idx].clone();
                raw_per_mention[m_idx].push(inherited);
                type_scores[m_idx].push(anchor_ts);
            }
        }
    }
}

/// Scores all candidates for all mentions in a document, incorporating
/// document-level coherence across mentions.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_all_mentions(
    config: &LinkerConfig,
    raw_per_mention: &[Vec<RawCandidate>],
    type_scores: &[Vec<f64>],
    context_scores: &[Vec<f64>],
    coref_boost: &[HashMap<(String, String), f64>],
) -> Vec<Vec<CandidateLink>> {
    let n_mentions = raw_per_mention.len();
    let (wp, wt, wx, wc) = config.weights.normalized();
    let pre_sum = (wp + wt + wx).max(1e-9);

    // Pass 1: find each mention's preliminary best candidate for coherence anchoring
    let mut anchors: Vec<Option<(&RawCandidate, f64)>> = Vec::with_capacity(n_mentions);
    for i in 0..n_mentions {
        let mut best: Option<(&RawCandidate, f64)> = None;
        for (j, cand) in raw_per_mention[i].iter().enumerate() {
            let ts = type_scores[i][j];
            if ts < config.min_type_compatibility {
                continue;
            }
            let ps = prior_score(cand);
            let xs = context_scores[i][j];
            let prelim = (wp * ps + wt * ts + wx * xs) / pre_sum;
            if best.is_none_or(|(_, b_score)| prelim > b_score) {
                best = Some((cand, prelim));
            }
        }
        anchors.push(best);
    }

    // Pass 2: compute coherence score and final combined score for each candidate
    let mut out = Vec::with_capacity(n_mentions);
    for i in 0..n_mentions {
        let mut scored = Vec::with_capacity(raw_per_mention[i].len());
        for (j, cand) in raw_per_mention[i].iter().enumerate() {
            let ts = type_scores[i][j];
            let ps = prior_score(cand);
            let xs = context_scores[i][j];

            let mut coh: f64 = if n_mentions <= 1 { 0.50 } else { 0.15 };
            if let Some(&boost) = coref_boost[i].get(&(cand.provider_id.clone(), cand.kb_id.clone())) {
                coh = coh.max(boost);
            }
            if n_mentions > 1 {
                let mut support = 0.0;
                for (other_i, other_anchor) in anchors.iter().enumerate() {
                    if other_i == i {
                        continue;
                    }
                    let Some((other_cand, other_prelim)) = other_anchor else {
                        continue;
                    };
                    if *other_prelim < 0.30 {
                        continue;
                    }
                    if cand.kb_id == other_cand.kb_id && cand.provider_id == other_cand.provider_id {
                        support += 0.85 * *other_prelim;
                    } else if cand.related_ids.contains(&other_cand.kb_id)
                        || other_cand.related_ids.contains(&cand.kb_id)
                    {
                        support += 1.0 * *other_prelim;
                    } else if !cand.related_ids.is_empty()
                        && cand.related_ids.iter().any(|r| other_cand.related_ids.contains(r))
                    {
                        support += 0.65 * *other_prelim;
                    }
                }
                if support > 0.0 {
                    coh = coh.max((0.20 + 0.80 * (support / (support + 0.8))).clamp(0.0, 1.0));
                }
            }

            let combined = if ts < config.min_type_compatibility {
                (0.10 * ps * ts).clamp(0.0, 1.0)
            } else {
                (wp * ps + wt * ts + wx * xs + wc * coh).clamp(0.0, 1.0)
            };

            scored.push(CandidateLink {
                kb_id: cand.kb_id.clone(),
                provider_id: cand.provider_id.clone(),
                canonical_name: cand.canonical_name.clone(),
                matched_alias: cand.matched_alias.clone(),
                score: combined,
                prior_score: ps,
                type_score: ts,
                context_score: xs,
                coherence_score: coh,
            });
        }
        // Ineligible candidates rank after every eligible one, and the layer
        // requires candidate scores in non-increasing order, so an ineligible
        // candidate's penalized score is capped at the lowest eligible score
        // of the same mention. Without the cap a high-prior but
        // type-incompatible candidate can outscore a weak eligible one and
        // the written layer fails `LinkedEntities::check`.
        let eligible_floor = scored
            .iter()
            .filter(|c| c.type_score >= config.min_type_compatibility)
            .map(|c| c.score)
            .min_by(f64::total_cmp);
        if let Some(floor) = eligible_floor {
            for c in scored.iter_mut().filter(|c| c.type_score < config.min_type_compatibility) {
                c.score = c.score.min(floor);
            }
        }
        scored.sort_by(|a, b| {
            let a_elig = a.type_score >= config.min_type_compatibility;
            let b_elig = b.type_score >= config.min_type_compatibility;
            b_elig
                .cmp(&a_elig)
                .then_with(|| b.score.total_cmp(&a.score))
                .then_with(|| b.prior_score.total_cmp(&a.prior_score))
                .then_with(|| a.kb_id.cmp(&b.kb_id))
        });
        out.push(scored);
    }
    out
}
