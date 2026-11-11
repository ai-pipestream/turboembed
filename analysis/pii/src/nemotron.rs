//! Mapping from `nvidia/Nemotron-PII`'s 55 entity types to [`PiiKind`],
//! value normalization for neural spans, and rule-precedence resolution
//! combining [`PiiDetector`] rules with scored neural candidates (#1888).
//!
//! Structured kinds that rules already cover (`card`, `us-ssn`, `email`,
//! `phone`, `aba-routing`, `bic`, `ipv4`, `ipv6`, `mac`, `url`, etc.) are
//! trained in the token-classification head, while checksum- and
//! format-validated rule detections win whenever their spans overlap a
//! neural candidate.

use turbo_analysis::{Span, ViewId};
use turbo_analysis_layers::{
    AllowScope, CustomKinds, KindError, Pii, PiiCandidates, PiiEvidence, PiiKind, PiiScore, PiiScoreKind,
};

use crate::detector::{DENY_LIST_DETECTOR, PiiDetector};
use crate::hits::{AllowDecision, Hits};
use crate::lists::AllowList;
use crate::scan::BUILTIN_VERSION;

/// All 55 `Nemotron-PII` label names (`snake_case`) paired with their
/// canonical `turbo:pii` wire names (`kebab-case`), sorted by `Nemotron-PII`
/// label name.
///
/// The 16 labels whose wire name matches a built-in [`PiiKind`] (`age`,
/// `aba-routing`, `city`, `county`, `card`, `date`, `email`, `ipv4`, `ipv6`,
/// `mac`, `phone`, `us-ssn`, `state`, `street-address`, `bic`, `url`) resolve
/// to that built-in variant; the remaining 39 personal, demographic,
/// credential, financial and medical/PHI labels resolve via [`CustomKinds`]
/// (or automatically to a built-in [`PiiKind`] if promoted later).
pub const NEMOTRON_TYPES: [(&str, &str); 55] = [
    ("account_number", "account-number"),
    ("age", "age"),
    ("api_key", "api-key"),
    ("bank_routing_number", "aba-routing"),
    ("biometric_identifier", "biometric-identifier"),
    ("blood_type", "blood-type"),
    ("certificate_license_number", "certificate-license-number"),
    ("city", "city"),
    ("company_name", "company-name"),
    ("coordinate", "coordinate"),
    ("country", "country"),
    ("county", "county"),
    ("credit_debit_card", "card"),
    ("customer_id", "customer-id"),
    ("cvv", "cvv"),
    ("date", "date"),
    ("date_of_birth", "date-of-birth"),
    ("date_time", "date-time"),
    ("device_identifier", "device-identifier"),
    ("education_level", "education-level"),
    ("email", "email"),
    ("employee_id", "employee-id"),
    ("employment_status", "employment-status"),
    ("fax_number", "fax-number"),
    ("first_name", "first-name"),
    ("gender", "gender"),
    ("health_plan_beneficiary_number", "health-plan-beneficiary-number"),
    ("http_cookie", "http-cookie"),
    ("ipv4", "ipv4"),
    ("ipv6", "ipv6"),
    ("language", "language"),
    ("last_name", "last-name"),
    ("license_plate", "license-plate"),
    ("mac_address", "mac"),
    ("medical_record_number", "medical-record-number"),
    ("national_id", "national-id"),
    ("occupation", "occupation"),
    ("password", "password"),
    ("phone_number", "phone"),
    ("pin", "pin"),
    ("political_view", "political-view"),
    ("postcode", "postcode"),
    ("race_ethnicity", "race-ethnicity"),
    ("religious_belief", "religious-belief"),
    ("sexuality", "sexuality"),
    ("ssn", "us-ssn"),
    ("state", "state"),
    ("street_address", "street-address"),
    ("swift_bic", "bic"),
    ("tax_id", "tax-id"),
    ("time", "time"),
    ("unique_id", "unique-id"),
    ("url", "url"),
    ("user_name", "user-name"),
    ("vehicle_identifier", "vehicle-identifier"),
];

/// Returns the canonical `turbo:pii` wire name for a `Nemotron-PII` type.
pub fn nemotron_wire_name(label: &str) -> Option<&'static str> {
    NEMOTRON_TYPES.binary_search_by_key(&label, |&(k, _)| k).ok().map(|i| NEMOTRON_TYPES[i].1)
}

/// Resolves a `Nemotron-PII` label (or an already kebab-cased kind name)
/// to a built-in [`PiiKind`] when one exists, or registers it in
/// `custom_kinds`.
pub fn resolve_nemotron_kind(label: &str, custom_kinds: &mut CustomKinds) -> Result<PiiKind, KindError> {
    let wire = nemotron_wire_name(label).unwrap_or(label);
    if let Some(builtin) = PiiKind::parse(wire) {
        return Ok(builtin);
    }
    custom_kinds.add(wire)
}

/// Registers every non-built-in `Nemotron-PII` wire name in `custom_kinds`
/// in deterministic table order.
pub fn register_nemotron_kinds(custom_kinds: &mut CustomKinds) -> Result<(), KindError> {
    for &(_, wire) in &NEMOTRON_TYPES {
        if PiiKind::parse(wire).is_none() {
            custom_kinds.add(wire)?;
        }
    }
    Ok(())
}

/// Returns a [`CustomKinds`] table pre-populated with all non-built-in
/// `Nemotron-PII` kinds.
pub fn nemotron_custom_kinds() -> CustomKinds {
    let mut kinds = CustomKinds::new();
    register_nemotron_kinds(&mut kinds).expect("valid Nemotron-PII custom kind names");
    kinds
}

/// Normalizes the text covered by a neural candidate of `kind`, matching
/// built-in rule normalizers on structured kinds where the surface form fits.
pub fn normalize_model_value(kind: PiiKind, covered: &str) -> String {
    let trimmed = covered.trim();
    let fallback = if trimmed.is_empty() { covered } else { trimmed };
    match kind {
        PiiKind::Card | PiiKind::AbaRouting => {
            let digits: String = trimmed.chars().filter(char::is_ascii_digit).collect();
            if digits.is_empty() { fallback.to_owned() } else { digits }
        }
        PiiKind::UsSsn => {
            let digits: Vec<u8> = trimmed.bytes().filter(u8::is_ascii_digit).collect();
            if digits.len() == 9 {
                format!(
                    "{}-{}-{}",
                    std::str::from_utf8(&digits[0..3]).expect("digits"),
                    std::str::from_utf8(&digits[3..5]).expect("digits"),
                    std::str::from_utf8(&digits[5..9]).expect("digits"),
                )
            } else {
                fallback.to_owned()
            }
        }
        PiiKind::Email => match trimmed.rsplit_once('@') {
            Some((local, domain)) if !local.is_empty() && !domain.is_empty() => {
                format!("{local}@{}", domain.to_ascii_lowercase())
            }
            _ => fallback.to_owned(),
        },
        PiiKind::Phone => {
            let plus = trimmed.starts_with('+');
            let digits: String = trimmed.chars().filter(char::is_ascii_digit).collect();
            if digits.is_empty() {
                fallback.to_owned()
            } else if plus {
                format!("+{digits}")
            } else {
                digits
            }
        }
        PiiKind::Bic => {
            let alnum: String =
                trimmed.chars().filter(char::is_ascii_alphanumeric).map(|c| c.to_ascii_uppercase()).collect();
            if alnum.is_empty() { fallback.to_owned() } else { alnum }
        }
        PiiKind::Mac => {
            let hex: Vec<u8> = trimmed.bytes().filter(u8::is_ascii_hexdigit).map(|b| b.to_ascii_lowercase()).collect();
            if hex.len() == 12 {
                (0..6).map(|i| std::str::from_utf8(&hex[i * 2..i * 2 + 2]).expect("hex")).collect::<Vec<_>>().join(":")
            } else {
                fallback.to_owned()
            }
        }
        _ => {
            let collapsed = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
            if collapsed.is_empty() { covered.to_owned() } else { collapsed }
        }
    }
}

/// One span decoded from a neural PII head before rule-overlap resolution.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ModelCandidate {
    pub span: Span,
    pub kind: PiiKind,
    pub probability: f32,
}

/// Detector and scoring-model identity recorded on neural PII candidates.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ModelProvenance<'a> {
    pub detector_id: &'a str,
    pub detector_version: &'a str,
    pub model_id: &'a str,
    pub model_version: &'a str,
    pub calibrated: bool,
}

/// Resolution options pairing optional structured `rules`, custom-kind table,
/// detector/model provenance, score threshold, and allow list for
/// [`resolve_model_with_rules`].
#[derive(Copy, Clone, Debug)]
pub struct ModelResolution<'a> {
    pub rules: Option<&'a PiiDetector>,
    pub model_kinds: &'a CustomKinds,
    pub provenance: ModelProvenance<'a>,
    pub min_score: Option<f32>,
    pub extra_allow: Option<&'a AllowList>,
}

/// Resolves neural `model_candidates` alongside optional checksum/format
/// `rules` into `(turbo:pii, turbo:pii-candidates)`.
///
/// Rules always win on span overlap:
/// - If a neural candidate matches an admitted rule mention's exact `(span,
///   kind)`, both candidates appear in `PiiCandidates` and the neural
///   candidate resolves to `PiiDisposition::Repeats` pointing at the kept
///   rule mention.
/// - If a neural candidate lies strictly inside an admitted rule mention
///   (and nesting is not kept), it resolves to `PiiDisposition::NestedIn`
///   pointing at the kept rule mention.
/// - If a neural candidate partially overlaps, encloses, or shares a span
///   under a different kind with an admitted rule mention, the neural span
///   yields to the rule mention so the checksum/format rule wins cleanly.
pub fn resolve_model_with_rules(
    text: &str,
    view: ViewId,
    opts: &ModelResolution<'_>,
    model_candidates: &[ModelCandidate],
) -> Result<(Pii, PiiCandidates), String> {
    let ModelResolution { rules, model_kinds, provenance, min_score, extra_allow } = *opts;
    if u32::try_from(text.len()).is_err() {
        return Err(format!("a text of {} bytes is past the u32 offset range", text.len()));
    }

    let mut merged_kinds = rules.map_or_else(CustomKinds::new, |r| r.custom_kinds().clone());
    for name in model_kinds.names() {
        if PiiKind::parse(name).is_none() {
            merged_kinds.add(name).map_err(|e| e.to_string())?;
        }
    }

    let admitted_rules: Vec<(Span, PiiKind, String)> = if let Some(det) = rules {
        let rule_pii = det.layer(text, view);
        rule_pii
            .mentions()
            .filter_map(|m| {
                let k = merged_kinds.translate(&rule_pii.custom_kinds, m.kind)?;
                Some((m.span, k, m.normalized.to_owned()))
            })
            .collect()
    } else {
        Vec::new()
    };

    let mut hits = Hits::default();
    if let Some(det) = rules {
        for s in &det.scanners {
            hits.begin_detector(s.extractor().id(), s.version());
            s.scan(text, &merged_kinds, &mut hits);
        }
        if !det.deny_lists.is_empty() {
            hits.begin_detector(DENY_LIST_DETECTOR, BUILTIN_VERSION);
            for (i, d) in det.deny_lists.iter().enumerate() {
                d.scan(text, i as u32, &mut hits);
            }
        }
        hits.apply_label_policy(text, &det.window);
        if let Some(min) = det.min_score {
            hits.apply_threshold(min);
        }
    }

    let model_ref =
        hits.detectors_mut().add(provenance.model_id, provenance.model_version).map_err(|e| e.to_string())?;
    hits.begin_detector(provenance.detector_id, provenance.detector_version);

    let threshold = match (min_score, rules.and_then(|r| r.min_score)) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (Some(a), None) | (None, Some(a)) => Some(a),
        (None, None) => None,
    };
    let keep_nested = rules.is_some_and(PiiDetector::keeps_nested);

    for mc in model_candidates {
        let (start, end) = (mc.span.start as usize, mc.span.end as usize);
        if start >= end || end > text.len() || !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            return Err(format!("model candidate span {}..{} is invalid for text of {} bytes", start, end, text.len()));
        }
        let kind = merged_kinds
            .translate(model_kinds, mc.kind)
            .ok_or_else(|| format!("model candidate kind {:?} is missing from custom_kinds", mc.kind))?;
        if !mc.probability.is_finite() {
            return Err(format!("model candidate probability {} is not finite", mc.probability));
        }
        let value = mc.probability.clamp(0.0, 1.0);
        let score = Some(PiiScore {
            value,
            kind: PiiScoreKind::Probability,
            model: model_ref,
            calibrated: provenance.calibrated,
        });
        let covered = &text[start..end];

        if threshold.is_some_and(|min| value < min) {
            let norm = normalize_model_value(kind, covered);
            hits.add_candidate(start, end, kind, &[PiiEvidence::Shape], score, |s| s.push_str(&norm));
            continue;
        }

        let overlaps_rule = admitted_rules.iter().any(|&(rs, _, _)| mc.span.start < rs.end && rs.start < mc.span.end);
        if !overlaps_rule {
            let norm = normalize_model_value(kind, covered);
            hits.add_candidate(start, end, kind, &[PiiEvidence::Shape], score, |s| s.push_str(&norm));
        } else if let Some((_, _, rule_norm)) = admitted_rules.iter().find(|&&(rs, rk, _)| rs == mc.span && rk == kind)
        {
            hits.add_candidate(start, end, kind, &[PiiEvidence::Shape], score, |s| s.push_str(rule_norm));
        } else if !keep_nested
            && admitted_rules
                .iter()
                .any(|&(rs, _, _)| rs.start <= mc.span.start && mc.span.end <= rs.end && rs != mc.span)
        {
            let norm = normalize_model_value(kind, covered);
            hits.add_candidate(start, end, kind, &[PiiEvidence::Shape], score, |s| s.push_str(&norm));
        }
    }

    if let Some(min) = threshold {
        hits.apply_threshold(min);
    }

    let mut out = Pii::with_custom_kinds(view, merged_kinds.clone());
    let mut candidates = PiiCandidates::with_tables(view, merged_kinds, Default::default());
    hits.resolve(
        keep_nested,
        |kind, start, end, normalized| -> AllowDecision {
            let covered = &text[start as usize..end as usize];
            if let Some(det) = rules
                && let Some(dec) = det.allow_rule(kind, covered, normalized)
            {
                return Some(dec);
            }
            extra_allow.and_then(|a| a.find_rule(covered, normalized)).map(|rule| (AllowScope::Global, rule))
        },
        &mut out,
        &mut candidates,
    );
    Ok((out, candidates))
}
