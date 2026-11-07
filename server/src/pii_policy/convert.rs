//! `RedactionPolicy` message to [`super::RedactionPolicy`].

use std::collections::HashSet;

use turbo_pii::rewrite::RewriteStrategy;
use turbo_pii::{AllowList, Case, ChecksumAlgorithm, CustomKinds, DenyList, LabelPolicy, Pattern, PiiKind};
use turbo_pii::{CustomKindId, PiiScoreKind};

use super::error::{At, PolicyError};
use super::resolve::{PolicyResolver, check_dictionary};
use super::strategy::{keyed_token, mask, strategy, whole_mask};
use super::{
    Condition, CustomIdentifier, DenyRule, KindRule, LabelOverride, PolicyDigest, PolicyFlag, RedactionPolicy,
    StrategyEntry, canonical,
};
use crate::proto::result as pb;
use pb::{pii_default_strategy, pii_kind_ref, pii_strategy_condition::Test, pii_term_source::Source};

const fn case(sensitive: bool) -> Case {
    if sensitive { Case::Sensitive } else { Case::Insensitive }
}

fn invalid(at: At, e: impl std::fmt::Display) -> PolicyError {
    PolicyError::Invalid { at, reason: e.to_string() }
}

/// What a kind reference resolved to.
enum KindOf {
    Known(PiiKind),
    Unknown(i32),
}

fn kind(at: At, r: Option<&pb::PiiKindRef>, kinds: &CustomKinds) -> Result<KindOf, PolicyError> {
    match r.and_then(|r| r.kind) {
        None => Err(PolicyError::Missing { at, field: "kind" }),
        Some(pii_kind_ref::Kind::Builtin(0)) => Err(PolicyError::Unspecified { at, field: "kind.builtin", value: 0 }),
        Some(pii_kind_ref::Kind::Builtin(id)) => {
            Ok(u16::try_from(id).ok().and_then(PiiKind::from_id).map_or(KindOf::Unknown(id), KindOf::Known))
        }
        Some(pii_kind_ref::Kind::Custom(index)) => match u16::try_from(index) {
            Ok(i) if usize::from(i) < kinds.len() => Ok(KindOf::Known(PiiKind::Custom(CustomKindId::new(i)))),
            _ => Err(PolicyError::CustomKindOutOfRange { at, index, table: kinds.len() }),
        },
    }
}

fn custom_kind(at: At, r: Option<&pb::PiiKindRef>, kinds: &CustomKinds) -> Result<PiiKind, PolicyError> {
    match kind(at, r, kinds)? {
        KindOf::Known(k) if k.is_custom() => Ok(k),
        _ => Err(PolicyError::BuiltinKind { at }),
    }
}

fn rule_id(at: At, id: &str, seen: &mut HashSet<String>) -> Result<String, PolicyError> {
    if id.is_empty() {
        return Err(PolicyError::Missing { at, field: "rule_id" });
    }
    if !seen.insert(id.to_owned()) {
        return Err(PolicyError::DuplicateRuleId { at });
    }
    Ok(id.to_owned())
}

fn terms(
    at: At,
    src: Option<&pb::PiiTermSource>,
    resolver: &dyn PolicyResolver,
    mut each: impl FnMut(&str) -> Result<(), PolicyError>,
) -> Result<(), PolicyError> {
    match src.and_then(|s| s.source.as_ref()) {
        None => Err(PolicyError::UnknownArm { at, field: "terms.source" }),
        Some(Source::Terms(t)) => t.terms.iter().try_for_each(|t| each(t)),
        Some(Source::Dictionary(d)) => {
            check_dictionary(at, &d.artifact_id, &d.sha256)?;
            let list = resolver.terms(&d.artifact_id, &d.sha256).map_err(|e| PolicyError::Unresolved {
                at,
                field: "terms.dictionary",
                reason: e.reason(),
            })?;
            list.iter().try_for_each(|t| each(t))
        }
    }
}

fn allow(at: At, a: Option<&pb::PiiAllowList>, resolver: &dyn PolicyResolver) -> Result<AllowList, PolicyError> {
    let mut list = AllowList::new();
    let Some(a) = a else { return Ok(list) };
    let c = case(a.case_sensitive);
    for (i, src) in a.terms.iter().enumerate() {
        terms(at.inner("allow.terms", i), Some(src), resolver, |t| {
            list.add_term_with_case(t, c).map(|_| ()).map_err(|e| invalid(at.inner("allow.terms", i), e))
        })?;
    }
    for (i, p) in a.patterns.iter().enumerate() {
        let at = at.inner("allow.patterns", i);
        let pattern = match pb::PiiPatternSyntax::try_from(p.syntax) {
            Ok(pb::PiiPatternSyntax::Shape) => Pattern::shape(&p.pattern),
            Ok(pb::PiiPatternSyntax::Glob) => Pattern::glob(&p.pattern),
            Ok(pb::PiiPatternSyntax::Unspecified) | Err(_) => {
                return Err(PolicyError::Unspecified { at, field: "syntax", value: p.syntax });
            }
        }
        .map_err(|e| invalid(at, e))?;
        list.add_pattern(pattern.with_case(case(p.case_sensitive)));
    }
    Ok(list)
}

/// The condition of `c`, or `None` when its arm is unset or unknown.
fn condition(at: At, c: &pb::PiiStrategyCondition) -> Result<Option<Condition>, PolicyError> {
    let Some(test) = &c.test else { return Ok(None) };
    Ok(Some(match test {
        Test::ChecksumPassed(p) => {
            let mut algorithms = Vec::with_capacity(p.algorithms.len());
            for &a in &p.algorithms {
                let alg = u32::try_from(a).ok().and_then(ChecksumAlgorithm::from_id);
                algorithms.push(alg.ok_or(PolicyError::Unspecified {
                    at,
                    field: "checksum_passed.algorithms",
                    value: a,
                })?);
            }
            Condition::ChecksumPassed(algorithms)
        }
        Test::LabelPresent(_) => Condition::LabelPresent,
        Test::Detector(d) => {
            if d.id.is_empty() {
                return Err(PolicyError::Missing { at, field: "detector.id" });
            }
            let version = (!d.version.is_empty()).then(|| d.version.clone());
            Condition::Detector { id: d.id.clone(), version }
        }
        Test::ScoreAtLeast(s) => {
            let kind = u32::try_from(s.kind).ok().and_then(PiiScoreKind::from_id);
            let kind = kind.ok_or(PolicyError::Unspecified { at, field: "score_at_least.kind", value: s.kind })?;
            if !s.min.is_finite() {
                return Err(invalid(at, "score_at_least.min is not finite"));
            }
            Condition::ScoreAtLeast { kind, min: s.min, calibrated: s.calibrated }
        }
    }))
}

fn labels(at: At, l: &pb::PiiLabelOverride) -> Result<LabelOverride, PolicyError> {
    let policy = match pb::PiiLabelRequirement::try_from(l.requirement) {
        Ok(pb::PiiLabelRequirement::None) => LabelPolicy::None,
        Ok(pb::PiiLabelRequirement::Required) => LabelPolicy::Required,
        Ok(pb::PiiLabelRequirement::Boosts) => LabelPolicy::Boosts,
        Ok(pb::PiiLabelRequirement::Unspecified) | Err(_) => {
            return Err(PolicyError::Unspecified { at, field: "labels.requirement", value: l.requirement });
        }
    };
    if l.extra_labels.iter().any(String::is_empty) {
        return Err(invalid(at, "labels.extra_labels has an empty label"));
    }
    let window_words = (l.window_words != 0).then_some(l.window_words as usize);
    Ok(LabelOverride { policy, window_words, extra_labels: l.extra_labels.clone() })
}

fn default_strategy(
    d: Option<&pb::PiiDefaultStrategy>,
    r: &dyn PolicyResolver,
) -> Result<RewriteStrategy, PolicyError> {
    let at = At::field("default_strategy");
    match d.map(|d| d.strategy.as_ref()) {
        None => Ok(whole_mask()),
        Some(None) => Err(PolicyError::UnknownArm { at, field: "default_strategy.strategy" }),
        Some(Some(pii_default_strategy::Strategy::Mask(m))) => mask(at, m),
        Some(Some(pii_default_strategy::Strategy::KeyedToken(k))) => keyed_token(at, k, r),
    }
}

struct Cx<'a> {
    kinds: &'a CustomKinds,
    default: &'a RewriteStrategy,
    resolver: &'a dyn PolicyResolver,
    flag_unknown: bool,
    flags: Vec<PolicyFlag>,
}

impl Cx<'_> {
    fn entry(&mut self, at: At, rule_id: &str, e: &pb::PiiStrategyEntry) -> Result<StrategyEntry, PolicyError> {
        let mut when = Vec::with_capacity(e.when.len());
        let mut unknown = false;
        for c in &e.when {
            match condition(at, c)? {
                Some(c) => when.push(c),
                None if self.flag_unknown => unknown = true,
                None => return Err(PolicyError::UnknownArm { at, field: "when.test" }),
            }
        }
        let s = match &e.strategy {
            None => None,
            Some(s) => strategy(at, s, self.default, self.resolver)?,
        };
        match s {
            Some(strategy) if !unknown => Ok(StrategyEntry { when, strategy }),
            _ if self.flag_unknown => {
                let entry = at.inner.map_or(0, |(_, i)| i);
                self.flags.push(PolicyFlag::UnknownStrategy { rule_id: rule_id.to_owned(), entry });
                Ok(StrategyEntry { when: Vec::new(), strategy: whole_mask() })
            }
            _ => Err(PolicyError::UnknownArm { at, field: "strategy" }),
        }
    }
}

impl RedactionPolicy {
    /// Converts `msg`, resolving its key and dictionary references with
    /// `resolver`, and computes its digest.
    pub fn from_proto(
        msg: &pb::RedactionPolicy,
        resolver: &dyn PolicyResolver,
    ) -> Result<RedactionPolicy, PolicyError> {
        if msg.id.is_empty() {
            return Err(PolicyError::Missing { at: At::field("id"), field: "id" });
        }
        let kinds = CustomKinds::from_names(&msg.custom_kinds).map_err(|e| PolicyError::CustomKinds(e.to_string()))?;
        let default = default_strategy(msg.default_strategy.as_ref(), resolver)?;
        let mut seen_ids = HashSet::new();
        let mut seen_kinds = HashSet::new();
        let mut cx = Cx {
            kinds: &kinds,
            default: &default,
            resolver,
            flag_unknown: msg.redact_and_flag_unknown,
            flags: Vec::new(),
        };
        let mut rules = Vec::with_capacity(msg.kind_rules.len());
        for (i, r) in msg.kind_rules.iter().enumerate() {
            let at = At::item("kind_rules", i);
            let id = rule_id(at, &r.rule_id, &mut seen_ids)?;
            let k = match kind(at, r.kind.as_ref(), cx.kinds)? {
                KindOf::Known(k) => k,
                KindOf::Unknown(id_num) if cx.flag_unknown => {
                    cx.flags.push(PolicyFlag::UnknownKind { rule_id: id, id: id_num });
                    continue;
                }
                KindOf::Unknown(id_num) => return Err(PolicyError::UnknownKind { at, id: id_num }),
            };
            if !seen_kinds.insert(k.code()) {
                return Err(PolicyError::DuplicateKind { at });
            }
            let enabled = match pb::PiiRuleState::try_from(r.state) {
                Ok(pb::PiiRuleState::Enabled) => true,
                Ok(pb::PiiRuleState::Disabled) => false,
                Ok(pb::PiiRuleState::Unspecified) | Err(_) => {
                    return Err(PolicyError::Unspecified { at, field: "state", value: r.state });
                }
            };
            let mut strategies = Vec::with_capacity(r.strategies.len());
            for (j, e) in r.strategies.iter().enumerate() {
                strategies.push(cx.entry(at.inner("strategies", j), &id, e)?);
            }
            let allow = allow(at, r.allow.as_ref(), resolver)?;
            let labels = r.labels.as_ref().map(|l| labels(at, l)).transpose()?;
            rules.push(KindRule { rule_id: id, kind: k, enabled, strategies, allow, labels });
        }
        let flags = std::mem::take(&mut cx.flags);
        let global_allow = allow(At::field("allow"), msg.allow.as_ref(), resolver)?;
        let mut deny = Vec::with_capacity(msg.deny.len());
        for (i, d) in msg.deny.iter().enumerate() {
            let at = At::item("deny", i);
            let id = rule_id(at, &d.rule_id, &mut seen_ids)?;
            let k = custom_kind(at, d.kind.as_ref(), &kinds)?;
            let mut list = DenyList::new(k).map_err(|e| invalid(at, e))?.with_case(case(d.case_sensitive));
            terms(at, d.terms.as_ref(), resolver, |t| list.add_term(t).map(|_| ()).map_err(|e| invalid(at, e)))?;
            let edits = u8::try_from(d.max_edits).unwrap_or(u8::MAX);
            let list =
                list.with_max_edits(edits).map_err(|e| invalid(at, e))?.with_capitalized_only(d.capitalized_only);
            deny.push(DenyRule { rule_id: id, list });
        }
        let mut custom_identifiers = Vec::with_capacity(msg.custom_identifiers.len());
        for (i, c) in msg.custom_identifiers.iter().enumerate() {
            let at = At::item("custom_identifiers", i);
            let id = rule_id(at, &c.rule_id, &mut seen_ids)?;
            let k = custom_kind(at, c.kind.as_ref(), &kinds)?;
            let pattern = Pattern::shape(&c.shape).map_err(|e| invalid(at, e))?.with_case(case(c.case_sensitive));
            custom_identifiers.push(CustomIdentifier { rule_id: id, kind: k, pattern });
        }
        let digest = PolicyDigest::of_canonical(&canonical::encode(msg));
        Ok(RedactionPolicy {
            id: msg.id.clone(),
            revision: msg.revision,
            digest,
            custom_kinds: kinds,
            rules,
            default_strategy: default,
            allow: global_allow,
            deny,
            custom_identifiers,
            flags,
        })
    }
}
