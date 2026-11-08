//! The `turbo:pii` and `turbo:pii-candidates` projections (#1903): each
//! mention with its detector, its evidence facts and its optional score,
//! and each candidate with the decision taken on it. The confidence tier
//! on the wire is derived from the facts. A candidate carries no value.

use std::any::TypeId;

use turbo_analysis::{Ix, KeyId, Layer, Span, ViewId};
use turbo_analysis_layers::{
    AllowRule, AllowScope, ChecksumAlgorithm, Pii, PiiCandidates, PiiDetectorRef, PiiDetectors, PiiDisposition,
    PiiEvidence, PiiKind, PiiScore, PiiScoreKind,
};

use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{
    PiiAllowRuleRef, PiiAllowedByRule, PiiCandidateAnnotation, PiiChecksumEvidence, PiiDenyHitEvidence,
    PiiDetectorRef as WireDetector, PiiDisposition as WireDisposition, PiiEvidence as WireEvidence, PiiKindRef,
    PiiLabelEvidence, PiiMention, PiiRuleEvidence, PiiScore as WireScore, PiiShapeEvidence, TypedAnnotation,
    TypedLayer, pii_allow_rule_ref, pii_disposition, pii_evidence, pii_kind_ref,
};

/// The wire reference of `kind`: its stable built-in id (the `PiiKindId`
/// value), or its index in the layer's `custom_pii_kinds`.
pub(crate) fn pii_kind_ref(kind: PiiKind) -> PiiKindRef {
    let kind = match kind {
        PiiKind::Custom(c) => pii_kind_ref::Kind::Custom(u32::from(c.index())),
        builtin => pii_kind_ref::Kind::Builtin(i32::from(builtin.builtin_id().unwrap_or(0))),
    };
    PiiKindRef { kind: Some(kind) }
}

/// The wire form of detector `r` of `table`; a reference outside the table
/// (which `Pii::check` refuses) becomes an empty id, which the validator
/// refuses in turn.
fn wire_detector(table: &PiiDetectors, r: PiiDetectorRef) -> WireDetector {
    let (id, version) = table.get(r).unwrap_or(("", ""));
    WireDetector { id: id.to_owned(), version: version.to_owned() }
}

/// The wire number of `algorithm`: the same number the binary codec uses.
fn wire_algorithm(algorithm: ChecksumAlgorithm) -> i32 {
    i32::from(algorithm.id())
}

/// The wire form of one fact, with a label span projected onto the target
/// view.
fn wire_fact(cx: &ProjectionCx<'_>, view: ViewId, fact: &PiiEvidence) -> WireEvidence {
    let fact = match *fact {
        PiiEvidence::Shape => pii_evidence::Fact::Shape(PiiShapeEvidence {}),
        PiiEvidence::Checksum { algorithm, passed } => {
            pii_evidence::Fact::Checksum(PiiChecksumEvidence { algorithm: wire_algorithm(algorithm), passed })
        }
        PiiEvidence::Label { span, phrase } => {
            let span = Some(cx.project_wire_span(view, Span::new(span.start, span.end)));
            pii_evidence::Fact::Label(PiiLabelEvidence { span, phrase })
        }
        PiiEvidence::DenyHit { list, term } => pii_evidence::Fact::DenyHit(PiiDenyHitEvidence { list, term }),
        PiiEvidence::Rule { rule } => pii_evidence::Fact::Rule(PiiRuleEvidence { rule }),
    };
    WireEvidence { fact: Some(fact) }
}

fn wire_evidence(cx: &ProjectionCx<'_>, view: ViewId, facts: &[PiiEvidence]) -> Vec<WireEvidence> {
    facts.iter().map(|f| wire_fact(cx, view, f)).collect()
}

/// The wire number of a score kind: the same number the binary codec uses.
fn wire_score_kind(kind: PiiScoreKind) -> i32 {
    i32::from(kind.id())
}

fn wire_score(table: &PiiDetectors, score: Option<PiiScore>) -> Option<WireScore> {
    score.map(|s| WireScore {
        value: s.value,
        kind: wire_score_kind(s.kind),
        model: Some(wire_detector(table, s.model)),
        calibrated: s.calibrated,
    })
}

fn wire_allow_rule(rule: AllowRule) -> PiiAllowRuleRef {
    let rule = match rule {
        AllowRule::Term(i) => pii_allow_rule_ref::Rule::Term(i),
        AllowRule::Pattern(i) => pii_allow_rule_ref::Rule::Pattern(i),
    };
    PiiAllowRuleRef { rule: Some(rule) }
}

fn wire_disposition(d: PiiDisposition) -> WireDisposition {
    let disposition = match d {
        PiiDisposition::Kept { mention } => pii_disposition::Disposition::Kept(mention.get() as u32),
        PiiDisposition::AllowedByRule { scope, rule } => {
            let scope = match scope {
                AllowScope::Global => 1,
                AllowScope::Kind => 2,
            };
            pii_disposition::Disposition::AllowedByRule(PiiAllowedByRule { scope, rule: Some(wire_allow_rule(rule)) })
        }
        PiiDisposition::NestedIn { mention } => pii_disposition::Disposition::NestedIn(mention.get() as u32),
        PiiDisposition::RejectedByRule { reason } => {
            pii_disposition::Disposition::RejectedByRule(i32::from(reason.id()))
        }
        PiiDisposition::Repeats { mention } => pii_disposition::Disposition::Repeats(mention.get() as u32),
    };
    WireDisposition { disposition: Some(disposition) }
}

pub(crate) fn project_pii_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    let ty = key.ty;
    if ty == TypeId::of::<Pii>() {
        let mut layer = build_layer::<Pii>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let m = l.get(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::Pii(PiiMention {
                            kind_ref: Some(pii_kind_ref(m.kind)),
                            confidence: i32::from(m.confidence().id()),
                            normalized: m.normalized.to_owned(),
                            detector: Some(wire_detector(&l.detectors, m.detector)),
                            evidence: wire_evidence(cx, view, m.evidence),
                            score: wire_score(&l.detectors, m.score),
                            ..PiiMention::default()
                        })),
                    }
                })
                .collect()
        });
        if let Some(pii) = cx.doc.get_key(&turbo_analysis::Key::<Pii>::named(key.name)) {
            layer.custom_pii_kinds = pii.custom_kinds.names().map(str::to_owned).collect();
        }
        return Some(layer);
    }
    if ty == TypeId::of::<PiiCandidates>() {
        let mut layer = build_layer::<PiiCandidates>(cx, key, |cx, l, mref| {
            let mref = mref.map(ToOwned::to_owned);
            let view = l.spans.view();
            (0..l.len())
                .map(|i| {
                    let ix = Ix::new(i as u32);
                    let c = l.get(ix);
                    TypedAnnotation {
                        spans: cx.project_spans(&l.spans, ix),
                        probability: None,
                        model_ref: mref.clone(),
                        value: Some(Value::PiiCandidate(PiiCandidateAnnotation {
                            kind: Some(pii_kind_ref(c.kind)),
                            detector: Some(wire_detector(&l.detectors, c.detector)),
                            evidence: wire_evidence(cx, view, c.evidence),
                            score: wire_score(&l.detectors, c.score),
                            disposition: Some(wire_disposition(c.disposition)),
                        })),
                    }
                })
                .collect()
        });
        if let Some(c) = cx.doc.get_key(&turbo_analysis::Key::<PiiCandidates>::named(key.name)) {
            layer.custom_pii_kinds = c.custom_kinds.names().map(str::to_owned).collect();
        }
        return Some(layer);
    }
    None
}
