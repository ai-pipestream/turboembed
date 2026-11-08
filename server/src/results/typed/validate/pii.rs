//! PII on the wire: the custom-kind table of a `turbo:pii` or
//! `turbo:pii-candidates` layer (or of the `turbo:pii-edits` layer that
//! covers one), and each mention's or candidate's kind
//! reference, detector, evidence facts, score and tier or disposition. A
//! kind is a built-in id [`PiiKind::from_id`] knows or an index into the
//! layer's `custom_pii_kinds`; a fact is one of the kinds the layer knows
//! (an unknown fact is refused, never skipped); the tier is the one the
//! facts derive to. `UNSPECIFIED` (0) is never a tier, an algorithm, a
//! score kind, a scope or a reason. A mention's value is `withheld`, a
//! keyed token or a clear value (#1904, #1909), and its retired
//! normalized form is unset. An error names the layer, the row and its wire spans, never the
//! mention's text, its normalized form, a custom kind name, a detector id,
//! or the refused number, which are caller input.

use std::fmt::Write as _;

use turbo_analysis::{Layer, Span};
use turbo_analysis_layers::{
    ChecksumAlgorithm, CustomKinds, KindError, Pii, PiiCandidates, PiiConfidence, PiiEdits, PiiEvidence, PiiKind,
    PiiRisk, PiiScoreKind, RejectReason,
};

use super::*;
use crate::proto::result::pii_kind_ref::Kind;
use crate::proto::result::pii_value::Form;
use crate::proto::result::{
    PiiCandidateAnnotation, PiiDetectorRef, PiiDisposition, PiiEvidence as WireEvidence, PiiMention, PiiScore,
    PiiWithheldReason, TypedLayer, pii_disposition, pii_evidence,
};

/// Whether `lid` is one of the four layers that carry a PII kind table.
fn is_pii_layer(lid: &str) -> bool {
    lid == Pii::ID || lid == PiiCandidates::ID || lid == PiiEdits::ID || lid == PiiRisk::ID
}

/// The number of custom kinds `layer` names: its `custom_pii_kinds`, which
/// only a PII layer may carry, held to [`CustomKinds::from_names`].
pub(super) fn custom_kinds(layer: &TypedLayer) -> Result<usize, Failure> {
    let lid = layer.id.as_str();
    let names = &layer.custom_pii_kinds;
    if !is_pii_layer(lid) {
        if !names.is_empty() {
            return Err(bad(format!(
                "layer {lid} carries custom_pii_kinds; only {}, {}, {} and {} may",
                Pii::ID,
                PiiCandidates::ID,
                PiiEdits::ID,
                PiiRisk::ID
            )));
        }
        return Ok(0);
    }
    CustomKinds::from_names(names).map_err(|e| bad(format!("layer {lid} custom_pii_kinds: {}", table_error(e))))?;
    Ok(names.len())
}

/// What is wrong with a custom-kind table, with no name in it.
fn table_error(e: KindError) -> String {
    match e {
        // The reserved kind is a built-in, never caller text.
        KindError::Reserved(_) => "a name is a built-in kind name".to_owned(),
        // Duplicate, too long, empty and full carry an index or a length.
        other => other.to_string(),
    }
}

/// One row of a PII layer: where an error points.
struct Row<'a> {
    ctx: &'a LayerCtx<'a>,
    i: usize,
    ann: &'a TypedAnnotation,
}

impl Row<'_> {
    fn at(&self) -> String {
        format!("{}[{}] at {}", self.ctx.lid, self.i, wire_spans(&self.ann.spans))
    }

    fn refuse(&self, what: &str) -> Failure {
        bad(format!("{} {what}", self.at()))
    }

    fn kind(&self, kind: Option<&crate::proto::result::PiiKindRef>) -> Result<(), Failure> {
        match kind.and_then(|r| r.kind) {
            None => Err(self.refuse("pii kind is missing")),
            Some(Kind::Builtin(id)) => {
                if u16::try_from(id).ok().and_then(PiiKind::from_id).is_none() {
                    return Err(self.refuse("pii kind id is not a built-in kind"));
                }
                Ok(())
            }
            Some(Kind::Custom(index)) => {
                let table = self.ctx.custom_pii_kinds;
                if usize::try_from(index).map_or(true, |n| n >= table) {
                    return Err(bad(format!(
                        "{} pii custom kind index is outside the layer's {table} custom_pii_kinds",
                        self.at()
                    )));
                }
                Ok(())
            }
        }
    }

    fn detector(&self, what: &str, detector: Option<&PiiDetectorRef>) -> Result<(), Failure> {
        match detector {
            None => Err(self.refuse(&format!("pii {what} is missing"))),
            Some(d) if d.id.is_empty() => Err(self.refuse(&format!("pii {what} id is empty"))),
            Some(_) => Ok(()),
        }
    }

    /// The facts as the layer records them, each one known; a label span
    /// is held to the text.
    fn evidence(&self, facts: &[WireEvidence]) -> Result<Vec<PiiEvidence>, Failure> {
        if facts.is_empty() {
            return Err(self.refuse("pii evidence is empty; every candidate records at least one fact"));
        }
        facts.iter().enumerate().map(|(n, f)| self.fact(n, f)).collect()
    }

    fn fact(&self, n: usize, f: &WireEvidence) -> Result<PiiEvidence, Failure> {
        let Some(fact) = &f.fact else {
            return Err(self.refuse(&format!("pii evidence[{n}] is an unknown or empty fact")));
        };
        Ok(match fact {
            pii_evidence::Fact::Shape(_) => PiiEvidence::Shape,
            pii_evidence::Fact::Checksum(c) => {
                let Some(algorithm) = u32::try_from(c.algorithm).ok().and_then(ChecksumAlgorithm::from_id) else {
                    return Err(self.refuse(&format!("pii evidence[{n}] checksum algorithm is not known")));
                };
                PiiEvidence::Checksum { algorithm, passed: c.passed }
            }
            pii_evidence::Fact::Label(l) => {
                let Some(s) = &l.span else {
                    return Err(self.refuse(&format!("pii evidence[{n}] label has no span")));
                };
                // One projected span under the outcome rules, like every
                // nested SpanFragment, before its offsets are read.
                let what = format!("pii evidence[{n}] label span");
                outcome::check_single(&self.at(), &what, s, self.ctx.own_view)?;
                let what = format!("{} {what}", self.at());
                let span = self.ctx.offsetter.from_wire(self.ctx.text, &what, s.start, s.end)?;
                // A label the target view deleted keeps an empty anchor, as
                // the mention does; the fact still counts toward the tier.
                PiiEvidence::Label { span: Span::new(span.start, span.end), phrase: l.phrase }
            }
            pii_evidence::Fact::DenyHit(d) => PiiEvidence::DenyHit { list: d.list, term: d.term },
            pii_evidence::Fact::Rule(r) => PiiEvidence::Rule { rule: r.rule },
        })
    }

    fn score(&self, score: Option<&PiiScore>) -> Result<(), Failure> {
        let Some(s) = score else { return Ok(()) };
        let Some(kind) = u32::try_from(s.kind).ok().and_then(PiiScoreKind::from_id) else {
            return Err(self.refuse("pii score kind is not known"));
        };
        if !s.value.is_finite() {
            return Err(self.refuse("pii score value is not finite"));
        }
        if kind != PiiScoreKind::Logit && !(0.0..=1.0).contains(&s.value) {
            return Err(self.refuse("pii score value is outside [0.0, 1.0] for its kind"));
        }
        self.detector("score model", s.model.as_ref())
    }
}

// Reads the retired string fields only to refuse them.
#[allow(deprecated)]
pub(super) fn mention(ctx: &LayerCtx<'_>, i: usize, ann: &TypedAnnotation, m: &PiiMention) -> Result<(), Failure> {
    let row = Row { ctx, i, ann };
    if ctx.lid != Pii::ID {
        return Err(bad(format!("{} is a pii mention outside a {} layer", row.at(), Pii::ID)));
    }
    if ann.spans.is_empty() {
        return Err(row.refuse("pii mention has no span"));
    }
    if !m.kind.is_empty() || !m.confidence_tier.is_empty() {
        return Err(row.refuse("pii mention sets a retired string kind or confidence_tier"));
    }
    // The clear value is never on the wire (#1904): `value` says how, if at
    // all, it was released, and only `withheld` is produced.
    if !m.normalized.is_empty() {
        return Err(row.refuse("pii mention sets the retired normalized value; use value"));
    }
    // Every mention says how its value was released (#1909): withheld
    // with a known reason (#1931), a
    // token of 32 bytes naming its key id and version, or a clear value
    // that is not empty. A ciphertext waits for #1885. The error names the
    // form, never its contents.
    let refused = match m.value.as_ref().map(|v| v.form.as_ref()) {
        Some(Some(Form::Withheld(w))) => match PiiWithheldReason::try_from(w.reason) {
            Ok(PiiWithheldReason::Unspecified) | Err(_) => Some("is withheld without a known reason"),
            Ok(_) => None,
        },
        None => Some("is missing"),
        Some(None) => Some("has no form"),
        Some(Some(Form::Clear(v))) => v.is_empty().then_some("is an empty clear value"),
        Some(Some(Form::Token(t))) if t.token.len() != 32 => Some("is a token that is not 32 bytes"),
        Some(Some(Form::Token(t))) if t.key_id.is_empty() || t.key_version == 0 => {
            Some("is a token without a key id and version")
        }
        Some(Some(Form::Token(_))) => None,
        Some(Some(Form::Ciphertext(_))) => Some("is a ciphertext, which waits for #1885"),
    };
    if let Some(why) = refused {
        return Err(row.refuse(&format!("pii mention value {why}")));
    }
    row.kind(m.kind_ref.as_ref())?;
    row.detector("detector", m.detector.as_ref())?;
    let facts = row.evidence(&m.evidence)?;
    row.score(m.score.as_ref())?;
    let Some(tier) = u32::try_from(m.confidence).ok().and_then(PiiConfidence::from_id) else {
        return Err(row.refuse("pii confidence is not a known tier"));
    };
    let derived = PiiConfidence::derive(&facts);
    if tier != derived {
        return Err(bad(format!(
            "{} pii confidence is {} but its evidence derives to {}",
            row.at(),
            tier.name(),
            derived.name()
        )));
    }
    Ok(())
}

pub(super) fn candidate(
    ctx: &LayerCtx<'_>,
    i: usize,
    ann: &TypedAnnotation,
    c: &PiiCandidateAnnotation,
) -> Result<(), Failure> {
    let row = Row { ctx, i, ann };
    if ctx.lid != PiiCandidates::ID {
        return Err(bad(format!("{} is a pii candidate outside a {} layer", row.at(), PiiCandidates::ID)));
    }
    if ann.spans.is_empty() {
        return Err(row.refuse("pii candidate has no span"));
    }
    row.kind(c.kind.as_ref())?;
    row.detector("detector", c.detector.as_ref())?;
    row.evidence(&c.evidence)?;
    row.score(c.score.as_ref())?;
    disposition(&row, c.disposition.as_ref())?;
    planned(&row, c)
}

/// A dry run's planned rewrite: only on a kept candidate, with a known
/// rule and strategy and the policy's digest.
fn planned(row: &Row<'_>, c: &PiiCandidateAnnotation) -> Result<(), Failure> {
    let Some(p) = &c.planned else { return Ok(()) };
    let kept = matches!(
        c.disposition.as_ref().and_then(|d| d.disposition.as_ref()),
        Some(pii_disposition::Disposition::Kept(_))
    );
    if !kept {
        return Err(row.refuse("pii candidate is planned but not kept"));
    }
    if p.rule.as_ref().and_then(|r| r.rule.as_ref()).is_none() {
        return Err(row.refuse("pii candidate planned rule is missing"));
    }
    if u8::try_from(p.strategy).ok().and_then(turbo_analysis_layers::PiiStrategyKind::from_id).is_none() {
        return Err(row.refuse("pii candidate planned strategy is not a known strategy"));
    }
    super::pii_edits::policy_ref(&p.rule_id, &p.policy_digest, true)
        .map_err(|what| bad(format!("{} pii candidate planned {what}", row.at())))
}

/// A disposition that is set and known; a mention index is a row of the
/// `turbo:pii` layer with this layer's instance name, when that layer is
/// in the message.
fn disposition(row: &Row<'_>, d: Option<&PiiDisposition>) -> Result<(), Failure> {
    let Some(d) = d.and_then(|d| d.disposition.as_ref()) else {
        return Err(row.refuse("pii candidate disposition is missing or unknown"));
    };
    let mention = |index: u32| -> Result<(), Failure> {
        let pii = row.ctx.layer_lengths.get(&(Pii::ID, row.ctx.instance)).copied();
        match pii {
            Some(len) if (index as usize) < len => Ok(()),
            Some(len) => Err(bad(format!("{} pii candidate mention index is outside the {len} mentions", row.at()))),
            None => Err(row.refuse("pii candidate refers to a mention but the message has no pii layer")),
        }
    };
    match d {
        pii_disposition::Disposition::Kept(m)
        | pii_disposition::Disposition::NestedIn(m)
        | pii_disposition::Disposition::Repeats(m) => mention(*m),
        pii_disposition::Disposition::AllowedByRule(a) => {
            if !(1..=2).contains(&a.scope) {
                return Err(row.refuse("pii candidate allow scope is not known"));
            }
            if a.rule.as_ref().and_then(|r| r.rule).is_none() {
                return Err(row.refuse("pii candidate allow rule is missing or unknown"));
            }
            Ok(())
        }
        pii_disposition::Disposition::RejectedByRule(reason) => {
            if u32::try_from(*reason).ok().and_then(RejectReason::from_id).is_none() {
                return Err(row.refuse("pii candidate reject reason is not known"));
            }
            Ok(())
        }
    }
}

/// The wire spans as `start..end`, comma separated, or `no span`.
fn wire_spans(spans: &[SpanFragment]) -> String {
    if spans.is_empty() {
        return "no span".to_owned();
    }
    let mut out = String::new();
    for (n, s) in spans.iter().enumerate() {
        let sep = if n == 0 { "" } else { ", " };
        let _ = write!(out, "{sep}{}..{}", s.start, s.end);
    }
    out
}
