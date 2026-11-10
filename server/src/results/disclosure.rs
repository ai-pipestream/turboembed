//! The one PII disclosure decision (#1904). Detection identifies PII;
//! what a response carries beyond the findings is an output option the
//! request asks for and the server's policy caps. Every surface that can
//! carry a detected value asks this module: the typed projection
//! (InspectDocumentGraph), DocumentGraphResponse (RoundTripDocumentGraph
//! and the progressive answer), progressive step snapshots and sink items.
//!
//! The rule applies to a document holding PII findings, one with a
//! `turbo:pii` layer instance. Its graph bytes and its source text carry
//! every detected value in clear (the layer codec writes the normalized
//! column, and the text holds the values themselves), so they leave only
//! with `full_text`. A PII rewrite view, one a `turbo-pii:` annotator wrote
//! (the masked view, and the rewrite views of #1902), leaves with
//! `rewritten_text`. Mention values are withheld here; only
//! InspectDocumentGraph releases them, through `crate::release` (#1909),
//! to verified services the server grants them to. A document with no
//! `turbo:pii` layer is answered as before.
//!
//! In-process C, JNI and Panama callers hand the library their own text;
//! these controls govern what the server copies out, not protection from
//! a caller who already holds the input (docs/grpc-analysis.md, PII output).

use std::any::TypeId;

use turbo_analysis::{Analyzer, Document, Evidence, Key, KeyId, Layer, View, ViewId};
use turbo_analysis_layers::{Entities, Pii, PiiCandidates, PiiEdits, PiiRisk};
use turbo_entity_linker::LinkedEntities;

use crate::api::{Failure, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED_OPTION};
use crate::proto::result::{
    PiiOutputOptions, PiiValue, PiiValueForm, PiiValueWithheld, PiiWithheldReason, TypedLayer, pii_value,
};
use crate::release::caps::{Form, Forms};

/// The annotator id prefix of the views a PII rewrite writes.
pub const REWRITE_ANNOTATOR_PREFIX: &str = "turbo-pii:";

/// What the server lets a request turn on (`--pii-output-allow`).
/// `values` are the forms a release may use; a grant (#1909) must name
/// the form too, and none is allowed unless listed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PiiOutputPolicy {
    pub rewritten_text: bool,
    pub full_text: bool,
    pub detail: bool,
    pub values: Forms,
}

impl PiiOutputPolicy {
    /// Every option but values: a caller of InspectDocumentGraph or
    /// RoundTripDocumentGraph sent the text it gets back.
    pub const ALL: PiiOutputPolicy =
        PiiOutputPolicy { rewritten_text: true, full_text: true, detail: true, values: Forms::NONE };
    /// Findings only.
    pub const NONE: PiiOutputPolicy =
        PiiOutputPolicy { rewritten_text: false, full_text: false, detail: false, values: Forms::NONE };

    /// One `--pii-output-allow` value: `none`, or a comma-separated list of
    /// `rewritten-text`, `full-text`, `detail`, and the value forms
    /// `token`, `ciphertext` and `clear`, each at most once.
    pub fn parse(s: &str) -> Result<PiiOutputPolicy, String> {
        let mut values = Forms::NONE;
        let mut rest = Vec::new();
        if s != "none" {
            for part in s.split(',') {
                match Form::parse(part) {
                    Some(f) if values.contains(f) => return Err(format!("`{part}` given twice")),
                    Some(f) => values = values.with(f),
                    None => rest.push(part),
                }
            }
        }
        let d = if rest.is_empty() { Disclosure::FINDINGS } else { Disclosure::parse(&rest.join(","))? };
        Ok(PiiOutputPolicy { rewritten_text: d.rewritten_text, full_text: d.full_text, detail: d.detail, values })
    }

    /// Whether `d` asks for nothing this policy refuses.
    pub fn allows(&self, d: &Disclosure) -> bool {
        (self.rewritten_text || !d.rewritten_text) && (self.full_text || !d.full_text) && (self.detail || !d.detail)
    }
}

impl Default for PiiOutputPolicy {
    fn default() -> PiiOutputPolicy {
        PiiOutputPolicy::ALL
    }
}

/// What one response about a PII-bearing document carries beyond its
/// findings. `Default` is findings only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Disclosure {
    pub rewritten_text: bool,
    pub full_text: bool,
    /// The candidates layer (#1903) in a findings-only answer.
    pub detail: bool,
}

impl Disclosure {
    /// Findings only: kinds, confidences, spans and provenance.
    pub const FINDINGS: Disclosure = Disclosure { rewritten_text: false, full_text: false, detail: false };
    /// Everything but values.
    pub const FULL: Disclosure = Disclosure { rewritten_text: true, full_text: true, detail: true };

    /// The decision for a request's options under `policy`: an option the
    /// policy does not allow, or any value form but withheld, refuses the
    /// call naming it. Unset options are findings only.
    pub fn decide(options: Option<&PiiOutputOptions>, policy: PiiOutputPolicy) -> Result<Disclosure, Failure> {
        let Some(o) = options else { return Ok(Disclosure::FINDINGS) };
        match PiiValueForm::try_from(o.values) {
            Ok(PiiValueForm::Unspecified | PiiValueForm::Withheld) => {}
            Ok(form) => {
                return Err(Failure::new(
                    TURBO_E_UNSUPPORTED_OPTION,
                    0,
                    format!(
                        "pii_output.values {}: PII values are released only by InspectDocumentGraph \
                         on a server with PII release configured (#1909)",
                        form.as_str_name()
                    ),
                ));
            }
            Err(_) => {
                return Err(Failure::new(
                    TURBO_E_UNSUPPORTED_OPTION,
                    0,
                    format!("pii_output.values {} is not a PiiValueForm", o.values),
                ));
            }
        }
        let d = Disclosure { rewritten_text: o.rewritten_text, full_text: o.full_text, detail: o.detail };
        for (asked, allowed, name) in [
            (d.rewritten_text, policy.rewritten_text, "rewritten_text"),
            (d.full_text, policy.full_text, "full_text"),
            (d.detail, policy.detail, "detail"),
        ] {
            if asked && !allowed {
                return Err(Failure::new(
                    TURBO_E_INVALID_STATE,
                    0,
                    format!("pii_output.{name} is not allowed by this server's PII output policy"),
                ));
            }
        }
        Ok(d)
    }

    /// A `none` or comma-separated option list, as the server flags spell
    /// it.
    pub fn parse(s: &str) -> Result<Disclosure, String> {
        let mut d = Disclosure::FINDINGS;
        if s == "none" {
            return Ok(d);
        }
        for part in s.split(',') {
            let slot = match part {
                "rewritten-text" => &mut d.rewritten_text,
                "full-text" => &mut d.full_text,
                "detail" => &mut d.detail,
                "values" => {
                    return Err(
                        "PII values are released only by InspectDocumentGraph to granted services (#1909)".into()
                    );
                }
                _ => return Err(format!("`{part}` is not none, rewritten-text, full-text or detail")),
            };
            if std::mem::replace(slot, true) {
                return Err(format!("`{part}` given twice"));
            }
        }
        Ok(d)
    }

    /// The request options that ask for exactly this disclosure.
    pub fn to_options(&self) -> PiiOutputOptions {
        PiiOutputOptions {
            rewritten_text: self.rewritten_text,
            full_text: self.full_text,
            detail: self.detail,
            values: PiiValueForm::Unspecified as i32,
        }
    }

    /// Whether the graph bytes and source text of `doc` may leave.
    pub fn graph(&self, doc: &Document) -> bool {
        self.full_text || !holds_pii(doc)
    }

    /// Whether the text of `view` on `doc` may leave.
    pub fn view_text(&self, doc: &Document, view: ViewId) -> bool {
        self.graph(doc) || (self.rewritten_text && is_rewrite_view(doc, view))
    }

    /// Refuses `rpc` on `doc` unless the text of `view` may leave: a
    /// surface that answers with view text (a window, an x-ray, chunk
    /// previews) has nothing to answer without it (#1904).
    pub fn require_view_text(&self, doc: &Document, view: ViewId, rpc: &str) -> Result<(), Failure> {
        if self.view_text(doc, view) {
            return Ok(());
        }
        Err(Failure::new(
            TURBO_E_INVALID_STATE,
            0,
            format!(
                "{rpc}: the document holds PII and the answer carries the text of view {}; ask \
                 pii_output.full_text, or pii_output.rewritten_text for a PII rewrite view",
                view.0
            ),
        ))
    }
}

/// Whether `doc` holds PII findings: a `turbo:pii` layer instance, in any
/// status.
pub fn holds_pii(doc: &Document) -> bool {
    doc.layers().any(|k| k.ty == TypeId::of::<Pii>())
}

/// Whether a step of `analyzer` writes PII findings: a document it runs
/// over will hold a `turbo:pii` layer instance.
pub fn plans_pii(analyzer: &Analyzer) -> bool {
    analyzer.writes().iter().any(|k| k.ty == TypeId::of::<Pii>())
}

/// Whether `view` is a PII rewrite view: not the source, written by a
/// `turbo-pii:` annotator, that rewrote every finding the document holds
/// (#1932). A `turbo:pii` instance with a mention, or without a layer to
/// say it has none, that neither the view nor a rewrite view it was made
/// from read (directly or as the source of a projected copy) may hold a
/// value the view carries in clear, so that view is not one. Every
/// `turbo:pii` layer counts, whoever wrote it: no rewrite step writes one.
pub fn is_rewrite_view(doc: &Document, view: ViewId) -> bool {
    view != ViewId::SOURCE
        && doc.layers().any(|k| {
            k.ty == TypeId::of::<View>()
                && doc.get_key(&Key::<View>::named(k.name)).is_some_and(|v| v.id() == view)
                && doc.provenance(k).is_some_and(|p| p.annotator.starts_with(REWRITE_ANNOTATOR_PREFIX))
                && rewrote_every_finding(doc, &lineage::rewritten_findings(doc, k))
        })
}

/// Whether every `turbo:pii` finding instance of `doc` is in `covered` or
/// holds no mention.
fn rewrote_every_finding(doc: &Document, covered: &[KeyId]) -> bool {
    doc.layers().filter(|k| k.ty == TypeId::of::<Pii>()).all(|k| {
        covered.contains(&k) || doc.get_key(&Key::<Pii>::named(k.name)).is_some_and(|p| p.mentions().next().is_none())
    })
}

/// The value of a finding when the request asks for none: withheld,
/// `NOT_REQUESTED`.
pub fn withheld() -> PiiValue {
    withheld_because(PiiWithheldReason::NotRequested)
}

/// A withheld value with its `reason` (#1931).
pub fn withheld_because(reason: PiiWithheldReason) -> PiiValue {
    PiiValue { form: Some(pii_value::Form::Withheld(PiiValueWithheld { reason: reason as i32 })) }
}

/// Whether layer `id` may carry annotations in a findings-only result:
/// `turbo:pii`; the `turbo:pii-edits` of a rewrite (spans, covered
/// mentions, rule and strategy); and, when the request asks for `detail`,
/// `turbo:pii-candidates` (every span a detector considered, with its
/// facts and the decision taken); and `turbo:pii-risk`, the document's
/// score, kinds and counts (#1889). None of them carries a value.
pub fn is_findings_layer(id: &str) -> bool {
    id == Pii::ID || id == PiiEdits::ID || id == PiiCandidates::ID || id == PiiRisk::ID
}

/// Findings only: every other layer carries the view's words (sentence
/// text, surfaces, lemmas, canonical names), so with the text withheld
/// those layers are listed with their status and no annotations. The
/// candidates are detail (#1903): listed bare unless `detail` is asked.
pub(crate) fn withhold_annotations(layers: &mut [TypedLayer], detail: bool) {
    let kept =
        |id: &str| id == Pii::ID || id == PiiEdits::ID || id == PiiRisk::ID || (detail && id == PiiCandidates::ID);
    for l in layers.iter_mut().filter(|l| !kept(&l.id)) {
        l.annotations.clear();
        l.indexes_layer.clear();
    }
}

/// A result shown on a PII rewrite view under `rewritten_text` alone: the
/// view's own text may leave, but a layer made from another view's text
/// (the source's lemmas, entity names, key-value strings, or a projected
/// copy of them that sits on the rewrite view) carries that view's words,
/// so it is listed bare unless its whole lineage may leave
/// ([`lineage::Lineage`]). An entity layer is judged with the linked
/// names and evidence joined into it. The findings layers are kept, the
/// candidates only with `detail` (#1904, #1932).
pub(crate) fn withhold_other_views(layers: &mut [TypedLayer], doc: &Document, disclosure: &Disclosure) {
    let key_of = |l: &TypedLayer| doc.layers().find(|k| k.layer.to_string() == l.id && k.name == l.instance);
    let own: Vec<(KeyId, Option<ViewId>)> = layers
        .iter()
        .filter_map(|l| key_of(l).map(|k| (k, l.native_view.and_then(|v| u16::try_from(v).ok()).map(ViewId))))
        .collect();
    let mut lineage =
        lineage::Lineage::new(doc, disclosure, |k| own.iter().find(|(o, _)| *o == k).and_then(|(_, v)| *v));
    let kept = |id: &str| {
        id == Pii::ID || id == PiiEdits::ID || id == PiiRisk::ID || (disclosure.detail && id == PiiCandidates::ID)
    };
    for l in layers.iter_mut().filter(|l| !kept(&l.id)) {
        let withheld = l.id == PiiCandidates::ID
            || key_of(l).is_none_or(|k| lineage.withheld(k) || joined(doc, k).any(|j| lineage.withheld(j)));
        if withheld {
            l.annotations.clear();
            l.indexes_layer.clear();
        }
    }
}

/// The side tables the typed projection joins into the layer `key`: the
/// linked names and evidence aligned to an entity layer.
fn joined(doc: &Document, key: KeyId) -> impl Iterator<Item = KeyId> + '_ {
    let side = [TypeId::of::<LinkedEntities>(), TypeId::of::<Evidence<Entities>>()];
    doc.layers().filter(move |k| side.contains(&k.ty) && doc.made_on(*k).first() == Some(&key))
}

#[path = "disclosure_lineage.rs"]
mod lineage;

#[cfg(test)]
#[path = "disclosure_tests.rs"]
mod tests;
