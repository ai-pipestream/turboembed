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
//! `rewritten_text`. Mention values are always withheld until #1909 gives
//! verified identities and grants. A document with no `turbo:pii` layer is
//! answered as before.
//!
//! In-process C, JNI and Panama callers hand the library their own text;
//! these controls govern what the server copies out, not protection from
//! a caller who already holds the input (docs/grpc-analysis.md, PII output).

use std::any::TypeId;

use turbo_analysis::{Analyzer, Document, Key, Layer, View, ViewId};
use turbo_analysis_layers::{Pii, PiiEdits};

use crate::api::{Failure, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED_OPTION};
use crate::proto::result::{PiiOutputOptions, PiiValue, PiiValueForm, PiiValueWithheld, TypedLayer, pii_value};

/// The annotator id prefix of the views a PII rewrite writes.
pub const REWRITE_ANNOTATOR_PREFIX: &str = "turbo-pii:";

/// What the server lets a request turn on (`--pii-output-allow`). Values
/// are not here: no policy allows them until #1909.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PiiOutputPolicy {
    pub rewritten_text: bool,
    pub full_text: bool,
    pub detail: bool,
}

impl PiiOutputPolicy {
    /// Every option but values: a caller of InspectDocumentGraph or
    /// RoundTripDocumentGraph sent the text it gets back.
    pub const ALL: PiiOutputPolicy = PiiOutputPolicy { rewritten_text: true, full_text: true, detail: true };
    /// Findings only.
    pub const NONE: PiiOutputPolicy = PiiOutputPolicy { rewritten_text: false, full_text: false, detail: false };

    /// One `--pii-output-allow` value: `none`, or a comma-separated list of
    /// `rewritten-text`, `full-text` and `detail`, each at most once.
    pub fn parse(s: &str) -> Result<PiiOutputPolicy, String> {
        Disclosure::parse(s).map(|d| PiiOutputPolicy {
            rewritten_text: d.rewritten_text,
            full_text: d.full_text,
            detail: d.detail,
        })
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
    /// Candidates and evidence detail; read by #1903's projection.
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
                        "pii_output.values {}: PII values are released only with verified grants (#1909), \
                         which this server does not have",
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
                "values" => return Err("PII values are released only with verified grants (#1909)".into()),
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
/// `turbo-pii:` annotator.
pub fn is_rewrite_view(doc: &Document, view: ViewId) -> bool {
    view != ViewId::SOURCE
        && doc.layers().any(|k| {
            k.ty == TypeId::of::<View>()
                && doc.get_key(&Key::<View>::named(k.name)).is_some_and(|v| v.id() == view)
                && doc.provenance(k).is_some_and(|p| p.annotator.starts_with(REWRITE_ANNOTATOR_PREFIX))
        })
}

/// The value every finding carries today.
pub fn withheld() -> PiiValue {
    PiiValue { form: Some(pii_value::Form::Withheld(PiiValueWithheld {})) }
}

/// Whether layer `id` is part of the findings: `turbo:pii`, and the
/// `turbo:pii-edits` of a rewrite (spans, covered mentions, rule and
/// strategy; never an original value).
pub fn is_findings_layer(id: &str) -> bool {
    id == Pii::ID || id == PiiEdits::ID
}

/// Findings only: every other layer carries the view's words (sentence
/// text, surfaces, lemmas, canonical names), so with the text withheld
/// those layers are listed with their status and no annotations.
pub(crate) fn withhold_annotations(layers: &mut [TypedLayer]) {
    for l in layers.iter_mut().filter(|l| !is_findings_layer(&l.id)) {
        l.annotations.clear();
        l.indexes_layer.clear();
    }
}

#[cfg(test)]
#[path = "disclosure_tests.rs"]
mod tests;
