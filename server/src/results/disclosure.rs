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

use turbo_analysis::export::{ErrorRecord, StatusRecord};
use turbo_analysis::{Document, Error, Key, Layer, Status, View, ViewId};
use turbo_analysis_layers::Pii;

use crate::api::{Failure, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED_OPTION};
use crate::proto::result::{PiiOutputOptions, PiiValue, PiiValueForm, PiiValueWithheld, TypedLayer, pii_value};
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
    /// `token` and `clear`, each at most once. `ciphertext` waits for
    /// #1885.
    pub fn parse(s: &str) -> Result<PiiOutputPolicy, String> {
        let mut values = Forms::NONE;
        let mut rest = Vec::new();
        if s != "none" {
            for part in s.split(',') {
                match Form::parse(part) {
                    Some(Form::Ciphertext) => return Err("ciphertext release waits for #1885".into()),
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
            ..PiiOutputOptions::default()
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

    /// Whether status details of `doc` must be cut to identities: free
    /// text in an error may quote the document.
    pub fn sanitize(&self, doc: &Document) -> bool {
        !self.graph(doc)
    }
}

/// Whether `doc` holds PII findings: a `turbo:pii` layer instance, in any
/// status.
pub fn holds_pii(doc: &Document) -> bool {
    doc.layers().any(|k| k.ty == TypeId::of::<Pii>())
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

/// Findings only: every layer but `turbo:pii` carries the view's words
/// (sentence text, surfaces, lemmas, canonical names), so with the text
/// withheld those layers are listed with their status and no annotations.
pub(crate) fn withhold_annotations(layers: &mut [TypedLayer]) {
    for l in layers.iter_mut().filter(|l| l.id != Pii::ID) {
        l.annotations.clear();
        l.indexes_layer.clear();
    }
}

/// A status detail cut to what kind of failure it is and which layer or
/// annotator it names, without the free text an annotator wrote.
pub fn sanitized_status(status: &Status) -> String {
    match status {
        Status::Complete => String::new(),
        Status::Skipped(_) => "skipped".into(),
        Status::Failed(e) => match e {
            Error::Missing(k) => format!("missing {k}"),
            Error::Duplicate(k) => format!("duplicate {k}"),
            Error::Clash(k) => format!("clash {k}"),
            Error::Bounds { layer, .. } => format!("bounds {layer}"),
            Error::Misaligned { layer, .. } => format!("misaligned {layer}"),
            Error::Unaligned { layer, .. } => format!("unaligned {layer}"),
            Error::Invalid { layer, .. } => format!("invalid {layer}"),
            Error::Pipeline(_) => "pipeline".into(),
            Error::Annotator { annotator, .. } => format!("annotator {annotator}"),
            Error::Skipped(_) => "skipped".into(),
            Error::Offset { unit, at, .. } => format!("offset {unit} {at}"),
            Error::Source(_) => "source".into(),
        },
    }
}

/// [`sanitized_status`] for an exported status record.
pub fn sanitized_record(status: &StatusRecord) -> String {
    match status {
        StatusRecord::Complete => String::new(),
        StatusRecord::Skipped(_) => "skipped".into(),
        StatusRecord::Failed(e) => match e {
            ErrorRecord::Missing(k) => format!("missing {k}"),
            ErrorRecord::Duplicate(k) => format!("duplicate {k}"),
            ErrorRecord::Clash(k) => format!("clash {k}"),
            ErrorRecord::Bounds { layer, .. } => format!("bounds {layer}"),
            ErrorRecord::Misaligned { layer, .. } => format!("misaligned {layer}"),
            ErrorRecord::Unaligned { layer, .. } => format!("unaligned {layer}"),
            ErrorRecord::Invalid { layer, .. } => format!("invalid {layer}"),
            ErrorRecord::Pipeline(_) => "pipeline".into(),
            ErrorRecord::Annotator { annotator, .. } => format!("annotator {annotator}"),
            ErrorRecord::Skipped(_) => "skipped".into(),
            ErrorRecord::Offset { unit, at, .. } => format!("offset {unit} {at}"),
            ErrorRecord::Source(_) => "source".into(),
        },
    }
}

#[cfg(test)]
#[path = "disclosure_tests.rs"]
mod tests;
