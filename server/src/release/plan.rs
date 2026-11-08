//! Which mentions of one InspectDocumentGraph answer receive a value, and
//! writing those values into their `PiiValue` once the audit service has
//! acknowledged the release (#1909). The projection is #1904's, with every
//! value withheld; this module only reads the layer's normalized column
//! for the mentions the effective forms allow.

use std::any::TypeId;

use turbo_analysis::Document;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Ix, Key, Layer};
use turbo_analysis_layers::{Pii, PiiEdits, PiiKind, Sensitive};

use super::caps::{Form, Forms, Grant, effective};
use super::keys::TokenKey;
use crate::api::{Failure, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT};
use crate::proto::audit::PiiAccessMention;
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{InspectDocumentGraphRequest, PiiValue, TypedDocumentResult, pii_value};
use crate::results::disclosure::PiiOutputPolicy;
use crate::results::document::codecs;
use crate::results::typed::inspect_document_graph_under;

/// One mention that receives a value.
#[derive(Debug)]
pub struct Planned {
    /// Its layer's index in `TypedDocumentResult.layers` and its index in
    /// that layer's annotations.
    pub layer: usize,
    pub index: usize,
    pub kind: PiiKind,
    /// The custom kind's name; empty for a built-in kind.
    pub custom: String,
    pub form: Form,
    /// The normalized value. Its `Debug` prints `<withheld>`.
    pub value: Sensitive<String>,
    /// The redaction policy of the edit that covers the mention, if any.
    pub cover: Option<Cover>,
}

/// The redaction policy fields of the `turbo:pii-edits` edit whose
/// members include a mention (#1882 part C), copied as the edit names
/// them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cover {
    pub rule_id: String,
    pub policy_revision: u64,
    pub policy_digest: [u8; 32],
}

impl Planned {
    pub fn audit(&self, typed: &TypedDocumentResult) -> PiiAccessMention {
        PiiAccessMention {
            layer_instance: typed.layers[self.layer].instance.clone(),
            index: self.index as u32,
            builtin_kind: self.kind.builtin_id().map_or(0, u32::from),
            custom_kind: self.custom.clone(),
            form: self.form.audit() as i32,
            rule_id: self.cover.as_ref().map(|c| c.rule_id.clone()).unwrap_or_default(),
            policy_revision: self.cover.as_ref().map_or(0, |c| c.policy_revision),
            redaction_policy_digest: self.cover.as_ref().map(|c| c.policy_digest.to_vec()).unwrap_or_default(),
        }
    }
}

/// The answer to `req` with every value withheld, and the mentions whose
/// value `asked` may release under `policy` and `grant`. `req` asks for
/// values withheld; its other options are decided as #1904 decides them.
pub fn project(
    req: &InspectDocumentGraphRequest,
    policy: PiiOutputPolicy,
    asked: Forms,
    grant: &Grant,
) -> Result<(TypedDocumentResult, Vec<Planned>), Failure> {
    let typed = inspect_document_graph_under(req, policy)?;
    let mut plan = Vec::new();
    if !typed.layers.iter().any(|l| l.id == Pii::ID) {
        return Ok((typed, plan));
    }
    // The projection imported the same bytes and accepted them.
    let bad = |e: String| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, e);
    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;
    let covers = covers(&doc)?;
    for (li, layer) in typed.layers.iter().enumerate().filter(|(_, l)| l.id == Pii::ID) {
        let Some(name) = doc.layers().find(|k| k.ty == TypeId::of::<Pii>() && k.name == layer.instance).map(|k| k.name)
        else {
            continue;
        };
        let Some(pii) = doc.get_key(&Key::<Pii>::named(name)) else { continue };
        // Edits cover the default pii layer's mentions only.
        let covers = if name == Key::<Pii>::DEFAULT.name { covers.as_slice() } else { &[] };
        if pii.len() != layer.annotations.len() {
            return Err(Failure::new(
                TURBO_E_INTERNAL,
                0,
                format!("layer {} projected {} of {} mentions", layer.instance, layer.annotations.len(), pii.len()),
            ));
        }
        for i in 0..pii.len() {
            let m = pii.get(Ix::new(i as u32));
            let Some(form) = effective(asked, policy.values, grant.forms(m.kind)).iter().next() else { continue };
            let custom =
                if m.kind.is_custom() { pii.custom_kinds.name(m.kind).unwrap_or("").to_owned() } else { String::new() };
            plan.push(Planned {
                layer: li,
                index: i,
                kind: m.kind,
                custom,
                form,
                value: Sensitive::new(m.normalized.to_owned()),
                cover: covers.get(i).cloned().flatten(),
            });
        }
    }
    Ok((typed, plan))
}

/// Each default pii mention's covering edit of a named redaction policy,
/// by mention index: the first edits layer, in the graph's layer order,
/// whose edit lists the mention among its members. Edits of no named
/// policy cover nothing. An edit that names a rule with no policy digest
/// never gets here: the projection's validator refuses the graph first,
/// before any audit call.
fn covers(doc: &Document) -> Result<Vec<Option<Cover>>, Failure> {
    let mut out: Vec<Option<Cover>> = Vec::new();
    let ty = TypeId::of::<PiiEdits>();
    for k in doc.layers().filter(|k| k.ty == ty) {
        let Some(edits) = doc.get_key(&Key::<PiiEdits>::named(k.name)) else { continue };
        let Some(digest) = edits.policy_digest else { continue };
        for e in edits.edits() {
            for m in e.members {
                let i = m.get();
                if out.len() <= i {
                    out.resize(i + 1, None);
                }
                out[i].get_or_insert_with(|| Cover {
                    rule_id: e.rule_id.to_owned(),
                    policy_revision: edits.policy_revision,
                    policy_digest: *digest.as_bytes(),
                });
            }
        }
    }
    Ok(out)
}

/// Writes each planned value into its mention: a token under `key`, or
/// the value in clear. Called only after the audit acknowledgement.
pub fn fill(typed: &mut TypedDocumentResult, plan: Vec<Planned>, key: Option<&TokenKey>) -> Result<(), Failure> {
    for p in plan {
        let form = match (p.form, key) {
            (Form::Token, Some(k)) => pii_value::Form::Token(k.token(p.value.expose())),
            (Form::Clear, _) => pii_value::Form::Clear(p.value.into_inner()),
            _ => return Err(Failure::new(TURBO_E_INTERNAL, 0, format!("no key for a {} release", p.form.name()))),
        };
        let ann = &mut typed.layers[p.layer].annotations[p.index];
        if let Some(Value::Pii(m)) = &mut ann.value {
            m.value = Some(PiiValue { form: Some(form) });
        }
    }
    Ok(())
}
