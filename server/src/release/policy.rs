//! The redaction policy a values request is released under (#1943, #1977):
//! the request names it as the redact step took it, inline or by name
//! (`InspectDocumentGraphRequest.pii_policy`), its digest must be the one
//! the graph's `turbo:pii-edits` recorded, and each kind rule's release
//! ceiling caps the forms that kind's mentions leave in. With
//! `ReleaseConfig::require_recorded_policy`, a request that names no policy
//! is refused when the graph's `turbo:pii-edits` records a policy digest.

use std::any::TypeId;
use std::sync::Arc;

use prost::Message;
use turbo_analysis::{Document, Key};
use turbo_analysis_layers::{PiiEdits, PiiKind};

use super::caps::Forms;
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::pii_policy::{NamedPolicies, PolicyResolver, RedactionPolicy};
use crate::proto::result::{PiiReleasePolicy, RedactionPolicy as WirePolicy, pii_release_policy::Policy};

/// `InspectDocumentGraphRequest.pii_policy`.
pub const PII_POLICY_FIELD: u32 = 8;

fn refused(why: &str) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, PII_POLICY_FIELD, format!("pii_policy {why}"))
}

/// The policy `sel` names: `None` when the request names none. Refused
/// when it names no arm, an unknown name (not echoed), bytes that are not
/// a `RedactionPolicy`, or a policy that does not convert. None of this
/// depends on the document.
pub fn select(
    sel: Option<&PiiReleasePolicy>,
    named: &NamedPolicies,
    resolver: &dyn PolicyResolver,
) -> Result<Option<Arc<RedactionPolicy>>, Failure> {
    let Some(sel) = sel else { return Ok(None) };
    match &sel.policy {
        None => Err(refused("names no policy: set inline or name")),
        Some(Policy::Name(n)) => {
            named.get(n).cloned().map(Some).ok_or_else(|| refused("name is not a policy this server serves"))
        }
        Some(Policy::Inline(bytes)) => {
            let msg = WirePolicy::decode(bytes.as_slice()).map_err(|_| refused("inline is not a RedactionPolicy"))?;
            let policy = RedactionPolicy::from_proto(&msg, resolver).map_err(|e| refused(&format!("inline: {e}")))?;
            Ok(Some(Arc::new(policy)))
        }
    }
}

/// Refused unless at least one `turbo:pii-edits` layer of `doc` records a
/// policy digest and every one that does records `policy`'s. The refusal
/// names no kind and no value.
pub fn check_digest(doc: &Document, policy: &RedactionPolicy) -> Result<(), Failure> {
    let ty = TypeId::of::<PiiEdits>();
    let mut recorded = false;
    for k in doc.layers().filter(|k| k.ty == ty) {
        let Some(edits) = doc.get_key(&Key::<PiiEdits>::named(k.name)) else { continue };
        let Some(digest) = edits.policy_digest else { continue };
        if digest != policy.digest {
            return Err(refused("is not the redaction policy the graph's turbo:pii-edits recorded"));
        }
        recorded = true;
    }
    if !recorded {
        return Err(refused("names a redaction policy, and the graph's turbo:pii-edits record none"));
    }
    Ok(())
}

/// Refused when `require_recorded_policy` is on and any `turbo:pii-edits`
/// layer of `doc` records a policy digest. The refusal names no kind and
/// no value (#1977).
pub fn check_unrecorded(doc: &Document, require_recorded_policy: bool) -> Result<(), Failure> {
    if !require_recorded_policy {
        return Ok(());
    }
    let ty = TypeId::of::<PiiEdits>();
    for k in doc.layers().filter(|k| k.ty == ty) {
        let Some(edits) = doc.get_key(&Key::<PiiEdits>::named(k.name)) else { continue };
        if edits.policy_digest.is_some() {
            return Err(refused("is required when the graph's turbo:pii-edits records a policy digest"));
        }
    }
    Ok(())
}

/// The release ceiling of `policy` for a mention of `kind`; a custom kind
/// is found by `custom`, its name in the pii layer's table, among the
/// policy's custom kinds. [`Forms::ALL`] without a policy, or for a kind
/// the policy sets no ceiling for.
pub fn ceiling(policy: Option<&RedactionPolicy>, kind: PiiKind, custom: &str) -> Forms {
    let Some(policy) = policy else { return Forms::ALL };
    let kind = if kind.is_custom() { policy.custom_kinds.find(custom) } else { Some(kind) };
    kind.map_or(Forms::ALL, |k| policy.release_ceiling(k))
}
