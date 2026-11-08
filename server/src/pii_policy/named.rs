//! The redaction policy a request runs `turbo-pii:redact` with: one given
//! inline, or one the server loaded and converted at start and serves by
//! its id. A bad policy file fails start-up; an unknown name is refused
//! per request.

use std::collections::BTreeMap;
use std::sync::Arc;

use prost::Message;

use super::{PolicyError, PolicyResolver, RedactionPolicy};
use crate::proto::result::{PiiRedactionSelection, RedactionPolicy as WirePolicy, pii_redaction_selection};

/// The policies a server serves by name, each named by its id. `Debug`
/// prints the ids only.
#[derive(Clone, Default)]
pub struct NamedPolicies {
    by_id: BTreeMap<String, Arc<RedactionPolicy>>,
}

impl std::fmt::Debug for NamedPolicies {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_set().entries(self.names()).finish()
    }
}

impl NamedPolicies {
    /// Converts every message once. Refused when one does not convert or
    /// two share an id; the error names the message by its position, never
    /// a file path, a term or key material.
    pub fn from_messages(msgs: &[WirePolicy], resolver: &dyn PolicyResolver) -> Result<NamedPolicies, String> {
        let mut by_id = BTreeMap::new();
        for (i, msg) in msgs.iter().enumerate() {
            let policy = RedactionPolicy::from_proto(msg, resolver).map_err(|e| format!("pii policy {i}: {e}"))?;
            if by_id.contains_key(&policy.id) {
                return Err(format!("pii policy {i}: the id repeats an earlier policy"));
            }
            by_id.insert(policy.id.clone(), Arc::new(policy));
        }
        Ok(NamedPolicies { by_id })
    }

    /// Decodes each binary `RedactionPolicy` message, then
    /// [`NamedPolicies::from_messages`].
    pub fn from_encoded(files: &[Vec<u8>], resolver: &dyn PolicyResolver) -> Result<NamedPolicies, String> {
        let msgs = files
            .iter()
            .enumerate()
            .map(|(i, b)| {
                WirePolicy::decode(b.as_slice()).map_err(|_| format!("pii policy {i}: not a RedactionPolicy"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        NamedPolicies::from_messages(&msgs, resolver)
    }

    /// The policy with id `name`.
    pub fn get(&self, name: &str) -> Option<&Arc<RedactionPolicy>> {
        self.by_id.get(name)
    }

    /// Every served policy id, sorted.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_id.keys().map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }
}

/// The policy of one request and whether it is a dry run.
#[derive(Clone, Debug)]
pub struct Redaction {
    pub policy: Arc<RedactionPolicy>,
    pub dry_run: bool,
}

/// Why a request's policy selection was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SelectError {
    /// Neither an inline policy nor a name.
    Unset,
    /// No policy of this server has the name.
    UnknownName,
    /// The inline policy does not convert.
    Policy(PolicyError),
}

impl std::fmt::Display for SelectError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SelectError::Unset => f.write_str("pii_redaction names no policy: set inline or name"),
            SelectError::UnknownName => f.write_str("pii_redaction.name is not a policy this server serves"),
            SelectError::Policy(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for SelectError {}

/// The policy `sel` selects: an inline policy converted now, or a named
/// one of `named`. An unknown name is refused, and the name is not echoed.
pub fn select(
    sel: &PiiRedactionSelection,
    named: &NamedPolicies,
    resolver: &dyn PolicyResolver,
) -> Result<Redaction, SelectError> {
    let policy = match &sel.policy {
        None => return Err(SelectError::Unset),
        Some(pii_redaction_selection::Policy::Name(n)) => Arc::clone(named.get(n).ok_or(SelectError::UnknownName)?),
        Some(pii_redaction_selection::Policy::Inline(p)) => {
            Arc::new(RedactionPolicy::from_proto(p, resolver).map_err(SelectError::Policy)?)
        }
    };
    Ok(Redaction { policy, dry_run: sel.dry_run })
}
