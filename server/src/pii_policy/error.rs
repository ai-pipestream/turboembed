//! Why a [`super::RedactionPolicy`] message was refused, with where in the
//! message. No error carries a term, a map entry or key material.

use std::fmt;

/// Where in a policy message a refusal points: a top-level field, its
/// index when repeated, and one nested repeated field with its index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct At {
    pub field: &'static str,
    pub index: Option<usize>,
    pub inner: Option<(&'static str, usize)>,
}

impl At {
    /// A top-level field that is not repeated.
    pub const fn field(field: &'static str) -> At {
        At { field, index: None, inner: None }
    }

    /// Entry `index` of a top-level repeated field.
    pub const fn item(field: &'static str, index: usize) -> At {
        At { field, index: Some(index), inner: None }
    }

    /// Entry `index` of the repeated field `inner` under this place.
    pub const fn inner(self, inner: &'static str, index: usize) -> At {
        At { inner: Some((inner, index)), ..self }
    }
}

impl fmt::Display for At {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.field)?;
        if let Some(i) = self.index {
            write!(f, "[{i}]")?;
        }
        if let Some((name, i)) = self.inner {
            write!(f, ".{name}[{i}]")?;
        }
        Ok(())
    }
}

/// A refused policy message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyError {
    /// A required field is unset or empty.
    Missing { at: At, field: &'static str },
    /// An enum field is UNSPECIFIED or a number this server does not know.
    Unspecified { at: At, field: &'static str, value: i32 },
    /// A built-in kind id this server does not know, without
    /// `redact_and_flag_unknown`.
    UnknownKind { at: At, id: i32 },
    /// A strategy or condition arm this server does not know (or none
    /// set), without `redact_and_flag_unknown`.
    UnknownArm { at: At, field: &'static str },
    /// A custom kind reference past the end of the policy's table.
    CustomKindOutOfRange { at: At, index: u32, table: usize },
    /// A built-in kind where only a custom kind is allowed.
    BuiltinKind { at: At },
    /// The custom kind table is invalid.
    CustomKinds(String),
    /// Two rules share a rule id.
    DuplicateRuleId { at: At },
    /// Two kind rules name the same kind.
    DuplicateKind { at: At },
    /// A map's fallback is itself a map.
    NestedMap { at: At },
    /// A reference that is not a registered reference: a file path, key
    /// material, or a malformed id or digest.
    BadRef { at: At, field: &'static str, reason: &'static str },
    /// The server could not resolve a reference; names the reference only.
    Unresolved { at: At, field: &'static str, reason: String },
    /// A value the native strategy or list refused.
    Invalid { at: At, reason: String },
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicyError::Missing { at, field } => write!(f, "redaction policy {at}: {field} is required"),
            PolicyError::Unspecified { at, field, value } => {
                write!(f, "redaction policy {at}: {field} {value} is UNSPECIFIED or unknown")
            }
            PolicyError::UnknownKind { at, id } => write!(f, "redaction policy {at}: unknown pii kind id {id}"),
            PolicyError::UnknownArm { at, field } => {
                write!(f, "redaction policy {at}: {field} is unset or not known to this server")
            }
            PolicyError::CustomKindOutOfRange { at, index, table } => {
                write!(f, "redaction policy {at}: custom kind {index} is outside the policy's {table} custom_kinds")
            }
            PolicyError::BuiltinKind { at } => write!(f, "redaction policy {at}: the kind must be a custom kind"),
            PolicyError::CustomKinds(e) => write!(f, "redaction policy custom_kinds: {e}"),
            PolicyError::DuplicateRuleId { at } => write!(f, "redaction policy {at}: rule_id repeats an earlier rule"),
            PolicyError::DuplicateKind { at } => write!(f, "redaction policy {at}: kind has an earlier rule"),
            PolicyError::NestedMap { at } => write!(f, "redaction policy {at}: a map's fallback cannot be a map"),
            PolicyError::BadRef { at, field, reason } => write!(f, "redaction policy {at}: {field} {reason}"),
            PolicyError::Unresolved { at, field, reason } => {
                write!(f, "redaction policy {at}: {field} is not resolved: {reason}")
            }
            PolicyError::Invalid { at, reason } => write!(f, "redaction policy {at}: {reason}"),
        }
    }
}

impl std::error::Error for PolicyError {}
