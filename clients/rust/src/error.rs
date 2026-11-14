//! Unified error type for `turbo-client` spanning in-process lexical/NLP
//! analysis, local `libturbo` sessions, handshake verification, and pooled
//! `turbo-kserve` gRPC offload.

use std::fmt;

/// Categorized client error mapping cleanly onto gRPC status codes and
/// downstream storage/coordinator failure modes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClientError {
    /// Caller supplied an invalid configuration, span, or request parameter.
    InvalidArgument(String),
    /// Handshake or state invariant failed (for example, an analyzer or
    /// bundle scoring fingerprint mismatch against an index generation).
    FailedPrecondition(String),
    /// Requested bundle, model, or vocabulary artifact was not found.
    NotFound(String),
    /// Missing or invalid bearer token or client certificate (`UNAUTHENTICATED`).
    Unauthenticated(String),
    /// Authenticated caller is not permitted to perform the operation (`PERMISSION_DENIED`).
    PermissionDenied(String),
    /// Request deadline expired or was cancelled before completion (`DEADLINE_EXCEEDED`).
    DeadlineExceeded(String),
    /// Remote endpoint or device pool is unreachable or saturated (`UNAVAILABLE` / `RESOURCE_EXHAUSTED`).
    Unavailable(String),
    /// In-process annotator, model execution, or wire decoding failure (`INTERNAL`).
    Internal(String),
}

impl ClientError {
    /// Short stable error category label.
    pub const fn code_name(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) => "INVALID_ARGUMENT",
            Self::FailedPrecondition(_) => "FAILED_PRECONDITION",
            Self::NotFound(_) => "NOT_FOUND",
            Self::Unauthenticated(_) => "UNAUTHENTICATED",
            Self::PermissionDenied(_) => "PERMISSION_DENIED",
            Self::DeadlineExceeded(_) => "DEADLINE_EXCEEDED",
            Self::Unavailable(_) => "UNAVAILABLE",
            Self::Internal(_) => "INTERNAL",
        }
    }

    /// Message text without category prefix.
    pub fn message(&self) -> &str {
        match self {
            Self::InvalidArgument(m)
            | Self::FailedPrecondition(m)
            | Self::NotFound(m)
            | Self::Unauthenticated(m)
            | Self::PermissionDenied(m)
            | Self::DeadlineExceeded(m)
            | Self::Unavailable(m)
            | Self::Internal(m) => m,
        }
    }
}

impl fmt::Display for ClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code_name(), self.message())
    }
}

impl std::error::Error for ClientError {}

impl From<turbo_analysis::Error> for ClientError {
    fn from(err: turbo_analysis::Error) -> Self {
        Self::Internal(err.to_string())
    }
}

#[cfg(feature = "local-core")]
impl From<turbo::status::Error> for ClientError {
    fn from(err: turbo::status::Error) -> Self {
        use turbo::status::{BUNDLE_INTEGRITY, BUNDLE_NOT_FOUND, INVALID_ARGUMENT, UNSUPPORTED_OPTION};
        match err.code {
            INVALID_ARGUMENT | UNSUPPORTED_OPTION => Self::InvalidArgument(err.to_string()),
            BUNDLE_NOT_FOUND => Self::NotFound(err.to_string()),
            BUNDLE_INTEGRITY => Self::FailedPrecondition(err.to_string()),
            _ => Self::Internal(err.to_string()),
        }
    }
}

#[cfg(feature = "grpc")]
impl From<tonic::Status> for ClientError {
    fn from(status: tonic::Status) -> Self {
        use tonic::Code;
        let msg = status.message().to_owned();
        match status.code() {
            Code::InvalidArgument => Self::InvalidArgument(msg),
            Code::FailedPrecondition => Self::FailedPrecondition(msg),
            Code::NotFound => Self::NotFound(msg),
            Code::Unauthenticated => Self::Unauthenticated(msg),
            Code::PermissionDenied => Self::PermissionDenied(msg),
            Code::DeadlineExceeded | Code::Cancelled => Self::DeadlineExceeded(msg),
            Code::Unavailable | Code::ResourceExhausted => Self::Unavailable(msg),
            _ => Self::Internal(format!("{}: {msg}", status.code())),
        }
    }
}

/// Convenience result alias for `turbo-client` operations.
pub type Result<T> = std::result::Result<T, ClientError>;
