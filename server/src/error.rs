//! One error type for every route: a Turbo status (code name, message and
//! the 1-based field index when an option was rejected) or a server-side
//! condition, mapped onto HTTP and gRPC statuses in one place so the two
//! surfaces agree.

use std::fmt;

use turbo::abi;

/// Why a request did not complete.
#[derive(Debug, Clone)]
pub struct ServeError {
    /// `TURBO_E_*` name, or a server condition (`NOT_FOUND`, `BAD_REQUEST`).
    pub status: String,
    /// Turbo status code when the error came from the library, else 0.
    pub code: i32,
    /// 1-based index of the rejected option field, or 0.
    pub field: u32,
    /// What went wrong, in the words of whoever rejected it.
    pub message: String,
}

impl ServeError {
    /// The caller sent something the server cannot act on.
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self { status: "BAD_REQUEST".into(), code: 0, field: 0, message: message.into() }
    }

    /// The named thing is not served.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self { status: "NOT_FOUND".into(), code: 0, field: 0, message: message.into() }
    }

    /// The server broke its own invariant.
    pub fn internal(message: impl Into<String>) -> Self {
        Self { status: "INTERNAL".into(), code: 0, field: 0, message: message.into() }
    }

    /// Bad request naming the option field the value belongs to.
    pub fn field(field: u32, message: impl Into<String>) -> Self {
        Self { status: "BAD_REQUEST".into(), code: 0, field, message: message.into() }
    }

    /// HTTP status for this error.
    pub fn http_status(&self) -> axum::http::StatusCode {
        use axum::http::StatusCode;
        if self.code == 0 {
            return match self.status.as_str() {
                "NOT_FOUND" => StatusCode::NOT_FOUND,
                "BAD_REQUEST" => StatusCode::BAD_REQUEST,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
        }
        match self.code {
            abi::TURBO_E_INVALID_ARGUMENT
            | abi::TURBO_E_INVALID_ENUM
            | abi::TURBO_E_INVALID_SHAPE
            | abi::TURBO_E_INVALID_UTF8
            | abi::TURBO_E_INVALID_STRUCT_SIZE => StatusCode::BAD_REQUEST,
            abi::TURBO_E_CAPACITY => StatusCode::UNPROCESSABLE_ENTITY,
            abi::TURBO_E_BUSY => StatusCode::SERVICE_UNAVAILABLE,
            abi::TURBO_E_DEVICE_NOT_FOUND | abi::TURBO_E_BUNDLE_NOT_FOUND => StatusCode::NOT_FOUND,
            c if is_unsupported(c) => StatusCode::NOT_IMPLEMENTED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    /// gRPC status for this error, with the same message.
    pub fn grpc_status(&self) -> tonic::Status {
        use tonic::Code;
        let code = if self.code == 0 {
            match self.status.as_str() {
                "NOT_FOUND" => Code::NotFound,
                "BAD_REQUEST" => Code::InvalidArgument,
                _ => Code::Internal,
            }
        } else {
            match self.code {
                abi::TURBO_E_INVALID_ARGUMENT
                | abi::TURBO_E_INVALID_ENUM
                | abi::TURBO_E_INVALID_SHAPE
                | abi::TURBO_E_INVALID_UTF8
                | abi::TURBO_E_INVALID_STRUCT_SIZE => Code::InvalidArgument,
                abi::TURBO_E_CAPACITY => Code::ResourceExhausted,
                abi::TURBO_E_BUSY => Code::Unavailable,
                abi::TURBO_E_DEVICE_NOT_FOUND | abi::TURBO_E_BUNDLE_NOT_FOUND => Code::NotFound,
                c if is_unsupported(c) => Code::Unimplemented,
                _ => Code::Internal,
            }
        };
        tonic::Status::new(code, self.to_string())
    }
}

/// The `TURBO_E_UNSUPPORTED*` family.
fn is_unsupported(code: i32) -> bool {
    matches!(
        code,
        abi::TURBO_E_UNSUPPORTED
            | abi::TURBO_E_UNSUPPORTED_OPTION
            | abi::TURBO_E_UNSUPPORTED_TASK
            | abi::TURBO_E_UNSUPPORTED_MODALITY
            | abi::TURBO_E_UNSUPPORTED_DTYPE
            | abi::TURBO_E_UNSUPPORTED_PLACEMENT
            | abi::TURBO_E_NOT_IMPLEMENTED
    )
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.field != 0 {
            write!(f, "{} (field {}): {}", self.status, self.field, self.message)
        } else {
            write!(f, "{}: {}", self.status, self.message)
        }
    }
}

impl std::error::Error for ServeError {}

impl From<turbo::Error> for ServeError {
    fn from(e: turbo::Error) -> Self {
        Self { status: e.code_name().to_string(), code: e.code(), field: e.field(), message: e.message().to_string() }
    }
}

/// A result carrying a [`ServeError`].
pub type Result<T> = std::result::Result<T, ServeError>;
