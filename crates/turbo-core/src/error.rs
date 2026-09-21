//! Error type shared by the core and providers.
//!
//! Every error carries a graded status code from `turbo-abi`, an optional
//! 1-based field index naming the offending descriptor field, and a message.
//! The C boundary copies these into the caller-owned `turbo_error`.

use std::fmt;

use turbo_abi as abi;

/// Library error. Never constructed with `TURBO_OK`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    code: i32,
    field: u32,
    message: String,
}

/// Result alias used throughout the core.
pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// Build an error with a status code and message.
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        debug_assert_ne!(code, abi::TURBO_OK, "Error::new called with TURBO_OK");
        Self { code, field: 0, message: message.into() }
    }

    /// Attach a 1-based descriptor field index.
    pub fn with_field(mut self, field: u32) -> Self {
        self.field = field;
        self
    }

    /// Status code.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// Offending field index, or 0.
    pub fn field(&self) -> u32 {
        self.field
    }

    /// Message text.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Symbolic name of the status code.
    pub fn code_name(&self) -> &'static str {
        status_name(self.code)
    }

    // Constructors for the common grades. Each names the code in the
    // function name so call sites read as the contract they enforce.

    /// `TURBO_E_INVALID_ARGUMENT`.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_INVALID_ARGUMENT, message)
    }
    /// `TURBO_E_INVALID_STRUCT_SIZE`.
    pub fn invalid_struct_size(what: &str, got: u32, expected: usize) -> Self {
        Self::new(
            abi::TURBO_E_INVALID_STRUCT_SIZE,
            format!("{what}.struct_size is {got}; this library understands {expected}"),
        )
    }
    /// `TURBO_E_INVALID_UTF8`.
    pub fn invalid_utf8(what: &str) -> Self {
        Self::new(abi::TURBO_E_INVALID_UTF8, format!("{what} is not valid UTF-8"))
    }
    /// `TURBO_E_INVALID_HANDLE`.
    pub fn invalid_handle(what: &str) -> Self {
        Self::new(abi::TURBO_E_INVALID_HANDLE, format!("{what} handle is null"))
    }
    /// `TURBO_E_INVALID_SHAPE`.
    pub fn invalid_shape(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_INVALID_SHAPE, message)
    }
    /// `TURBO_E_INVALID_STATE`.
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_INVALID_STATE, message)
    }
    /// `TURBO_E_INVALID_ENUM`.
    pub fn invalid_enum(what: &str, value: u32) -> Self {
        Self::new(abi::TURBO_E_INVALID_ENUM, format!("{what} value {value} is not a known constant"))
    }
    /// `TURBO_E_UNSUPPORTED`.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_UNSUPPORTED, message)
    }
    /// `TURBO_E_UNSUPPORTED_OPTION` naming the field.
    pub fn unsupported_option(field: u32, name: &str, provider: &str) -> Self {
        Self::new(
            abi::TURBO_E_UNSUPPORTED_OPTION,
            format!(
                "option `{name}` is not honored by provider `{provider}` for this model; the capability bit is clear"
            ),
        )
        .with_field(field)
    }
    /// `TURBO_E_UNSUPPORTED_TASK`.
    pub fn unsupported_task(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_UNSUPPORTED_TASK, message)
    }
    /// `TURBO_E_UNSUPPORTED_DTYPE`.
    pub fn unsupported_dtype(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_UNSUPPORTED_DTYPE, message)
    }
    /// `TURBO_E_UNSUPPORTED_PLACEMENT`.
    pub fn unsupported_placement(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_UNSUPPORTED_PLACEMENT, message)
    }
    /// `TURBO_E_NOT_IMPLEMENTED`.
    pub fn not_implemented(what: &str) -> Self {
        Self::new(
            abi::TURBO_E_NOT_IMPLEMENTED,
            format!("{what} is declared in the ABI but not implemented in this build"),
        )
    }
    /// `TURBO_E_UNSUPPORTED_MODALITY`.
    pub fn unsupported_modality(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_UNSUPPORTED_MODALITY, message)
    }
    /// `TURBO_E_OUT_OF_MEMORY`.
    pub fn out_of_memory(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_OUT_OF_MEMORY, message)
    }
    /// `TURBO_E_BUSY`.
    pub fn busy(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_BUSY, message)
    }
    /// `TURBO_E_CAPACITY`.
    pub fn capacity(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_CAPACITY, message)
    }
    /// `TURBO_E_DEVICE_NOT_FOUND`.
    pub fn device_not_found(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_DEVICE_NOT_FOUND, message)
    }
    /// `TURBO_E_DEVICE_UNAVAILABLE`.
    pub fn device_unavailable(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_DEVICE_UNAVAILABLE, message)
    }
    /// `TURBO_E_RUNTIME`.
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_RUNTIME, message)
    }
    /// `TURBO_E_PROVIDER_LOAD`.
    pub fn provider_load(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_PROVIDER_LOAD, message)
    }
    /// `TURBO_E_ABI_MISMATCH`.
    pub fn abi_mismatch(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_ABI_MISMATCH, message)
    }
    /// `TURBO_E_CANCELLED`.
    pub fn cancelled() -> Self {
        Self::new(abi::TURBO_E_CANCELLED, "operation was cancelled")
    }
    /// `TURBO_E_BUNDLE_NOT_FOUND`.
    pub fn bundle_not_found(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_BUNDLE_NOT_FOUND, message)
    }
    /// `TURBO_E_BUNDLE_INVALID`.
    pub fn bundle_invalid(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_BUNDLE_INVALID, message)
    }
    /// `TURBO_E_BUNDLE_INTEGRITY`.
    pub fn bundle_integrity(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_BUNDLE_INTEGRITY, message)
    }
    /// `TURBO_E_BUNDLE_NO_ARTIFACT`.
    pub fn bundle_no_artifact(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_BUNDLE_NO_ARTIFACT, message)
    }
    /// `TURBO_E_INTERNAL`.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_INTERNAL, message)
    }
    /// `TURBO_E_PANIC`.
    pub fn panic(message: impl Into<String>) -> Self {
        Self::new(abi::TURBO_E_PANIC, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.field != 0 {
            write!(f, "{} (field {}): {}", self.code_name(), self.field, self.message)
        } else {
            write!(f, "{}: {}", self.code_name(), self.message)
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => Error::bundle_not_found(e.to_string()),
            std::io::ErrorKind::OutOfMemory => Error::out_of_memory(e.to_string()),
            _ => Error::runtime(format!("I/O error: {e}")),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::bundle_invalid(format!("manifest JSON: {e}"))
    }
}

/// Symbolic name for a status code, including `TURBO_OK`.
pub fn status_name(code: i32) -> &'static str {
    match code {
        abi::TURBO_OK => "TURBO_OK",
        abi::TURBO_E_INVALID_ARGUMENT => "TURBO_E_INVALID_ARGUMENT",
        abi::TURBO_E_INVALID_STRUCT_SIZE => "TURBO_E_INVALID_STRUCT_SIZE",
        abi::TURBO_E_INVALID_UTF8 => "TURBO_E_INVALID_UTF8",
        abi::TURBO_E_INVALID_HANDLE => "TURBO_E_INVALID_HANDLE",
        abi::TURBO_E_INVALID_SHAPE => "TURBO_E_INVALID_SHAPE",
        abi::TURBO_E_INVALID_STATE => "TURBO_E_INVALID_STATE",
        abi::TURBO_E_INVALID_ENUM => "TURBO_E_INVALID_ENUM",
        abi::TURBO_E_UNSUPPORTED => "TURBO_E_UNSUPPORTED",
        abi::TURBO_E_UNSUPPORTED_OPTION => "TURBO_E_UNSUPPORTED_OPTION",
        abi::TURBO_E_UNSUPPORTED_TASK => "TURBO_E_UNSUPPORTED_TASK",
        abi::TURBO_E_UNSUPPORTED_DTYPE => "TURBO_E_UNSUPPORTED_DTYPE",
        abi::TURBO_E_UNSUPPORTED_PLACEMENT => "TURBO_E_UNSUPPORTED_PLACEMENT",
        abi::TURBO_E_NOT_IMPLEMENTED => "TURBO_E_NOT_IMPLEMENTED",
        abi::TURBO_E_UNSUPPORTED_MODALITY => "TURBO_E_UNSUPPORTED_MODALITY",
        abi::TURBO_E_OUT_OF_MEMORY => "TURBO_E_OUT_OF_MEMORY",
        abi::TURBO_E_BUSY => "TURBO_E_BUSY",
        abi::TURBO_E_OVERLOADED => "TURBO_E_OVERLOADED",
        abi::TURBO_E_CAPACITY => "TURBO_E_CAPACITY",
        abi::TURBO_E_DEVICE_NOT_FOUND => "TURBO_E_DEVICE_NOT_FOUND",
        abi::TURBO_E_DEVICE_UNAVAILABLE => "TURBO_E_DEVICE_UNAVAILABLE",
        abi::TURBO_E_RUNTIME => "TURBO_E_RUNTIME",
        abi::TURBO_E_PROVIDER_LOAD => "TURBO_E_PROVIDER_LOAD",
        abi::TURBO_E_ABI_MISMATCH => "TURBO_E_ABI_MISMATCH",
        abi::TURBO_E_CANCELLED => "TURBO_E_CANCELLED",
        abi::TURBO_E_BUNDLE_NOT_FOUND => "TURBO_E_BUNDLE_NOT_FOUND",
        abi::TURBO_E_BUNDLE_INVALID => "TURBO_E_BUNDLE_INVALID",
        abi::TURBO_E_BUNDLE_INTEGRITY => "TURBO_E_BUNDLE_INTEGRITY",
        abi::TURBO_E_BUNDLE_NO_ARTIFACT => "TURBO_E_BUNDLE_NO_ARTIFACT",
        abi::TURBO_E_INTERNAL => "TURBO_E_INTERNAL",
        abi::TURBO_E_PANIC => "TURBO_E_PANIC",
        _ => "TURBO_E_UNKNOWN",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_includes_field_when_set() {
        let e = Error::unsupported_option(3, "pooling", "mock");
        assert_eq!(e.field(), 3);
        assert!(e.to_string().starts_with("TURBO_E_UNSUPPORTED_OPTION (field 3)"));
    }

    #[test]
    fn every_abi_code_has_a_name() {
        for code in [
            abi::TURBO_E_INVALID_ARGUMENT,
            abi::TURBO_E_UNSUPPORTED_MODALITY,
            abi::TURBO_E_BUNDLE_NO_ARTIFACT,
            abi::TURBO_E_PANIC,
        ] {
            assert_ne!(status_name(code), "TURBO_E_UNKNOWN");
        }
        assert_eq!(status_name(12345), "TURBO_E_UNKNOWN");
    }
}
