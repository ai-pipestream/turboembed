//! Status codes from turbo.h and the error every fallible call returns.

use std::ffi::CStr;

pub const OK: i32 = 0;

pub const INVALID_ARGUMENT: i32 = 256;
pub const INVALID_STRUCT_SIZE: i32 = 257;
pub const INVALID_UTF8: i32 = 258;
pub const INVALID_HANDLE: i32 = 259;
pub const INVALID_SHAPE: i32 = 260;
pub const INVALID_STATE: i32 = 261;
pub const INVALID_ENUM: i32 = 262;

pub const UNSUPPORTED: i32 = 512;
pub const UNSUPPORTED_OPTION: i32 = 513;
pub const UNSUPPORTED_TASK: i32 = 514;

pub const OUT_OF_MEMORY: i32 = 768;
pub const BUSY: i32 = 769;
pub const CAPACITY: i32 = 771;

pub const DEVICE_NOT_FOUND: i32 = 1024;
pub const DEVICE_UNAVAILABLE: i32 = 1025;
pub const RUNTIME: i32 = 1026;

pub const BUNDLE_NOT_FOUND: i32 = 1280;
pub const BUNDLE_INVALID: i32 = 1281;
pub const BUNDLE_INTEGRITY: i32 = 1282;
pub const BUNDLE_NO_ARTIFACT: i32 = 1283;

pub const INTERNAL: i32 = 1536;
pub const PANIC: i32 = 1537;

pub fn name(code: i32) -> &'static CStr {
    match code {
        OK => c"TURBO_OK",
        INVALID_ARGUMENT => c"TURBO_E_INVALID_ARGUMENT",
        INVALID_STRUCT_SIZE => c"TURBO_E_INVALID_STRUCT_SIZE",
        INVALID_UTF8 => c"TURBO_E_INVALID_UTF8",
        INVALID_HANDLE => c"TURBO_E_INVALID_HANDLE",
        INVALID_SHAPE => c"TURBO_E_INVALID_SHAPE",
        INVALID_STATE => c"TURBO_E_INVALID_STATE",
        INVALID_ENUM => c"TURBO_E_INVALID_ENUM",
        UNSUPPORTED => c"TURBO_E_UNSUPPORTED",
        UNSUPPORTED_OPTION => c"TURBO_E_UNSUPPORTED_OPTION",
        UNSUPPORTED_TASK => c"TURBO_E_UNSUPPORTED_TASK",
        OUT_OF_MEMORY => c"TURBO_E_OUT_OF_MEMORY",
        BUSY => c"TURBO_E_BUSY",
        CAPACITY => c"TURBO_E_CAPACITY",
        DEVICE_NOT_FOUND => c"TURBO_E_DEVICE_NOT_FOUND",
        DEVICE_UNAVAILABLE => c"TURBO_E_DEVICE_UNAVAILABLE",
        RUNTIME => c"TURBO_E_RUNTIME",
        BUNDLE_NOT_FOUND => c"TURBO_E_BUNDLE_NOT_FOUND",
        BUNDLE_INVALID => c"TURBO_E_BUNDLE_INVALID",
        BUNDLE_INTEGRITY => c"TURBO_E_BUNDLE_INTEGRITY",
        BUNDLE_NO_ARTIFACT => c"TURBO_E_BUNDLE_NO_ARTIFACT",
        INTERNAL => c"TURBO_E_INTERNAL",
        PANIC => c"TURBO_E_PANIC",
        _ => c"TURBO_E_UNKNOWN",
    }
}

/// A failed call: the status code, the 1-based field for the two codes that
/// name one, and a message for turbo_error.message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub code: i32,
    pub field: u32,
    pub message: String,
}

impl Error {
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Error { code, field: 0, message: message.into() }
    }

    pub fn field(code: i32, field: u32, message: impl Into<String>) -> Self {
        Error { code, field, message: message.into() }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", name(self.code).to_str().unwrap_or_default(), self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub fn invalid(message: impl Into<String>) -> Error {
    Error::new(BUNDLE_INVALID, message)
}
