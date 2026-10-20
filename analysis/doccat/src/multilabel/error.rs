//! Errors returned when building, opening, serializing, training, or
//! calibrating a multi-label document categorizer model.

use std::fmt;

/// Why a multi-label model, threshold calibration, or model package could
/// not be built, opened, or fitted.
#[derive(Debug)]
pub enum MultiLabelError {
    /// A model field failed validation: label or threshold count, a
    /// threshold outside the open interval `(0.0, 1.0)`, a NaN or
    /// infinite weight, unsorted or duplicate feature names, and the
    /// like. The detail is the sentence a reader needs.
    Invalid(String),
    /// The model bytes ended before a field could be read. Truncated
    /// archives fail with this error; reading never panics.
    Truncated(&'static str),
    /// The entry does not start with the multi-label model type string.
    BadMagic,
    /// The format version is not one this reader writes.
    UnsupportedVersion(i32),
    /// The trailing CRC-32C does not match the bytes before it.
    Checksum { want: u32, got: u32 },
    /// Opening the model package (the zip archive or its manifest)
    /// failed; the detail is the underlying error's message.
    Package(String),
    /// The training samples hold no document or no label at all.
    Train(String),
}

impl fmt::Display for MultiLabelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MultiLabelError::Invalid(detail) => write!(f, "invalid multi-label model: {detail}"),
            MultiLabelError::Truncated(what) => write!(f, "truncated multi-label model: {what}"),
            MultiLabelError::BadMagic => write!(f, "not a multi-label model entry"),
            MultiLabelError::UnsupportedVersion(v) => {
                write!(f, "unsupported multi-label model format version {v}")
            }
            MultiLabelError::Checksum { want, got } => {
                write!(f, "multi-label model checksum mismatch: computed {got:#010x}, stored {want:#010x}")
            }
            MultiLabelError::Package(detail) => write!(f, "multi-label model package: {detail}"),
            MultiLabelError::Train(detail) => write!(f, "multi-label training: {detail}"),
        }
    }
}

impl std::error::Error for MultiLabelError {}
