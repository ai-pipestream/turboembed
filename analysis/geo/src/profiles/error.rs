//! Why a profile table does not load.

use std::fmt;

/// Why a profile table cannot be loaded. Every variant but [`Io`] is an
/// `InvalidFormatException` in OpenNLP, and [`Display`](fmt::Display)
/// writes its message. Lines and columns are one-based, as OpenNLP counts
/// them: every line of the file counts, blank lines and comments included.
///
/// [`Io`]: ProfileError::Io
#[derive(Debug)]
pub enum ProfileError {
    Io(std::io::Error),
    /// The bytes are not UTF-8; `line` is the line of the first bad byte.
    /// OpenNLP reads up to that line first, so a problem on an earlier line
    /// is reported instead.
    Utf8 {
        line: usize,
    },
    /// The table holds nothing but blank lines and comments.
    NoHeader,
    /// The first line is not `id` followed by at least one metric name.
    Header {
        line: usize,
    },
    EmptyMetricName {
        line: usize,
        column: usize,
    },
    DuplicateMetricName {
        line: usize,
        column: usize,
        name: String,
    },
    /// A row without one value per metric.
    FieldCount {
        line: usize,
        found: usize,
        expected: usize,
    },
    EmptyId {
        line: usize,
    },
    DuplicateId {
        line: usize,
        id: String,
    },
    EmptyValue {
        line: usize,
        column: usize,
        metric: String,
    },
    /// A value that is not a decimal number: a hexadecimal number, a Java
    /// type suffix, or text `Double.parseDouble` refuses.
    MalformedValue {
        line: usize,
        value: String,
    },
    /// `NaN`, an infinity, or a number beyond the double range.
    NonFiniteValue {
        line: usize,
        value: String,
    },
    /// A header and no places.
    NoPlaces,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::Io(e) => e.fmt(f),
            ProfileError::Utf8 { .. } => f.write_str("profile table is not valid UTF-8"),
            ProfileError::NoHeader => f.write_str("the profile table has no header"),
            ProfileError::Header { .. } => f.write_str("the header must be: id, then at least one metric"),
            ProfileError::EmptyMetricName { column, .. } => write!(f, "empty metric name in header column {column}"),
            ProfileError::DuplicateMetricName { column, name, .. } => {
                write!(f, "duplicate metric name in header column {column}: {name}")
            }
            ProfileError::FieldCount { line, found, expected } => {
                write!(f, "row {line} has {found} fields, expected {expected}")
            }
            ProfileError::EmptyId { line } => write!(f, "empty id in row {line}"),
            ProfileError::DuplicateId { line, id } => write!(f, "duplicate id in row {line}: {id}"),
            ProfileError::EmptyValue { line, column, metric } => {
                write!(f, "empty value in row {line}, column {column} ({metric})")
            }
            ProfileError::MalformedValue { line, value } => write!(f, "malformed value in row {line}: {value}"),
            ProfileError::NonFiniteValue { line, value } => write!(f, "non-finite value in row {line}: {value}"),
            ProfileError::NoPlaces => f.write_str("the profile table lists no places"),
        }
    }
}

impl std::error::Error for ProfileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ProfileError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ProfileError {
    fn from(e: std::io::Error) -> ProfileError {
        ProfileError::Io(e)
    }
}
