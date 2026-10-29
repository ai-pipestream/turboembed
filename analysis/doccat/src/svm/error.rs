//! Why an SVM document categorizer model could not be opened.

use std::fmt;
use std::path::PathBuf;

use super::libsvm::SvmType;

/// Which of the two files a [`ModelError::Line`] is about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum File {
    /// The LIBSVM text model.
    Libsvm,
    /// The sidecar: vocabulary, weighting, scaling and categories.
    Sidecar,
}

/// What is wrong with one line of a model file.
#[derive(Clone, Debug, PartialEq)]
pub enum LineError {
    /// A keyword the format does not have.
    UnknownKeyword(String),
    /// A keyword given a second time.
    Repeated(&'static str),
    /// A field that does not parse as its type.
    BadValue { field: &'static str, text: String },
    /// A record with too few or too many fields or values.
    Count { field: &'static str, expected: usize, found: usize },
    /// A record that needs another before it (`rho` before `nr_class`).
    Before { field: &'static str, needs: &'static str },
    /// A support vector or feature index not above the one before it.
    Unordered { field: &'static str, index: i64 },
    /// A precomputed kernel, whose input is the kernel's values.
    PrecomputedKernel,
    /// A first line other than the format's.
    Header(String),
    /// A line after the end of the model.
    Trailing,
}

impl fmt::Display for LineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LineError::UnknownKeyword(k) => write!(f, "unknown keyword {k:?}"),
            LineError::Repeated(k) => write!(f, "{k} is given twice"),
            LineError::BadValue { field, text } => write!(f, "{field}: {text:?} does not parse"),
            LineError::Count { field, expected, found } => {
                write!(f, "{field}: {expected} values expected, {found} found")
            }
            LineError::Before { field, needs } => write!(f, "{field} comes before {needs}"),
            LineError::Unordered { field, index } => write!(f, "{field}: index {index} is not above the one before it"),
            LineError::PrecomputedKernel => f.write_str("a precomputed kernel needs the kernel's values as input"),
            LineError::Header(found) => write!(f, "the first line is {found:?}"),
            LineError::Trailing => f.write_str("a line after the end of the model"),
        }
    }
}

/// A model that is not a usable SVM document categorizer model. Each
/// variant names what was wrong.
#[derive(Debug)]
pub enum ModelError {
    Io {
        path: PathBuf,
        error: std::io::Error,
    },
    /// A line of one of the files, numbered from 1.
    Line {
        file: File,
        line: usize,
        error: LineError,
    },
    /// A record the file must have and lacks.
    Missing {
        file: File,
        field: &'static str,
    },
    /// A value out of its range: a count that does not add up, a label
    /// outside the categories, a category given twice.
    Invalid {
        file: File,
        detail: String,
    },
    /// The categorizer reads class labels: one-class and regression models
    /// give none.
    NotClassification(SvmType),
    /// `categorize` asks LIBSVM for probability estimates, which a model
    /// trained without them cannot give. OpenNLP throws
    /// `ArrayIndexOutOfBoundsException` on every document of such a model
    /// with two or more categories: zlibsvm hands LIBSVM empty `probA` and
    /// `probB` arrays rather than none.
    NoProbabilityEstimates,
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = |file: &File| match file {
            File::Libsvm => "LIBSVM model",
            File::Sidecar => "SVM doccat sidecar",
        };
        match self {
            ModelError::Io { path, error } => write!(f, "{}: {error}", path.display()),
            ModelError::Line { file, line, error } => write!(f, "{} line {line}: {error}", name(file)),
            ModelError::Missing { file, field } => write!(f, "{}: no {field} record", name(file)),
            ModelError::Invalid { file, detail } => write!(f, "{}: {detail}", name(file)),
            ModelError::NotClassification(t) => {
                write!(f, "LIBSVM model: a {} model gives no class labels to categorize with", t.name())
            }
            ModelError::NoProbabilityEstimates => {
                f.write_str("LIBSVM model: no probA and probB, so it gives no probability estimates")
            }
        }
    }
}

impl std::error::Error for ModelError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ModelError::Io { error, .. } => Some(error),
            _ => None,
        }
    }
}
