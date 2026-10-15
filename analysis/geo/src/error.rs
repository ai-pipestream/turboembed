//! What can go wrong building or loading a gazetteer.

use std::fmt;

use crate::entry::EntryError;
use crate::store::CapacityError;
use crate::suppression::SuppressionError;

/// Why a set of records cannot be indexed.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexError {
    /// Two records with the same source and record id.
    Duplicate {
        source: String,
        record_id: String,
    },
    /// A name of a record folds to the empty match key, so no lookup could
    /// reach the record by it: the canonical name always, an alternate name
    /// where the alternates are not skipped.
    Unmatchable {
        name: String,
        source: String,
        record_id: String,
    },
    Capacity(CapacityError),
}

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexError::Duplicate { source, record_id } => {
                write!(f, "Duplicate gazetteer record for source {source} and recordId {record_id}")
            }
            IndexError::Unmatchable { name, source, record_id } => write!(
                f,
                "Name '{name}' of record {source};{record_id} folds to an empty match key, so the record would be \
                 unreachable by lookup"
            ),
            IndexError::Capacity(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for IndexError {}

impl From<CapacityError> for IndexError {
    fn from(e: CapacityError) -> IndexError {
        IndexError::Capacity(e)
    }
}

/// A number field Java's `Double.parseDouble` or `Long.parseLong` refuses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NumberFormat(pub String);

impl fmt::Display for NumberFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() { f.write_str("empty String") } else { write!(f, "For input string: \"{}\"", self.0) }
    }
}

/// What is wrong with one row of a table.
#[derive(Clone, Debug, PartialEq)]
pub enum RowProblem {
    /// The row has the wrong number of tab separated columns.
    Columns {
        found: usize,
        expected: ColumnCount,
    },
    Number(NumberFormat),
    Entry(EntryError),
    /// A user table's row with one coordinate and not the other.
    HalfLocation,
    /// A user table's row with neither coordinates nor a bounding box.
    NoLocation,
    /// A user table's bounding box that is not four edges.
    BoxEdges(String),
    /// An attribute that is not `key=value`.
    AttributePair(String),
    DuplicateAttribute(String),
    /// The record id of an earlier row.
    RepeatedId(String),
    Unmatchable {
        name: String,
        source: String,
        record_id: String,
    },
    /// A suppression rule that could never match or has a bad filter.
    Suppression(SuppressionError),
    /// A row of the bundled table without its eleven fields.
    Bundled(String),
}

/// How many columns a row must have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColumnCount {
    Exactly(usize),
    AtLeast(usize),
    AtMost(usize),
}

impl fmt::Display for RowProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowProblem::Columns { found, expected } => match expected {
                ColumnCount::Exactly(n) => write!(f, "has {found} columns, expected {n}"),
                ColumnCount::AtLeast(_) => write!(f, "has {found} columns, expected at least record id and name"),
                ColumnCount::AtMost(n) => write!(f, "has {found} columns, expected at most {n}"),
            },
            RowProblem::Number(e) => e.fmt(f),
            RowProblem::Entry(e) => e.fmt(f),
            RowProblem::HalfLocation => {
                f.write_str("latitude and longitude must be present together or absent together")
            }
            RowProblem::NoLocation => f.write_str("a row needs coordinates or a bounding box"),
            RowProblem::BoxEdges(field) => write!(f, "a bounding box must read west,south,east,north, got: {field}"),
            RowProblem::AttributePair(pair) => write!(f, "an attribute must read key=value, got: {pair}"),
            RowProblem::DuplicateAttribute(key) => write!(f, "the attribute key repeats: {key}"),
            RowProblem::RepeatedId(id) => write!(f, "repeats record id: {id}"),
            RowProblem::Unmatchable { name, source, record_id } => write!(
                f,
                "Name '{name}' of record {source};{record_id} folds to an empty match key, so the record would be \
                 unreachable by lookup"
            ),
            RowProblem::Suppression(e) => write!(f, "not a suppression rule: {e}"),
            RowProblem::Bundled(line) => write!(f, "malformed gazetteer data: {line}"),
        }
    }
}

impl From<EntryError> for RowProblem {
    fn from(e: EntryError) -> RowProblem {
        RowProblem::Entry(e)
    }
}

impl From<NumberFormat> for RowProblem {
    fn from(e: NumberFormat) -> RowProblem {
        RowProblem::Number(e)
    }
}

/// Why a table cannot be loaded.
#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    /// The bytes are not UTF-8; `line` is the line the first bad byte is
    /// on.
    Utf8 {
        line: usize,
    },
    /// A row the format refuses, with its one-based line number.
    Row {
        line: usize,
        problem: RowProblem,
    },
    /// The table holds no data rows.
    NoRows,
    Capacity(CapacityError),
    /// The source identifier of a user table is blank.
    BlankSource,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Io(e) => e.fmt(f),
            LoadError::Utf8 { line } => write!(f, "line {line} is not valid UTF-8"),
            LoadError::Row { line, problem } => write!(f, "line {line}: {problem}"),
            LoadError::NoRows => f.write_str("the table contains no rows"),
            LoadError::Capacity(e) => e.fmt(f),
            LoadError::BlankSource => f.write_str("source must not be null or blank"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            LoadError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> LoadError {
        LoadError::Io(e)
    }
}

impl From<CapacityError> for LoadError {
    fn from(e: CapacityError) -> LoadError {
        LoadError::Capacity(e)
    }
}
