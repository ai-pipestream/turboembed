//! Why a MeCab dictionary could not be read, and why a text could not be
//! segmented.

use std::fmt;

/// A file of a dictionary directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DictionaryFile {
    /// `matrix.def`, the connection costs.
    Matrix,
    /// A lexicon file, `*.csv`, by its name.
    Lexicon(String),
    /// `char.def`, the character categories.
    CharDef,
    /// `unk.def`, the unknown word templates.
    UnkDef,
}

impl fmt::Display for DictionaryFile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DictionaryFile::Matrix => f.write_str("matrix.def"),
            DictionaryFile::Lexicon(name) => f.write_str(name),
            DictionaryFile::CharDef => f.write_str("char.def"),
            DictionaryFile::UnkDef => f.write_str("unk.def"),
        }
    }
}

/// What is wrong with one line of a lexicon file or of `unk.def`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryProblem {
    /// Fewer than four fields.
    TooFewFields,
    /// The surface (the first field) is empty.
    EmptySurface,
    /// The left context id is not below the matrix's second dimension.
    LeftId(i32),
    /// The right context id is not below the matrix's first dimension.
    RightId(i32),
    /// The word cost is outside the 16-bit range.
    Cost(i32),
}

/// What is wrong with one line of `char.def`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CharDefProblem {
    /// A code point is not `0x` and a hexadecimal number.
    MalformedCodePoint,
    /// A code point is above U+10FFFF or negative.
    CodePointOutOfRange,
    /// A mapping names no category.
    MappingWithoutCategory,
    /// A range's end is below its start.
    RangeDescends,
    /// A category line has fewer than four fields.
    MalformedCategory,
    /// A category's INVOKE or GROUP field is not `0` or `1`.
    MalformedFlag,
    /// A category's LENGTH is not a number from 0 to 15.
    Length,
    /// A category is defined twice.
    DuplicateCategory(String),
    /// The line would define an eighteenth category.
    TooManyCategories,
}

/// Why a dictionary could not be read. The checks and their order are
/// those of OpenNLP's `MecabDictionary.load`; docs/segmentation.md says
/// where they differ.
#[derive(Debug)]
pub enum DictionaryError {
    /// A prebuilt image is malformed or has different whitespace settings.
    Prebuilt(super::PrebuiltError),
    /// A file could not be read.
    Io { file: DictionaryFile, source: std::io::Error },
    /// A required file is not in the directory.
    Missing(DictionaryFile),
    /// A file is not UTF-8; `offset` is where the first bad sequence
    /// starts.
    Utf8 { file: DictionaryFile, offset: usize },
    /// A number field is not a Java `int`.
    Number { file: DictionaryFile, line: usize },
    /// `matrix.def` is empty or its first line is blank.
    MatrixEmpty,
    /// The header of `matrix.def` is not two fields.
    MatrixHeader,
    /// A dimension of `matrix.def` is below 1.
    MatrixDimensions { rows: i32, cols: i32 },
    /// The dimensions pass a limit: either one passes the entry limit, or
    /// the cells pass the cell limit.
    MatrixTooLarge { rows: i32, cols: i32 },
    /// The file is too short to list the cells the header declares.
    MatrixShort { cells: u64, bytes: usize },
    /// A line of `matrix.def` is not three fields.
    MatrixLine { line: usize },
    /// A line's context ids are outside the dimensions.
    MatrixIds { line: usize, right: i32, left: i32 },
    /// A line's cost is outside the 16-bit range.
    MatrixCost { line: usize, cost: i32 },
    /// A pair is listed twice.
    MatrixDuplicate { line: usize, right: i32, left: i32 },
    /// Some pairs are not listed.
    MatrixIncomplete { listed: u64, cells: u64 },
    /// A line of a lexicon file or of `unk.def` is malformed.
    Entry { file: DictionaryFile, line: usize, problem: EntryProblem },
    /// The lexicon files, or `unk.def`, list more entries than the limit.
    TooManyEntries { limit: usize },
    /// No lexicon file lists an entry.
    NoLexiconEntries,
    /// A line of `char.def` is malformed.
    CharDef { line: usize, problem: CharDefProblem },
    /// `char.def` defines no `DEFAULT` category.
    NoDefaultCategory,
    /// `unk.def` names a category `char.def` does not define.
    UnknownCategory(String),
    /// A `char.def` mapping names a category it does not define; the code
    /// point is the mapping's first.
    UndefinedMapping { code_point: u32, category: String },
    /// The dictionary is too large for the 32-bit indexes of the trie or
    /// the feature store.
    TooLarge,
    /// The entries' first feature columns hold more than 65536 distinct
    /// values, more part of speech tags than a tag set holds.
    TooManyTags,
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use DictionaryError::*;
        match self {
            Prebuilt(e) => e.fmt(f),
            Io { file, source } => write!(f, "{file}: {source}"),
            Missing(file) => write!(f, "required dictionary file is missing: {file}"),
            Utf8 { file, offset } => write!(f, "{file}: not UTF-8 at byte {offset}"),
            Number { file, line } => write!(f, "malformed number in {file} line {line}"),
            MatrixEmpty => f.write_str("empty matrix.def"),
            MatrixHeader => f.write_str("malformed matrix.def header"),
            MatrixDimensions { rows, cols } => {
                write!(f, "matrix.def dimensions must be positive, got {rows} {cols}")
            }
            MatrixTooLarge { rows, cols } => write!(f, "matrix.def dimensions {rows} x {cols} exceed safe limit"),
            MatrixShort { cells, bytes } => {
                write!(f, "matrix.def declares {cells} connection costs but holds only {bytes} bytes")
            }
            MatrixLine { line } => write!(f, "malformed matrix.def line {line}"),
            MatrixIds { line, right, left } => {
                write!(
                    f,
                    "malformed matrix.def line {line}: context ids {right} {left} are outside the declared dimensions"
                )
            }
            MatrixCost { line, cost } => write!(
                f,
                "malformed matrix.def line {line}: connection cost {cost} is outside the 16-bit range the format defines"
            ),
            MatrixDuplicate { line, right, left } => {
                write!(f, "duplicate matrix.def entry {right} {left} at line {line}")
            }
            MatrixIncomplete { listed, cells } => {
                write!(f, "matrix.def declares {cells} connection costs but only {listed} pairs are listed")
            }
            Entry { file, line, problem } => {
                write!(f, "malformed entry at {file} line {line}")?;
                match problem {
                    EntryProblem::TooFewFields => Ok(()),
                    EntryProblem::EmptySurface => f.write_str(": surface must not be empty"),
                    EntryProblem::LeftId(id) => {
                        write!(f, ": left context id {id} is outside the matrix.def dimensions")
                    }
                    EntryProblem::RightId(id) => {
                        write!(f, ": right context id {id} is outside the matrix.def dimensions")
                    }
                    EntryProblem::Cost(cost) => {
                        write!(f, ": word cost {cost} is outside the 16-bit range the format defines")
                    }
                }
            }
            TooManyEntries { limit } => write!(f, "lexicon entry count exceeds safe limit of {limit}"),
            NoLexiconEntries => f.write_str("no lexicon entries found"),
            CharDef { line, problem } => {
                let what = match problem {
                    CharDefProblem::MalformedCodePoint => "malformed code point".to_string(),
                    CharDefProblem::CodePointOutOfRange => "code point out of range".to_string(),
                    CharDefProblem::MappingWithoutCategory => "mapping without category".to_string(),
                    CharDefProblem::RangeDescends => "code point range descends".to_string(),
                    CharDefProblem::MalformedCategory => "malformed category".to_string(),
                    CharDefProblem::MalformedFlag => "malformed category flag".to_string(),
                    CharDefProblem::Length => "category LENGTH must be between 0 and 15".to_string(),
                    CharDefProblem::DuplicateCategory(name) => format!("duplicate char.def category {name}"),
                    CharDefProblem::TooManyCategories => {
                        "char.def defines 18 categories; MeCab supports at most 17".into()
                    }
                };
                write!(f, "{what} at char.def line {line}")
            }
            NoDefaultCategory => f.write_str("char.def defines no DEFAULT category"),
            UnknownCategory(name) => write!(f, "unk.def names the undefined category {name}"),
            UndefinedMapping { code_point, category } => {
                write!(f, "char.def declaration at U+{code_point:04X} names the undefined category {category}")
            }
            TooLarge => f.write_str("the dictionary is too large to index"),
            TooManyTags => f.write_str("the entries have more than 65536 distinct first feature columns"),
        }
    }
}

impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DictionaryError::Io { source, .. } => Some(source),
            DictionaryError::Prebuilt(e) => Some(e),
            _ => None,
        }
    }
}

/// Why a text could not be segmented: at `offset` (a byte offset of the
/// text) neither the lexicon, nor the character's category, nor the
/// `DEFAULT` category of `unk.def` offers a candidate. A dictionary whose
/// `unk.def` has a `DEFAULT` template never fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoCandidate {
    pub offset: usize,
}

impl fmt::Display for NoCandidate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "dictionary provides no candidate at byte {}; unk.def lacks a DEFAULT template", self.offset)
    }
}

impl std::error::Error for NoCandidate {}
