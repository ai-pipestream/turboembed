//! Why a dictionary did not load. Each error's message is the message of
//! the `IOException` OpenNLP throws for the same input, so the goldens
//! compare them as text; [`LoadError::EncodingNotPorted`] and
//! [`LoadError::Io`] have no Java counterpart.

use std::fmt;

/// Which of the two files a size or decoding error is about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Stream {
    Affix,
    Dictionary,
}

impl Stream {
    fn label(self) -> &'static str {
        match self {
            Stream::Affix => "affix stream",
            Stream::Dictionary => "dictionary stream",
        }
    }
}

/// A dictionary that did not load.
#[derive(Debug)]
#[non_exhaustive]
pub enum LoadError {
    /// Reading a file failed.
    Io(std::io::Error),
    /// A file is larger than [`crate::MAX_FILE_BYTES`].
    TooLarge {
        stream: Stream,
        limit: usize,
    },
    /// A file is not valid in the encoding `SET` declares.
    Malformed {
        stream: Stream,
        charset: &'static str,
    },
    /// `SET` names an encoding Java does not know, or an illegal name.
    UnsupportedEncoding {
        name: String,
    },
    /// `SET` names an encoding Java decodes but this port has no table for.
    EncodingNotPorted {
        name: String,
    },
    /// A strict load met a directive the reader does not support.
    UnsupportedDirective {
        directive: String,
        source: String,
        line: usize,
    },
    MultipleFlagModes {
        line: usize,
    },
    FlagModeMissing {
        line: usize,
    },
    UnsupportedFlagMode {
        mode: String,
        line: usize,
    },
    AliasValueMissing {
        line: usize,
    },
    NegativeAliasCount {
        line: usize,
    },
    AliasCount {
        expected: i32,
        found: usize,
    },
    /// A directive that takes one value has none.
    ValueMissing {
        directive: String,
        line: usize,
    },
    /// A directive's value is not an integer.
    MalformedValue {
        directive: String,
        line: usize,
    },
    FlagMissing {
        directive: String,
        line: usize,
    },
    NegativeCompoundMin {
        line: usize,
    },
    CompoundMinTooLarge {
        line: usize,
    },
    NegativeCompoundWordMax {
        line: usize,
    },
    InvalidCompoundSyllable {
        line: usize,
    },
    InvalidSyllableNum {
        line: usize,
    },
    InvalidLanguage {
        line: usize,
    },
    InvalidIgnore {
        line: usize,
    },
    InvalidBreak {
        line: usize,
    },
    MalformedAffixHeader {
        line: usize,
    },
    InvalidCrossProduct {
        line: usize,
    },
    MalformedRuleCount {
        line: usize,
    },
    NegativeRuleCount {
        line: usize,
    },
    TruncatedAffixBlock {
        line: usize,
    },
    MalformedAffixRule {
        line: usize,
    },
    RuleFlagMismatch {
        line: usize,
    },
    InvalidMorphologyAlias {
        line: usize,
    },
    MalformedFlagAlias {
        alias: String,
        line: usize,
    },
    FlagAliasOutOfRange {
        alias: i32,
        line: usize,
        size: usize,
    },
    MalformedNumericFlag {
        line: usize,
    },
    NumericFlagRange {
        line: usize,
        value: String,
    },
    OddLongFlags {
        line: usize,
    },
    ExpectedOneFlag {
        line: usize,
    },
    UnterminatedClass {
        line: usize,
    },
    /// The count line of a table (`ICONV`, `REP`, `AM`, `BREAK`, ...).
    InvalidTableCount {
        tag: &'static str,
        line: usize,
    },
    InvalidTableEntry {
        tag: &'static str,
        line: usize,
    },
    TableCount {
        tag: &'static str,
        expected: i32,
        found: usize,
    },
    InvalidCompoundRule {
        line: usize,
    },
    CompoundRuleTooLong {
        line: usize,
    },
    EmptyCompoundRule {
        line: usize,
    },
    EmptyConversionPattern {
        tag: &'static str,
        line: usize,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LoadError::*;
        match self {
            Io(e) => write!(f, "{e}"),
            TooLarge { stream, limit } => write!(f, "{} size exceeds safe limit of {limit}", stream.label()),
            Malformed { stream, charset } => write!(f, "{} is not valid {charset}", stream.label()),
            UnsupportedEncoding { name } => write!(f, "unsupported SET encoding: {name}"),
            EncodingNotPorted { name } => write!(f, "SET encoding {name} is not one this port decodes"),
            UnsupportedDirective { directive, source, line } => write!(
                f,
                "unsupported affix directive {directive} in {source} at line {line}; \
                 use LoadMode.ALLOW_PARTIAL to load without this behavior"
            ),
            MultipleFlagModes { line } => write!(f, "multiple FLAG directives at line {line}"),
            FlagModeMissing { line } => write!(f, "FLAG line without a mode at line {line}"),
            UnsupportedFlagMode { mode, line } => write!(f, "unsupported FLAG mode '{mode}' at line {line}"),
            AliasValueMissing { line } => write!(f, "AF line without a value at line {line}"),
            NegativeAliasCount { line } => write!(f, "negative AF count at line {line}"),
            AliasCount { expected, found } => write!(f, "AF header specifies {expected} aliases but found {found}"),
            ValueMissing { directive, line } => write!(f, "{directive} line without a value at line {line}"),
            MalformedValue { directive, line } => write!(f, "malformed {directive} at line {line}"),
            FlagMissing { directive, line } => write!(f, "{directive} line without a flag at line {line}"),
            NegativeCompoundMin { line } => write!(f, "negative COMPOUNDMIN at line {line}"),
            CompoundMinTooLarge { line } => write!(f, "COMPOUNDMIN exceeds {} at line {line}", i32::MAX / 2),
            NegativeCompoundWordMax { line } => write!(f, "negative COMPOUNDWORDMAX at line {line}"),
            InvalidCompoundSyllable { line } => write!(f, "invalid COMPOUNDSYLLABLE at line {line}"),
            InvalidSyllableNum { line } => write!(f, "invalid SYLLABLENUM at line {line}"),
            InvalidLanguage { line } => write!(f, "invalid LANG at line {line}"),
            InvalidIgnore { line } => write!(f, "invalid IGNORE at line {line}"),
            InvalidBreak { line } => write!(f, "invalid BREAK at line {line}"),
            MalformedAffixHeader { line } => write!(f, "malformed affix header at line {line}"),
            InvalidCrossProduct { line } => write!(f, "invalid cross-product marker at line {line}"),
            MalformedRuleCount { line } => write!(f, "malformed affix rule count at line {line}"),
            NegativeRuleCount { line } => write!(f, "negative affix rule count at line {line}"),
            TruncatedAffixBlock { line } => write!(f, "affix block truncated at line {line}"),
            MalformedAffixRule { line } => write!(f, "malformed affix rule at line {line}"),
            RuleFlagMismatch { line } => write!(f, "affix rule flag does not match header at line {line}"),
            InvalidMorphologyAlias { line } => write!(f, "invalid AM alias at line {line}"),
            MalformedFlagAlias { alias, line } => write!(f, "malformed flag alias '{alias}' at line {line}"),
            FlagAliasOutOfRange { alias, line, size } => {
                write!(f, "flag alias {alias} at line {line} is outside the AF table of {size} aliases")
            }
            MalformedNumericFlag { line } => write!(f, "malformed numeric flag at line {line}"),
            NumericFlagRange { line, value } => write!(f, "numeric flag outside 1..65535 at line {line}: {value}"),
            OddLongFlags { line } => write!(f, "odd long-flag run at line {line}"),
            ExpectedOneFlag { line } => write!(f, "expected exactly one flag at line {line}"),
            UnterminatedClass { line } => write!(f, "unterminated character class at line {line}"),
            InvalidTableCount { tag, line } => write!(f, "invalid {tag} count at line {line}"),
            InvalidTableEntry { tag, line } => write!(f, "invalid {tag} entry at line {line}"),
            TableCount { tag, expected, found } => {
                write!(f, "{tag} header specifies {expected} entries but found {found}")
            }
            InvalidCompoundRule { line } => write!(f, "invalid COMPOUNDRULE at line {line}"),
            CompoundRuleTooLong { line } => write!(f, "COMPOUNDRULE exceeds 4096 elements at line {line}"),
            EmptyCompoundRule { line } => write!(f, "empty COMPOUNDRULE at line {line}"),
            EmptyConversionPattern { tag, line } => write!(f, "empty {tag} pattern at line {line}"),
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
