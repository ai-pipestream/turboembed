//! Head rules and gap labelers for constituency parsing (`opennlp.tools.parser.HeadRules`,
//! `opennlp.tools.parser.GapLabeler`, `opennlp.tools.parser.lang.en.HeadRules`, and
//! `opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`).
//!
//! Both rule classes read the same text format, one rule per line: the number of fields, the
//! constituent type, the direction (`1` searches the children left to right, anything else right
//! to left) and the tags in order of priority, separated by `java.util.StringTokenizer`
//! whitespace (space, tab, line feed, carriage return, form feed). The field count is the number
//! of tags plus two. A later rule for the same type replaces the earlier one. The English tags
//! are type names; the AnCora tags are Java regular expressions ([`TagPattern`]).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, Write};

use turbo_jvm::hash_map::{spread, string_hash};

use crate::parse::{Constituent, Head, Parse, ParseError};

mod ancora;
mod english;

pub use ancora::{AncoraSpanishHeadRules, TagPattern};
pub use english::EnglishHeadRules;

/// Maximum number of tags permitted in a single head rule (`MAX_TAGS_PER_RULE = 1_000`).
pub const MAX_TAGS_PER_RULE: usize = 1_000;

pub use turbo_jvm::MAX_PATTERN_ATOMS;

const PUNCTUATION_TAGS: [&str; 4] = [".", ",", "``", "''"];

/// Labels constituents on the reader's stack that contain a trace gap (`opennlp.tools.parser.GapLabeler`).
pub trait GapLabeler {
    /// Inspects the constituent stack when the reader meets a `-NONE-` trace token and changes the
    /// labels of a matching extraction chain. `at` is the input offset of the trace's bracket, for
    /// the error when a label it reads is missing ([`ParseError::MissingType`]).
    fn label_gaps(&self, stack: &mut [Constituent<'_>], at: usize) -> Result<(), ParseError>;
}

/// Finds the head of a constituent from its children (`opennlp.tools.parser.HeadRules`).
pub trait HeadRules {
    /// Returns the head for `constituents` under parent type `kind`, or `None` if
    /// `constituents` starts with a token node (`TOK_NODE`) or is empty.
    fn get_head(&self, constituents: &[Parse], kind: &str) -> Option<Head>;

    /// Returns the set of POS tags treated as punctuation (`getPunctuationTags`).
    fn punctuation_tags(&self) -> &HashSet<String>;
}

/// Errors returned when reading head rules. Each variant but [`HeadRulesError::UnsupportedPattern`]
/// stands for the exception OpenNLP throws on the same line ([`HeadRulesError::java_class`]).
/// Lines are numbered from 1.
#[derive(Debug)]
pub enum HeadRulesError {
    /// The reader failed (`IOException`).
    Io(io::Error),
    /// A line without a field count, type or direction (`NoSuchElementException`), blank lines
    /// included.
    MissingField { line: usize },
    /// A field count that is not an `int`, or that gives fewer than 0 or more than
    /// [`MAX_TAGS_PER_RULE`] tags (`IOException`).
    InvalidTagCount { line: usize, count: String },
    /// More tags than the field count gives (`ArrayIndexOutOfBoundsException`).
    TooManyTags { line: usize },
    /// Fewer tags than the field count gives (`NullPointerException`).
    MissingTags { line: usize },
    /// An AnCora tag pattern outside the regular expressions [`TagPattern`] reads, or one Java
    /// would refuse (`PatternSyntaxException`).
    UnsupportedPattern { line: usize, pattern: String, reason: String },
}

impl HeadRulesError {
    /// The class of the exception OpenNLP throws in the same place, if this reader knows it.
    pub fn java_class(&self) -> Option<&'static str> {
        match self {
            Self::Io(_) | Self::InvalidTagCount { .. } => Some("java.io.IOException"),
            Self::MissingField { .. } => Some("java.util.NoSuchElementException"),
            Self::TooManyTags { .. } => Some("java.lang.ArrayIndexOutOfBoundsException"),
            Self::MissingTags { .. } => Some("java.lang.NullPointerException"),
            Self::UnsupportedPattern { .. } => None,
        }
    }
}

impl fmt::Display for HeadRulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::InvalidTagCount { count, .. } => write!(f, "Invalid tag count in head rules: {count}"),
            Self::MissingField { line } => write!(f, "head rules line {line}: missing field"),
            Self::TooManyTags { line } => write!(f, "head rules line {line}: more tags than the field count"),
            Self::MissingTags { line } => write!(f, "head rules line {line}: fewer tags than the field count"),
            Self::UnsupportedPattern { line, pattern, reason } => {
                write!(f, "head rules line {line}: unsupported tag pattern {pattern}: {reason}")
            }
        }
    }
}

impl std::error::Error for HeadRulesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for HeadRulesError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// A single head-finding rule: search direction and tags in order of priority.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRule {
    /// Whether constituents are searched left-to-right (`true`) or right-to-left (`false`).
    pub left_to_right: bool,
    /// The tags as the rule file gives them: type names, or regular expressions for AnCora.
    pub tags: Vec<String>,
}

/// The rules read from a file, with the types in the order they were first read (the order a
/// Java `HashMap` inserts them in).
#[derive(Clone, Debug)]
struct RuleTable<T> {
    order: Vec<String>,
    rules: HashMap<String, (HeadRule, T)>,
}

impl<T: Clone> RuleTable<T> {
    /// Reads rules the way both OpenNLP classes do (`readHeadRules`), with `compile` making the
    /// matcher data of each rule's tags.
    fn read(
        s: &str,
        capacity: usize,
        mut compile: impl FnMut(&[String], usize) -> Result<T, HeadRulesError>,
    ) -> Result<Self, HeadRulesError> {
        let mut order = Vec::with_capacity(capacity);
        let mut rules: HashMap<String, (HeadRule, T)> = HashMap::with_capacity(capacity);
        for (i, line) in JavaLineIter::new(s).enumerate() {
            let line_number = i + 1;
            let mut tokens = split_string_tokenizer(line);
            let (Some(num), Some(kind), Some(dir)) = (tokens.next(), tokens.next(), tokens.next()) else {
                return Err(HeadRulesError::MissingField { line: line_number });
            };
            let invalid = || HeadRulesError::InvalidTagCount { line: line_number, count: num.to_string() };
            let raw_count = turbo_jvm::text::parse_int(num).map_err(|_| invalid())?;
            let num_tags = i64::from(raw_count) - 2;
            if !(0..=MAX_TAGS_PER_RULE as i64).contains(&num_tags) {
                return Err(invalid());
            }
            let num_tags = num_tags as usize;
            let mut tags = Vec::with_capacity(num_tags);
            for tok in tokens {
                if tags.len() == num_tags {
                    return Err(HeadRulesError::TooManyTags { line: line_number });
                }
                tags.push(tok.to_string());
            }
            if tags.len() < num_tags {
                return Err(HeadRulesError::MissingTags { line: line_number });
            }
            let compiled = compile(&tags, line_number)?;
            let rule = HeadRule { left_to_right: dir == "1", tags };
            if !rules.contains_key(kind) {
                order.push(kind.to_string());
            }
            rules.insert(kind.to_string(), (rule, compiled));
        }
        Ok(Self { order, rules })
    }

    /// Writes the rules as `serialize` does: one line per rule in the iteration order of the Java
    /// `HashMap` of initial capacity `capacity` that holds them.
    fn serialize<W: Write>(&self, capacity: usize, mut writer: W) -> io::Result<()> {
        for i in java_hash_map_order(&self.order, capacity) {
            let kind = &self.order[i];
            let (rule, _) = &self.rules[kind];
            write!(writer, "{} {} {}", rule.tags.len() + 2, kind, if rule.left_to_right { "1" } else { "0" })?;
            for tag in &rule.tags {
                write!(writer, " {tag}")?;
            }
            writer.write_all(b"\n")?;
        }
        writer.flush()
    }

    fn rules_equal(&self, other: &Self) -> bool {
        self.rules.len() == other.rules.len()
            && self.rules.iter().all(|(k, (r, _))| other.rules.get(k).is_some_and(|(o, _)| o == r))
    }
}

/// The order in which a `java.util.HashMap` created with `new HashMap<>(capacity)` iterates the
/// distinct `keys`, inserted with `put` in the order given.
///
/// The table starts at `tableSizeFor(capacity)` bins and doubles when its size passes three
/// quarters of that, or when a bin of 8 keys gets a ninth while the table has fewer than 64 bins.
/// Iteration walks the bins in order, each in insertion order. A bin that reaches 9 keys at 64 bins
/// or more becomes a tree in Java, whose order is not modelled; such a bin is kept in insertion
/// order. Rule files with that many types in one bin are not known.
fn java_hash_map_order(keys: &[String], capacity: usize) -> Vec<usize> {
    let mut cap = capacity.max(1).next_power_of_two();
    let spreads: Vec<u32> = keys.iter().map(|k| spread(string_hash(k))).collect();
    let mut bins: Vec<Vec<usize>> = vec![Vec::new(); cap];
    let rehash = |bins: &mut Vec<Vec<usize>>, cap: usize| {
        let old = std::mem::replace(bins, vec![Vec::new(); cap]);
        for i in old.into_iter().flatten() {
            bins[spreads[i] as usize & (cap - 1)].push(i);
        }
    };
    for (i, &h) in spreads.iter().enumerate() {
        let bin = h as usize & (cap - 1);
        let before = bins[bin].len();
        bins[bin].push(i);
        if before >= 8 && cap < 64 {
            cap *= 2;
            rehash(&mut bins, cap);
        }
        if i + 1 > cap / 4 * 3 {
            cap *= 2;
            rehash(&mut bins, cap);
        }
    }
    bins.into_iter().flatten().collect()
}

/// The gap labeling both rule classes do, with their labels for subject extraction (`subject`,
/// from the second constituent from the top down) and object extraction (`object`). Labels are
/// compared in Java's order, so a missing label fails at the same comparison.
fn label_extraction_gaps(
    stack: &mut [Constituent<'_>],
    at: usize,
    subject: [&str; 3],
    object: [&str; 4],
) -> Result<(), ParseError> {
    let len = stack.len();
    if len <= 4 {
        return Ok(());
    }
    let con = |k: usize| &stack[len - 1 - k];
    let chain =
        if con(1).label_is(subject[0], at)? && con(2).label_is(subject[1], at)? && con(3).label_is(subject[2], at)? {
            3
        } else if con(1).label_is(object[0], at)?
            && con(2).label_is(object[1], at)?
            && con(3).label_is(object[2], at)?
            && con(4).label_is(object[3], at)?
        {
            4
        } else {
            return Ok(());
        };
    for c in &mut stack[len - 1 - chain..len - 1] {
        let label = format!("{}-G", c.label().expect("compared above"));
        c.set_label(label);
    }
    Ok(())
}

fn punctuation_set() -> HashSet<String> {
    PUNCTUATION_TAGS.iter().map(|&p| p.to_string()).collect()
}

/// Splits a string on `java.util.StringTokenizer` default delimiters (`' '`, `'\t'`, `'\n'`,
/// `'\r'`, `'\x0C'`).
#[inline]
fn split_string_tokenizer(line: &str) -> impl Iterator<Item = &str> {
    line.split([' ', '\t', '\n', '\r', '\x0C']).filter(|s| !s.is_empty())
}

/// Iterates lines with `java.io.BufferedReader.readLine()` semantics (`\n`, `\r\n`, `\r`).
pub(crate) struct JavaLineIter<'a> {
    text: &'a str,
    pos: usize,
}

impl<'a> JavaLineIter<'a> {
    #[inline]
    pub(crate) fn new(text: &'a str) -> Self {
        Self { text, pos: 0 }
    }
}

impl<'a> Iterator for JavaLineIter<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        if self.pos >= bytes.len() {
            return None;
        }
        let start = self.pos;
        let mut i = start;
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'\n' {
                self.pos = i + 1;
                return Some(&self.text[start..i]);
            }
            if b == b'\r' {
                if i + 1 < bytes.len() && bytes[i + 1] == b'\n' {
                    self.pos = i + 2;
                } else {
                    self.pos = i + 1;
                }
                return Some(&self.text[start..i]);
            }
            i += 1;
        }
        self.pos = bytes.len();
        Some(&self.text[start..])
    }
}
