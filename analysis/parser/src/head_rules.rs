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
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::sync::LazyLock;

use turbo_jvm::hash_map::{spread, string_hash};

use crate::parse::{Constituent, Head, Parse, ParseError, TOK_NODE};

/// Maximum number of tags permitted in a single head rule (`MAX_TAGS_PER_RULE = 1_000`).
pub const MAX_TAGS_PER_RULE: usize = 1_000;

/// Maximum number of atoms in a compiled [`TagPattern`]: one bit each in a `u64`, and one more
/// for the end of the pattern.
pub const MAX_PATTERN_ATOMS: usize = 63;

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

const EN_NP_TAGS1: [&str; 7] = ["NN", "NNP", "NNPS", "NNS", "NX", "JJR", "POS"];
const EN_NP_TAGS2: [&str; 3] = ["$", "ADJP", "PRN"];
const EN_NP_TAGS3: [&str; 4] = ["JJ", "JJS", "RB", "QP"];

/// English constituency head rules and trace gap labeler (`opennlp.tools.parser.lang.en.HeadRules`).
///
/// `NP` and `NX` have fixed rules, read before the file's: the last child whose type is one of
/// `NN NNP NNPS NNS NX JJR POS`, else the first `NP` child, else the last child of type
/// `$ ADJP PRN`, else the last of `JJ JJS RB QP`, else the last child. A type without a rule takes
/// the head of its last child. OpenNLP's class has no other special cases (none for coordination
/// or punctuation); its punctuation tags (`.`, `,`, ` `` `, `''`) are only reported by
/// [`HeadRules::punctuation_tags`].
#[derive(Clone, Debug)]
pub struct EnglishHeadRules {
    table: RuleTable<()>,
    punct_set: HashSet<String>,
}

impl PartialEq for EnglishHeadRules {
    fn eq(&self, other: &Self) -> bool {
        self.table.rules_equal(&other.table) && self.punct_set == other.punct_set
    }
}

impl Eq for EnglishHeadRules {}

impl EnglishHeadRules {
    /// `new HashMap<>(30)` in `readHeadRules`.
    const CAPACITY: usize = 30;

    /// Reads English head rules (`new HeadRules(Reader)`).
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, HeadRulesError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_str(&buf)
    }

    /// Returns the rule registered for constituent `kind`, if any.
    pub fn rule(&self, kind: &str) -> Option<&HeadRule> {
        self.table.rules.get(kind).map(|(r, _)| r)
    }

    /// Returns the number of constituent head rules loaded.
    pub fn len(&self) -> usize {
        self.table.rules.len()
    }

    /// Returns `true` if no constituent head rules are loaded.
    pub fn is_empty(&self) -> bool {
        self.table.rules.is_empty()
    }

    /// Writes the head rules in the file format (`HeadRules.serialize`), in the iteration order of
    /// OpenNLP's `HashMap`, so the bytes are the ones Java writes.
    pub fn serialize<W: Write>(&self, writer: W) -> io::Result<()> {
        self.table.serialize(Self::CAPACITY, writer)
    }

    /// Serializes the head rules into a `String`.
    pub fn serialize_to_string(&self) -> String {
        let mut out = Vec::new();
        self.serialize(&mut out).expect("writing to Vec<u8> cannot fail");
        String::from_utf8(out).expect("serialized head rules are valid UTF-8")
    }
}

impl FromStr for EnglishHeadRules {
    type Err = HeadRulesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let table = RuleTable::read(s, Self::CAPACITY, |_, _| Ok(()))?;
        Ok(Self { table, punct_set: punctuation_set() })
    }
}

impl HeadRules for EnglishHeadRules {
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.punct_set
    }

    fn get_head(&self, constituents: &[Parse], kind: &str) -> Option<Head> {
        if constituents.is_empty() || constituents[0].kind() == TOK_NODE {
            return None;
        }
        if kind == "NP" || kind == "NX" {
            let last_of = |tags: &[&str]| {
                constituents.iter().rev().find(|c| tags.iter().rev().any(|t| c.kind() == *t)).map(Parse::head)
            };
            return last_of(&EN_NP_TAGS1)
                .or_else(|| constituents.iter().find(|c| c.kind() == "NP").map(Parse::head))
                .or_else(|| last_of(&EN_NP_TAGS2))
                .or_else(|| last_of(&EN_NP_TAGS3))
                .or_else(|| Some(constituents[constituents.len() - 1].head()));
        }
        if let Some((hr, _)) = self.table.rules.get(kind) {
            if hr.left_to_right {
                for tag in &hr.tags {
                    if let Some(c) = constituents.iter().find(|c| c.kind() == tag) {
                        return Some(c.head());
                    }
                }
                return Some(constituents[0].head());
            }
            for tag in &hr.tags {
                if let Some(c) = constituents.iter().rev().find(|c| c.kind() == tag) {
                    return Some(c.head());
                }
            }
            return Some(constituents[constituents.len() - 1].head());
        }
        Some(constituents[constituents.len() - 1].head())
    }
}

impl GapLabeler for EnglishHeadRules {
    /// Subject extraction (`NP` under `S` under `SBAR`) and object extraction (`NP` under `VP`
    /// under `S` under `SBAR`), read from the second to fifth constituents from the top of a stack
    /// of more than four; each constituent of the chain gets `-G` appended.
    fn label_gaps(&self, stack: &mut [Constituent<'_>], at: usize) -> Result<(), ParseError> {
        label_extraction_gaps(stack, at, ["NP", "S", "SBAR"], ["NP", "VP", "S", "SBAR"])
    }
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

/// A Java regular expression from the subset the AnCora head rules use, compiled to a bit-parallel
/// matcher that tests a whole tag (`Pattern.matcher(tag).matches()`) without backtracking.
///
/// It reads a sequence of atoms, each a literal character, `.` (any character but a line
/// terminator), a class (`[MAS]`, `[a-z0-9]`, `[^...]`) or `\` before a character that is not a
/// letter or digit (that character), each optionally followed by `*`, `+` or `?`. `^` at the start
/// and `$` at the end are accepted and change nothing, since the whole tag must match. Groups,
/// alternation, bounded repeats, lazy and possessive quantifiers, escapes such as `\d`, nested
/// classes and class intersections are refused; OpenNLP's rule files use none of them.
///
/// OpenNLP's `es-head-rules` file doubles its backslashes (`GRUP\\.VERB`). In a Java regular
/// expression that is a backslash followed by any character, so those tags never match a type
/// such as `GRUP.VERB`; this reads them the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TagPattern {
    atoms: Vec<Atom>,
    /// For each ASCII character, the atoms that match it.
    ascii: [u64; 128],
    /// The atoms that repeat (`*`).
    repeat: u64,
    /// The atoms that may be passed over (`*` and `?`).
    skip: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Atom {
    Char(char),
    Any,
    Class { negated: bool, ranges: Vec<(char, char)> },
}

impl Atom {
    fn matches(&self, c: char) -> bool {
        match self {
            Atom::Char(x) => *x == c,
            Atom::Any => !matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}'),
            Atom::Class { negated, ranges } => ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != *negated,
        }
    }
}

impl TagPattern {
    /// Compiles a Java regular expression from the subset described on [`TagPattern`].
    pub fn compile(regex: &str) -> Result<Self, String> {
        let mut atoms: Vec<Atom> = Vec::new();
        let mut repeat = 0u64;
        let mut optional = 0u64;
        let mut chars = regex.char_indices().peekable();
        let mut quantifiable = false;
        while let Some((i, c)) = chars.next() {
            match c {
                '*' | '+' | '?' => {
                    if !quantifiable {
                        return Err(format!("dangling {c}"));
                    }
                    if matches!(chars.peek(), Some((_, '*' | '+' | '?' | '{'))) {
                        return Err("lazy, possessive and stacked quantifiers are not supported".into());
                    }
                    let last = atoms.len() - 1;
                    match c {
                        '*' => repeat |= 1 << last,
                        '?' => optional |= 1 << last,
                        _ => {
                            atoms.push(atoms[last].clone());
                            repeat |= 1 << (last + 1);
                        }
                    }
                    quantifiable = false;
                    continue;
                }
                '^' if i == 0 => {
                    quantifiable = false;
                    continue;
                }
                '$' if regex[i..].bytes().all(|b| b == b'$') => break,
                '(' | ')' | '|' | '{' | '}' | '^' | '$' => {
                    return Err(format!("{c} is not supported here"));
                }
                '.' => atoms.push(Atom::Any),
                '\\' => atoms.push(Atom::Char(escaped(chars.next().map(|(_, c)| c))?)),
                '[' => atoms.push(class(&mut chars)?),
                _ => atoms.push(Atom::Char(c)),
            }
            quantifiable = true;
            if atoms.len() > MAX_PATTERN_ATOMS {
                return Err(format!("more than {MAX_PATTERN_ATOMS} atoms"));
            }
        }
        if atoms.len() > MAX_PATTERN_ATOMS {
            return Err(format!("more than {MAX_PATTERN_ATOMS} atoms"));
        }
        let mut ascii = [0u64; 128];
        for (i, atom) in atoms.iter().enumerate() {
            for (b, slot) in ascii.iter_mut().enumerate() {
                if atom.matches(b as u8 as char) {
                    *slot |= 1 << i;
                }
            }
        }
        Ok(Self { atoms, ascii, repeat, skip: repeat | optional })
    }

    /// Whether the pattern matches all of `tag` (`Pattern.matcher(tag).matches()`).
    #[inline]
    pub fn matches(&self, tag: &str) -> bool {
        let mut states = self.closure(1);
        for c in tag.chars() {
            let matching = if c.is_ascii() {
                self.ascii[c as usize]
            } else {
                self.atoms.iter().enumerate().filter(|(_, a)| a.matches(c)).fold(0u64, |m, (i, _)| m | 1 << i)
            };
            let hit = states & matching;
            states = self.closure(((hit & !self.repeat) << 1) | (hit & self.repeat));
            if states == 0 {
                return false;
            }
        }
        states & (1u64 << self.atoms.len()) != 0
    }

    /// Adds the states reached by passing over optional and repeated atoms.
    #[inline]
    fn closure(&self, mut states: u64) -> u64 {
        loop {
            let next = states | ((states & self.skip) << 1);
            if next == states {
                return states;
            }
            states = next;
        }
    }
}

/// The literal character of `\c`: Java reads a backslash before a character that is not an ASCII
/// letter or digit as that character.
fn escaped(c: Option<char>) -> Result<char, String> {
    match c {
        None => Err("trailing backslash".into()),
        Some(c) if c.is_ascii_alphanumeric() => Err(format!("\\{c} is not supported")),
        Some(c) => Ok(c),
    }
}

/// A character class after its `[`, up to and with its `]`.
fn class(chars: &mut std::iter::Peekable<std::str::CharIndices<'_>>) -> Result<Atom, String> {
    let negated = chars.next_if(|&(_, c)| c == '^').is_some();
    let mut ranges = Vec::new();
    loop {
        let lo = match chars.next() {
            None => return Err("unterminated class".into()),
            Some((_, ']')) if !ranges.is_empty() => return Ok(Atom::Class { negated, ranges }),
            Some((_, ']' | '[')) => return Err("empty or nested classes are not supported".into()),
            Some((_, '&')) if chars.peek().is_some_and(|&(_, c)| c == '&') => {
                return Err("class intersections are not supported".into());
            }
            Some((_, '\\')) => escaped(chars.next().map(|(_, c)| c))?,
            Some((_, c)) => c,
        };
        let is_range = chars.peek().is_some_and(|&(_, c)| c == '-') && {
            let mut ahead = chars.clone();
            ahead.next();
            ahead.next().is_some_and(|(_, c)| c != ']')
        };
        if !is_range {
            ranges.push((lo, lo));
            continue;
        }
        chars.next();
        let hi = match chars.next() {
            Some((_, '\\')) => escaped(chars.next().map(|(_, c)| c))?,
            Some((_, '[')) => return Err("nested classes are not supported".into()),
            Some((_, c)) => c,
            None => return Err("unterminated class".into()),
        };
        if hi < lo {
            return Err(format!("illegal range {lo}-{hi}"));
        }
        ranges.push((lo, hi));
    }
}

/// `TAGS1`, `TAGS2` and `TAGS3` of `AncoraSpanishHeadRules`.
static ANCORA_TAGS: LazyLock<[Vec<TagPattern>; 3]> = LazyLock::new(|| {
    let compile = |tags: &[&str]| tags.iter().map(|t| TagPattern::compile(t).expect("built-in pattern")).collect();
    [
        compile(&["AQA.*", "AQC.*", r"GRUP\.A", r"S\.A", "NC.*S.*", "NP.*", "NC.*P.*", r"GRUP\.NOM"]),
        compile(&[r"\$", r"GRUP\.A", "SA"]),
        compile(&["AQ0.*", "AQ[AC].*", "AO.*", r"GRUP\.A", r"S\.A", "RG", "RN", r"GRUP\.NOM"]),
    ]
});

/// AnCora Spanish constituency head rules and trace gap labeler
/// (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`).
///
/// The tags of the rules are Java regular expressions ([`TagPattern`]). `SN` and `GRUP.NOM` have
/// fixed rules, read before the file's. As in OpenNLP, a child chosen by a pattern is itself the
/// head ([`Parse::as_head`]), where the English rules take the child's head; only the fallbacks
/// (the first or last child when nothing matches, and the last child for a type without a rule)
/// take the child's head.
#[derive(Clone, Debug)]
pub struct AncoraSpanishHeadRules {
    table: RuleTable<Vec<TagPattern>>,
    punct_set: HashSet<String>,
}

impl PartialEq for AncoraSpanishHeadRules {
    fn eq(&self, other: &Self) -> bool {
        self.table.rules_equal(&other.table) && self.punct_set == other.punct_set
    }
}

impl Eq for AncoraSpanishHeadRules {}

impl AncoraSpanishHeadRules {
    /// `new HashMap<>(60)` in `readHeadRules`.
    const CAPACITY: usize = 60;

    /// Reads AnCora Spanish head rules (`new AncoraSpanishHeadRules(Reader)`).
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, HeadRulesError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_str(&buf)
    }

    /// Returns the rule registered for constituent `kind`, if any.
    pub fn rule(&self, kind: &str) -> Option<&HeadRule> {
        self.table.rules.get(kind).map(|(r, _)| r)
    }

    /// Returns the number of constituent head rules loaded.
    pub fn len(&self) -> usize {
        self.table.rules.len()
    }

    /// Returns `true` if no constituent head rules are loaded.
    pub fn is_empty(&self) -> bool {
        self.table.rules.is_empty()
    }

    /// Writes the head rules in the file format, the tags as read
    /// (`AncoraSpanishHeadRules.serialize`), in the iteration order of OpenNLP's `HashMap`.
    pub fn serialize<W: Write>(&self, writer: W) -> io::Result<()> {
        self.table.serialize(Self::CAPACITY, writer)
    }

    /// Serializes the head rules into a `String`.
    pub fn serialize_to_string(&self) -> String {
        let mut out = Vec::new();
        self.serialize(&mut out).expect("writing to Vec<u8> cannot fail");
        String::from_utf8(out).expect("serialized head rules are valid UTF-8")
    }
}

impl FromStr for AncoraSpanishHeadRules {
    type Err = HeadRulesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let table = RuleTable::read(s, Self::CAPACITY, |tags, line| {
            tags.iter()
                .map(|t| {
                    TagPattern::compile(t).map_err(|reason| HeadRulesError::UnsupportedPattern {
                        line,
                        pattern: t.clone(),
                        reason,
                    })
                })
                .collect()
        })?;
        Ok(Self { table, punct_set: punctuation_set() })
    }
}

impl HeadRules for AncoraSpanishHeadRules {
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.punct_set
    }

    fn get_head(&self, constituents: &[Parse], kind: &str) -> Option<Head> {
        if constituents.is_empty() || constituents[0].kind() == TOK_NODE {
            return None;
        }
        let last = &constituents[constituents.len() - 1];
        if kind == "SN" || kind == "GRUP.NOM" {
            let [tags1, tags2, tags3] = &*ANCORA_TAGS;
            let last_of = |tags: &[TagPattern]| {
                constituents.iter().rev().find(|c| tags.iter().rev().any(|t| t.matches(c.kind()))).map(Parse::as_head)
            };
            return constituents
                .iter()
                .find(|c| tags1.iter().rev().any(|t| t.matches(c.kind())))
                .map(Parse::as_head)
                .or_else(|| {
                    constituents.iter().find(|c| c.kind() == "SN" || c.kind() == "GRUP.NOM").map(Parse::as_head)
                })
                .or_else(|| last_of(tags2))
                .or_else(|| last_of(tags3))
                .or_else(|| Some(last.head()));
        }
        if let Some((hr, patterns)) = self.table.rules.get(kind) {
            if hr.left_to_right {
                for pattern in patterns {
                    if let Some(c) = constituents.iter().find(|c| pattern.matches(c.kind())) {
                        return Some(c.as_head());
                    }
                }
                return Some(constituents[0].head());
            }
            for pattern in patterns {
                if let Some(c) = constituents.iter().rev().find(|c| pattern.matches(c.kind())) {
                    return Some(c.as_head());
                }
            }
            return Some(last.head());
        }
        Some(last.head())
    }
}

impl GapLabeler for AncoraSpanishHeadRules {
    /// Subject extraction (`SN` under `S` under `GRUP.NOM`) and object extraction (`SN` under
    /// `GRUP.VERB` under `S` under `GRUP.NOM`), as [`EnglishHeadRules`] labels its gaps.
    fn label_gaps(&self, stack: &mut [Constituent<'_>], at: usize) -> Result<(), ParseError> {
        label_extraction_gaps(stack, at, ["SN", "S", "GRUP.NOM"], ["SN", "GRUP.VERB", "S", "GRUP.NOM"])
    }
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
