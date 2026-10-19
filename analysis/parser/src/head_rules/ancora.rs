//! AnCora Spanish head rules (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`) and the
//! Java regular expression subset their tags use ([`TagPattern`]).

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::sync::LazyLock;

use crate::parse::{Constituent, Head, Parse, ParseError, TOK_NODE};

use super::*;

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
