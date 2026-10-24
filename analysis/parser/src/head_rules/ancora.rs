//! AnCora Spanish head rules (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`); their
//! tags use the Java regular expression subset in [`turbo_jvm::TagPattern`].

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::str::FromStr;
use std::sync::LazyLock;

use crate::parse::{Constituent, Head, Parse, ParseError, TOK_NODE};

pub use turbo_jvm::TagPattern;

use super::*;

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
