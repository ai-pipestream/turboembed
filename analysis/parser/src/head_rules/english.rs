//! English constituency head rules and trace gap labeler
//! (`opennlp.tools.parser.lang.en.HeadRules`).

use std::collections::HashSet;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

use crate::parse::{Constituent, Head, Parse, ParseError, TOK_NODE};

use super::*;

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
