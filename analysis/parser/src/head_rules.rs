//! Head rules and gap labelers for constituency parsing (`opennlp.tools.parser.HeadRules`,
//! `opennlp.tools.parser.GapLabeler`, `opennlp.tools.parser.lang.en.HeadRules`, and
//! `opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::{self, BufRead, Write};
use std::str::FromStr;

use crate::parse::{Constituent, Parse, TOK_NODE};

/// Maximum number of tags permitted in a single head rule (`MAX_TAGS_PER_RULE = 1_000`).
pub const MAX_TAGS_PER_RULE: usize = 1_000;

/// Maximum number of states in a compiled [`TagPattern`] bitmask automaton (`MAX_PATTERN_STATES = 62`).
pub const MAX_PATTERN_STATES: usize = 62;

/// Version 2 format header for AnCora Spanish head rules (`AncoraSpanishHeadRules.FORMAT_HEADER`).
pub const ANCORA_FORMAT_HEADER: &str = "# opennlp-ancora-head-rules 2";

const PUNCTUATION_TAGS: [&str; 4] = [".", ",", "``", "''"];

/// Labels constituents on the parser stack that contain a trace gap (`opennlp.tools.parser.GapLabeler`).
pub trait GapLabeler {
    /// Inspects the constituent stack when a `-NONE-` trace node is encountered and mutates
    /// labels on matching extraction chains.
    fn label_gaps(&self, stack: &mut [Constituent]);
}

/// Resolves the lexical head child of a constituent (`opennlp.tools.parser.HeadRules`).
pub trait HeadRules {
    /// Returns the head constituent for `constituents` under parent label `kind`, or `None`
    /// if `constituents` starts with a terminal token node (`TOK_NODE`).
    fn get_head<'a>(&self, constituents: &'a [Parse], kind: &str) -> Option<&'a Parse>;

    /// Returns the set of POS tags treated as punctuation when assigning heads.
    fn punctuation_tags(&self) -> &HashSet<String>;
}

/// Errors returned when loading or serializing head rules.
#[derive(Debug)]
pub enum HeadRulesError {
    /// An underlying I/O failure.
    Io(io::Error),
    /// A malformed head rules line, header, or tag pattern.
    InvalidFormat(String),
}

impl fmt::Display for HeadRulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "{err}"),
            Self::InvalidFormat(msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for HeadRulesError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidFormat(_) => None,
        }
    }
}

impl From<io::Error> for HeadRulesError {
    fn from(err: io::Error) -> Self {
        Self::Io(err)
    }
}

/// A single head-finding rule specifying search direction and priority-ordered tags.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadRule {
    /// Whether constituents are searched left-to-right (`true`) or right-to-left (`false`).
    pub left_to_right: bool,
    /// Priority-ordered tag strings or globs.
    pub tags: Vec<String>,
}

/// A 62-state `u64` bitmask automaton matching AnCora Spanish POS tag globs without regex or
/// backtracking (`AncoraSpanishHeadRules.TagPattern`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagPattern {
    character_states: [u64; 128],
    wildcard_states: u64,
    accept_state: u64,
}

impl TagPattern {
    /// Compiles an AnCora Spanish tag glob into a 62-state bitmask automaton.
    ///
    /// Supported syntax:
    /// - Case-sensitive ASCII literals (`A..=Z`, `0..=9`, `.`, `$`)
    /// - `*` for zero or more characters (consecutive `**` is rejected)
    /// - Bounded character choices `[MAS]`, `[CS]`, `[IS]`, `[12]`, `[AC]`
    pub fn compile(glob: &str) -> Result<Self, String> {
        let bytes = glob.as_bytes();
        let mut character_states = [0u64; 128];
        let mut wildcards = 0u64;
        let mut states = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            if states == MAX_PATTERN_STATES {
                return Err(format!("pattern exceeds {MAX_PATTERN_STATES} states"));
            }
            let state = 1u64 << states;
            let b = bytes[i];
            if b == b'*' {
                if states > 0 && (wildcards & (state >> 1)) != 0 {
                    return Err("consecutive wildcards are not supported".to_string());
                }
                wildcards |= state;
                i += 1;
            } else if b == b'[' {
                let Some(rel_end) = glob[i + 1..].find(']') else {
                    return Err("unterminated character choice".to_string());
                };
                let end = i + 1 + rel_end;
                let choice = &glob[i + 1..end];
                if !is_supported_choice(choice) {
                    return Err(format!("unsupported character choice [{choice}]"));
                }
                for &ch in choice.as_bytes() {
                    character_states[ch as usize] |= state;
                }
                i = end + 1;
            } else {
                let c = glob[i..].chars().next().unwrap();
                if !is_ancora_literal(c) {
                    return Err(format!("unsupported character '{c}'"));
                }
                character_states[b as usize] |= state;
                i += 1;
            }
            states += 1;
        }
        if states == 0 {
            return Err("pattern must not be empty".to_string());
        }
        Ok(Self { character_states, wildcard_states: wildcards, accept_state: 1u64 << states })
    }

    const fn compile_const(glob: &str) -> Self {
        let bytes = glob.as_bytes();
        let mut character_states = [0u64; 128];
        let mut wildcards = 0u64;
        let mut states = 0usize;
        let mut i = 0usize;
        while i < bytes.len() {
            assert!(states < MAX_PATTERN_STATES, "pattern exceeds 62 states");
            let state = 1u64 << states;
            let b = bytes[i];
            if b == b'*' {
                assert!(!(states > 0 && (wildcards & (state >> 1)) != 0), "consecutive wildcards are not supported");
                wildcards |= state;
                i += 1;
            } else if b == b'[' {
                let mut end = i + 1;
                while end < bytes.len() && bytes[end] != b']' {
                    character_states[bytes[end] as usize] |= state;
                    end += 1;
                }
                assert!(end < bytes.len(), "unterminated character choice");
                i = end + 1;
            } else {
                character_states[b as usize] |= state;
                i += 1;
            }
            states += 1;
        }
        assert!(states > 0, "pattern must not be empty");
        Self { character_states, wildcard_states: wildcards, accept_state: 1u64 << states }
    }

    /// Advances all active states in parallel across `tag` with zero allocation or backtracking.
    #[inline]
    pub fn matches(&self, tag: &str) -> bool {
        let mut states = self.epsilon_closure(1u64);
        for &b in tag.as_bytes() {
            let matching = if b < 128 { self.character_states[b as usize] } else { 0 };
            let next = ((states & matching) << 1) | (states & self.wildcard_states);
            states = self.epsilon_closure(next);
            if states == 0 {
                return false;
            }
        }
        (self.epsilon_closure(states) & self.accept_state) != 0
    }

    #[inline]
    const fn epsilon_closure(&self, states: u64) -> u64 {
        states | ((states & self.wildcard_states) << 1)
    }
}

#[inline]
fn is_supported_choice(choice: &str) -> bool {
    matches!(choice, "MAS" | "CS" | "IS" | "12" | "AC")
}

#[inline]
fn is_ancora_literal(c: char) -> bool {
    c.is_ascii_uppercase() || c.is_ascii_digit() || c == '.' || c == '$'
}

const ANCORA_TAGS1: [TagPattern; 8] = [
    TagPattern::compile_const("AQA*"),
    TagPattern::compile_const("AQC*"),
    TagPattern::compile_const("GRUP.A"),
    TagPattern::compile_const("S.A"),
    TagPattern::compile_const("NC*S*"),
    TagPattern::compile_const("NP*"),
    TagPattern::compile_const("NC*P*"),
    TagPattern::compile_const("GRUP.NOM"),
];

const ANCORA_TAGS2: [TagPattern; 3] =
    [TagPattern::compile_const("$"), TagPattern::compile_const("GRUP.A"), TagPattern::compile_const("SA")];

const ANCORA_TAGS3: [TagPattern; 8] = [
    TagPattern::compile_const("AQ0*"),
    TagPattern::compile_const("AQ[AC]*"),
    TagPattern::compile_const("AO*"),
    TagPattern::compile_const("GRUP.A"),
    TagPattern::compile_const("S.A"),
    TagPattern::compile_const("RG"),
    TagPattern::compile_const("RN"),
    TagPattern::compile_const("GRUP.NOM"),
];

const EN_NP_TAGS1: [&str; 7] = ["NN", "NNP", "NNPS", "NNS", "NX", "JJR", "POS"];
const EN_NP_TAGS2: [&str; 3] = ["$", "ADJP", "PRN"];
const EN_NP_TAGS3: [&str; 4] = ["JJ", "JJS", "RB", "QP"];

/// English constituency head rules and trace gap labeler (`opennlp.tools.parser.lang.en.HeadRules`).
#[derive(Clone, Debug)]
pub struct EnglishHeadRules {
    rule_order: Vec<String>,
    head_rules: HashMap<String, HeadRule>,
    punct_set: HashSet<String>,
}

impl PartialEq for EnglishHeadRules {
    fn eq(&self, other: &Self) -> bool {
        self.head_rules == other.head_rules && self.punct_set == other.punct_set
    }
}

impl Eq for EnglishHeadRules {}

impl EnglishHeadRules {
    /// Reads English head rules from a buffered reader (`new HeadRules(Reader)`).
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, HeadRulesError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_str(&buf)
    }

    /// Returns the rule registered for constituent `kind`, if any.
    pub fn rule(&self, kind: &str) -> Option<&HeadRule> {
        self.head_rules.get(kind)
    }

    /// Returns the number of constituent head rules loaded.
    pub fn len(&self) -> usize {
        self.head_rules.len()
    }

    /// Returns `true` if no constituent head rules are loaded.
    pub fn is_empty(&self) -> bool {
        self.head_rules.is_empty()
    }

    /// Serializes the head rules in OpenNLP text format (`HeadRules.serialize`).
    pub fn serialize<W: Write>(&self, mut writer: W) -> io::Result<()> {
        for kind in &self.rule_order {
            if let Some(rule) = self.head_rules.get(kind) {
                write!(writer, "{} {} {}", rule.tags.len() + 2, kind, if rule.left_to_right { "1" } else { "0" })?;
                for tag in &rule.tags {
                    write!(writer, " {tag}")?;
                }
                writeln!(writer)?;
            }
        }
        writer.flush()
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
        let mut rule_order = Vec::with_capacity(30);
        let mut head_rules: HashMap<String, HeadRule> = HashMap::with_capacity(30);
        for line in JavaLineIter::new(s) {
            let mut tokens = split_string_tokenizer(line);
            let Some(num) = tokens.next() else {
                return Err(HeadRulesError::InvalidFormat("Missing tag count in head rules".to_string()));
            };
            let Some(kind) = tokens.next() else {
                return Err(HeadRulesError::InvalidFormat("Missing constituent type in head rules".to_string()));
            };
            let Some(dir) = tokens.next() else {
                return Err(HeadRulesError::InvalidFormat("Missing direction in head rules".to_string()));
            };
            let raw_count: i64 = num
                .parse()
                .map_err(|_| HeadRulesError::InvalidFormat(format!("Invalid tag count in head rules: {num}")))?;
            let num_tags = raw_count - 2;
            if num_tags < 0 || num_tags as usize > MAX_TAGS_PER_RULE {
                return Err(HeadRulesError::InvalidFormat(format!("Invalid tag count in head rules: {num}")));
            }
            let num_tags = num_tags as usize;
            let mut tags = Vec::with_capacity(num_tags);
            for tok in tokens {
                tags.push(tok.to_string());
            }
            if tags.len() != num_tags {
                return Err(HeadRulesError::InvalidFormat(format!("Invalid tag count in head rules: {num}")));
            }
            if !head_rules.contains_key(kind) {
                rule_order.push(kind.to_string());
            }
            head_rules.insert(kind.to_string(), HeadRule { left_to_right: dir == "1", tags });
        }
        let punct_set = PUNCTUATION_TAGS.iter().map(|&s| s.to_string()).collect();
        Ok(Self { rule_order, head_rules, punct_set })
    }
}

impl HeadRules for EnglishHeadRules {
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.punct_set
    }

    fn get_head<'a>(&self, constituents: &'a [Parse], kind: &str) -> Option<&'a Parse> {
        if constituents.is_empty() || constituents[0].kind() == TOK_NODE {
            return None;
        }
        if kind == "NP" || kind == "NX" {
            for ci in (0..constituents.len()).rev() {
                for ti in (0..EN_NP_TAGS1.len()).rev() {
                    if constituents[ci].kind() == EN_NP_TAGS1[ti] {
                        return Some(constituents[ci].head());
                    }
                }
            }
            for constituent in constituents {
                if constituent.kind() == "NP" {
                    return Some(constituent.head());
                }
            }
            for ci in (0..constituents.len()).rev() {
                for ti in (0..EN_NP_TAGS2.len()).rev() {
                    if constituents[ci].kind() == EN_NP_TAGS2[ti] {
                        return Some(constituents[ci].head());
                    }
                }
            }
            for ci in (0..constituents.len()).rev() {
                for ti in (0..EN_NP_TAGS3.len()).rev() {
                    if constituents[ci].kind() == EN_NP_TAGS3[ti] {
                        return Some(constituents[ci].head());
                    }
                }
            }
            return Some(constituents[constituents.len() - 1].head());
        }
        if let Some(hr) = self.head_rules.get(kind) {
            let cl = constituents.len();
            if hr.left_to_right {
                for tag in &hr.tags {
                    for constituent in constituents {
                        if constituent.kind() == tag {
                            return Some(constituent.head());
                        }
                    }
                }
                return Some(constituents[0].head());
            } else {
                for tag in &hr.tags {
                    for ci in (0..cl).rev() {
                        if constituents[ci].kind() == tag {
                            return Some(constituents[ci].head());
                        }
                    }
                }
                return Some(constituents[cl - 1].head());
            }
        }
        Some(constituents[constituents.len() - 1].head())
    }
}

impl GapLabeler for EnglishHeadRules {
    fn label_gaps(&self, stack: &mut [Constituent]) {
        if stack.len() > 4 {
            let len = stack.len();
            let c1 = stack[len - 2].label() == "NP";
            let c2_s = stack[len - 3].label() == "S";
            let c2_vp = stack[len - 3].label() == "VP";
            let c3_sbar = stack[len - 4].label() == "SBAR";
            let c3_s = stack[len - 4].label() == "S";
            let c4_sbar = stack[len - 5].label() == "SBAR";

            if c1 && c2_s && c3_sbar {
                append_gap_label(&mut stack[len - 2]);
                append_gap_label(&mut stack[len - 3]);
                append_gap_label(&mut stack[len - 4]);
            } else if c1 && c2_vp && c3_s && c4_sbar {
                append_gap_label(&mut stack[len - 2]);
                append_gap_label(&mut stack[len - 3]);
                append_gap_label(&mut stack[len - 4]);
                append_gap_label(&mut stack[len - 5]);
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CompiledSpanishRule {
    rule: HeadRule,
    patterns: Vec<TagPattern>,
}

impl PartialEq for CompiledSpanishRule {
    fn eq(&self, other: &Self) -> bool {
        self.rule == other.rule
    }
}

impl Eq for CompiledSpanishRule {}

/// AnCora Spanish constituency head rules and trace gap labeler
/// (`opennlp.tools.parser.lang.es.AncoraSpanishHeadRules`).
///
/// Version 2 rule files start with `# opennlp-ancora-head-rules 2`. Tag patterns use
/// case-sensitive ASCII literals, `*` for zero or more characters, and the bounded character
/// choices used by the shipped AnCora tagset (`[MAS]`, `[CS]`, `[IS]`, `[12]`, `[AC]`).
/// Headerless legacy files are imported when they use the subset represented by the shipped
/// OpenNLP rules.
#[derive(Clone, Debug)]
pub struct AncoraSpanishHeadRules {
    rule_order: Vec<String>,
    head_rules: HashMap<String, CompiledSpanishRule>,
    punct_set: HashSet<String>,
}

impl PartialEq for AncoraSpanishHeadRules {
    fn eq(&self, other: &Self) -> bool {
        self.head_rules == other.head_rules && self.punct_set == other.punct_set
    }
}

impl Eq for AncoraSpanishHeadRules {}

impl AncoraSpanishHeadRules {
    /// Reads AnCora Spanish head rules from a buffered reader (`new AncoraSpanishHeadRules(Reader)`).
    pub fn read<R: BufRead>(mut reader: R) -> Result<Self, HeadRulesError> {
        let mut buf = String::new();
        reader.read_to_string(&mut buf)?;
        Self::from_str(&buf)
    }

    /// Returns the rule registered for constituent `kind`, if any.
    pub fn rule(&self, kind: &str) -> Option<&HeadRule> {
        self.head_rules.get(kind).map(|c| &c.rule)
    }

    /// Returns the number of constituent head rules loaded.
    pub fn len(&self) -> usize {
        self.head_rules.len()
    }

    /// Returns `true` if no constituent head rules are loaded.
    pub fn is_empty(&self) -> bool {
        self.head_rules.is_empty()
    }

    /// Serializes the head rules in version 2 format (`AncoraSpanishHeadRules.serialize`).
    pub fn serialize<W: Write>(&self, mut writer: W) -> io::Result<()> {
        writeln!(writer, "{ANCORA_FORMAT_HEADER}")?;
        for kind in &self.rule_order {
            if let Some(compiled) = self.head_rules.get(kind) {
                let rule = &compiled.rule;
                write!(writer, "{} {} {}", rule.tags.len() + 2, kind, if rule.left_to_right { "1" } else { "0" })?;
                for tag in &rule.tags {
                    write!(writer, " {tag}")?;
                }
                writeln!(writer)?;
            }
        }
        writer.flush()
    }

    /// Serializes the head rules in version 2 format into a `String`.
    pub fn serialize_to_string(&self) -> String {
        let mut out = Vec::new();
        self.serialize(&mut out).expect("writing to Vec<u8> cannot fail");
        String::from_utf8(out).expect("serialized head rules are valid UTF-8")
    }

    fn read_head_rule(&mut self, line: &str, line_number: usize, legacy: bool) -> Result<(), HeadRulesError> {
        let tokens: Vec<&str> = split_string_tokenizer(line).collect();
        if tokens.is_empty() {
            return Err(ancora_rule_error(line_number, None, "blank lines are not supported"));
        }
        let count_token = tokens[0];
        let count: i64 = count_token
            .parse()
            .map_err(|_| ancora_rule_error(line_number, None, &format!("invalid field count '{count_token}'")))?;
        let remaining = tokens.len() - 1;
        if count < 2 || (count - 2) as usize > MAX_TAGS_PER_RULE || remaining != count as usize {
            return Err(ancora_rule_error(line_number, None, "declared field count does not match the rule"));
        }
        let kind = tokens[1];
        let direction = tokens[2];
        if direction != "0" && direction != "1" {
            return Err(ancora_rule_error(line_number, Some(kind), "direction must be 0 or 1"));
        }
        if self.head_rules.contains_key(kind) {
            return Err(ancora_rule_error(line_number, Some(kind), "duplicate constituent rule"));
        }
        let tag_count = (count - 2) as usize;
        let mut tags = Vec::with_capacity(tag_count);
        let mut patterns = Vec::with_capacity(tag_count);
        for &source in &tokens[3..] {
            let glob = if legacy { convert_legacy_pattern(source, line_number, kind)? } else { source.to_string() };
            let pattern = TagPattern::compile(&glob).map_err(|msg| {
                ancora_rule_error(line_number, Some(kind), &format!("unsupported tag pattern '{source}': {msg}"))
            })?;
            tags.push(glob);
            patterns.push(pattern);
        }
        self.rule_order.push(kind.to_string());
        self.head_rules.insert(
            kind.to_string(),
            CompiledSpanishRule { rule: HeadRule { left_to_right: direction == "1", tags }, patterns },
        );
        Ok(())
    }
}

impl FromStr for AncoraSpanishHeadRules {
    type Err = HeadRulesError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut rules = Self {
            rule_order: Vec::with_capacity(60),
            head_rules: HashMap::with_capacity(60),
            punct_set: PUNCTUATION_TAGS.iter().map(|&p| p.to_string()).collect(),
        };
        let mut lines = JavaLineIter::new(s);
        let first_line = lines.next();
        let version_two = first_line == Some(ANCORA_FORMAT_HEADER);
        let mut line_number = 1usize;
        if let Some(line) = first_line {
            if line.starts_with('#') && !version_two {
                return Err(HeadRulesError::InvalidFormat(format!(
                    "Unsupported Ancora head rules header at line 1: {line}"
                )));
            }
            if !version_two {
                rules.read_head_rule(line, line_number, true)?;
            }
        }
        for line in lines {
            line_number += 1;
            rules.read_head_rule(line, line_number, !version_two)?;
        }
        Ok(rules)
    }
}

impl HeadRules for AncoraSpanishHeadRules {
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.punct_set
    }

    fn get_head<'a>(&self, constituents: &'a [Parse], kind: &str) -> Option<&'a Parse> {
        if constituents.is_empty() || constituents[0].kind() == TOK_NODE {
            return None;
        }
        if kind == "SN" || kind == "GRUP.NOM" {
            for constituent in constituents {
                for t in (0..ANCORA_TAGS1.len()).rev() {
                    if ANCORA_TAGS1[t].matches(constituent.kind()) {
                        return Some(constituent);
                    }
                }
            }
            for constituent in constituents {
                if constituent.kind() == "SN" || constituent.kind() == "GRUP.NOM" {
                    return Some(constituent);
                }
            }
            for ci in (0..constituents.len()).rev() {
                for ti in (0..ANCORA_TAGS2.len()).rev() {
                    if ANCORA_TAGS2[ti].matches(constituents[ci].kind()) {
                        return Some(&constituents[ci]);
                    }
                }
            }
            for ci in (0..constituents.len()).rev() {
                for ti in (0..ANCORA_TAGS3.len()).rev() {
                    if ANCORA_TAGS3[ti].matches(constituents[ci].kind()) {
                        return Some(&constituents[ci]);
                    }
                }
            }
            return Some(constituents[constituents.len() - 1].head());
        }
        if let Some(compiled) = self.head_rules.get(kind) {
            let cl = constituents.len();
            if compiled.rule.left_to_right {
                for pattern in &compiled.patterns {
                    for constituent in constituents {
                        if pattern.matches(constituent.kind()) {
                            return Some(constituent);
                        }
                    }
                }
                return Some(constituents[0].head());
            } else {
                for pattern in &compiled.patterns {
                    for ci in (0..cl).rev() {
                        if pattern.matches(constituents[ci].kind()) {
                            return Some(&constituents[ci]);
                        }
                    }
                }
                return Some(constituents[cl - 1].head());
            }
        }
        Some(constituents[constituents.len() - 1].head())
    }
}

impl GapLabeler for AncoraSpanishHeadRules {
    fn label_gaps(&self, stack: &mut [Constituent]) {
        if stack.len() > 4 {
            let len = stack.len();
            let c1 = stack[len - 2].label() == "SN";
            let c2_s = stack[len - 3].label() == "S";
            let c2_gv = stack[len - 3].label() == "GRUP.VERB";
            let c3_gn = stack[len - 4].label() == "GRUP.NOM";
            let c3_s = stack[len - 4].label() == "S";
            let c4_gn = stack[len - 5].label() == "GRUP.NOM";

            if c1 && c2_s && c3_gn {
                append_gap_label(&mut stack[len - 2]);
                append_gap_label(&mut stack[len - 3]);
                append_gap_label(&mut stack[len - 4]);
            } else if c1 && c2_gv && c3_s && c4_gn {
                append_gap_label(&mut stack[len - 2]);
                append_gap_label(&mut stack[len - 3]);
                append_gap_label(&mut stack[len - 4]);
                append_gap_label(&mut stack[len - 5]);
            }
        }
    }
}

fn append_gap_label(constituent: &mut Constituent) {
    let mut next = String::with_capacity(constituent.label().len() + 2);
    next.push_str(constituent.label());
    next.push_str("-G");
    constituent.set_label(next);
}

fn convert_legacy_pattern(source: &str, line_number: usize, kind: &str) -> Result<String, HeadRulesError> {
    if source == "GRUP.ADV" {
        return Ok(source.to_string());
    }
    let bytes = source.as_bytes();
    let mut glob = String::with_capacity(source.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'.' && i + 1 < bytes.len() && bytes[i + 1] == b'*' {
            glob.push('*');
            i += 2;
        } else if b == b'\\' {
            let mut escaped = i + 1;
            if escaped < bytes.len() && bytes[escaped] == b'\\' {
                escaped += 1;
            }
            if escaped >= bytes.len() || (bytes[escaped] != b'.' && bytes[escaped] != b'$') {
                return Err(ancora_rule_error(
                    line_number,
                    Some(kind),
                    &format!("unsupported legacy escape in '{source}'"),
                ));
            }
            glob.push(bytes[escaped] as char);
            i = escaped + 1;
        } else if matches!(b, b'.' | b'$' | b'*' | b'+' | b'|' | b'(' | b')' | b'?' | b'{' | b'}' | b'^') {
            return Err(ancora_rule_error(
                line_number,
                Some(kind),
                &format!("unsupported legacy regex construct in '{source}'"),
            ));
        } else {
            let ch = source[i..].chars().next().unwrap();
            glob.push(ch);
            i += ch.len_utf8();
        }
    }
    Ok(glob)
}

fn ancora_rule_error(line: usize, kind: Option<&str>, message: &str) -> HeadRulesError {
    match kind {
        Some(k) => {
            HeadRulesError::InvalidFormat(format!("Invalid Ancora head rules at line {line} for rule {k}: {message}"))
        }
        None => HeadRulesError::InvalidFormat(format!("Invalid Ancora head rules at line {line}: {message}")),
    }
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
