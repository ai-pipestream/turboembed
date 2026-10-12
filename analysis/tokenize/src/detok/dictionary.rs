//! The detokenizer dictionary, as OpenNLP's `DetokenizationDictionary`
//! reads it: one entry per token, with its operation in the entry's
//! `operation` attribute.
//!
//! ```xml
//! <dictionary>
//!   <entry operation="MOVE_LEFT"><token>.</token></entry>
//!   <entry operation="RIGHT_LEFT_MATCHING"><token>"</token></entry>
//! </dictionary>
//! ```
//!
//! The rules follow the Java: an entry must have exactly one token, its
//! `operation` must name one of the four operations exactly, tokens are
//! compared exactly (the dictionary's `case_sensitive` attribute is
//! ignored, as the Java ignores it), and when a token appears twice the
//! later entry wins.

use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};

use turbo_maxent::{Entries, Entry};

use super::Operation;

/// Why a detokenizer dictionary could not be built.
#[derive(Debug)]
pub enum DictionaryError {
    /// The XML could not be read as an OpenNLP dictionary.
    Read(turbo_maxent::Error),
    /// The entry at this index (0-based, in document order) does not have
    /// exactly one token. OpenNLP refuses it too
    /// (`Each entry must have exactly one token!`).
    TokenCount { index: usize, tokens: Box<[Box<str>]> },
    /// The entry at this index has no `operation` attribute. OpenNLP
    /// refuses it as `Unknown operation type: null`.
    MissingOperation { index: usize, token: Box<str> },
    /// The entry at this index has an `operation` attribute that names no
    /// operation.
    UnknownOperation { index: usize, token: Box<str>, value: Box<str> },
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DictionaryError::Read(error) => write!(f, "detokenizer dictionary: {error}"),
            DictionaryError::TokenCount { index, tokens } => {
                write!(f, "detokenizer dictionary entry {index}: {} tokens {tokens:?}, expected one", tokens.len())
            }
            DictionaryError::MissingOperation { index, token } => {
                write!(f, "detokenizer dictionary entry {index}: token {token:?} has no operation attribute")
            }
            DictionaryError::UnknownOperation { index, token, value } => {
                write!(f, "detokenizer dictionary entry {index}: token {token:?} has unknown operation {value:?}")
            }
        }
    }
}

impl std::error::Error for DictionaryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            DictionaryError::Read(error) => Some(error),
            _ => None,
        }
    }
}

impl From<turbo_maxent::Error> for DictionaryError {
    fn from(error: turbo_maxent::Error) -> DictionaryError {
        DictionaryError::Read(error)
    }
}

/// What the dictionary holds for one token.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Rule {
    pub(crate) operation: Operation,
    /// For a `RightLeftMatching` token, its own index among the matching
    /// tokens, which the detokenizer uses to keep one open-or-closed flag
    /// per token. Unused for the other operations.
    pub(crate) slot: u32,
}

/// The 64-bit FNV-1a hash as a `Hasher`. Dictionary tokens are short, and
/// this hashes them faster than the standard library's default while
/// giving the same lookups.
#[derive(Clone, Debug)]
pub(crate) struct Fnv(u64);

impl Default for Fnv {
    fn default() -> Fnv {
        Fnv(0xCBF2_9CE4_8422_2325)
    }
}

impl Hasher for Fnv {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3);
        }
    }
}

/// A map from tokens to their [`Operation`].
#[derive(Clone, Debug, Default)]
pub struct DetokenizerDictionary {
    rules: HashMap<Box<str>, Rule, BuildHasherDefault<Fnv>>,
    /// The number of matching slots handed out so far.
    slots: u32,
}

impl DetokenizerDictionary {
    /// An empty dictionary.
    pub fn new() -> DetokenizerDictionary {
        DetokenizerDictionary::default()
    }

    /// Reads a dictionary from OpenNLP's dictionary XML.
    pub fn read(xml: &[u8]) -> Result<DetokenizerDictionary, DictionaryError> {
        DetokenizerDictionary::from_entries(&Entries::read(xml)?)
    }

    /// Builds a dictionary from entries already read, checking each as
    /// OpenNLP does.
    pub fn from_entries(entries: &Entries) -> Result<DetokenizerDictionary, DictionaryError> {
        let mut dictionary = DetokenizerDictionary::new();
        for (index, entry) in entries.iter().enumerate() {
            let (token, operation) = check_entry(index, entry)?;
            dictionary.insert(token, operation);
        }
        Ok(dictionary)
    }

    /// Sets the operation of `token`, replacing any earlier one, as
    /// OpenNLP's array constructor does. Setting
    /// [`NoOperation`](Operation::NoOperation) makes the token behave as
    /// if the dictionary did not hold it.
    pub fn insert(&mut self, token: &str, operation: Operation) {
        if let Some(rule) = self.rules.get_mut(token) {
            if operation == Operation::RightLeftMatching && rule.operation != Operation::RightLeftMatching {
                rule.slot = self.slots;
                self.slots += 1;
            }
            rule.operation = operation;
            return;
        }
        let slot = if operation == Operation::RightLeftMatching {
            self.slots += 1;
            self.slots - 1
        } else {
            0
        };
        self.rules.insert(token.into(), Rule { operation, slot });
    }

    /// The operation the dictionary gives `token`, before any
    /// `RightLeftMatching` is resolved, or `None` when it has none.
    pub fn get(&self, token: &str) -> Option<Operation> {
        self.rule(token).map(|r| r.operation)
    }

    /// The number of distinct tokens.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    pub(crate) fn rule(&self, token: &str) -> Option<Rule> {
        self.rules.get(token).copied()
    }

    /// The number of matching slots the detokenizer must track: an upper
    /// bound on every `slot` of a `RightLeftMatching` rule.
    pub(crate) fn slots(&self) -> u32 {
        self.slots
    }
}

/// The token and operation of one entry, or why the entry is not valid.
fn check_entry(index: usize, entry: &Entry) -> Result<(&str, Operation), DictionaryError> {
    let [token] = entry.tokens.as_slice() else {
        return Err(DictionaryError::TokenCount { index, tokens: entry.tokens.clone().into_boxed_slice() });
    };
    let Some(value) = entry.attribute("operation") else {
        return Err(DictionaryError::MissingOperation { index, token: token.clone() });
    };
    match Operation::from_attribute(value) {
        Some(operation) => Ok((token, operation)),
        None => Err(DictionaryError::UnknownOperation { index, token: token.clone(), value: value.into() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn later_entries_win() {
        let mut d = DetokenizerDictionary::new();
        d.insert("-", Operation::MergeToLeft);
        d.insert("-", Operation::MergeBoth);
        assert_eq!(d.get("-"), Some(Operation::MergeBoth));
        assert_eq!(d.len(), 1);
        assert_eq!(d.get("+"), None);
    }

    #[test]
    fn matching_slots_are_handed_out_once_per_token() {
        let mut d = DetokenizerDictionary::new();
        d.insert("\"", Operation::RightLeftMatching);
        d.insert("'", Operation::RightLeftMatching);
        // A repeated matching entry keeps its slot.
        d.insert("\"", Operation::RightLeftMatching);
        assert_eq!(d.rule("\"").unwrap().slot, 0);
        assert_eq!(d.rule("'").unwrap().slot, 1);
        assert_eq!(d.slots(), 2);
        // A token that becomes matching later gets a new slot.
        d.insert("|", Operation::MergeToLeft);
        d.insert("|", Operation::RightLeftMatching);
        assert_eq!(d.rule("|").unwrap().slot, 2);
        assert_eq!(d.slots(), 3);
    }

    #[test]
    fn fnv_reference_value() {
        let mut h = Fnv::default();
        h.write(b"a");
        assert_eq!(h.finish(), 0xAF63_DC4C_8601_EC8C);
    }
}
