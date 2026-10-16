//! The readings found for one input: OpenNLP's `Results`, an insertion
//! ordered set of stems or analyses.

use std::collections::HashSet;

use crate::affix::Affix;
use crate::dictionary::{Dictionary, EntryId};

/// The most readings one input returns.
pub(crate) const MAX_ANALYSES: usize = 2048;

/// Distinct strings in the order they were first added. Small sets are
/// searched in place; a larger one keeps an index.
#[derive(Default)]
pub(crate) struct OrderedSet {
    values: Vec<String>,
    index: Option<HashSet<String>>,
}

/// The size from which the set keeps an index.
const INDEXED_FROM: usize = 16;

impl OrderedSet {
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub(crate) fn contains(&self, value: &str) -> bool {
        match &self.index {
            Some(index) => index.contains(value),
            None => self.values.iter().any(|v| v == value),
        }
    }

    /// Adds `value` unless the set holds it.
    pub(crate) fn insert(&mut self, value: String) {
        if self.contains(&value) {
            return;
        }
        if let Some(index) = &mut self.index {
            index.insert(value.clone());
        } else if self.values.len() + 1 >= INDEXED_FROM {
            let mut index: HashSet<String> = self.values.iter().cloned().collect();
            index.insert(value.clone());
            self.index = Some(index);
        }
        self.values.push(value);
    }

    pub(crate) fn into_vec(self) -> Vec<String> {
        self.values
    }

    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[String] {
        &self.values
    }
}

/// One selected reading of a compound part.
#[derive(Clone)]
pub(crate) struct CompoundPart<'d> {
    /// The part as written (or with its first letter made upper case).
    pub(crate) surface: String,
    /// The spelling of the entry.
    pub(crate) root: String,
    pub(crate) entry: EntryId,
    /// The affixes undone, in the order they were applied.
    pub(crate) affixes: Vec<&'d Affix>,
}

/// The readings of one input or piece of it.
pub(crate) struct Results {
    pub(crate) morphological: bool,
    pub(crate) values: OrderedSet,
    /// A direct or affixed reading reached a forbidden entry, which rules
    /// out the compound and break readings of the input.
    pub(crate) forbidden: bool,
    /// The input is the part of a Hungarian word before a hyphen, read as
    /// a compound under the relaxed rules for it.
    pub(crate) hyphenated_first_part: bool,
}

impl Results {
    pub(crate) fn new(morphological: bool) -> Results {
        Results { morphological, values: OrderedSet::default(), forbidden: false, hyphenated_first_part: false }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Adds readings until the set holds [`MAX_ANALYSES`].
    pub(crate) fn add_all(&mut self, additions: impl IntoIterator<Item = String>) {
        for value in additions {
            if self.values.len() >= MAX_ANALYSES {
                return;
            }
            self.values.insert(value);
        }
    }

    /// Adds a complete compound: every part's stems, or each combination
    /// of the parts' analyses, each part introduced by `pa:` and its text.
    pub(crate) fn add_compound(&mut self, dictionary: &Dictionary, parts: &[CompoundPart<'_>]) {
        if !self.morphological {
            for part in parts {
                self.add_all(dictionary.morphological_stems(&part.root, part.entry, &part.affixes));
            }
            return;
        }
        let mut accumulated = vec![String::new()];
        for (i, part) in parts.iter().enumerate() {
            let mut next = Vec::new();
            let last = i == parts.len() - 1;
            for fields in dictionary.morphological_analyses(&part.root, part.entry, last, &part.affixes) {
                for prior in &accumulated {
                    if next.len() >= MAX_ANALYSES {
                        break;
                    }
                    let mut s = String::with_capacity(prior.len() + part.surface.len() + fields.len() + 5);
                    s.push_str(prior);
                    if !prior.is_empty() {
                        s.push(' ');
                    }
                    s.push_str("pa:");
                    s.push_str(&part.surface);
                    if !fields.is_empty() {
                        s.push(' ');
                    }
                    s.push_str(&fields);
                    next.push(s);
                }
            }
            accumulated = next;
        }
        self.add_all(accumulated);
    }

    /// Adds the readings of a word broken in two: both sides' stems, or each
    /// pair of analyses, a side that is no compound introduced by `pa:` and
    /// its text.
    pub(crate) fn add_broken(&mut self, left_text: &str, left: &[String], right_text: &str, right: &[String]) {
        if !self.morphological {
            self.add_all(left.iter().cloned());
            self.add_all(right.iter().cloned());
            return;
        }
        for first in left {
            for last in right {
                if self.values.len() >= MAX_ANALYSES {
                    return;
                }
                let mut s = String::new();
                if !first.starts_with("pa:") {
                    s.push_str("pa:");
                    s.push_str(left_text);
                    s.push(' ');
                }
                s.push_str(first);
                s.push(' ');
                if !last.starts_with("pa:") {
                    s.push_str("pa:");
                    s.push_str(right_text);
                    s.push(' ');
                }
                s.push_str(last);
                self.values.insert(s);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_set_keeps_first_insertion_order() {
        let mut set = OrderedSet::default();
        for i in 0..40 {
            set.insert(format!("{}", i % 25));
        }
        assert_eq!(set.len(), 25);
        assert_eq!(set.as_slice()[24], "24");
        assert!(set.contains("3"));
    }
}
