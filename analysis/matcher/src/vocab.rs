//! The strings a matcher's patterns name, as ids, and its labels.

use std::collections::HashMap;

use crate::error::MatchError;
use crate::hash::BuildFoldHasher;

/// The id of a string no pattern names. A token whose value is not in the
/// vocabulary gets it, and it equals no pattern's value.
pub(crate) const NONE: u32 = u32::MAX;

/// Every string a matcher's patterns name, each with an id. A document's
/// token values are looked up here once per call, so the tests the
/// patterns make are comparisons of ids.
#[derive(Clone, Debug, Default)]
pub(crate) struct Vocab {
    ids: HashMap<Box<str>, u32, BuildFoldHasher>,
    bytes: usize,
}

impl Vocab {
    pub(crate) fn intern(&mut self, s: &str) -> u32 {
        if let Some(&id) = self.ids.get(s) {
            return id;
        }
        let id = self.ids.len() as u32;
        self.bytes += s.len();
        self.ids.insert(s.into(), id);
        id
    }

    #[inline]
    pub(crate) fn get(&self, s: &str) -> u32 {
        self.ids.get(s).copied().unwrap_or(NONE)
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// The table's slots and the strings' own bytes; the allocator's
    /// rounding is not counted.
    pub(crate) fn heap_bytes(&self) -> usize {
        // A slot holds a `Box<str>` (16 bytes) and a `u32`, padded to 24,
        // plus one control byte.
        self.ids.capacity() * 25 + self.bytes
    }
}

/// A label a matcher gives its matches: the key patterns and phrases are
/// added under, as spaCy's `match_id`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub u32);

/// A matcher's labels. Removing a label retires its id: adding the name
/// again gives a new id, so nothing added before the removal comes back.
#[derive(Clone, Debug, Default)]
pub(crate) struct Labels {
    names: Vec<Box<str>>,
    active: Vec<bool>,
    by_name: HashMap<Box<str>, u32, BuildFoldHasher>,
}

impl Labels {
    pub(crate) fn get_or_add(&mut self, name: &str) -> Label {
        if let Some(&id) = self.by_name.get(name) {
            return Label(id);
        }
        let id = self.names.len() as u32;
        self.names.push(name.into());
        self.active.push(true);
        self.by_name.insert(name.into(), id);
        Label(id)
    }

    pub(crate) fn get(&self, name: &str) -> Option<Label> {
        self.by_name.get(name).map(|&id| Label(id))
    }

    pub(crate) fn remove(&mut self, name: &str) -> Result<Label, MatchError> {
        let id = self.by_name.remove(name).ok_or_else(|| MatchError::NoLabel(name.into()))?;
        self.active[id as usize] = false;
        Ok(Label(id))
    }

    #[inline]
    pub(crate) fn is_active(&self, l: Label) -> bool {
        self.active[l.0 as usize]
    }

    pub(crate) fn name(&self, l: Label) -> &str {
        &self.names[l.0 as usize]
    }

    /// How many labels are in use.
    pub(crate) fn len(&self) -> usize {
        self.by_name.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removing_a_label_retires_its_id() {
        let mut l = Labels::default();
        let a = l.get_or_add("A");
        assert_eq!(l.get_or_add("A"), a);
        assert_eq!(l.len(), 1);
        assert_eq!(l.remove("A"), Ok(a));
        assert!(!l.is_active(a));
        assert_eq!(l.remove("A"), Err(MatchError::NoLabel("A".into())));
        let again = l.get_or_add("A");
        assert_ne!(again, a);
        assert_eq!(l.name(again), "A");
        assert_eq!((l.len(), l.names.len()), (1, 2));
    }

    #[test]
    fn unknown_strings_get_none() {
        let mut v = Vocab::default();
        assert_eq!(v.intern("a"), 0);
        assert_eq!(v.intern("b"), 1);
        assert_eq!(v.intern("a"), 0);
        assert_eq!(v.get("b"), 1);
        assert_eq!(v.get("c"), NONE);
        assert_eq!(v.len(), 2);
    }
}
