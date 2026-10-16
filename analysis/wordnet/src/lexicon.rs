//! Core WordNet domain types (`Synset`, `LexicalKnowledgeBase`, `LemmaKey`, `InMemoryWordNetLexicon`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::error::WordNetError;
use crate::folding::fold;
use crate::pos::WordNetPos;
use crate::relation::WordNetRelation;

/// An immutable WordNet synset (`Synset`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Synset {
    id: String,
    pos: WordNetPos,
    lemmas: Vec<String>,
    gloss: String,
    relations: BTreeMap<WordNetRelation, Vec<String>>,
}

impl Synset {
    /// Creates a validated [`Synset`].
    ///
    /// Returns an error if `id` is empty, if any lemma in `lemmas` is empty, or if any
    /// entry in `relations` has an empty target list or an empty target synset ID.
    pub fn new(
        id: impl Into<String>,
        pos: WordNetPos,
        lemmas: Vec<String>,
        gloss: impl Into<String>,
        relations: BTreeMap<WordNetRelation, Vec<String>>,
    ) -> Result<Self, WordNetError> {
        let id = id.into();
        if id.is_empty() {
            return Err(WordNetError::invalid("id must not be null or empty"));
        }
        for lemma in &lemmas {
            if lemma.is_empty() {
                return Err(WordNetError::invalid(format!(
                    "lemmas must not contain a null or empty element for synset {id}"
                )));
            }
        }
        let gloss = gloss.into();
        for (rel, targets) in &relations {
            if targets.is_empty() {
                return Err(WordNetError::invalid(format!(
                    "Relation {rel:?} must map to a non-empty target list for synset {id}"
                )));
            }
            for target in targets {
                if target.is_empty() {
                    return Err(WordNetError::invalid(format!(
                        "Relation {rel:?} must not contain a null or empty target id for synset {id}"
                    )));
                }
            }
        }
        Ok(Self { id, pos, lemmas, gloss, relations })
    }

    /// Returns the opaque, source-qualified synset identifier.
    #[inline]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the coarse part of speech of this synset.
    #[inline]
    pub fn pos(&self) -> WordNetPos {
        self.pos
    }

    /// Returns the member lemmas in source order.
    #[inline]
    pub fn lemmas(&self) -> &[String] {
        &self.lemmas
    }

    /// Returns the definition gloss of this synset (empty string if none).
    #[inline]
    pub fn gloss(&self) -> &str {
        &self.gloss
    }

    /// Returns all outgoing relations mapped to target synset IDs in [`WordNetRelation`] order.
    #[inline]
    pub fn relations(&self) -> &BTreeMap<WordNetRelation, Vec<String>> {
        &self.relations
    }

    /// Returns the target synset IDs for a given [`WordNetRelation`], or an empty slice if none.
    #[inline]
    pub fn related(&self, relation: WordNetRelation) -> &[String] {
        self.relations.get(&relation).map_or(&[], Vec::as_slice)
    }
}

/// A folded sense-index key (`LemmaKey`).
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LemmaKey {
    lemma: String,
    pos: WordNetPos,
}

impl LemmaKey {
    /// Creates a [`LemmaKey`] without additional folding.
    pub fn new(lemma: impl Into<String>, pos: WordNetPos) -> Self {
        Self { lemma: lemma.into(), pos }
    }

    /// Folds `written_form` using [`crate::folding::fold`] and pairs it with `pos`.
    pub fn of(written_form: &str, pos: WordNetPos) -> Self {
        Self { lemma: fold(written_form), pos }
    }

    /// Returns the lemma string.
    #[inline]
    pub fn lemma(&self) -> &str {
        &self.lemma
    }

    /// Returns the part of speech.
    #[inline]
    pub fn pos(&self) -> WordNetPos {
        self.pos
    }
}

/// Read-only query interface over a WordNet lexical database (`LexicalKnowledgeBase`).
pub trait LexicalKnowledgeBase {
    /// Finds the synsets containing `lemma` with `pos`, in the source's sense order.
    fn lookup(&self, lemma: &str, pos: WordNetPos) -> Vec<&Synset>;

    /// Finds a synset by its opaque identifier.
    fn synset(&self, synset_id: &str) -> Option<&Synset>;

    /// Navigates one typed relation from `synset_id`, returning target synset IDs in source order.
    fn related(&self, synset_id: &str, relation: WordNetRelation) -> &[String] {
        self.synset(synset_id).map_or(&[], |s| s.related(relation))
    }

    /// Navigates one typed relation from `synset_id`, resolving each target synset in the lexicon.
    fn related_synsets(&self, synset_id: &str, relation: WordNetRelation) -> Vec<&Synset> {
        self.related(synset_id, relation).iter().filter_map(|id| self.synset(id)).collect()
    }

    /// Tests whether the lexicon contains `lemma` with `pos`.
    fn contains(&self, lemma: &str, pos: WordNetPos) -> bool {
        !self.lookup(lemma, pos).is_empty()
    }
}

impl<T: LexicalKnowledgeBase + ?Sized> LexicalKnowledgeBase for &T {
    #[inline]
    fn lookup(&self, lemma: &str, pos: WordNetPos) -> Vec<&Synset> {
        (**self).lookup(lemma, pos)
    }

    #[inline]
    fn synset(&self, synset_id: &str) -> Option<&Synset> {
        (**self).synset(synset_id)
    }

    #[inline]
    fn related(&self, synset_id: &str, relation: WordNetRelation) -> &[String] {
        (**self).related(synset_id, relation)
    }

    #[inline]
    fn related_synsets(&self, synset_id: &str, relation: WordNetRelation) -> Vec<&Synset> {
        (**self).related_synsets(synset_id, relation)
    }

    #[inline]
    fn contains(&self, lemma: &str, pos: WordNetPos) -> bool {
        (**self).contains(lemma, pos)
    }
}

impl<T: LexicalKnowledgeBase + ?Sized> LexicalKnowledgeBase for std::sync::Arc<T> {
    #[inline]
    fn lookup(&self, lemma: &str, pos: WordNetPos) -> Vec<&Synset> {
        (**self).lookup(lemma, pos)
    }

    #[inline]
    fn synset(&self, synset_id: &str) -> Option<&Synset> {
        (**self).synset(synset_id)
    }

    #[inline]
    fn related(&self, synset_id: &str, relation: WordNetRelation) -> &[String] {
        (**self).related(synset_id, relation)
    }

    #[inline]
    fn related_synsets(&self, synset_id: &str, relation: WordNetRelation) -> Vec<&Synset> {
        (**self).related_synsets(synset_id, relation)
    }

    #[inline]
    fn contains(&self, lemma: &str, pos: WordNetPos) -> bool {
        (**self).contains(lemma, pos)
    }
}

/// Immutable, referentially validated in-memory WordNet lexicon (`InMemoryWordNetLexicon`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InMemoryWordNetLexicon {
    synsets_by_id: HashMap<String, Synset>,
    sense_order: HashMap<LemmaKey, Vec<String>>,
}

impl InMemoryWordNetLexicon {
    /// Constructs a validated [`InMemoryWordNetLexicon`] from synset and sense-order maps.
    ///
    /// Verifies referential integrity:
    /// - Every key in `synsets_by_id` equals its [`Synset::id`].
    /// - Every relation target resolves to a synset in `synsets_by_id`.
    /// - Every sense-order entry resolves to existing synsets containing the key's folded lemma
    ///   and part of speech, with no duplicate synset IDs per key.
    pub fn new(
        synsets_by_id: HashMap<String, Synset>,
        sense_order: HashMap<LemmaKey, Vec<String>>,
    ) -> Result<Self, WordNetError> {
        for (id, synset) in &synsets_by_id {
            if id != synset.id() {
                return Err(WordNetError::invalid(format!("Synset table key {id} does not match its synset")));
            }
        }

        for synset in synsets_by_id.values() {
            for (rel, targets) in synset.relations() {
                for target in targets {
                    if !synsets_by_id.contains_key(target) {
                        return Err(WordNetError::invalid(format!(
                            "Synset {} has a {} relation to unknown synset {target}",
                            synset.id(),
                            rel.lmf_name().to_ascii_uppercase()
                        )));
                    }
                }
            }
        }

        let mut index = HashMap::with_capacity(sense_order.len());
        for (key, ordered_ids) in sense_order {
            let mut seen = HashSet::with_capacity(ordered_ids.len());
            for synset_id in &ordered_ids {
                if !seen.insert(synset_id.as_str()) {
                    return Err(WordNetError::invalid(format!(
                        "Sense index entry {} ({}) contains duplicate synset {synset_id}",
                        key.lemma(),
                        key.pos()
                    )));
                }
                let Some(synset) = synsets_by_id.get(synset_id) else {
                    return Err(WordNetError::invalid(format!(
                        "Sense index entry {} ({}) references unknown synset {synset_id}",
                        key.lemma(),
                        key.pos()
                    )));
                };
                let contains_lemma = synset.lemmas().iter().any(|lemma| key.lemma() == fold(lemma));
                if synset.pos() != key.pos() || !contains_lemma {
                    return Err(WordNetError::invalid(format!(
                        "Sense index entry {} ({}) references synset {synset_id}, which does not contain that lemma and part of speech",
                        key.lemma(),
                        key.pos()
                    )));
                }
            }
            index.insert(key, ordered_ids);
        }

        Ok(Self { synsets_by_id, sense_order: index })
    }

    /// Returns the number of synsets in this lexicon.
    #[inline]
    pub fn len(&self) -> usize {
        self.synsets_by_id.len()
    }

    /// Returns the number of synsets in this lexicon (Java `size()` parity alias).
    #[inline]
    pub fn size(&self) -> usize {
        self.synsets_by_id.len()
    }

    /// Returns `true` if this lexicon contains no synsets.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.synsets_by_id.is_empty()
    }

    /// Returns an iterator over all synsets in this lexicon.
    #[inline]
    pub fn synsets(&self) -> impl Iterator<Item = &Synset> {
        self.synsets_by_id.values()
    }

    /// Returns the map of all synsets keyed by synset ID.
    #[inline]
    pub fn synsets_by_id(&self) -> &HashMap<String, Synset> {
        &self.synsets_by_id
    }

    /// Returns the map of sense-ordered synset IDs keyed by [`LemmaKey`].
    #[inline]
    pub fn sense_order(&self) -> &HashMap<LemmaKey, Vec<String>> {
        &self.sense_order
    }

    /// Estimates the in-memory heap footprint in bytes of all stored synsets, relations, and sense orders.
    pub fn memory_footprint_bytes(&self) -> usize {
        let mut bytes = std::mem::size_of::<Self>();
        bytes += self.synsets_by_id.capacity() * (std::mem::size_of::<String>() + std::mem::size_of::<Synset>() + 8);
        for (k, s) in &self.synsets_by_id {
            bytes += k.capacity() + s.id.capacity() + s.gloss.capacity();
            bytes += s.lemmas.capacity() * std::mem::size_of::<String>();
            for l in &s.lemmas {
                bytes += l.capacity();
            }
            for targets in s.relations.values() {
                bytes += std::mem::size_of::<(WordNetRelation, Vec<String>)>();
                bytes += targets.capacity() * std::mem::size_of::<String>();
                for t in targets {
                    bytes += t.capacity();
                }
            }
        }
        bytes +=
            self.sense_order.capacity() * (std::mem::size_of::<LemmaKey>() + std::mem::size_of::<Vec<String>>() + 8);
        for (k, ids) in &self.sense_order {
            bytes += k.lemma.capacity();
            bytes += ids.capacity() * std::mem::size_of::<String>();
            for id in ids {
                bytes += id.capacity();
            }
        }
        bytes
    }
}

impl LexicalKnowledgeBase for InMemoryWordNetLexicon {
    fn lookup(&self, lemma: &str, pos: WordNetPos) -> Vec<&Synset> {
        let key = LemmaKey::of(lemma, pos);
        let Some(ids) = self.sense_order.get(&key) else {
            return Vec::new();
        };
        let mut resolved = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(synset) = self.synsets_by_id.get(id) {
                resolved.push(synset);
            }
        }
        resolved
    }

    #[inline]
    fn synset(&self, synset_id: &str) -> Option<&Synset> {
        self.synsets_by_id.get(synset_id)
    }
}
