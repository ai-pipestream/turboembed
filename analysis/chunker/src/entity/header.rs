//! Per-chunk entity headers (`turbo:chunk-entities`): the entities each
//! chunk mentions, then the ones in scope from earlier in its section,
//! as entity keys and the `opennlp:entities` index naming each, never as
//! copied strings. [`ChunkEntities::header`] projects a chunk's entries
//! onto the text when an embedder wants them as a prefix. With
//! `DocumentEntities` and `EntitySalience` present, [`SalienceHeaders`]
//! can order a chunk's entries by document salience and append the
//! document's most salient entities after the carried ones.

use turbo_analysis::{Anchor, Annotator, Error, Groups, Ix, Key, KeyId, Layer, LayerName, Refs, View};
use turbo_analysis_layers::{Chunks, DocumentEntities, Entities, EntitySalience, Sentences};

use super::salient::{self, SalienceHeaders};
use super::{Chains, EntityKeys, Keying, chunk_entities, fnv, same_view};
use crate::coref::Mentions;

/// What a chunk header holds besides the entities the chunk mentions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct HeaderPolicy {
    /// The most entities carried into a chunk from earlier in its section
    /// (its parent row, or the document before it when it has none),
    /// most recently mentioned first. Default 4; 0 carries none.
    pub max_carried: u32,
}

impl Default for HeaderPolicy {
    fn default() -> HeaderPolicy {
        HeaderPolicy { max_carried: 4 }
    }
}

/// The entity header of every row of `turbo:chunks`. Row `c`'s entries
/// are `row[c]..row[c + 1]`: first the keys with a mention whose bytes
/// overlap the chunk's fragments, by the sentence of their first such
/// mention and then by key, then the keys carried from mentions before
/// the chunk in its section, latest sentence first and then by key,
/// with `mentions` zero. [`SalienceHeaders`] may reorder the first group
/// and append the document's most salient keys, also with `mentions`
/// zero, after the second.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkEntities {
    /// For each entity key, the entity that names it (its first mention).
    pub first: Vec<Ix<Entities>>,
    /// Entry offsets, one per chunk and one past the last.
    pub row: Vec<u32>,
    /// The key of each entry, an index into `first`.
    pub key: Vec<u32>,
    /// The distinct mentions of the key in the chunk; zero when carried.
    pub mentions: Vec<u32>,
}

impl ChunkEntities {
    /// Chunk `c`'s entries: (key, mentions in the chunk).
    pub fn entries(&self, c: Ix<Chunks>) -> impl Iterator<Item = (u32, u32)> + '_ {
        let i = c.get();
        let r = match (self.row.get(i), self.row.get(i + 1)) {
            (Some(&a), Some(&b)) => a as usize..b as usize,
            _ => 0..0,
        };
        self.key[r.clone()].iter().copied().zip(self.mentions[r].iter().copied())
    }

    /// The entities that name chunk `c`'s header entries, in entry order.
    pub fn entities(&self, c: Ix<Chunks>) -> impl Iterator<Item = Ix<Entities>> + '_ {
        self.entries(c).map(|(k, _)| self.first[k as usize])
    }

    /// The surface of key `k` in `text`: its naming entity's fragments,
    /// joined by a space.
    pub fn surface(&self, k: u32, entities: &Entities, text: &str) -> Option<String> {
        let e = *self.first.get(k as usize)?;
        let parts: Vec<&str> = entities.spans.fragments(e).filter_map(|f| text.get(f.range())).collect();
        (!parts.is_empty()).then(|| parts.join(" ").split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// Chunk `c`'s header as text, `[Entities: A; B | in scope: C]`, or
    /// `None` when it has no entries.
    pub fn header(&self, c: Ix<Chunks>, entities: &Entities, text: &str) -> Option<String> {
        let (mut named, mut carried) = (Vec::new(), Vec::new());
        for (k, m) in self.entries(c) {
            let s = self.surface(k, entities, text)?;
            if m > 0 { named.push(s) } else { carried.push(s) }
        }
        match (named.is_empty(), carried.is_empty()) {
            (true, true) => None,
            (false, true) => Some(format!("[Entities: {}]", named.join("; "))),
            (true, false) => Some(format!("[In scope: {}]", carried.join("; "))),
            (false, false) => Some(format!("[Entities: {} | in scope: {}]", named.join("; "), carried.join("; "))),
        }
    }
}

impl Layer for ChunkEntities {
    const ID: &'static str = "turbo:chunk-entities";
    const OF: Option<&'static LayerName> = Some(Chunks::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }

    fn len(&self) -> usize {
        self.row.len().saturating_sub(1)
    }

    fn indexes(&self) -> Vec<(KeyId, usize)> {
        match self.first.iter().map(|e| e.get() + 1).max() {
            Some(n) => vec![(Key::<Entities>::DEFAULT.id(), n)],
            None => Vec::new(),
        }
    }

    fn check(&self) -> Result<(), String> {
        if self.row.first() != Some(&0)
            || self.row.last().copied().unwrap_or(0) as usize != self.key.len()
            || !self.row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("row must start at 0, be non-decreasing, and end at key.len()".into());
        }
        if self.mentions.len() != self.key.len() {
            return Err("a mention count for every entry".into());
        }
        if self.key.iter().any(|&k| k as usize >= self.first.len()) {
            return Err("an entry key outside first".into());
        }
        for (c, w) in self.row.windows(2).enumerate() {
            let r = w[0] as usize..w[1] as usize;
            let carried_at = self.mentions[r.clone()].iter().position(|&m| m == 0).unwrap_or(r.len());
            if self.mentions[r.start + carried_at..r.end].iter().any(|&m| m > 0) {
                return Err(format!("chunk {c}: a mentioned entry after a carried one"));
            }
            let mut keys = self.key[r].to_vec();
            keys.sort_unstable();
            if keys.windows(2).any(|k| k[0] == k[1]) {
                return Err(format!("chunk {c}: an entity key twice"));
            }
        }
        Ok(())
    }
}

/// Writes [`ChunkEntities`] for the chunks.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkEntityHeaders {
    pub keying: Keying,
    pub policy: HeaderPolicy,
    pub salience: SalienceHeaders,
}

impl Annotator for ChunkEntityHeaders {
    const ID: &'static str = "turbo:chunk-entity-headers";
    type Reads = (
        Sentences,
        Chunks,
        View,
        Option<Entities>,
        Option<Mentions>,
        Option<Groups<Mentions>>,
        Option<Groups<Entities>>,
        Option<DocumentEntities>,
        Option<EntitySalience>,
    );
    type Writes = (ChunkEntities,);

    fn annotate(
        &self,
        (sentences, chunks, view, entities, mentions, mention_groups, entity_groups, document, salience): Refs<
            '_,
            Self,
        >,
    ) -> Result<(ChunkEntities,), Error> {
        same_view(Self::ID, sentences, view, entities, mentions)?;
        if chunks.spans.view() != sentences.spans.view() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let chains = Chains { mentions, mention_groups, entity_groups };
        let keys = EntityKeys::build(sentences, entities, view.text(), self.keying, chains);
        let mut out = chunk_entities(chunks, &keys, self.policy);
        if let (false, Some(d), Some(s)) = (self.salience.is_off(), document, salience) {
            if d.view != view.id() || d.entities.len() != s.scores.len() {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: "salience headers need matching consolidated entities and view".into(),
                });
            }
            s.check().map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
            salient::apply(&mut out, &keys, d, s, self.salience);
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        if self.salience.is_off() {
            fnv(&[self.keying.bits(), u64::from(self.policy.max_carried)])
        } else {
            fnv(&[self.keying.bits(), u64::from(self.policy.max_carried), self.salience.bits()])
        }
    }
}
