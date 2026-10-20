//! Per-chunk entity headers (`turbo:chunk-entities`): the entities each
//! chunk mentions, then the ones in scope from earlier in its section,
//! as entity keys and the `opennlp:entities` index naming each, never as
//! copied strings. [`ChunkEntities::header`] projects a chunk's entries
//! onto the text when an embedder wants them as a prefix.

use turbo_analysis::{Anchor, Annotator, Error, Groups, Ix, Key, KeyId, Layer, LayerName, Refs, View};
use turbo_analysis_layers::{Chunks, Entities, Sentences};

use super::{Chains, EntityKeys, Keying, fnv, same_view};
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
/// are `row[c]..row[c + 1]`: first the keys it mentions, in order of
/// first mention, then the keys carried from earlier in its section,
/// most recent first, with `mentions` zero.
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

/// Builds the header of every row of `chunks` from `keys`.
pub fn chunk_entities(
    sentences: &Sentences,
    chunks: &Chunks,
    keys: &EntityKeys,
    policy: HeaderPolicy,
) -> ChunkEntities {
    let n = sentences.spans.len();
    // The keys each sentence mentions, ascending, with their counts.
    let mut by_sentence: Vec<Vec<(u32, u32)>> = vec![Vec::new(); n];
    for (k, occ) in keys.occurrences.iter().enumerate() {
        for &(s, count) in occ {
            if let Some(v) = by_sentence.get_mut(s as usize) {
                v.push((k as u32, count));
            }
        }
    }
    let mut out = ChunkEntities { first: keys.first.clone(), row: vec![0], key: Vec::new(), mentions: Vec::new() };
    let mut count = vec![0u32; keys.len()];
    let mut taken = vec![false; keys.len()];
    let clamp = |x: usize| x.min(n);
    for c in 0..chunks.spans.len() {
        let r = chunks.sentences[c];
        let (a, b) = (clamp(r.start.get()), clamp(r.end.get()));
        let start = out.key.len();
        for sentence in &by_sentence[a..b] {
            for &(k, m) in sentence {
                if count[k as usize] == 0 {
                    out.key.push(k);
                }
                count[k as usize] += m;
            }
        }
        for i in start..out.key.len() {
            let k = out.key[i] as usize;
            out.mentions.push(count[k]);
            count[k] = 0;
            taken[k] = true;
        }
        let scope = match chunks.parent[c] {
            Some(p) => Some(clamp(chunks.sentences[p.get()].start.get())),
            None if chunks.level[c] == 0 => Some(0),
            None => None,
        };
        let mut carried = 0;
        if let Some(lo) = scope.filter(|&lo| lo < a) {
            'back: for sentence in by_sentence[lo..a].iter().rev() {
                for &(k, _) in sentence {
                    if carried >= policy.max_carried {
                        break 'back;
                    }
                    if !taken[k as usize] {
                        taken[k as usize] = true;
                        out.key.push(k);
                        out.mentions.push(0);
                        carried += 1;
                    }
                }
            }
        }
        for &k in &out.key[start..] {
            taken[k as usize] = false;
        }
        out.row.push(out.key.len() as u32);
    }
    out
}

/// Writes [`ChunkEntities`] for the chunks.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkEntityHeaders {
    pub keying: Keying,
    pub policy: HeaderPolicy,
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
    );
    type Writes = (ChunkEntities,);

    fn annotate(
        &self,
        (sentences, chunks, view, entities, mentions, mention_groups, entity_groups): Refs<'_, Self>,
    ) -> Result<(ChunkEntities,), Error> {
        same_view(Self::ID, sentences, view, entities, mentions)?;
        if chunks.spans.view() != sentences.spans.view() {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        let chains = Chains { mentions, mention_groups, entity_groups };
        let keys = EntityKeys::build(sentences, entities, view.text(), self.keying, chains);
        Ok((chunk_entities(sentences, chunks, &keys, self.policy),))
    }

    fn config_hash(&self) -> u64 {
        fnv(&[self.keying.bits(), u64::from(self.policy.max_carried)])
    }
}
