//! Entity keys: which mentions of `opennlp:entities` name the same
//! entity, and in which sentences each entity is mentioned. Two entities
//! share a key when their normalized surfaces match (NFKC, full case
//! folding, whitespace collapsed, edge punctuation dropped), optionally
//! only within one entity kind, or when a coreference chain
//! (`Groups<Entities>`, or `Groups<Mentions>` with each mention tied to
//! an entity) puts them together. The mentions of such a chain that are
//! not entities (a pronoun, a nominal) count as mentions of its entity.

use std::collections::BTreeMap;

use turbo_analysis::{Groups, Ix, Span};
use turbo_analysis_layers::{Entities, Sentences};
use turbo_unicode::casefold;
use turbo_unicode::normalize::{Form, normalize};

use crate::coref::Mentions;

/// How entities are keyed.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Keying {
    /// Whether two entities must also have the same kind (label) to
    /// share a key. Off by default: a recognizer that tags one name
    /// as an organization once and a location once still names one
    /// thing to a reader.
    pub kind: bool,
    /// Whether coreference chains present in the document merge keys
    /// and add their non-entity mentions. On by default.
    pub chains: bool,
}

impl Default for Keying {
    fn default() -> Keying {
        Keying { kind: false, chains: true }
    }
}

impl Keying {
    /// The bits the config hashes of the steps that key entities fold in.
    pub(crate) fn bits(self) -> u64 {
        u64::from(self.kind) | u64::from(self.chains) << 1
    }
}

/// The coreference layers keying may read.
#[derive(Copy, Clone, Default)]
pub struct Chains<'a> {
    pub mentions: Option<&'a Mentions>,
    pub mention_groups: Option<&'a Groups<Mentions>>,
    pub entity_groups: Option<&'a Groups<Entities>>,
}

/// The surface of an entity as it is compared: NFKC, full case folding
/// and NFKC again, whitespace runs collapsed to one space, and the
/// characters that are not letters or digits trimmed from both ends.
pub fn normalize_surface(surface: &str) -> String {
    let mut nfkc = String::new();
    normalize(Form::Nfkc, surface, &mut nfkc);
    let mut folded = String::new();
    casefold::fold(&nfkc, &mut folded);
    nfkc.clear();
    normalize(Form::Nfkc, &folded, &mut nfkc);
    let mut out = String::with_capacity(nfkc.len());
    for word in nfkc.split(char::is_whitespace).filter(|w| !w.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out.trim_matches(|c: char| !c.is_alphanumeric()).to_owned()
}

/// The entity keys of one document.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EntityKeys {
    /// The key of each entity: `None` for one outside every sentence or
    /// whose surface normalizes to nothing.
    pub key: Vec<Option<u32>>,
    /// For each key, the entity that names it: its first in text order.
    pub first: Vec<Ix<Entities>>,
    /// For each key, the sentences it is mentioned in, ascending, with
    /// how many distinct mentions each holds.
    pub occurrences: Vec<Vec<(u32, u32)>>,
    /// Every distinct mention, by start, then end, then key.
    pub mentions: Vec<KeyMention>,
    /// The byte fragments of the mentions, `KeyMention::fragments` into it.
    pub fragments: Vec<Span>,
}

/// One distinct mention of a key: an entity, or a chain mention tied to
/// one, with its envelope span and its own fragments.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KeyMention {
    pub key: u32,
    pub sentence: u32,
    pub span: Span,
    /// `EntityKeys::fragments[fragments.0..fragments.1]`.
    pub fragments: (u32, u32),
}

impl EntityKeys {
    /// The byte fragments of mention `m`.
    pub fn fragments_of(&self, m: &KeyMention) -> &[Span] {
        &self.fragments[m.fragments.0 as usize..m.fragments.1 as usize]
    }
}

/// The sentence whose span holds `at`, if any.
fn sentence_at(sentences: &Sentences, at: u32) -> Option<u32> {
    let (starts, ends) = sentences.spans.columns();
    let i = ends.partition_point(|&e| e <= at);
    (i < starts.len() && starts[i] <= at).then_some(i as u32)
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        let up = parent[parent[x as usize] as usize];
        parent[x as usize] = up;
        x = up;
    }
    x
}

fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    // The smaller index is the root, so roots are first in text order.
    if ra < rb {
        parent[rb as usize] = ra;
    } else if rb < ra {
        parent[ra as usize] = rb;
    }
}

impl EntityKeys {
    /// Keys the entities of one document. `text` is the text of the view
    /// the sentences and entities are in.
    pub fn build(
        sentences: &Sentences,
        entities: Option<&Entities>,
        text: &str,
        keying: Keying,
        chains: Chains<'_>,
    ) -> EntityKeys {
        let Some(ents) = entities else { return EntityKeys::default() };
        let n = ents.spans.len();
        let surface = |i: usize| -> String {
            let parts: Vec<&str> =
                ents.spans.fragments(Ix::new(i as u32)).filter_map(|f: Span| text.get(f.range())).collect();
            normalize_surface(&parts.join(" "))
        };
        let sentence: Vec<Option<u32>> =
            (0..n).map(|i| sentence_at(sentences, ents.spans.get(Ix::new(i as u32)).start)).collect();
        let mut valid = vec![false; n];
        let mut parent: Vec<u32> = (0..n as u32).collect();
        let mut by_surface: BTreeMap<(Option<u16>, String), u32> = BTreeMap::new();
        for i in 0..n {
            let norm = surface(i);
            if sentence[i].is_none() || norm.is_empty() {
                continue;
            }
            valid[i] = true;
            let kind = keying.kind.then(|| ents.kind[i].0);
            let root = *by_surface.entry((kind, norm)).or_insert(i as u32);
            union(&mut parent, root, i as u32);
        }
        // Mentions of a chain that are not entities: (any entity of the
        // chain, sentence, span).
        let mut extra: Vec<(u32, u32, Span, usize)> = Vec::new();
        if keying.chains {
            if let Some(groups) = chains.entity_groups {
                for (_, members) in groups.iter() {
                    let live: Vec<u32> = members
                        .iter()
                        .map(|m| m.get() as u32)
                        .filter(|&m| (m as usize) < n && valid[m as usize])
                        .collect();
                    for w in live.windows(2) {
                        union(&mut parent, w[0], w[1]);
                    }
                }
            }
            if let (Some(m), Some(groups)) = (chains.mentions, chains.mention_groups) {
                let by_span: BTreeMap<(u32, u32), u32> = (0..n)
                    .rev()
                    .filter(|&i| valid[i])
                    .map(|i| {
                        let s = ents.spans.get(Ix::new(i as u32));
                        ((s.start, s.end), i as u32)
                    })
                    .collect();
                let entity_of = |j: usize| -> Option<u32> {
                    let lifted = m.entity.as_ref().and_then(|e| e.get(j).copied().flatten()).map(|e| e.get() as u32);
                    let lifted = lifted.filter(|&e| (e as usize) < n && valid[e as usize]);
                    let s = m.spans.get(Ix::new(j as u32));
                    lifted.or_else(|| by_span.get(&(s.start, s.end)).copied())
                };
                for (_, members) in groups.iter() {
                    let members: Vec<usize> = members.iter().map(|x| x.get()).filter(|&j| j < m.spans.len()).collect();
                    let linked: Vec<u32> = members.iter().filter_map(|&j| entity_of(j)).collect();
                    let Some(&anchor) = linked.first() else { continue };
                    for &e in &linked[1..] {
                        union(&mut parent, anchor, e);
                    }
                    for &j in &members {
                        if entity_of(j).is_some() {
                            continue;
                        }
                        let span = m.spans.get(Ix::new(j as u32));
                        let s =
                            m.sentence.get(j).map(|s| s.get() as u32).or_else(|| sentence_at(sentences, span.start));
                        if let Some(s) = s.filter(|&s| (s as usize) < sentences.spans.len()) {
                            extra.push((anchor, s, span, j));
                        }
                    }
                }
            }
        }
        let mut key = vec![None; n];
        let mut first = Vec::new();
        let mut id_of_root: BTreeMap<u32, u32> = BTreeMap::new();
        for i in 0..n {
            if !valid[i] {
                continue;
            }
            let root = find(&mut parent, i as u32);
            let id = *id_of_root.entry(root).or_insert_with(|| {
                first.push(Ix::new(i as u32));
                first.len() as u32 - 1
            });
            key[i] = Some(id);
        }
        // Distinct mentions per key: two entities (or an entity and a
        // chain mention) over the same span are one mention.
        // The last field is the source: entity i, or chain mention j as n + j.
        let mut seen: Vec<(u32, u32, u32, u32, usize)> = Vec::new();
        for i in 0..n {
            if let (Some(k), Some(s)) = (key[i], sentence[i]) {
                let sp = ents.spans.get(Ix::new(i as u32));
                seen.push((k, s, sp.start, sp.end, i));
            }
        }
        for &(anchor, s, sp, j) in &extra {
            if let Some(k) = key[anchor as usize] {
                seen.push((k, s, sp.start, sp.end, n + j));
            }
        }
        seen.sort_unstable();
        seen.dedup_by(|a, b| a.0 == b.0 && a.2 == b.2 && a.3 == b.3);
        let mut by_start: Vec<(u32, u32, u32, u32, usize)> =
            seen.iter().map(|&(k, s, a, b, src)| (a, b, k, s, src)).collect();
        by_start.sort_unstable();
        let (mut mentions, mut fragments) = (Vec::with_capacity(by_start.len()), Vec::new());
        for (a, b, k, s, src) in by_start {
            let from = fragments.len() as u32;
            match (src < n, chains.mentions) {
                (true, _) => fragments.extend(ents.spans.fragments(Ix::new(src as u32))),
                (false, Some(m)) => fragments.extend(m.spans.fragments(Ix::new((src - n) as u32))),
                (false, None) => fragments.push(Span::new(a, b)),
            }
            let fragments_end = fragments.len() as u32;
            let span = Span::new(a, b);
            mentions.push(KeyMention { key: k, sentence: s, span, fragments: (from, fragments_end) });
        }
        let mut occurrences = vec![Vec::new(); first.len()];
        for (k, s, _, _, _) in seen {
            let occ: &mut Vec<(u32, u32)> = &mut occurrences[k as usize];
            match occ.last_mut() {
                Some((last, count)) if *last == s => *count += 1,
                _ => occ.push((s, 1)),
            }
        }
        EntityKeys { key, first, occurrences, mentions, fragments }
    }

    /// How many distinct entities the document mentions.
    pub fn len(&self) -> usize {
        self.first.len()
    }

    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }
}
