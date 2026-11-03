//! The nodes of a sketched dependency tree: one per word, or one per
//! entity once its words are contracted, each with a `u64` anchor and
//! no string.

use turbo_analysis::{FilterRule, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Entities, MorphLemmas, MorphPosTags, Word, WordOrTokenAnchor};

use crate::hash;

const P: u64 = 0x9e37_79b9_7f4a_7c15;
const ENTITY: u64 = 0x656e_7469_7479_0001;
const FUNCTION: u64 = 0x6675_6e63_7469_6f6e;

/// One fold step over the sketch's node and treelet hashes.
#[inline]
pub(super) fn mix(h: u64, x: u64) -> u64 {
    hash::fold(h, x, P)
}

/// Per tag of a POS tag set, whether the tag marks a content head: a
/// verb, a noun, a proper noun or an adjective, in Universal
/// Dependencies or Penn Treebank labels.
fn content_tags(set: &TagSet) -> Vec<bool> {
    set.labels()
        .iter()
        .map(|l| {
            matches!(&**l, "VERB" | "NOUN" | "PROPN" | "ADJ")
                || l.starts_with("VB")
                || l.starts_with("NN")
                || l.starts_with("JJ")
        })
        .collect()
}

/// Per tag of a relation tag set, whether the relation is a core
/// argument: `nsubj`, `obj`, `iobj`, `ccomp`, `xcomp` or `obl`, with or
/// without a subtype after a colon.
fn core_relations(set: &TagSet) -> Vec<bool> {
    set.labels()
        .iter()
        .map(|l| {
            let base = l.split(':').next().unwrap_or("");
            matches!(base, "nsubj" | "obj" | "iobj" | "ccomp" | "xcomp" | "obl")
        })
        .collect()
}

/// A word's form, lowercased code point by code point into `buf`, as a
/// 64-bit hash. The buffer is reused across words, so no word allocates.
fn lower_hash(form: &str, buf: &mut Vec<u8>) -> u64 {
    buf.clear();
    if form.is_ascii() {
        buf.extend(form.bytes().map(|b| b.to_ascii_lowercase()));
    } else {
        let mut utf8 = [0u8; 4];
        for c in form.chars().flat_map(char::to_lowercase) {
            buf.extend_from_slice(c.encode_utf8(&mut utf8).as_bytes());
        }
    }
    hash::word(buf)
}

/// The contracted dependency graph of one document: words are nodes,
/// except that the words of one kept entity are one node, the entity's
/// first word, and the arcs inside it are gone.
pub(super) struct Graph {
    /// Per word, its node: its own index, or its entity's first word.
    pub node: Vec<u32>,
    /// Per node, the node of its head, if any. Unused for a word that is
    /// not a node.
    pub head: Vec<Option<u32>>,
    /// Per node, the relation of the arc to its head: the word's own, or
    /// for an entity the relation of the arc that leaves it.
    pub rel: Vec<Tag>,
    /// Per word, its POS tag.
    pub pos: Vec<Tag>,
    /// Per word, the exact anchor: the lowercased form, the POS tag and
    /// the relation.
    pub exact: Vec<u64>,
    /// Per node, the delexicalized anchor.
    pub shape: Vec<u64>,
    /// Per relation tag, whether it is a core argument.
    pub core: Vec<bool>,
    /// The children of each node, in word order: node `n`'s are
    /// `child[child_start[n]..child_start[n + 1]]`.
    child_start: Vec<u32>,
    child: Vec<u32>,
}

impl Graph {
    pub fn build<A: WordOrTokenAnchor>(
        seed: u64,
        words: &[Word<'_>],
        pos: &A::Pos,
        deps: &Dependencies<A>,
        lemmas: Option<&A::Lemmas>,
        entities: Option<&Entities>,
    ) -> Result<Graph, String> {
        let n = words.len();
        if deps.len() != n || pos.len() != n || lemmas.is_some_and(|l| l.len() != n) {
            return Err(format!(
                "{n} words, {} dependencies, {} tags and {} lemmas",
                deps.len(),
                pos.len(),
                lemmas.map_or(n, Layer::len)
            ));
        }
        let content = content_tags(pos.tagset());
        let core = core_relations(&deps.relations);
        let mut pos_tags = Vec::with_capacity(n);
        for w in 0..n {
            pos_tags.push(pos.tag(w).ok_or_else(|| format!("word {w} has no tag in {}", pos.tagset().name))?);
        }
        let mut node: Vec<u32> = (0..n as u32).collect();
        let mut kind: Vec<Option<Tag>> = vec![None; n];
        if let Some(e) = entities {
            contract(e, words, &mut node, &mut kind);
        }
        let mut buf = Vec::new();
        let mut exact = Vec::with_capacity(n);
        let mut shape = vec![0u64; n];
        let mut head = vec![None; n];
        let mut rel = deps.relation.clone();
        for w in 0..n {
            let form = words[w].form;
            let lower = lower_hash(form, &mut buf);
            exact.push(mix(mix(mix(seed, lower), u64::from(pos_tags[w].0)), u64::from(deps.relation[w].0)));
            if node[w] as usize != w {
                continue;
            }
            shape[w] = match kind[w] {
                Some(k) => mix(mix(seed, ENTITY), u64::from(k.0)),
                None if hash::is_number(form) => mix(mix(seed, hash::NUMBER), u64::from(pos_tags[w].0)),
                None if content[pos_tags[w].0 as usize] => {
                    let lemma = lemmas.and_then(|l| l.lemma(w)).filter(|l| !l.is_empty() && *l != "_");
                    let anchor = match lemma {
                        Some(l) => lower_hash(l, &mut buf),
                        None => lower,
                    };
                    mix(mix(seed, anchor), u64::from(pos_tags[w].0))
                }
                None => mix(mix(seed, FUNCTION), u64::from(pos_tags[w].0)),
            };
            // The arc that leaves the node: the word's own, or for an
            // entity the first of its words whose head is outside it.
            let last = (w + 1..n).take_while(|&x| node[x] as usize == w).last().unwrap_or(w);
            for x in w..=last {
                let outside = deps.head[x].map(|h| node[h.get()]);
                if outside.is_none_or(|h| h as usize != w) {
                    head[w] = outside;
                    rel[w] = deps.relation[x];
                    break;
                }
            }
        }
        let mut child_start = vec![0u32; n + 1];
        for w in 0..n {
            if node[w] as usize == w
                && let Some(h) = head[w]
            {
                child_start[h as usize + 1] += 1;
            }
        }
        for i in 0..n {
            child_start[i + 1] += child_start[i];
        }
        let mut fill = child_start.clone();
        let mut child = vec![0u32; child_start[n] as usize];
        for w in 0..n {
            if node[w] as usize == w
                && let Some(h) = head[w]
            {
                child[fill[h as usize] as usize] = w as u32;
                fill[h as usize] += 1;
            }
        }
        Ok(Graph { node, head, rel, pos: pos_tags, exact, shape, core, child_start, child })
    }

    /// Whether word `w` is a node.
    #[inline]
    pub fn is_node(&self, w: usize) -> bool {
        self.node[w] as usize == w
    }

    /// The children of node `n`, in word order.
    #[inline]
    pub fn children(&self, n: usize) -> &[u32] {
        &self.child[self.child_start[n] as usize..self.child_start[n + 1] as usize]
    }
}

/// Contracts the kept entities (overlaps resolved longest first) to
/// their first word. An entity whose words another kept entity already
/// holds is left alone.
fn contract(entities: &Entities, words: &[Word<'_>], node: &mut [u32], kind: &mut [Option<Tag>]) {
    let kept = entities.spans.filter(Some(&entities.score), FilterRule::LongestFirst);
    for e in kept {
        let r = entities.tokens[e.get()];
        let (ts, te) = (r.start.get() as u32, r.end.get() as u32);
        // Words are in non-decreasing token order: those whose tokens
        // meet the entity's range are one run.
        let a = words.partition_point(|w| (w.tokens.end.get() as u32) <= ts);
        let b = a + words[a..].partition_point(|w| (w.tokens.start.get() as u32) < te);
        if a >= b || (a..b).any(|w| node[w] as usize != w) {
            continue;
        }
        node[a..b].fill(a as u32);
        kind[a] = Some(entities.kind[e.get()]);
    }
}
