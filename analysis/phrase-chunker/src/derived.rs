//! Noun and verb chunks read off a dependency tree (#1650), in place of a
//! trained chunker: a base noun phrase is a nominal and the dependents
//! that sit inside it (determiners, adjectives, numbers, compounds,
//! possessors), a verb group a verb and its auxiliaries and negation.
//! The rules follow spaCy's `noun_chunks` (MIT), restated for Universal
//! Dependencies relations; nothing of spaCy's runs here.
//!
//! [`derive`] works on columns; [`DependencyChunker`] reads the tokens,
//! their tags and the parse and writes `turbo:phrase-chunks` (`NP`, `VP`). It
//! reads UPOS tags (`NOUN`, `VERB`) or Penn tags (`NN`, `VBZ`), and the
//! parse's relations are UD's (`nsubj`, `obj`, `aux`, ...).
//!
//! A chunk is contiguous: a dependent the rules take is in it only when
//! every token between it and the head is in it too. Chunks never
//! overlap; where a verb group would cross a noun chunk (`the running
//! water`, where `running` is an adjective), the noun chunk wins.

use std::sync::Arc;

use turbo_analysis::{Annotator, Error, IxRange, Refs, Span, Spans, View};
use turbo_analysis_layers::{Dependencies, PhraseChunks, PosTags, TagSet, Tokens};

/// What a chunk is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Noun,
    Verb,
}

impl ChunkKind {
    pub fn label(self) -> &'static str {
        match self {
            ChunkKind::Noun => "NP",
            ChunkKind::Verb => "VP",
        }
    }
}

/// A chunk: tokens `start..end` of the sequence, and its head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Derived {
    pub start: usize,
    pub end: usize,
    pub head: usize,
    pub kind: ChunkKind,
}

/// Relations that put a token inside its head's noun chunk.
const IN_NOUN: [&str; 11] = [
    "det",
    "det:predet",
    "det:poss",
    "amod",
    "nummod",
    "compound",
    "nmod:poss",
    "flat",
    "flat:name",
    "fixed",
    "goeswith",
];
/// Relations that put a token inside its head's verb group.
const IN_VERB: [&str; 3] = ["aux", "aux:pass", "compound:prt"];
/// Words that negate a verb inside its group (`advmod` of the verb).
const NEGATION: [&str; 4] = ["not", "n't", "never", "nt"];

fn base(rel: &str) -> &str {
    rel.split(':').next().unwrap_or(rel)
}

/// Whether `tag` (UPOS or Penn) is a nominal that can head a noun chunk.
pub fn nominal(tag: &str) -> bool {
    matches!(tag, "NOUN" | "PROPN" | "PRON" | "NN" | "NNS" | "NNP" | "NNPS" | "PRP" | "WP" | "EX")
}

/// Whether `tag` (UPOS or Penn) is a verb that can head a verb group.
pub fn verbal(tag: &str) -> bool {
    matches!(tag, "VERB" | "AUX" | "MD") || tag.starts_with("VB")
}

/// The noun and verb chunks of a parsed sequence: `tags[i]` its tag,
/// `heads[i]` its head (None for a root) and `rels[i]` its relation, as
/// written, in order and never overlapping.
///
/// # Panics
///
/// When the columns differ in length or a head is out of range.
pub fn derive(tags: &[&str], heads: &[Option<usize>], rels: &[&str], words: &[&str]) -> Vec<Derived> {
    let n = tags.len();
    assert!(heads.len() == n && rels.len() == n && words.len() == n, "a tag, head, relation and word per token");
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, h) in heads.iter().enumerate() {
        if let Some(h) = *h {
            assert!(h < n, "head {h} of token {i} is past the {n} tokens");
            children[h].push(i);
        }
    }
    // The tokens a head takes into its chunk: the dependents the rules
    // name, and theirs, transitively.
    let take = |head: usize, inside: &dyn Fn(usize) -> bool| {
        let mut set = vec![head];
        let mut stack = vec![head];
        while let Some(t) = stack.pop() {
            for &c in &children[t] {
                if inside(c) && !set.contains(&c) {
                    set.push(c);
                    stack.push(c);
                }
            }
        }
        // The contiguous run around the head.
        let has = |i: usize| set.contains(&i);
        let mut start = head;
        while start > 0 && has(start - 1) {
            start -= 1;
        }
        let mut end = head + 1;
        while end < n && has(end) {
            end += 1;
        }
        (start, end)
    };
    let in_noun = |c: usize| {
        let r = rels[c];
        IN_NOUN.contains(&r)
            // An adverb of an adjective or a number (`very large`), and the
            // possessive marker of a possessor (`John 's`).
            || (base(r) == "advmod" && heads[c].is_some_and(|h| matches!(rels[h], "amod" | "nummod")))
            || (r == "case" && heads[c].is_some_and(|h| rels[h] == "nmod:poss"))
    };
    let in_verb = |c: usize| {
        let r = rels[c];
        IN_VERB.contains(&r) || (base(r) == "advmod" && NEGATION.contains(&words[c].to_lowercase().as_str()))
    };
    let mut out: Vec<Derived> = Vec::new();
    for (i, tag) in tags.iter().enumerate() {
        if nominal(tag) && !in_noun(i) {
            let (start, end) = take(i, &in_noun);
            out.push(Derived { start, end, head: i, kind: ChunkKind::Noun });
        }
    }
    let nouns = out.clone();
    for (i, tag) in tags.iter().enumerate() {
        // A verb heads a group unless it is inside another's group or
        // modifies a noun from inside its chunk.
        if verbal(tag) && !in_verb(i) && !in_noun(i) {
            let (start, end) = take(i, &in_verb);
            if !nouns.iter().any(|c| start < c.end && c.start < end) {
                out.push(Derived { start, end, head: i, kind: ChunkKind::Verb });
            }
        }
    }
    out.sort_by_key(|c| (c.start, c.end));
    // Two noun chunks can only meet when a parse is odd (a nominal taken
    // into another's chunk across a gap is not); the first one wins.
    let mut kept: Vec<Derived> = Vec::with_capacity(out.len());
    for c in out {
        if kept.last().is_none_or(|k| k.end <= c.start) {
            kept.push(c);
        }
    }
    kept
}

/// Writes `turbo:phrase-chunks` from the tokens, their tags and their parse.
#[derive(Clone, Debug)]
pub struct DependencyChunker {
    kinds: Arc<TagSet>,
}

impl Default for DependencyChunker {
    fn default() -> Self {
        DependencyChunker { kinds: TagSet::new("chunks", &["NP", "VP"]) }
    }
}

impl Annotator for DependencyChunker {
    const ID: &'static str = "turbo-phrase-chunker:dependencies";
    type Reads = (View, Tokens, PosTags, Dependencies);
    type Writes = (PhraseChunks,);

    fn annotate(&self, (view, tokens, tags, deps): Refs<'_, Self>) -> Result<(PhraseChunks,), Error> {
        let n = crate::pattern_chunker::validate_tokens_and_tags(Self::ID, view, tokens, tags)?;
        let refuse = |detail: String| Err(Error::Annotator { annotator: Self::ID, detail });
        if deps.head.len() != n || deps.relation.len() != n {
            return refuse(format!("{} heads and {} relations for {n} tokens", deps.head.len(), deps.relation.len()));
        }
        // Relations in the tag set, heads in range and never their own,
        // and no cycle: what a document checks on `add`, for a parse
        // passed in directly.
        if let Err(detail) = deps.check_forest() {
            return refuse(detail);
        }
        let text = view.text();
        let (starts, ends) = tokens.spans.columns();
        let words: Vec<&str> = (0..n).map(|i| &text[starts[i] as usize..ends[i] as usize]).collect();
        let tag: Vec<&str> = tags.tags.iter().map(|&t| tags.tagset.label(t)).collect();
        let heads: Vec<Option<usize>> = deps.head.iter().map(|h| h.map(|h| h.get())).collect();
        let rels: Vec<&str> = deps.relation.iter().map(|&r| deps.relations.label(r)).collect();
        let mut spans = Spans::new(view.id());
        let (mut ranges, mut kind) = (Vec::new(), Vec::new());
        for c in derive(&tag, &heads, &rels, &words) {
            spans.push(Span::new(starts[c.start], ends[c.end - 1]));
            ranges.push(IxRange::new(c.start as u32, c.end as u32));
            kind.push(self.kinds.tag(c.kind.label()).expect("NP and VP are the chunker's kinds"));
        }
        Ok((PhraseChunks { spans, tokens: ranges, kind, kinds: self.kinds.clone() },))
    }
}
