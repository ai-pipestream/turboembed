//! Sentence-quark alignment, Penn tokenization, and named-entity attachment.

use std::collections::{HashMap, HashSet};

use turbo_jvm::hash_map::put_order;

use super::super::error::MascError;
use super::xml::{NamedEntities, Penn, Region};
use super::{
    MascDocument, MascEntity, MascName, MascSentence, MascToken, MascUnplacedEntity, MascWarning, Quark, Unplaced,
};

impl MascDocument {
    pub(super) fn add_penn_tags(&mut self, penn: &Penn) -> Result<(), MascError> {
        // Each quark's tokens, built in the order the token map iterates,
        // each token put in front of the ones before it.
        let mut quark_tokens: HashMap<i32, Vec<usize>> = HashMap::new();
        for t in put_order(penn.tokens.iter().map(|t| t.id)).0 {
            for &q in &penn.tokens[t].quarks {
                let tokens = quark_tokens.entry(q).or_default();
                if !tokens.is_empty() {
                    self.warnings.push(MascWarning::QuarkInSeveralTokens { quark: q });
                }
                tokens.insert(0, t);
            }
        }
        for si in 0..self.sentences.len() {
            let tokens = tokenize(si, &self.sentences[si], &self.quarks, penn, &quark_tokens, &mut self.warnings)?;
            let sentence = &mut self.sentences[si];
            sentence.token_places = tokens.iter().enumerate().map(|(i, t)| (t.id, i)).collect();
            sentence.tokens = Some(tokens);
        }
        self.has_penn_tags = true;
        Ok(())
    }

    /// The entities of `ne` that are in no sentence, and why: whether a
    /// token of one has a quark in a region of the sentence file, those
    /// the reader keeps (the sentences) or those it dropped (`dropped`). A
    /// region holds a quark it overlaps, as `combineAnnotations` gives it
    /// the quark.
    pub(super) fn unplaced(&self, ne: &NamedEntities, penn: &Penn, dropped: &[Region]) -> Vec<MascUnplacedEntity> {
        let placed: HashSet<i32> = self.sentences.iter().flat_map(|s| s.token_places.keys().copied()).collect();
        let mut quark_places: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, q) in self.quarks.iter().enumerate() {
            quark_places.entry(q.id).or_default().push(i);
        }
        let regions: Vec<Region> =
            self.sentences.iter().map(|s| (s.start, s.end)).chain(dropped.iter().copied()).collect();
        let in_a_region = |tokens: &[i32]| {
            tokens.iter().filter_map(|t| penn.places.get(t)).any(|&p| {
                penn.tokens[p].quarks.iter().filter_map(|q| quark_places.get(q)).flatten().any(|&w| {
                    let q = &self.quarks[w];
                    regions.iter().any(|&(start, end)| end > q.start && start < q.end)
                })
            })
        };
        let mut out = Vec::new();
        for entity in &ne.entities {
            if entity.tokens.iter().any(|t| placed.contains(t)) {
                continue;
            }
            let cause =
                if in_a_region(&entity.tokens) { Unplaced::DroppedSentence } else { Unplaced::OutsideSentences };
            out.push(MascUnplacedEntity { kind: ne.types.get(&entity.id).cloned().flatten(), cause });
        }
        out
    }

    pub(super) fn add_named_entities(&mut self, ne: &NamedEntities) {
        if !self.has_penn_tags && !self.sentences.is_empty() {
            // the first sentence throws "Named entity labels provided for an
            // un-tokenized sentence.", which the document logs
            self.warnings.push(MascWarning::NamesWithoutTokens);
            return;
        }
        for si in 0..self.sentences.len() {
            let sentence = &mut self.sentences[si];
            let count = sentence.tokens.as_ref().map_or(0, Vec::len);
            let mut names = Vec::new();
            for &e in &ne.order {
                let entity = &ne.entities[e];
                let kind = ne.types.get(&entity.id).cloned().flatten();
                let mut start = count;
                let mut end = 0;
                let mut inside = false;
                for token in &entity.tokens {
                    if let Some(&i) = sentence.token_places.get(token) {
                        inside = true;
                        if i < start {
                            start = i;
                        }
                        // compared with the end so far, which is one past a
                        // token: a name of the tokens 3 and 4 ends at 4
                        if i > end {
                            end = i + 1;
                        }
                    }
                }
                if inside {
                    names.push(MascName { start: start as u32, end: end as u32, kind });
                }
            }
            names.sort_by_key(|n| n.start);
            if let Some(kept) = remove_overlaps(&names) {
                self.warnings.push(MascWarning::OverlappingNames { sentence: si });
                names = kept;
            }
            sentence.names = names;
            let mut entities = Vec::new();
            for entity in &ne.entities {
                let mut tokens: Vec<u32> =
                    entity.tokens.iter().filter_map(|t| sentence.token_places.get(t)).map(|&i| i as u32).collect();
                tokens.sort_unstable();
                tokens.dedup();
                let (Some(&first), Some(&last)) = (tokens.first(), tokens.last()) else {
                    continue;
                };
                let kind = ne.types.get(&entity.id).cloned().flatten();
                entities.push(MascEntity { start: first, end: last + 1, kind, tokens });
            }
            sentence.entities = entities;
        }
        self.has_named_entities = true;
    }
}

/// The list of names `addNamedEntities` keeps of names sorted by start,
/// or `None` when it finds no overlap and keeps them all.
///
/// It walks a left and a right name: when they overlap (the left contains
/// or crosses the right) it marks the shorter (the right on a tie) and
/// moves the right on; otherwise it moves the left on, until the left
/// reaches the right and finds that a name contains itself. Then, if
/// anything was marked, it keeps the unmarked names but the last.
pub(super) fn remove_overlaps(names: &[MascName]) -> Option<Vec<MascName>> {
    let contains = |a: &MascName, b: &MascName| a.start <= b.start && b.end <= a.end;
    let crosses = |a: &MascName, b: &MascName| {
        !contains(a, b)
            && !contains(b, a)
            && (a.start <= b.start && b.start < a.end || b.start <= a.start && a.start < b.end)
    };
    let mut marked = HashSet::new();
    let mut overlaps = false;
    let mut left = 0;
    let mut right = 1;
    while right < names.len() {
        let (l, r) = (&names[left], &names[right]);
        if contains(l, r) || crosses(l, r) {
            if r.end - r.start > l.end - l.start {
                marked.insert(left);
            } else {
                marked.insert(right);
            }
            overlaps = true;
            right += 1;
        } else {
            left += 1;
        }
    }
    overlaps.then(|| {
        names[..names.len() - 1]
            .iter()
            .enumerate()
            .filter(|(i, _)| !marked.contains(i))
            .map(|(_, n)| n.clone())
            .collect()
    })
}

/// `MascSentence.tokenizePenn` for sentence `si`: the tokens of its
/// quarks, in the order of the quarks and, for a quark in several tokens,
/// of the quark's token list; a token is taken once, and a token whose
/// quarks span nothing is left out.
fn tokenize(
    si: usize,
    sentence: &MascSentence,
    quarks: &[Quark],
    penn: &Penn,
    quark_tokens: &HashMap<i32, Vec<usize>>,
    warnings: &mut Vec<MascWarning>,
) -> Result<Vec<MascToken>, MascError> {
    let mut tokens = Vec::new();
    let mut processed = HashSet::new();
    for &w in &sentence.quarks {
        let Some(list) = quark_tokens.get(&quarks[w].id) else {
            continue;
        };
        for &t in list {
            if processed.contains(&t) {
                continue;
            }
            let token = &penn.tokens[t];
            for &q in &token.quarks {
                if !sentence.quark_places.contains_key(&q) {
                    warnings.push(MascWarning::TokenCrossesSentence { sentence: si, token: token.id, quark: q });
                }
            }
            let mut places = Vec::with_capacity(token.quarks.len());
            for &q in &token.quarks {
                // the sentence's own quark, else the first in the document
                let place = match sentence.quark_places.get(&q) {
                    Some(&p) => p,
                    None => quarks.iter().position(|x| x.id == q).ok_or(MascError::MissingQuark { id: q })?,
                };
                places.push(place);
            }
            let (Some(&first), Some(&last)) = (places.first(), places.last()) else {
                continue;
            };
            let start = quarks[first].start;
            let end = quarks[last].end;
            if end - start > 0 {
                tokens.push(MascToken {
                    id: token.id,
                    start,
                    end,
                    tag: penn.tags.get(&token.id).cloned().flatten(),
                    base: penn.bases.get(&token.id).cloned().flatten(),
                    quarks: places,
                });
                processed.insert(t);
            }
        }
    }
    Ok(tokens)
}

/// `combineAnnotations`: each non-empty region with the quarks that
/// overlap it. The quarks are walked once, in file order, stepping back
/// at the start of a region over the quarks that end after it starts.
pub(super) fn combine(regions: &[(i32, i32)], quarks: &[Quark]) -> Result<Vec<MascSentence>, MascError> {
    let mut wi = 0usize;
    let count = quarks.len();
    let mut sentences = Vec::new();
    for &(start, end) in regions {
        if end - start <= 0 {
            continue;
        }
        let mut list = Vec::new();
        let mut next = quarks.get(wi).ok_or(MascError::IndexOutOfBounds { index: wi as i64, len: count })?;
        while start < next.end && wi > 0 {
            wi -= 1;
            next = &quarks[wi];
        }
        let mut over = false;
        while !over && wi < count {
            let q = &quarks[wi];
            if end > q.start && start < q.end {
                list.push(wi);
                if end == q.end {
                    over = true;
                }
                wi += 1;
            } else if end <= q.start {
                over = true;
            } else {
                wi += 1;
            }
        }
        // counts the sentences made against the regions, empty ones too
        if !over && sentences.len() != regions.len() - 1 {
            return Err(MascError::QuarksEndInSentence { end });
        }
        sentences.push(MascSentence::new(start, end, list, quarks));
    }
    Ok(sentences)
}
