//! The dependency-path rule baseline for Re-DocRED (#1825), the row the
//! #1734 receipt lists as missing: each document parsed by our own
//! dependency head, an ordered entity pair labelled by the most frequent
//! training relation on the shortest path between the two entities' head
//! words, and no relation when the path is one training never gave.
//!
//! An entity's anchor is its first mention's head word: the first word
//! of the mention whose parse head is the sentence root or a word outside
//! the mention, the rule the analysis annotator follows
//! (`analysis/relation/src/prepare.rs`). The path is the word-to-word path
//! through the lowest common ancestor in the sentence's tree, written as
//! the analysis pattern language writes it: `<rel` a step toward the
//! head, `>rel` a step toward the dependent, up steps first. A pair in
//! different sentences, or sharing one head word, has no path.

use std::collections::BTreeMap;

use crate::redocred::{Document, Triple};

/// A parsed sentence: each word's head (1-based, 0 the root) and its
/// relation, as our parser decodes them.
pub type Parse = Vec<(u32, String)>;

/// A document's parses, one a sentence.
pub type Parsed = Vec<Parse>;

/// The first word of `first..=last` (0-based, of one sentence) whose head
/// is the root or a word outside the range; None when every word of the
/// mention hangs inside it.
pub fn head_word(parse: &Parse, first: usize, last: usize) -> Option<usize> {
    (first..=last).find(|&i| {
        let h = parse[i].0 as usize;
        h == 0 || !(first + 1..=last + 1).contains(&h)
    })
}

/// The path between words `a` and `b` of one sentence (0-based): up from
/// `a` to the lowest common ancestor, then down to `b`, each step
/// `<rel` or `>rel`, joined by single spaces. None when `a` is `b`, when
/// a head index is out of range, or when the two are not in one tree
/// (a broken parse gives no path rather than a wrong one).
pub fn path(parse: &Parse, a: usize, b: usize) -> Option<String> {
    if a == b || a >= parse.len() || b >= parse.len() {
        return None;
    }
    let parent = |i: usize| -> Option<usize> {
        let h = parse.get(i)?.0 as usize;
        if h == 0 { None } else { h.checked_sub(1).filter(|&p| p != i) }
    };
    // a's ancestors, itself first, the root last.
    let mut up = Vec::new();
    let mut seen = BTreeMap::new();
    let mut at = Some(a);
    while let Some(i) = at {
        if seen.insert(i, up.len()).is_some() {
            return None; // a cycle
        }
        up.push(i);
        at = parent(i);
    }
    // b's ancestors until one is a's; then the steps a up to it, it down to b.
    let mut down = Vec::new();
    let mut at = Some(b);
    let up_to = loop {
        let i = at?;
        if let Some(&up_to) = seen.get(&i) {
            break up_to;
        }
        down.push(i);
        if down.len() > parse.len() {
            return None; // b's chain cycles without meeting a's
        }
        at = Some(parent(i)?);
    };
    let mut steps: Vec<String> = up[..up_to].iter().map(|&i| format!("<{}", parse[i].1)).collect();
    steps.extend(down.iter().rev().map(|&i| format!(">{}", parse[i].1)));
    Some(steps.join(" "))
}

/// Each entity's anchor word: its first mention's head word, as
/// `(sentence, word)`, or None when the mention has none.
fn anchors(doc: &Document, parsed: &Parsed) -> Vec<Option<(usize, usize)>> {
    let mut starts = Vec::with_capacity(doc.sentences.len());
    let mut at = 0;
    for &n in &doc.sentences {
        starts.push(at);
        at += n;
    }
    (0..doc.types.len())
        .map(|e| {
            let m = doc.mentions.iter().find(|m| m.entity == e as u32)?;
            let (s, start) = starts.iter().enumerate().map(|(s, &start)| (s, start)).find(|&(s, start)| {
                let end = start + doc.sentences[s];
                m.first as usize >= start && (m.last as usize) < end
            })?;
            let (first, last) = (m.first as usize - start, m.last as usize - start);
            head_word(&parsed[s], first, last).map(|w| (s, w))
        })
        .collect()
}

/// The path between two entities' anchor words, or None.
fn pair_path(parsed: &Parsed, anchors: &[Option<(usize, usize)>], h: usize, t: usize) -> Option<String> {
    let (sh, ah) = anchors.get(h).copied()??;
    let (st, at) = anchors.get(t).copied()??;
    if sh != st {
        return None;
    }
    path(&parsed[sh], ah, at)
}

/// The path-to-relation table of the training documents: each path's
/// commonest relation over the gold triples that walk it, ties to the
/// smaller label.
pub fn mine(train: &[Document], parses: &[Parsed]) -> BTreeMap<String, String> {
    let mut counts: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    for (d, parsed) in train.iter().zip(parses) {
        let anchors = anchors(d, parsed);
        for &(h, t, ref r) in &d.relations {
            if let Some(p) = pair_path(parsed, &anchors, h as usize, t as usize) {
                *counts.entry(p).or_default().entry(r.as_str()).or_default() += 1;
            }
        }
    }
    counts
        .into_iter()
        .map(|(p, rs)| {
            let best = rs.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(a.0)));
            (p, best.map_or_else(String::new, |(r, _)| r.to_owned()))
        })
        .collect()
}

/// The rule's relations of each document: every ordered pair of entities
/// whose path the table names, with that relation.
pub fn predict(docs: &[Document], parses: &[Parsed], table: &BTreeMap<String, String>) -> Vec<Vec<Triple>> {
    docs.iter()
        .zip(parses)
        .map(|(d, parsed)| {
            let anchors = anchors(d, parsed);
            let mut out = Vec::new();
            for h in 0..d.types.len() {
                for t in 0..d.types.len() {
                    if h == t {
                        continue;
                    }
                    if let Some(r) = pair_path(parsed, &anchors, h, t).and_then(|p| table.get(&p)) {
                        out.push((h as u32, t as u32, r.clone()));
                    }
                }
            }
            out
        })
        .collect()
}
