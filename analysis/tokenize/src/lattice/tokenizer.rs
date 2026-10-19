//! The Viterbi search: OpenNLP's `LatticeTokenizer.analyze` over a loaded
//! [`MecabDictionary`].

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use super::chardef::AssignmentId;
use super::dictionary::{BOUNDARY, EntryId, MecabDictionary};
use super::error::NoCandidate;
use crate::TokenizerOptions;
use crate::whitespace::whitespace_spans;

/// No node.
const NONE: u32 = u32::MAX;

/// One morpheme of a segmentation: its span in the text, the dictionary
/// entry it is, and whether unknown word handling made it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Morpheme {
    pub span: Span,
    pub entry: EntryId,
    pub unknown: bool,
}

/// One lattice node: a candidate morpheme of the current stretch with the
/// cost of the cheapest path that ends with it. The nodes ending at one
/// position are chained through `next`, the latest first, as the Java
/// chains them.
#[derive(Copy, Clone, Debug)]
struct Node {
    /// Byte offsets in the stretch.
    start: u32,
    end: u32,
    entry: u32,
    /// The entry's right context id, which the nodes after it look up.
    right: u32,
    unknown: bool,
    cost: i64,
    previous: u32,
    next: u32,
}

/// The buffers one segmentation reuses, so that repeated calls allocate
/// nothing once they have grown: the whitespace-free stretches, the
/// lattice nodes, the head of the chain of nodes ending at each byte of a
/// stretch, each character's categories and run end, and the morphemes
/// [`LatticeTokenizer::tokenize_with`] takes the spans of.
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    stretches: Vec<Span>,
    nodes: Vec<Node>,
    head: Vec<u32>,
    assignment: Vec<AssignmentId>,
    run_end: Vec<u32>,
    morphemes: Vec<Morpheme>,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// The lattice tokenizer: a dictionary and the whitespace mode the text is
/// split in.
#[derive(Debug, Clone)]
pub struct LatticeTokenizer {
    pub(crate) dictionary: MecabDictionary,
    pub(crate) whitespace: WhitespaceMode,
}

impl LatticeTokenizer {
    /// A tokenizer over `dictionary` that splits the text at whitespace in
    /// `whitespace` mode. The dictionary keeps the mode its files were
    /// split in; OpenNLP reads one process-wide mode for both.
    pub fn new(dictionary: MecabDictionary, whitespace: WhitespaceMode) -> LatticeTokenizer {
        LatticeTokenizer { dictionary, whitespace }
    }

    pub fn dictionary(&self) -> &MecabDictionary {
        &self.dictionary
    }

    pub fn whitespace(&self) -> WhitespaceMode {
        self.whitespace
    }

    /// Appends the morphemes of `text` to `out`, in text order, with byte
    /// spans of `text`. Whitespace is never part of a morpheme. Fails, with
    /// `out` holding the morphemes of the stretches before, where the
    /// dictionary offers no candidate.
    pub fn analyze(&self, text: &str, out: &mut Vec<Morpheme>) -> Result<(), NoCandidate> {
        self.analyze_with(text, &mut Scratch::new(), out)
    }

    /// [`analyze`](Self::analyze) with the caller's buffers.
    pub fn analyze_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Morpheme>) -> Result<(), NoCandidate> {
        scratch.stretches.clear();
        let options = TokenizerOptions { whitespace: self.whitespace, keep_newlines: false };
        whitespace_spans(text, options, &mut scratch.stretches);
        for i in 0..scratch.stretches.len() {
            let stretch = scratch.stretches[i];
            self.decode(text, stretch, scratch, out)?;
        }
        Ok(())
    }

    /// The spans of the morphemes of `text`, appended to `out`; fails as
    /// [`analyze`](Self::analyze) fails.
    pub fn tokenize_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Span>) -> Result<(), NoCandidate> {
        let mut morphemes = std::mem::take(&mut scratch.morphemes);
        morphemes.clear();
        let result = self.analyze_with(text, scratch, &mut morphemes);
        out.extend(morphemes.iter().map(|m| m.span));
        scratch.morphemes = morphemes;
        result
    }

    /// The cheapest path through one whitespace-free stretch.
    ///
    /// From each character boundary that a node ends at (and the start),
    /// in order, the candidates are the lexicon entries of every surface
    /// that starts there, shortest surface first and each surface's
    /// entries in file order, then the unknown word candidates. Each
    /// candidate takes the predecessor whose path plus connection cost is
    /// strictly lowest, the latest added first, and is then chained before
    /// the nodes already ending where it ends. At the end of the stretch
    /// the first node of that chain with the strictly lowest total wins.
    fn decode(
        &self,
        text: &str,
        stretch: Span,
        scratch: &mut Scratch,
        out: &mut Vec<Morpheme>,
    ) -> Result<(), NoCandidate> {
        let dict = &self.dictionary;
        let part = &text[stretch.range()];
        let bytes = part.as_bytes();
        let n = bytes.len();
        let Scratch { nodes, head, assignment, run_end, .. } = scratch;
        nodes.clear();
        head.clear();
        head.resize(n + 1, NONE);
        self.category_runs(part, assignment, run_end);

        let mut i = 0;
        while i < n {
            let width = char_width(bytes[i]);
            if i > 0 && head[i] == NONE {
                i += width;
                continue;
            }
            let first = nodes.len();
            let start = i as u32;
            dict.trie.prefixes(&bytes[i..], |len, value| {
                for e in dict.group(value) {
                    let right = dict.entries[e as usize].right;
                    nodes.push(Node::new(start, (i + len) as u32, e, right, false));
                }
            });
            let matched = nodes.len() > first;
            let primary = dict.chars.assignment(assignment[i]).primary;
            let category = dict.category(primary);
            if (!matched || category.invoke)
                && let Some(templates) = dict.templates(primary)
            {
                let run = run_end[i] as usize;
                let mut push = |end: usize| {
                    for e in templates.clone() {
                        let right = dict.entries[e as usize].right;
                        nodes.push(Node::new(start, end as u32, e, right, true));
                    }
                };
                if category.group {
                    push(run);
                }
                let mut end = i;
                for _ in 0..category.length {
                    if end >= run {
                        break;
                    }
                    end += char_width(bytes[end]);
                    if category.group && end == run {
                        continue;
                    }
                    push(end);
                }
            }
            if nodes.len() == first {
                // A category that offered nothing still gives its templates
                // one character; without templates, DEFAULT's do.
                let fallback = dict.templates(primary).or_else(|| dict.templates(dict.default_category()));
                for e in fallback.into_iter().flatten() {
                    let right = dict.entries[e as usize].right;
                    nodes.push(Node::new(start, (i + width) as u32, e, right, true));
                }
            }
            if nodes.len() == first {
                return Err(NoCandidate { offset: stretch.start as usize + i });
            }
            let predecessors = if i == 0 { None } else { Some(head[i]) };
            for k in first..nodes.len() {
                self.relax(nodes, k, predecessors);
                if nodes[k].cost < i64::MAX {
                    let end = nodes[k].end as usize;
                    nodes[k].next = head[end];
                    head[end] = k as u32;
                }
            }
            i += width;
        }

        let mut best = NONE;
        let mut best_total = i64::MAX;
        let mut at = head[n];
        while at != NONE {
            let node = &nodes[at as usize];
            let total = node.cost + i64::from(dict.matrix.cost(node.right, BOUNDARY));
            if best == NONE || total < best_total {
                best = at;
                best_total = total;
            }
            at = node.next;
        }
        // Every reachable position offers a candidate that ends further on
        // and no later than the stretch's end, so the end is reached.
        debug_assert!(best != NONE || n == 0);
        let first = out.len();
        while best != NONE {
            let node = &nodes[best as usize];
            out.push(Morpheme {
                span: Span { start: stretch.start + node.start, end: stretch.start + node.end },
                entry: EntryId(node.entry),
                unknown: node.unknown,
            });
            best = node.previous;
        }
        out[first..].reverse();
        Ok(())
    }

    /// Gives node `k` the cheapest path to it: from the start of the
    /// stretch when `predecessors` is `None`, else through the chain of
    /// nodes that starts at `predecessors`, a strictly lower total
    /// replacing the one found before.
    #[inline(always)]
    fn relax(&self, nodes: &mut [Node], k: usize, predecessors: Option<u32>) {
        let dict = &self.dictionary;
        let entry = dict.entries[nodes[k].entry as usize];
        let own = i64::from(entry.cost);
        let Some(mut at) = predecessors else {
            nodes[k].cost = own + i64::from(dict.matrix.cost(BOUNDARY, entry.left));
            return;
        };
        let mut best = i64::MAX;
        let mut previous = NONE;
        while at != NONE {
            let p = &nodes[at as usize];
            let total = p.cost + i64::from(dict.matrix.cost(p.right, entry.left)) + own;
            if total < best {
                best = total;
                previous = at;
            }
            at = p.next;
        }
        nodes[k].cost = best;
        nodes[k].previous = previous;
    }

    /// Each character's categories, and the end of the run that starts at
    /// it: the run goes on while each character shares a category with the
    /// one after it, as MeCab's `seekToOtherType` does. One pass from the
    /// end of the stretch.
    fn category_runs(&self, part: &str, assignment: &mut Vec<AssignmentId>, run_end: &mut Vec<u32>) {
        let dict = &self.dictionary;
        let n = part.len();
        assignment.clear();
        assignment.resize(n, AssignmentId::NONE);
        run_end.clear();
        run_end.resize(n, 0);
        let mut next_mask = 0u32;
        let mut next_run_end = n as u32;
        for (i, c) in part.char_indices().rev() {
            let id = dict.assignment_of(c as u32);
            let mask = dict.chars.assignment(id).mask;
            let end = if mask & next_mask != 0 { next_run_end } else { (i + c.len_utf8()) as u32 };
            assignment[i] = id;
            run_end[i] = end;
            next_mask = mask;
            next_run_end = end;
        }
    }
}

impl Node {
    #[inline(always)]
    fn new(start: u32, end: u32, entry: u32, right: u32, unknown: bool) -> Node {
        Node { start, end, entry, right, unknown, cost: i64::MAX, previous: NONE, next: NONE }
    }
}

/// The length of the UTF-8 character whose first byte is `b`.
#[inline(always)]
fn char_width(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests;
