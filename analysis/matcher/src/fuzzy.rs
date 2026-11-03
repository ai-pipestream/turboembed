//! Fuzzy lookup over a sorted lexicon held as a trie, by bounded
//! Damerau-Levenshtein distance.
//!
//! The metric is the optimal string alignment distance (restricted
//! Damerau-Levenshtein), as in OpenNLP's `DamerauOSADistance` and the
//! `turbo-analysis` spellcheck: insertions, deletions and substitutions of
//! one Unicode scalar cost one, a transposition of two adjacent scalars
//! costs one, and no substring is edited twice. A supplementary character
//! is one scalar, as elsewhere in this workspace. The distance is
//! symmetric, so the search computes each trie prefix against the query and
//! reads the entry's distance off the last column.
//!
//! [`Lexicon::search`] walks the trie once per query, extending the dynamic
//! program one column set per edge and descending only where the row's
//! minimum stays within the bound: a word within `max_distance` of the
//! query has every prefix within `max_distance` of some query prefix, so a
//! pruned subtree cannot contain a match. Entries are never scanned and
//! re-verified pairwise. `max_distance` 0 is an exact lookup; 1 and 2 are
//! the intended bounds, and larger values work at a higher traversal cost.
//!
//! Results are sorted by `(distance, entry index)`, so the output is
//! deterministic for a given lexicon and query.

/// A lexicon entry within the bound of a query: `index` is the entry's
/// position in the slice the [`Lexicon`] was built from, `distance` its
/// distance from the query.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FuzzyMatch {
    pub index: u32,
    pub distance: u32,
}

const NIL: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Node {
    /// The scalar on the edge from the parent; NUL at the root.
    ch: char,
    /// The next sibling, or NIL.
    sibling: u32,
    /// The first child, or NIL.
    child: u32,
    /// Whether a word ends at this node.
    term: bool,
    /// The entry index when `term`.
    index: u32,
}

/// A sorted word list held as a trie for bounded fuzzy lookup. Built once
/// from entries sorted by bytes (the usual `&str` order; UTF-8 byte order
/// is scalar order, so siblings are stored ascending), then searched any
/// number of times; the entries themselves stay with the caller, and a
/// [`FuzzyMatch`] addresses them by position. Entries must be distinct; a
/// repeated entry keeps the first index.
pub struct Lexicon {
    nodes: Vec<Node>,
    terms: u32,
}

/// Reusable scratch for [`Lexicon::search`], so repeated searches do not
/// allocate: the query scalars, one dynamic-program row per depth, and the
/// traversal stack.
#[derive(Default)]
pub struct Scratch {
    chars: Vec<char>,
    rows: Vec<Vec<u32>>,
    stack: Vec<Frame>,
}

#[derive(Copy, Clone)]
struct Frame {
    node: u32,
    depth: u32,
    next_child: u32,
}

impl Lexicon {
    /// The lexicon over `entries`, sorted ascending by bytes and distinct.
    pub fn new(entries: &[&str]) -> Lexicon {
        debug_assert!(entries.windows(2).all(|w| w[0] < w[1]), "entries must be sorted and distinct");
        let mut lex =
            Lexicon { nodes: vec![Node { ch: '\0', sibling: NIL, child: NIL, term: false, index: 0 }], terms: 0 };
        // path[d] is the node at depth d, tip[d] its last child so a new
        // sibling links in O(1); both shrink to the common prefix with the
        // previous entry before the rest of the word is added.
        let mut path: Vec<u32> = vec![0];
        let mut tip: Vec<u32> = vec![NIL];
        for (index, word) in entries.iter().enumerate() {
            let chars: Vec<char> = word.chars().collect();
            let mut depth = 0;
            while depth < chars.len()
                && depth + 1 < path.len()
                && lex.nodes[path[depth + 1] as usize].ch == chars[depth]
            {
                depth += 1;
            }
            path.truncate(depth + 1);
            tip.truncate(depth + 1);
            for &c in &chars[depth..] {
                let id = lex.nodes.len() as u32;
                lex.nodes.push(Node { ch: c, sibling: NIL, child: NIL, term: false, index: 0 });
                let parent = *path.last().expect("path holds the root") as usize;
                let tail = *tip.last().expect("tip aligns with path");
                if tail == NIL {
                    lex.nodes[parent].child = id;
                } else {
                    lex.nodes[tail as usize].sibling = id;
                }
                *tip.last_mut().expect("tip aligns with path") = id;
                path.push(id);
                tip.push(NIL);
            }
            let last = *path.last().expect("path holds the root") as usize;
            if !lex.nodes[last].term {
                lex.nodes[last].term = true;
                lex.nodes[last].index = index as u32;
                lex.terms += 1;
            }
        }
        lex
    }

    /// The number of distinct entries stored.
    pub fn len(&self) -> usize {
        self.terms as usize
    }

    /// Whether no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.terms == 0
    }

    /// The bytes the trie holds on the heap.
    pub fn heap_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<Node>()
    }

    /// Every entry within `max_distance` of `query`, sorted by
    /// `(distance, entry index)`, replacing `out`'s contents.
    pub fn search(&self, query: &str, max_distance: u8, scratch: &mut Scratch, out: &mut Vec<FuzzyMatch>) {
        self.search_counted(query, max_distance, scratch, out, &mut 0);
    }

    pub(crate) fn search_counted(
        &self,
        query: &str,
        max_distance: u8,
        scratch: &mut Scratch,
        out: &mut Vec<FuzzyMatch>,
        visited: &mut usize,
    ) {
        out.clear();
        scratch.chars.clear();
        scratch.chars.extend(query.chars());
        let m = scratch.chars.len();
        let max = u32::from(max_distance);
        let rows = &mut scratch.rows;
        rows.clear();
        rows.resize_with(2, Vec::new);
        rows[0].clear();
        rows[0].extend(0..=m as u32);
        let mut stack = std::mem::take(&mut scratch.stack);
        stack.clear();
        stack.push(Frame { node: 0, depth: 0, next_child: self.nodes[0].child });
        if self.nodes[0].term && rows[0][m] <= max {
            out.push(FuzzyMatch { index: self.nodes[0].index, distance: rows[0][m] });
        }
        while let Some(top) = stack.last_mut() {
            let node = top.node;
            let depth = top.depth as usize;
            let child = top.next_child;
            if child == NIL {
                stack.pop();
                continue;
            }
            top.next_child = self.nodes[child as usize].sibling;
            let edge = self.nodes[child as usize].ch;
            let prev_char = (depth > 0).then(|| self.nodes[node as usize].ch);
            *visited += 1;
            if rows.len() <= depth + 1 {
                rows.resize_with(depth + 2, Vec::new);
            }
            // Split so the parent row, the grandparent row and the row
            // being written are borrowed disjointly.
            let row_min = if depth == 0 {
                let (parent, rest) = rows.split_at_mut(1);
                fill_row(edge, prev_char, &parent[0], None, &scratch.chars, &mut rest[0])
            } else {
                let (upto, rest) = rows.split_at_mut(depth + 1);
                fill_row(edge, prev_char, &upto[depth], Some(&upto[depth - 1]), &scratch.chars, &mut rest[0])
            };
            if row_min > max {
                continue;
            }
            if self.nodes[child as usize].term && rows[depth + 1][m] <= max {
                out.push(FuzzyMatch { index: self.nodes[child as usize].index, distance: rows[depth + 1][m] });
            }
            stack.push(Frame { node: child, depth: depth as u32 + 1, next_child: self.nodes[child as usize].child });
        }
        scratch.stack = stack;
        out.sort_by(|a, b| a.distance.cmp(&b.distance).then(a.index.cmp(&b.index)));
    }
}

/// The next dynamic-program row: the entry prefix one scalar longer, ended
/// by `edge`, against every prefix of the query. Returns the row minimum.
/// `prev` is the row above, `grandparent` the row above that (read only for
/// a transposition), `prev_char` the entry scalar before `edge`.
fn fill_row(
    edge: char,
    prev_char: Option<char>,
    prev: &[u32],
    grandparent: Option<&[u32]>,
    query: &[char],
    out: &mut Vec<u32>,
) -> u32 {
    out.clear();
    out.push(prev[0] + 1);
    let mut row_min = out[0];
    for j in 1..=query.len() {
        let cost = u32::from(edge != query[j - 1]);
        let mut v = (prev[j] + 1).min(out[j - 1] + 1).min(prev[j - 1] + cost);
        if j >= 2
            && prev_char == Some(query[j - 1])
            && edge == query[j - 2]
            && let Some(gp) = grandparent
        {
            v = v.min(gp[j - 2] + 1);
        }
        out.push(v);
        row_min = row_min.min(v);
    }
    row_min
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The search must cut whole subtrees off, not test every entry: a
    /// query far from most of the lexicon visits a fraction of the edges.
    #[test]
    fn search_prunes_subtrees_instead_of_scanning() {
        let mut entries: Vec<String> = (0..200).map(|i| format!("prefix{i:03}suffix")).collect();
        entries.push("zzzz".to_string());
        entries.sort();
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        let lexicon = Lexicon::new(&refs);
        let mut scratch = Scratch::default();
        let mut out = Vec::new();
        let mut visited = 0;
        lexicon.search_counted("zzzz", 1, &mut scratch, &mut out, &mut visited);
        assert_eq!(out, &[FuzzyMatch { index: (refs.len() - 1) as u32, distance: 0 }]);
        // One edge out of the root and one to the hit: nowhere near the
        // two hundred unrelated words.
        assert!(visited < 10, "{visited} edges visited");
        // A scan would compute a row for every one of the edges; pruning
        // keeps even an exact search to the path plus a few sibling probes
        // at the branch levels.
        let mut visited = 0;
        lexicon.search_counted("prefix042suffix", 0, &mut scratch, &mut out, &mut visited);
        assert!(visited < lexicon.nodes.len() / 10, "{visited} edges visited");
    }

    /// Every entry within the bound is found, even where the trie shares
    /// long prefixes, and results come sorted by (distance, index).
    #[test]
    fn exhaustive_small_lexicon_is_fully_matched() {
        let alphabet = ["a", "b", "é"];
        let mut entries = Vec::new();
        for &a in &alphabet {
            for &b in &alphabet {
                for &c in &alphabet {
                    for &d in &alphabet {
                        entries.push(format!("{a}{b}{c}{d}"));
                    }
                }
            }
        }
        entries.sort();
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        let lexicon = Lexicon::new(&refs);
        let mut scratch = Scratch::default();
        let mut out = Vec::new();
        for query in ["aabb", "éabé", "😀aab", "aaaaa", ""] {
            for max in 0..=3u8 {
                lexicon.search(query, max, &mut scratch, &mut out);
                let mut want = Vec::new();
                let q: Vec<char> = query.chars().collect();
                for (index, e) in refs.iter().enumerate() {
                    let e: Vec<char> = e.chars().collect();
                    let d = crate::fuzzy::tests::osa(&e, &q);
                    if d <= max as usize {
                        want.push(FuzzyMatch { index: index as u32, distance: d as u32 });
                    }
                }
                want.sort_by(|a, b| a.distance.cmp(&b.distance).then(a.index.cmp(&b.index)));
                assert_eq!(out, want, "{query:?} max {max}");
            }
        }
    }

    /// The full optimal-string-alignment table, as the definition states it.
    fn osa(a: &[char], b: &[char]) -> usize {
        let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
        for (i, row) in d.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in d[0].iter_mut().enumerate() {
            *cell = j;
        }
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
                if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                    v = v.min(d[i - 2][j - 2] + 1);
                }
                d[i][j] = v;
            }
        }
        d[a.len()][b.len()]
    }
}
