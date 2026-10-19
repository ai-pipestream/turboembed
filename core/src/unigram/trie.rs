/// The pieces as a trie over their bytes: each node's children are one
/// run of `nodes`, in byte order; a walk scans a node's few children, or
/// reads a table of its many. SentencePiece's `.model` tokenizer
/// (sentencepiece.rs) walks the same trie.
pub(crate) struct Trie {
    pub(super) nodes: Vec<Node>,
    /// Each node's `byte`, apart, so a search through a node's children
    /// reads one byte each.
    bytes: Vec<u8>,
    /// For a node of many children, each byte's child as its place among
    /// them, or u16::MAX: no search there.
    pub(super) wide: Vec<[u16; 256]>,
    /// The root's child for each first byte, or NONE: no search there,
    /// where every walk starts.
    root: Box<[u32; 256]>,
}

#[derive(Clone, Copy)]
pub(super) struct Node {
    /// The byte on the edge into this node.
    byte: u8,
    /// The piece that ends here, or NONE.
    id: u32,
    /// The children: nodes[first..first + count].
    first: u32,
    pub(super) count: u32,
    /// The node's table in `wide`, or NONE.
    pub(super) wide: u32,
}

impl Trie {
    pub(super) const NONE: u32 = u32::MAX;

    /// The trie of `pieces`, each the id of its place; Err is a piece
    /// listed twice.
    pub(super) fn new(pieces: &[Box<str>]) -> std::result::Result<Trie, String> {
        Self::of(pieces.iter().enumerate().map(|(id, p)| (p.as_bytes(), id as u32)).collect())
    }

    /// The trie of (piece, id) pairs, none of them empty; Err is a piece
    /// listed twice.
    pub(crate) fn of(mut pieces: Vec<(&[u8], u32)>) -> std::result::Result<Trie, String> {
        pieces.sort_unstable();
        if let Some(w) = pieces.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(String::from_utf8_lossy(w[0].0).into_owned());
        }
        let mut nodes = vec![Node { byte: 0, id: Self::NONE, first: 0, count: 0, wide: Self::NONE }];
        // Depth first, a node's children appended together when it is
        // reached, so that a walk down the trie stays near where it
        // started: (node, the pieces under it, its depth).
        let mut stack = vec![(0usize, 0usize, pieces.len(), 0usize)];
        while let Some((node, mut lo, hi, depth)) = stack.pop() {
            if lo < hi && pieces[lo].0.len() == depth {
                nodes[node].id = pieces[lo].1;
                lo += 1;
            }
            nodes[node].first = nodes.len() as u32;
            let pushed = stack.len();
            while lo < hi {
                let byte = pieces[lo].0[depth];
                let end = lo + pieces[lo..hi].partition_point(|p| p.0[depth] == byte);
                stack.push((nodes.len(), lo, end, depth + 1));
                nodes.push(Node { byte, id: Self::NONE, first: 0, count: 0, wide: Self::NONE });
                lo = end;
            }
            nodes[node].count = nodes.len() as u32 - nodes[node].first;
            // The first child next.
            stack[pushed..].reverse();
        }
        let mut root = Box::new([Self::NONE; 256]);
        let r = nodes[0];
        for k in r.first..r.first + r.count {
            root[nodes[k as usize].byte as usize] = k;
        }
        let bytes = nodes.iter().map(|n| n.byte).collect();
        let mut wide = Vec::new();
        for i in 0..nodes.len() {
            let n = nodes[i];
            if n.count as usize > Self::NARROW {
                let mut t = [u16::MAX; 256];
                for k in 0..n.count {
                    t[nodes[(n.first + k) as usize].byte as usize] = k as u16;
                }
                nodes[i].wide = wide.len() as u32;
                wide.push(t);
            }
        }
        Ok(Trie { nodes, bytes, wide, root })
    }

    /// The id of the piece that is exactly `piece`.
    pub(crate) fn get(&self, piece: &[u8]) -> Option<u32> {
        let (&b0, rest) = piece.split_first()?;
        let mut node = *self.nodes.get(self.root[b0 as usize] as usize)?;
        for &b in rest {
            node = self.nodes[self.child(node, b)?];
        }
        (node.id != Self::NONE).then_some(node.id)
    }

    /// `found(len, id)` for every piece that `text` starts with, shortest
    /// first.
    #[inline]
    pub(crate) fn prefixes(&self, text: &[u8], mut found: impl FnMut(usize, u32)) {
        let Some(&b0) = text.first() else { return };
        let Some(&first) = self.nodes.get(self.root[b0 as usize] as usize) else { return };
        if first.id != Self::NONE {
            found(1, first.id);
        }
        let mut node = first;
        for (i, &b) in text.iter().enumerate().skip(1) {
            let Some(k) = self.child(node, b) else { return };
            node = self.nodes[k];
            if node.id != Self::NONE {
                found(i + 1, node.id);
            }
        }
    }

    /// The most children a node has with no table of them.
    const NARROW: usize = 8;

    /// The index of `node`'s child on byte `b`: a scan of a few children,
    /// a table of many.
    #[inline]
    fn child(&self, node: Node, b: u8) -> Option<usize> {
        if node.wide != Self::NONE {
            let k = self.wide[node.wide as usize][b as usize];
            return (k != u16::MAX).then_some(node.first as usize + k as usize);
        }
        let first = node.first as usize;
        let kids = &self.bytes[first..first + node.count as usize];
        kids.iter().position(|&k| k == b).map(|k| first + k)
    }
}
