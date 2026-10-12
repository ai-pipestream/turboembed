//! The blocks as the parser finds them, and the sections made from the
//! headings once every block is closed.

use turbo_analysis::{Ix, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Structure};

/// One block. Blocks are kept in the order they open, which is text order
/// with every parent before its children, the order the layer needs.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Node {
    pub kind: Block,
    pub start: u32,
    pub end: u32,
    pub parent: Option<u32>,
    pub depth: u8,
}

#[derive(Default)]
pub(crate) struct Tree {
    pub nodes: Vec<Node>,
}

impl Tree {
    pub(crate) fn open(&mut self, kind: Block, start: usize, end: usize, parent: Option<u32>, depth: u8) -> u32 {
        let i = self.nodes.len() as u32;
        self.nodes.push(Node { kind, start: start as u32, end: end as u32, parent, depth });
        i
    }

    /// Ends a block: its parent reaches at least as far. An open block's
    /// end is updated only on itself, so this is the one place an end
    /// moves up the tree.
    pub(crate) fn close(&mut self, i: u32) {
        let n = self.nodes[i as usize];
        if let Some(p) = n.parent {
            let p = &mut self.nodes[p as usize];
            p.end = p.end.max(n.end);
        }
    }

    pub(crate) fn node(&mut self, i: u32) -> &mut Node {
        &mut self.nodes[i as usize]
    }

    /// Whether block `i` has a child: blocks open in order, so while `i`
    /// is open every block after it is inside it.
    pub(crate) fn has_child(&self, i: u32) -> bool {
        self.nodes.len() > i as usize + 1
    }

    /// The layer, with a section around each top-level heading: from the
    /// heading to the end of the last top-level block before the next
    /// heading of the same or a higher level (a smaller number), nested by
    /// level. Headings inside a quote or a list open no section. Blocks
    /// before the first heading are in no section.
    pub(crate) fn into_structure(self, view: ViewId) -> Structure {
        let sections = self.nodes.iter().filter(|x| x.parent.is_none() && x.kind == Block::Heading).count();
        let mut out = Columns::with_capacity(self.nodes.len() + sections);
        let mut new_index = Vec::with_capacity(self.nodes.len());
        // The open sections, outermost first: (level, index in the layer).
        let mut open: Vec<(u8, usize)> = Vec::new();
        for x in &self.nodes {
            let parent = match x.parent {
                Some(p) => Some(new_index[p as usize]),
                None => {
                    if x.kind == Block::Heading {
                        while open.last().is_some_and(|&(level, _)| level >= x.depth) {
                            open.pop();
                        }
                        let up = open.last().map(|&(_, i)| i);
                        open.push((x.depth, out.push(Block::Section, x.start, x.end, up, x.depth)));
                    }
                    for &(_, i) in &open {
                        out.end[i] = out.end[i].max(x.end);
                    }
                    open.last().map(|&(_, i)| i)
                }
            };
            new_index.push(out.push(x.kind, x.start, x.end, parent, x.depth));
        }
        let mut spans = Spans::with_capacity(view, out.start.len());
        for (&a, &b) in out.start.iter().zip(&out.end) {
            spans.push(Span::new(a, b));
        }
        Structure { spans, kind: out.kind, parent: out.parent, depth: out.depth }
    }
}

/// The layer's columns while sections are added: a section's end is known
/// only once its last block is.
struct Columns {
    start: Vec<u32>,
    end: Vec<u32>,
    kind: Vec<Block>,
    parent: Vec<Option<Ix<Structure>>>,
    depth: Vec<u8>,
}

impl Columns {
    fn with_capacity(n: usize) -> Columns {
        Columns {
            start: Vec::with_capacity(n),
            end: Vec::with_capacity(n),
            kind: Vec::with_capacity(n),
            parent: Vec::with_capacity(n),
            depth: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, kind: Block, start: u32, end: u32, parent: Option<usize>, depth: u8) -> usize {
        self.start.push(start);
        self.end.push(end);
        self.kind.push(kind);
        self.parent.push(parent.map(|p| Ix::new(p as u32)));
        self.depth.push(depth);
        self.start.len() - 1
    }
}
