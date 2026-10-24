//! The block parser: one pass over the lines, keeping the open containers
//! (quotes, lists, list items) and at most one open leaf (a paragraph, a
//! code block or a table) at the innermost of them, the way CommonMark's
//! reference parser does.
//!
//! For each line: the open containers it continues are matched first (a
//! quote by its `>`, a list item by its indentation); then new quotes and
//! list items are opened; then the rest of the line goes to the open leaf
//! or starts a new one. A line that continues none of the inner containers
//! and starts nothing new can still continue a paragraph inside them (a
//! lazy line). Every span starts at a block's first character that is not
//! whitespace (its marker, for a container, a heading or a fence) and ends
//! after its last such character.

use turbo_analysis_layers::Block;

use crate::Syntax;
use crate::line::{Cursor, Lines};
use crate::marker::{ListMarker, atx, closes_fence, fence, list_marker};
use crate::tree::Tree;

#[derive(Copy, Clone, Debug)]
enum Container {
    Quote {
        node: u32,
    },
    List {
        node: u32,
        kind: char,
    },
    /// `indent`: how many columns past where its parent's content starts
    /// an item's own content starts, which a line must be indented by to
    /// continue it. Measured from the parent's content, not from the
    /// line's start, since the markers of the containers around it can
    /// take different widths on different lines (`> >` and `>>`).
    /// `blank_start`: the marker's line held nothing else.
    Item {
        node: u32,
        indent: usize,
        blank_start: bool,
    },
}

impl Container {
    fn node(self) -> u32 {
        match self {
            Container::Quote { node } | Container::List { node, .. } | Container::Item { node, .. } => node,
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum Leaf {
    Html {
        node: u32,
        end: crate::html::End,
    },
    Footnote {
        node: u32,
    },
    Columns {
        node: u32,
    },
    None,
    /// `last_start`: where its last line's content starts; `prev_end`:
    /// where the line before that ends. A table's header is the last line.
    Paragraph {
        node: u32,
        lines: u32,
        last_start: usize,
        prev_end: usize,
    },
    Fenced {
        node: u32,
        ch: u8,
        len: usize,
    },
    Indented {
        node: u32,
    },
    Table {
        node: u32,
    },
}

pub(crate) struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    markdown: bool,
    tree: Tree,
    stack: Vec<Container>,
    /// Where in `stack` the quotes are, in order; how many there are is a
    /// new quote's depth.
    quote_at: Vec<usize>,
    /// How many lists `stack` holds: a new one's depth.
    lists: usize,
    leaf: Leaf,
    /// Plain text: the headings found by their capitals alone, which are
    /// paragraphs again when nothing but such headings follows them.
    caps: Vec<u32>,
    /// The character and the stop of the last scan for a thematic break
    /// that failed before its line's end.
    not_thematic: Option<(u8, usize)>,
    runs: Vec<(usize, usize)>,
    cells: Vec<(usize, usize)>,
    column_starts: Vec<usize>,
    /// Last separator offsets on the previous/current physical lines.
    /// Compare them with the content bounds after matching containers.
    column_separators: [Option<usize>; 2],
}

/// A depth, at most 255.
fn saturate(n: usize) -> u8 {
    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Where a Markdown text's blocks start: after a front matter block (YAML
/// metadata, as static site generators read it) when the text starts with
/// one at `from`, else `from`. Front matter is a line `---` at the start,
/// up to and including the next line `---` or `...`; without that closing
/// line there is none. Spaces and tabs may end either line.
fn front_matter(text: &str, from: usize) -> usize {
    let b = text.as_bytes();
    let is = |(s, e): (usize, usize), mark: &[u8]| {
        let t = &b[s..e];
        t.starts_with(mark) && t[mark.len()..].iter().all(|&x| x == b' ' || x == b'\t')
    };
    let mut lines = Lines::new(text, from);
    if !lines.next().is_some_and(|l| is(l, b"---")) {
        return from;
    }
    for l in lines.by_ref() {
        if is(l, b"---") || is(l, b"...") {
            return lines.at();
        }
    }
    from
}

/// Consumes a quote marker and the space or tab it may have after it.
fn after_quote_marker(c: &mut Cursor<'_>) {
    c.advance(1, 1);
    if c.indent().0 > 0 {
        c.skip_cols(1);
    }
}

mod extra;
mod leaf;

impl<'a> Parser<'a> {
    pub(crate) fn new(text: &'a str, syntax: Syntax) -> Parser<'a> {
        Parser {
            text,
            bytes: text.as_bytes(),
            markdown: syntax == Syntax::Markdown,
            tree: Tree::default(),
            stack: Vec::new(),
            quote_at: Vec::new(),
            lists: 0,
            leaf: Leaf::None,
            caps: Vec::new(),
            not_thematic: None,
            runs: Vec::new(),
            cells: Vec::new(),
            column_starts: Vec::new(),
            column_separators: [None; 2],
        }
    }

    /// Every block of the text, closed. A byte order mark at the start is
    /// in no block, and neither is a Markdown text's front matter.
    pub(crate) fn run(mut self) -> Tree {
        let mut from = if self.text.starts_with('\u{FEFF}') { '\u{FEFF}'.len_utf8() } else { 0 };
        if self.markdown {
            from = front_matter(self.text, from);
        }
        if self.markdown {
            for (start, end) in Lines::new(self.text, from) {
                self.line(start, end);
            }
        } else {
            let (mut lines, mut previous) = (Lines::new(self.text, from), None);
            while let Some((start, end, columns)) = lines.next_columns() {
                self.column_separators = [previous, columns];
                self.line(start, end);
                previous = columns;
            }
        }
        self.close_leaf();
        self.close_containers(0);
        self.uncap_signature();
        self.tree.list_tightness(self.text);
        self.tree
    }

    /// Plain text: headings in capitals with nothing after them but more
    /// of the same (a name and a country under a letter's closing) open
    /// no section; they are paragraphs.
    fn uncap_signature(&mut self) {
        let mut last = self.tree.nodes.len();
        while let Some(&h) = self.caps.last() {
            // The top-level block after the heading, if any.
            let after = self.tree.nodes[h as usize + 1..last].iter().any(|n| n.parent.is_none());
            if after {
                return;
            }
            self.caps.pop();
            let n = self.tree.node(h);
            n.kind = Block::Paragraph;
            n.depth = 0;
            last = h as usize;
        }
    }

    fn line(&mut self, ls: usize, le: usize) {
        let mut c = Cursor::new(self.bytes, ls, le);
        let matched = self.match_containers(&mut c, le);
        let all = matched == self.stack.len();

        if all && matches!(self.leaf, Leaf::Html { .. } | Leaf::Footnote { .. }) && self.extra_continue(&mut c, le) {
            return;
        }
        if all {
            match self.leaf {
                Leaf::Fenced { node, ch, len } => {
                    let (cols, p) = c.indent();
                    if cols < 4 && p < le && closes_fence(self.bytes, p, le, ch, len) {
                        self.extend(node, p, le);
                        self.close_leaf();
                    } else {
                        self.extend(node, c.pos, le);
                    }
                    return;
                }
                Leaf::Indented { node } => {
                    let (cols, p) = c.indent();
                    if self.is_blank(p, le) || cols >= 4 {
                        self.extend(node, p, le);
                        return;
                    }
                }
                _ => {}
            }
        }

        let opened = self.open_containers(&mut c, le, matched, all);
        let blank = self.is_blank(c.pos, le);

        if !opened && !all && !blank && self.lazy(&mut c, le) {
            return;
        }
        if !opened {
            self.close_unmatched(matched);
        }
        if blank {
            if matches!(self.leaf, Leaf::Paragraph { .. } | Leaf::Table { .. } | Leaf::Columns { .. }) {
                self.close_leaf();
            }
            return;
        }
        self.leaf_line(&mut c, le);
    }

    /// How many of the open containers the line continues, consuming
    /// their markers and indentation from `c`. A list always continues:
    /// whether its item does decides, and a list whose item ends is closed
    /// once something other than a new item of it is added.
    fn match_containers(&mut self, c: &mut Cursor<'_>, le: usize) -> usize {
        let mut matched = 0;
        while matched < self.stack.len() {
            let (_, p) = c.indent();
            if self.is_blank(p, le) {
                return self.match_blank(matched);
            }
            let ok = match self.stack[matched] {
                Container::Quote { .. } => {
                    let (cols, p) = c.indent();
                    if cols < 4 && p < le && self.bytes[p] == b'>' {
                        c.skip_cols(cols);
                        after_quote_marker(c);
                        true
                    } else {
                        false
                    }
                }
                Container::List { .. } => true,
                Container::Item { indent, .. } => {
                    let (cols, _) = c.indent();
                    if cols >= indent {
                        c.skip_cols(indent);
                        true
                    } else {
                        false
                    }
                }
            };
            if !ok {
                break;
            }
            matched += 1;
        }
        matched
    }

    /// How many containers a line continues whose rest, from container
    /// `from` on, is blank, found without visiting the containers one by
    /// one, which would take time quadratic in the text for deep nesting
    /// followed by many blank lines. A blank rest continues every list
    /// and list item and no quote, except an item that started with a
    /// blank line and holds nothing yet: an item starts with at most one
    /// blank line. Such an item has no child, so it is the innermost
    /// container.
    fn match_blank(&self, from: usize) -> usize {
        let k = self.quote_at.partition_point(|&q| q < from);
        if let Some(&q) = self.quote_at.get(k) {
            return q;
        }
        match self.stack.last() {
            Some(&Container::Item { node, blank_start: true, .. }) if !self.tree.has_child(node) => {
                self.stack.len() - 1
            }
            _ => self.stack.len(),
        }
    }

    /// Opens the quotes and list items that start the rest of the line.
    /// Whether any opened.
    fn open_containers(&mut self, c: &mut Cursor<'_>, le: usize, matched: usize, all: bool) -> bool {
        let mut opened = false;
        loop {
            let (cols, p) = c.indent();
            if cols >= 4 || p >= le {
                return opened;
            }
            if self.bytes[p] == b'>' {
                if !opened {
                    self.close_leaf();
                    self.close_unmatched(matched);
                    opened = true;
                }
                c.skip_cols(cols);
                self.close_lists();
                let depth = saturate(self.quote_at.len() + 1);
                let node = self.tree.open(Block::Quote, p, p + 1, self.parent(), depth);
                self.push(Container::Quote { node });
                after_quote_marker(c);
                continue;
            }
            if self.thematic(p, le) {
                return opened;
            }
            let Some(m) = list_marker(self.text, p, le, !self.markdown) else { return opened };
            let after = p + m.bytes;
            let blank_after = self.is_blank(after, le);
            let interrupts = !opened && all && matches!(self.leaf, Leaf::Paragraph { .. });
            if interrupts && (blank_after || m.number.is_some_and(|n| n != 1)) {
                return opened;
            }
            if !opened {
                self.close_leaf();
                self.close_unmatched(matched);
                opened = true;
            }
            let base = c.col;
            c.skip_cols(cols);
            self.open_item(c, base, p, m, blank_after);
        }
    }

    /// Opens a list item whose marker `m` is at `p`, its parent's content
    /// starting at column `base`.
    fn open_item(&mut self, c: &mut Cursor<'_>, base: usize, p: usize, m: ListMarker, blank_after: bool) {
        let same_list = matches!(self.stack.last(), Some(&Container::List { kind, .. }) if kind == m.kind);
        if !same_list {
            self.close_lists();
            let depth = saturate(self.lists + 1);
            let node = self.tree.open(Block::List, p, p + m.bytes, self.parent(), depth);
            self.push(Container::List { node, kind: m.kind });
        }
        c.advance(m.bytes, m.cols);
        let marker_end = c.col;
        let (cols, _) = c.indent();
        let content = if blank_after || cols > 4 { marker_end + 1 } else { marker_end + cols };
        if !blank_after {
            c.skip_cols(content - marker_end);
        }
        let node = self.tree.open(Block::ListItem, p, p + m.bytes, self.parent(), saturate(self.lists));
        self.push(Container::Item { node, indent: content - base, blank_start: blank_after });
    }

    /// A lazy line: the paragraph open inside containers the line does
    /// not continue takes it, unless it starts a block of its own.
    fn lazy(&mut self, c: &mut Cursor<'_>, le: usize) -> bool {
        let Leaf::Paragraph { .. } = self.leaf else { return false };
        let (cols, p) = c.indent();
        if cols < 4 && (self.markdown && (atx(self.bytes, p, le).is_some() || fence(self.bytes, p, le).is_some())) {
            return false;
        }
        if self.markdown && cols < 4 && crate::html::start(&self.text[p..le], true).is_some() {
            return false;
        }
        if cols < 4 && self.thematic(p, le) {
            return false;
        }
        self.paragraph_line(p, le);
        true
    }
}
