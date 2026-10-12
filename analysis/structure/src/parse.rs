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
use crate::marker::{ListMarker, atx, caps_heading, closes_fence, fence, list_marker, setext, thematic};
use crate::table::{cells, delimiter_row};
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
    /// How many quotes and lists `stack` holds: a new one's depth.
    quotes: usize,
    lists: usize,
    leaf: Leaf,
    runs: Vec<(usize, usize)>,
    cells: Vec<(usize, usize)>,
}

/// A depth, at most 255.
fn saturate(n: usize) -> u8 {
    u8::try_from(n).unwrap_or(u8::MAX)
}

/// Consumes a quote marker and the space or tab it may have after it.
fn after_quote_marker(c: &mut Cursor<'_>) {
    c.advance(1, 1);
    if c.indent().0 > 0 {
        c.skip_cols(1);
    }
}

impl<'a> Parser<'a> {
    pub(crate) fn new(text: &'a str, syntax: Syntax) -> Parser<'a> {
        Parser {
            text,
            bytes: text.as_bytes(),
            markdown: syntax == Syntax::Markdown,
            tree: Tree::default(),
            stack: Vec::new(),
            quotes: 0,
            lists: 0,
            leaf: Leaf::None,
            runs: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// Every block of the text, closed. A byte order mark at the start is
    /// in no block.
    pub(crate) fn run(mut self) -> Tree {
        let from = if self.text.starts_with('\u{FEFF}') { '\u{FEFF}'.len_utf8() } else { 0 };
        for (start, end) in Lines::new(self.bytes, from) {
            self.line(start, end);
        }
        self.close_leaf();
        self.close_containers(0);
        self.tree
    }

    fn line(&mut self, ls: usize, le: usize) {
        let mut c = Cursor::new(self.bytes, ls, le);
        let matched = self.match_containers(&mut c, le);
        let all = matched == self.stack.len();

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
            if matches!(self.leaf, Leaf::Paragraph { .. } | Leaf::Table { .. }) {
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
                Container::Item { node, indent, blank_start } => {
                    let (cols, p) = c.indent();
                    if self.is_blank(p, le) {
                        // An item can start with at most one blank line.
                        !(blank_start && !self.tree.has_child(node))
                    } else if cols >= indent {
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
                let depth = saturate(self.quotes + 1);
                let node = self.tree.open(Block::Quote, p, p + 1, self.parent(), depth);
                self.push(Container::Quote { node });
                after_quote_marker(c);
                continue;
            }
            if thematic(self.bytes, p, le) {
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
        if cols < 4 && thematic(self.bytes, p, le) {
            return false;
        }
        self.paragraph_line(p, le);
        true
    }

    /// The rest of a line that is not blank, once the containers are
    /// settled: it continues the open leaf or starts a new one.
    fn leaf_line(&mut self, c: &mut Cursor<'_>, le: usize) {
        let (cols, p) = c.indent();
        let b = self.bytes;
        let in_text = matches!(self.leaf, Leaf::Paragraph { .. } | Leaf::Table { .. });
        if self.markdown && cols >= 4 && !in_text {
            self.close_leaf();
            self.close_lists();
            let start = self.first_char(p, le);
            let node = self.tree.open(Block::Code, start, self.trim_end(start, le), self.parent(), 0);
            self.leaf = Leaf::Indented { node };
            return;
        }
        let shallow = cols < 4;
        if self.markdown && shallow {
            if let Some(level) = atx(b, p, le) {
                self.close_leaf();
                self.close_lists();
                let node = self.tree.open(Block::Heading, p, self.trim_end(p, le), self.parent(), level);
                self.tree.close(node);
                return;
            }
            if let Some((ch, len)) = fence(b, p, le) {
                self.close_leaf();
                self.close_lists();
                let node = self.tree.open(Block::Code, p, self.trim_end(p, le), self.parent(), 0);
                self.leaf = Leaf::Fenced { node, ch, len };
                return;
            }
        }
        if let Leaf::Paragraph { node, lines, .. } = self.leaf
            && shallow
            && let Some(level) = setext(b, p, le)
        {
            // Plain text underlines a heading of one line; under a longer
            // paragraph the line is a separator.
            if !self.markdown && lines > 1 {
                self.close_leaf();
                return;
            }
            let end = self.trim_end(p, le) as u32;
            let n = self.tree.node(node);
            n.kind = Block::Heading;
            n.depth = level;
            n.end = end;
            self.tree.close(node);
            self.leaf = Leaf::None;
            return;
        }
        if shallow && thematic(b, p, le) {
            self.close_leaf();
            self.close_lists();
            return;
        }
        if self.markdown && self.table_start(p, le) {
            return;
        }
        match self.leaf {
            Leaf::Table { node } => self.table_row(node, p, le),
            Leaf::Paragraph { .. } => self.paragraph_line(p, le),
            _ => {
                self.close_leaf();
                self.close_lists();
                let start = self.first_char(p, le);
                let end = self.trim_end(start, le);
                let node = self.tree.open(Block::Paragraph, start, end, self.parent(), 0);
                self.leaf = Leaf::Paragraph { node, lines: 1, last_start: start, prev_end: start };
            }
        }
    }

    /// A delimiter row under a paragraph whose last line has as many
    /// cells turns that line into a table's header.
    fn table_start(&mut self, p: usize, le: usize) -> bool {
        let Leaf::Paragraph { node, lines, last_start, prev_end } = self.leaf else { return false };
        let Some(n) = delimiter_row(self.text, p, le) else { return false };
        let header_end = self.tree.nodes[node as usize].end as usize;
        self.cells.clear();
        cells(self.text, last_start, header_end, &mut self.runs, &mut self.cells);
        if self.cells.len() != n {
            return false;
        }
        let end = self.trim_end(p, le);
        let table = if lines == 1 {
            let t = self.tree.node(node);
            t.kind = Block::Table;
            t.end = end as u32;
            node
        } else {
            self.tree.node(node).end = prev_end as u32;
            self.tree.close(node);
            let parent = self.tree.nodes[node as usize].parent;
            self.tree.open(Block::Table, last_start, end, parent, 0)
        };
        self.leaf = Leaf::Table { node: table };
        self.push_row(table, last_start, header_end);
        true
    }

    fn table_row(&mut self, table: u32, p: usize, le: usize) {
        let start = self.first_char(p, le);
        let end = self.trim_end(start, le);
        self.cells.clear();
        cells(self.text, start, end, &mut self.runs, &mut self.cells);
        self.push_row(table, start, end);
        self.tree.node(table).end = end as u32;
    }

    /// A row of the cells in `self.cells`.
    fn push_row(&mut self, table: u32, start: usize, end: usize) {
        let row = self.tree.open(Block::TableRow, start, end, Some(table), 0);
        for &(a, b) in &self.cells {
            let cell = self.tree.open(Block::TableCell, a, b, Some(row), 0);
            self.tree.close(cell);
        }
        self.tree.close(row);
    }

    fn paragraph_line(&mut self, p: usize, le: usize) {
        let Leaf::Paragraph { node, lines, .. } = self.leaf else { return };
        let start = self.first_char(p, le);
        let end = self.trim_end(start, le);
        let prev_end = self.tree.nodes[node as usize].end as usize;
        self.tree.node(node).end = end as u32;
        self.leaf = Leaf::Paragraph { node, lines: lines + 1, last_start: start, prev_end };
    }

    /// Moves a leaf's end to the end of a line's content, if the line
    /// from `from` holds any.
    fn extend(&mut self, node: u32, from: usize, le: usize) {
        let end = self.trim_end(from, le);
        if end > from {
            let n = self.tree.node(node);
            n.end = n.end.max(end as u32);
        }
    }

    fn close_leaf(&mut self) {
        match std::mem::replace(&mut self.leaf, Leaf::None) {
            Leaf::None => {}
            Leaf::Paragraph { node, lines, .. } => {
                // Plain text: a line on its own, at the top level, in
                // capitals, is a heading.
                let n = self.tree.nodes[node as usize];
                if !self.markdown
                    && lines == 1
                    && n.parent.is_none()
                    && caps_heading(&self.text[n.start as usize..n.end as usize])
                {
                    let n = self.tree.node(node);
                    n.kind = Block::Heading;
                    n.depth = 1;
                }
                self.tree.close(node);
            }
            Leaf::Fenced { node, .. } | Leaf::Indented { node } | Leaf::Table { node } => self.tree.close(node),
        }
    }

    /// Closes the containers from `from` on, innermost first, with the
    /// leaf inside them.
    fn close_unmatched(&mut self, from: usize) {
        if from < self.stack.len() {
            self.close_leaf();
            self.close_containers(from);
        }
    }

    fn close_containers(&mut self, from: usize) {
        while self.stack.len() > from {
            let c = self.pop();
            self.tree.close(c.node());
        }
    }

    /// A list holds only list items: anything else added where a list is
    /// innermost ends the list.
    fn close_lists(&mut self) {
        while let Some(&Container::List { node, .. }) = self.stack.last() {
            self.pop();
            self.tree.close(node);
        }
    }

    fn parent(&self) -> Option<u32> {
        self.stack.last().map(|c| c.node())
    }

    fn push(&mut self, c: Container) {
        match c {
            Container::Quote { .. } => self.quotes += 1,
            Container::List { .. } => self.lists += 1,
            Container::Item { .. } => {}
        }
        self.stack.push(c);
    }

    fn pop(&mut self) -> Container {
        let c = self.stack.pop().expect("an open container");
        match c {
            Container::Quote { .. } => self.quotes -= 1,
            Container::List { .. } => self.lists -= 1,
            Container::Item { .. } => {}
        }
        c
    }

    fn is_blank(&self, from: usize, le: usize) -> bool {
        from >= le || self.text[from..le].trim_start().is_empty()
    }

    /// The first character at or after `from` that is not whitespace.
    fn first_char(&self, from: usize, le: usize) -> usize {
        le - self.text[from..le].trim_start().len()
    }

    /// The end of the last character before `le` that is not whitespace,
    /// or `from` when there is none.
    fn trim_end(&self, from: usize, le: usize) -> usize {
        from + self.text[from..le].trim_end().len()
    }
}
