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
use crate::marker::{ListMarker, atx, caps_heading, closes_fence, fence, list_marker, setext, thematic_scan};
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
fn front_matter(b: &[u8], from: usize) -> usize {
    let is = |(s, e): (usize, usize), mark: &[u8]| {
        let t = &b[s..e];
        t.starts_with(mark) && t[mark.len()..].iter().all(|&x| x == b' ' || x == b'\t')
    };
    let mut lines = Lines::new(b, from);
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
        }
    }

    /// Every block of the text, closed. A byte order mark at the start is
    /// in no block, and neither is a Markdown text's front matter.
    pub(crate) fn run(mut self) -> Tree {
        let mut from = if self.text.starts_with('\u{FEFF}') { '\u{FEFF}'.len_utf8() } else { 0 };
        if self.markdown {
            from = front_matter(self.bytes, from);
        }
        for (start, end) in Lines::new(self.bytes, from) {
            self.line(start, end);
        }
        self.close_leaf();
        self.close_containers(0);
        self.uncap_signature();
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
        if cols < 4 && self.thematic(p, le) {
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
            && let Some((level, run)) = setext(b, p, le)
            && (self.markdown || run >= 3)
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
        if shallow && self.thematic(p, le) {
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
        // Plain text: a first line in capitals over a second with lower
        // case letters is a heading over its paragraph ("INTRODUCTION",
        // then "This report covers the year.").
        if lines == 1 && self.text[start..end].chars().any(char::is_lowercase) && self.is_caps_heading(node) {
            self.cap(node);
            self.tree.close(node);
            let parent = self.tree.nodes[node as usize].parent;
            let node = self.tree.open(Block::Paragraph, start, end, parent, 0);
            self.leaf = Leaf::Paragraph { node, lines: 1, last_start: start, prev_end: start };
            return;
        }
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
                if lines == 1 && self.is_caps_heading(node) {
                    self.cap(node);
                }
                self.tree.close(node);
            }
            Leaf::Fenced { node, .. } | Leaf::Indented { node } | Leaf::Table { node } => self.tree.close(node),
        }
    }

    /// Whether a thematic break is at `p`. Every marker of a line like
    /// "- - - … a" is tested as one, and each test from scratch would scan
    /// to the `a`, time quadratic in the line; a test from inside a failed
    /// scan's run of the same character fails where that scan did.
    fn thematic(&mut self, p: usize, le: usize) -> bool {
        let ch = self.bytes[p];
        if let Some((c, stop)) = self.not_thematic
            && c == ch
            && p < stop
        {
            return false;
        }
        match thematic_scan(self.bytes, p, le) {
            Ok(yes) => yes,
            Err(stop) => {
                self.not_thematic = Some((ch, stop));
                false
            }
        }
    }

    /// Whether a paragraph's first line, in plain text at the top level, is
    /// a heading by its capitals.
    fn is_caps_heading(&self, node: u32) -> bool {
        let n = self.tree.nodes[node as usize];
        let first = &self.text[n.start as usize..n.end as usize];
        let first = first.split(['\n', '\r']).next().unwrap_or(first);
        !self.markdown && n.parent.is_none() && caps_heading(first)
    }

    /// Makes a one-line paragraph a heading of level 1, found by its
    /// capitals.
    fn cap(&mut self, node: u32) {
        let n = self.tree.node(node);
        n.kind = Block::Heading;
        n.depth = 1;
        self.caps.push(node);
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
            Container::Quote { .. } => self.quote_at.push(self.stack.len()),
            Container::List { .. } => self.lists += 1,
            Container::Item { .. } => {}
        }
        self.stack.push(c);
    }

    fn pop(&mut self) -> Container {
        let c = self.stack.pop().expect("an open container");
        match c {
            Container::Quote { .. } => {
                self.quote_at.pop();
            }
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
