use turbo_analysis_layers::Block;

use crate::line::Cursor;
use crate::marker::{atx, caps_heading, fence, setext, thematic_scan};
use crate::table::{cells, delimiter_row};

use super::{Container, Leaf, Parser};

impl<'a> Parser<'a> {
    /// The rest of a line that is not blank, once the containers are
    /// settled: it continues the open leaf or starts a new one.
    pub(super) fn leaf_line(&mut self, c: &mut Cursor<'_>, le: usize) {
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
    pub(super) fn table_start(&mut self, p: usize, le: usize) -> bool {
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

    pub(super) fn table_row(&mut self, table: u32, p: usize, le: usize) {
        let start = self.first_char(p, le);
        let end = self.trim_end(start, le);
        self.cells.clear();
        cells(self.text, start, end, &mut self.runs, &mut self.cells);
        self.push_row(table, start, end);
        self.tree.node(table).end = end as u32;
    }

    /// A row of the cells in `self.cells`.
    pub(super) fn push_row(&mut self, table: u32, start: usize, end: usize) {
        let row = self.tree.open(Block::TableRow, start, end, Some(table), 0);
        for &(a, b) in &self.cells {
            let cell = self.tree.open(Block::TableCell, a, b, Some(row), 0);
            self.tree.close(cell);
        }
        self.tree.close(row);
    }

    pub(super) fn paragraph_line(&mut self, p: usize, le: usize) {
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
    pub(super) fn extend(&mut self, node: u32, from: usize, le: usize) {
        let end = self.trim_end(from, le);
        if end > from {
            let n = self.tree.node(node);
            n.end = n.end.max(end as u32);
        }
    }

    pub(super) fn close_leaf(&mut self) {
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
    pub(super) fn thematic(&mut self, p: usize, le: usize) -> bool {
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
    pub(super) fn is_caps_heading(&self, node: u32) -> bool {
        let n = self.tree.nodes[node as usize];
        let first = &self.text[n.start as usize..n.end as usize];
        let first = first.split(['\n', '\r']).next().unwrap_or(first);
        !self.markdown && n.parent.is_none() && caps_heading(first)
    }

    /// Makes a one-line paragraph a heading of level 1, found by its
    /// capitals.
    pub(super) fn cap(&mut self, node: u32) {
        let n = self.tree.node(node);
        n.kind = Block::Heading;
        n.depth = 1;
        self.caps.push(node);
    }

    /// Closes the containers from `from` on, innermost first, with the
    /// leaf inside them.
    pub(super) fn close_unmatched(&mut self, from: usize) {
        if from < self.stack.len() {
            self.close_leaf();
            self.close_containers(from);
        }
    }

    pub(super) fn close_containers(&mut self, from: usize) {
        while self.stack.len() > from {
            let c = self.pop();
            self.tree.close(c.node());
        }
    }

    /// A list holds only list items: anything else added where a list is
    /// innermost ends the list.
    pub(super) fn close_lists(&mut self) {
        while let Some(&Container::List { node, .. }) = self.stack.last() {
            self.pop();
            self.tree.close(node);
        }
    }

    pub(super) fn parent(&self) -> Option<u32> {
        self.stack.last().map(|c| c.node())
    }

    pub(super) fn push(&mut self, c: Container) {
        match c {
            Container::Quote { .. } => self.quote_at.push(self.stack.len()),
            Container::List { .. } => self.lists += 1,
            Container::Item { .. } => {}
        }
        self.stack.push(c);
    }

    pub(super) fn pop(&mut self) -> Container {
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

    pub(super) fn is_blank(&self, from: usize, le: usize) -> bool {
        from >= le || self.text[from..le].trim_start().is_empty()
    }

    /// The first character at or after `from` that is not whitespace.
    pub(super) fn first_char(&self, from: usize, le: usize) -> usize {
        le - self.text[from..le].trim_start().len()
    }

    /// The end of the last character before `le` that is not whitespace,
    /// or `from` when there is none.
    pub(super) fn trim_end(&self, from: usize, le: usize) -> usize {
        from + self.text[from..le].trim_end().len()
    }
}
