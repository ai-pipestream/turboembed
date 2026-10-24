use super::{Leaf, Parser};
use crate::line::Cursor;
use turbo_analysis_layers::Block;

impl Parser<'_> {
    pub(super) fn extra_continue(&mut self, c: &mut Cursor<'_>, le: usize) -> bool {
        match self.leaf {
            Leaf::Html { node, end } => {
                if matches!(end, crate::html::End::Blank) && end.closes(&self.text[c.pos..le]) {
                    self.close_leaf();
                } else {
                    self.extend(node, c.pos, le);
                    if end.closes(&self.text[c.pos..le]) {
                        self.close_leaf();
                    }
                }
                true
            }
            Leaf::Footnote { node } => {
                let (cols, p) = c.indent();
                if self.is_blank(p, le) {
                    return true;
                }
                if cols >= 4 {
                    self.extend(node, p, le);
                    return true;
                }
                self.close_leaf();
                false
            }
            _ => false,
        }
    }

    pub(super) fn extra_start(&mut self, p: usize, le: usize, cols: usize) -> bool {
        if !self.markdown || cols >= 4 {
            return false;
        }
        let line = &self.text[p..le];
        if let Some(end) = crate::html::start(line, matches!(self.leaf, Leaf::Paragraph { .. })) {
            self.close_leaf();
            self.close_lists();
            let node = self.tree.open(Block::Html, p, self.trim_end(p, le), self.parent(), 0);
            self.leaf = Leaf::Html { node, end };
            if end.closes(line) {
                self.close_leaf();
            }
            return true;
        }
        if let Some(label) = line.strip_prefix("[^").and_then(|s| s.split_once("]:"))
            && !label.0.is_empty()
            && !label.0.chars().any(|c| c.is_whitespace() || matches!(c, '[' | ']'))
        {
            self.close_leaf();
            self.close_lists();
            let node = self.tree.open(Block::Footnote, p, self.trim_end(p, le), self.parent(), 0);
            self.leaf = Leaf::Footnote { node };
            return true;
        }
        false
    }

    pub(super) fn columns_start(&mut self, p: usize, le: usize) -> bool {
        if !self.columns_candidate {
            return false;
        }
        let Leaf::Paragraph { node, lines, last_start, prev_end } = self.leaf else { return false };
        let header_end = self.tree.nodes[node as usize].end as usize;
        if !crate::columns::aligned(self.text, last_start, header_end, p, le) {
            return false;
        }
        let (starts, header) = crate::columns::row(self.text, last_start, header_end);
        let (_, row) = crate::columns::row(self.text, p, le);
        let end = self.trim_end(p, le);
        let table = if lines == 1 {
            self.tree.node(node).kind = Block::Table;
            node
        } else {
            self.tree.node(node).end = prev_end as u32;
            self.tree.close(node);
            self.tree.open(Block::Table, last_start, header_end, self.tree.nodes[node as usize].parent, 0)
        };
        self.column_starts = starts;
        self.cells = header;
        self.push_row(table, last_start, header_end);
        self.cells = row;
        self.push_row(table, p, end);
        self.tree.node(table).end = end as u32;
        self.leaf = Leaf::Columns { node: table };
        true
    }

    pub(super) fn columns_row(&mut self, table: u32, p: usize, le: usize) {
        let (starts, row) = crate::columns::row(self.text, p, le);
        if starts == self.column_starts {
            self.cells = row;
            self.push_row(table, p, self.trim_end(p, le));
            self.tree.node(table).end = self.trim_end(p, le) as u32;
        } else {
            self.close_leaf();
            self.leaf_line(&mut Cursor::new(self.bytes, p, le), le);
        }
    }
}
