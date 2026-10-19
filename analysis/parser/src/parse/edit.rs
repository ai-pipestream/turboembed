//! Tree surgery on a parse: child mutation, adjunction, pruning, possessives and names.

use std::sync::Arc;

use turbo_analysis::Span;

use crate::head_rules::HeadRules;

use super::error::ParseError;
use super::{Head, Parse, insert_by_prob};

/// `Span.crosses`.
#[inline]
fn spans_cross(a: Span, b: Span) -> bool {
    !a.contains(b)
        && !b.contains(a)
        && ((a.start <= b.start && b.start < a.end) || (b.start <= a.start && a.start < b.end))
}

/// The path of `Parse.getCommonParent` for the nodes at paths `a` and `b`: the parent when they
/// are the same node, otherwise their deepest common ancestor (one of them, when it contains the
/// other).
fn common_parent_path(a: &[usize], b: &[usize]) -> Vec<usize> {
    if a == b {
        return a[..a.len().saturating_sub(1)].to_vec();
    }
    let lcp = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    a[..lcp].to_vec()
}

/// A tag node found by walking the tree: its type, span, head index and the child indices that
/// lead to it from the root.
struct TagPath {
    kind: String,
    span: Span,
    head_index: i32,
    path: Vec<usize>,
}

impl TagPath {
    /// The path of the node's parent.
    fn parent(&self) -> &[usize] {
        &self.path[..self.path.len() - 1]
    }
}

impl Parse {
    /// Replaces the label of the child at `index` (`Parse.setChild`).
    ///
    /// Fails with [`ParseError::ChildIndex`] when `index` is past the children, where Java throws
    /// `IndexOutOfBoundsException`.
    pub fn set_child(&mut self, index: usize, label: impl Into<String>) -> Result<(), ParseError> {
        let len = self.parts.len();
        if index >= len {
            return Err(ParseError::ChildIndex { index, len });
        }
        self.parts[index].label = Some(label.into());
        Ok(())
    }

    /// Appends `daughter`, after copies of its preceding punctuation, to this node and updates
    /// its span and head (`Parse.add`). The daughter keeps its punctuation set, as in Java.
    ///
    /// Fails with [`ParseError::MissingHead`] when `rules` give no head, after the children and
    /// span have changed, as Java does.
    pub fn add(&mut self, daughter: Parse, rules: &dyn HeadRules) -> Result<(), ParseError> {
        self.parts.extend(daughter.prev_punct.iter().cloned());
        let end = daughter.span.end;
        self.parts.push(daughter);
        self.span = Span::new(self.span.start, end);
        self.take_head_from_children(rules)
    }

    /// Removes the child at `index` and updates the span when the first or last child is removed
    /// (`Parse.remove`).
    ///
    /// Fails with [`ParseError::ChildIndex`] when `index` is past the children, where Java throws
    /// `IndexOutOfBoundsException`.
    pub fn remove(&mut self, index: usize) -> Result<(), ParseError> {
        let len = self.parts.len();
        if index >= len {
            return Err(ParseError::ChildIndex { index, len });
        }
        self.parts.remove(index);
        if !self.parts.is_empty() && (index == 0 || index == self.parts.len()) {
            self.update_span();
        }
        Ok(())
    }

    /// Adjoins `node` to the child at `parse_index` and returns the node made to hold both
    /// (`Parse.adjoinRoot`). That node has the child's type and the head `rules` give for the
    /// child and `node`, and holds copies of `node`'s preceding punctuation between them. This
    /// node's span is left as it is.
    ///
    /// Fails with [`ParseError::ChildIndex`] when `parse_index` is past the children, where Java
    /// throws `IndexOutOfBoundsException`.
    pub fn adjoin_root(
        &mut self,
        node: Parse,
        rules: &dyn HeadRules,
        parse_index: usize,
    ) -> Result<&mut Parse, ParseError> {
        let len = self.parts.len();
        if parse_index >= len {
            return Err(ParseError::ChildIndex { index: parse_index, len });
        }
        Self::adjoin_pair(&mut self.parts[parse_index], node, rules);
        Ok(&mut self.parts[parse_index])
    }

    /// Sister-adjoins this node's last child and `sister` under a node made with the last child's
    /// type, as [`Parse::adjoin_root`] does, then updates this node's span and head
    /// (`Parse.adjoin`). Returns the node holding both.
    ///
    /// Fails with [`ParseError::ChildIndex`] when this node has no children, where Java throws
    /// `IndexOutOfBoundsException` reading its last child, and with [`ParseError::MissingHead`]
    /// when `rules` give no head for this node's children, after the children and span have
    /// changed, as Java does.
    pub fn adjoin(&mut self, sister: Parse, rules: &dyn HeadRules) -> Result<&mut Parse, ParseError> {
        let len = self.parts.len();
        if len == 0 {
            return Err(ParseError::ChildIndex { index: 0, len });
        }
        let last = len - 1;
        let sister_end = sister.span.end;
        Self::adjoin_pair(&mut self.parts[last], sister, rules);
        self.span = Span::new(self.span.start, sister_end);
        self.take_head_from_children(rules)?;
        Ok(&mut self.parts[last])
    }

    /// Replaces `slot` by a node holding its old content and `node`, as `adjoin` and
    /// `adjoin_root` build it.
    fn adjoin_pair(slot: &mut Parse, node: Parse, rules: &dyn HeadRules) {
        let placeholder = Parse::new(Arc::clone(&slot.text), slot.span, String::new(), 0.0, 0);
        let pair = [std::mem::replace(slot, placeholder), node];
        let head = rules.get_head(&pair, &pair[0].kind);
        let [last_child, node] = pair;
        let span = Span::new(last_child.span.start, node.span.end);
        let mut adj = Parse::with_head(Arc::clone(&last_child.text), span, last_child.kind.clone(), 1.0, head);
        adj.parts.reserve(node.prev_punct.len() + 2);
        adj.parts.push(last_child);
        adj.parts.extend(node.prev_punct.iter().cloned());
        adj.parts.push(node);
        *slot = adj;
    }

    /// Moves all children of `self` other than the child at `root_index` into that child, those
    /// before it in front of its own children and those after it behind them, and updates that
    /// child's span (`Parse.expandTopNode`). This node's span and head are left as they are.
    ///
    /// Fails with [`ParseError::ChildIndex`], changing nothing, when `root_index` is past the
    /// children, where Java throws `IndexOutOfBoundsException`.
    pub fn expand_top_node(&mut self, root_index: usize) -> Result<(), ParseError> {
        let len = self.parts.len();
        if root_index >= len {
            return Err(ParseError::ChildIndex { index: root_index, len });
        }
        let mut after = self.parts.split_off(root_index + 1);
        let mut root = self.parts.pop().expect("root_index is a child of this node");
        let mut before = std::mem::take(&mut self.parts);
        before.reserve(root.parts.len() + after.len());
        before.append(&mut root.parts);
        before.append(&mut after);
        root.parts = before;
        root.update_span();
        self.parts.push(root);
        Ok(())
    }

    /// Computes heads bottom-up for this parse and all sub-parses (`Parse.updateHeads`). A node
    /// whose rules give no head (a part-of-speech node over its token) becomes its own head and
    /// keeps its head index.
    pub fn update_heads(&mut self, rules: &dyn HeadRules) {
        if self.parts.is_empty() {
            self.head = None;
            return;
        }
        for c in &mut self.parts {
            c.update_heads(rules);
        }
        match rules.get_head(&self.parts, &self.kind) {
            Some(h) => {
                self.head = Some(h);
                self.head_index = h.head_index;
            }
            None => self.head = None,
        }
    }

    /// Sets the head `rules` give for the children, as `add` and `adjoin` do.
    fn take_head_from_children(&mut self, rules: &dyn HeadRules) -> Result<(), ParseError> {
        let h = rules.get_head(&self.parts, &self.kind).ok_or(ParseError::MissingHead)?;
        self.head = Some(h);
        self.head_index = h.head_index;
        Ok(())
    }

    /// Recomputes `span` from the first and last child (`Parse.updateSpan`).
    pub fn update_span(&mut self) {
        if let (Some(first), Some(last)) = (self.parts.first(), self.parts.last()) {
            self.span = Span::new(first.span.start, last.span.end);
        }
    }

    /// Replaces every node whose only child has the node's own type by that child
    /// (`Parse.pruneParse`), repeatedly, so a chain `A -> A -> A` keeps its last node.
    ///
    /// Fails with [`ParseError::MissingParent`], changing nothing, when `parse` itself has a single
    /// child of its own type, where Java dereferences the root's missing parent.
    pub fn prune_parse(parse: &mut Parse) -> Result<(), ParseError> {
        if parse.parts.len() == 1 && parse.kind == parse.parts[0].kind {
            return Err(ParseError::MissingParent);
        }
        Self::prune_children(parse);
        Ok(())
    }

    fn prune_children(node: &mut Parse) {
        for child in &mut node.parts {
            while child.parts.len() == 1 && child.kind == child.parts[0].kind {
                let grand_child = child.parts.pop().expect("one child");
                *child = grand_child;
            }
            Self::prune_children(child);
        }
    }

    /// Wraps the tag nodes after a possessive (`POS`) in an `NP` whose head is the first of them
    /// (`Parse.fixPossesives`): when the tag after a `POS` node is a child of the `POS` node's
    /// grandparent, it and the following tags that share its parent are put under an added `NP`.
    ///
    /// OpenNLP compares the tag nodes' current parents, so an `NP` inserted for one possessive
    /// changes the parents seen for the next; the tag paths are read again after every insert.
    pub fn fix_possessives(parse: &mut Parse) {
        let mut tags: Vec<TagPath> = Vec::new();
        let mut path = Vec::new();
        Self::collect_tag_paths(parse, &mut path, &mut tags);
        for ti in 0..tags.len() {
            if tags[ti].kind != "POS" || ti + 1 >= tags.len() {
                continue;
            }
            let grandparent = tags[ti].path.len().checked_sub(2).map(|n| &tags[ti].path[..n]);
            if grandparent != Some(tags[ti + 1].parent()) {
                continue;
            }
            let start = tags[ti + 1].span.start;
            let mut end = tags[ti + 1].span.end;
            for npi in (ti + 2)..tags.len() {
                if tags[npi].parent() == tags[npi - 1].parent() {
                    end = tags[npi].span.end;
                } else {
                    break;
                }
            }
            let head = Head { span: tags[ti + 1].span, head_index: tags[ti + 1].head_index };
            let np = Parse::with_head(Arc::clone(&parse.text), Span::new(start, end), "NP", 1.0, Some(head));
            parse.insert(np).expect("tag nodes lie inside the root");
            tags.clear();
            Self::collect_tag_paths(parse, &mut path, &mut tags);
        }
    }

    /// Lists the tag nodes as `getTagNodes` does, each with its path. Inserting a node over whole
    /// tag nodes, as `fix_possessives` and `add_names` do, never changes which nodes are tags or
    /// their order, so listing them again finds the same nodes as Java's array holds.
    fn collect_tag_paths(node: &Parse, path: &mut Vec<usize>, out: &mut Vec<TagPath>) {
        for (i, child) in node.parts.iter().enumerate() {
            path.push(i);
            if child.is_pos_tag() {
                out.push(TagPath {
                    kind: child.kind.clone(),
                    span: child.span,
                    head_index: child.head_index,
                    path: path.clone(),
                });
            } else {
                Self::collect_tag_paths(child, path, out);
            }
            path.pop();
        }
    }

    /// Inserts a node of type `tag` for each name (`Parse.addNames`). Each name is a range of
    /// indices into the tag nodes of `parse` (the `getTagNodes` array OpenNLP's callers pass as
    /// `tokens`), end exclusive.
    ///
    /// A name's node goes under the common parent of its first and last token when the name's span
    /// is that parent's span or crosses none of its children. When it crosses a child of an `NP`
    /// and covers the last child of that `NP`'s first child, the node spans the whole `NP`
    /// instead; otherwise the name is skipped. The tokens are the same nodes for every name, so
    /// inserting one name's node can move the tokens of the next, as in Java.
    ///
    /// Fails, after inserting the names before it, on a token index outside the tag nodes or a
    /// name that ends before it starts.
    pub fn add_names(tag: &str, names: &[Span], parse: &mut Parse) -> Result<(), ParseError> {
        let mut tokens: Vec<TagPath> = Vec::new();
        let mut path = Vec::new();
        Self::collect_tag_paths(parse, &mut path, &mut tokens);
        for name in names {
            let token = |index: i64| {
                usize::try_from(index)
                    .ok()
                    .and_then(|i| tokens.get(i))
                    .ok_or(ParseError::TokenIndex { index, len: tokens.len() })
            };
            let first = token(i64::from(name.start))?;
            let last = token(i64::from(name.end) - 1)?;
            if first.span.start > last.span.end {
                return Err(ParseError::ReversedName { name: *name });
            }
            let common = common_parent_path(&first.path, &last.path);
            let name_span = Span::new(first.span.start, last.span.end);
            let end_head_index = last.head_index;
            let parent = parse.node_at_path_mut(&common);
            let node = if name_span == parent.span || !parent.parts.iter().any(|kid| spans_cross(name_span, kid.span)) {
                Some(Parse::new(Arc::clone(&parent.text), name_span, tag, 1.0, end_head_index))
            } else if parent.kind == "NP"
                && let Some(grand_kids) = parent.parts.first().map(|kid| &kid.parts)
                && grand_kids.len() > 1
                && name_span.contains(grand_kids[grand_kids.len() - 1].span)
            {
                Some(Parse::new(Arc::clone(&parent.text), parent.span, tag, 1.0, parent.head_index))
            } else {
                None
            };
            if let Some(node) = node {
                parent.insert(node).expect("a name lies inside its common parent");
                tokens.clear();
                Self::collect_tag_paths(parse, &mut path, &mut tokens);
            }
        }
        Ok(())
    }

    /// The node at `path`, the child indices from this node down; this node for an empty path.
    /// Panics when an index is past the children of the node it indexes.
    pub fn node_at_path(&self, path: &[usize]) -> &Parse {
        let mut cur = self;
        for &idx in path {
            cur = &cur.parts[idx];
        }
        cur
    }

    /// The node at `path`, as [`Parse::node_at_path`] finds it, to change.
    pub fn node_at_path_mut(&mut self, path: &[usize]) -> &mut Parse {
        let mut cur = self;
        for &idx in path {
            cur = &mut cur.parts[idx];
        }
        cur
    }

    /// Returns the punctuation parses immediately preceding this node, in the order of
    /// `Parse.getPreviousPunctuationSet`.
    pub fn previous_punctuation(&self) -> &[Parse] {
        &self.prev_punct
    }

    /// Adds a punctuation parse preceding this node (`Parse.addPreviousPunctuation`).
    ///
    /// OpenNLP keeps the set in a `TreeSet` ordered by [`Parse::cmp_prob`], so the marks are in
    /// order of descending probability and a mark with the same probability as one already there
    /// is dropped. Every node `Parse::parse` makes has probability 1, so only the first mark added
    /// to such a node is kept.
    pub fn add_previous_punctuation(&mut self, punct: Parse) {
        insert_by_prob(&mut self.prev_punct, punct);
    }

    /// Sets the punctuation parses preceding this node, in the order given
    /// (`Parse.setPrevPunctuation`).
    pub fn set_previous_punctuation(&mut self, punct: Vec<Parse>) {
        self.prev_punct = punct;
    }

    /// Returns the punctuation parses immediately following this node, in the order of
    /// `Parse.getNextPunctuationSet`.
    pub fn next_punctuation(&self) -> &[Parse] {
        &self.next_punct
    }

    /// Adds a punctuation parse following this node (`Parse.addNextPunctuation`), ordered and
    /// dropped as [`Parse::add_previous_punctuation`] describes.
    pub fn add_next_punctuation(&mut self, punct: Parse) {
        insert_by_prob(&mut self.next_punct, punct);
    }

    /// Sets the punctuation parses following this node, in the order given
    /// (`Parse.setNextPunctuation`).
    pub fn set_next_punctuation(&mut self, punct: Vec<Parse>) {
        self.next_punct = punct;
    }

    /// Returns the derivation string, if any (`Parse.getDerivation`).
    pub fn derivation(&self) -> Option<&str> {
        self.derivation.as_deref()
    }

    /// Sets the derivation string (`Parse.setDerivation`).
    pub fn set_derivation(&mut self, derivation: Option<String>) {
        self.derivation = derivation;
    }
}
