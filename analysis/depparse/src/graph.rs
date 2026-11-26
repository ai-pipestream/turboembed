use std::sync::OnceLock;

use crate::{Error, Result, error};
use turbo_analysis::{CodePointSpan, Ix, Layer, Span, Tag, TagSet, Utf16Span, View};
use turbo_analysis_layers::{Dependencies, MorphPosTags, Tokens, Word, WordOrTokenAnchor};

/// One labeled edge. A root has no head index.
#[derive(Debug, PartialEq, Eq)]
pub struct DependencyArc<'a, A: WordOrTokenAnchor = Tokens> {
    pub dependent: Ix<A>,
    pub head: Option<Ix<A>>,
    pub relation: &'a str,
}

/// A dependency node resolved against its document [`View`], surface [`Tokens`],
/// anchor layer (`A = Tokens` or `A = Words`), and optional POS layer (`A::Pos`).
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedDependencyNode<'a, A: WordOrTokenAnchor = Tokens> {
    /// Strongly-typed index into the anchor layer (`Ix<Tokens>` or `Ix<Words>`).
    pub index: Ix<A>,
    /// Resolved syntactic word with surface `IxRange<Tokens>`, `token_span`,
    /// optional clitic `subspan`, and borrowed `form`.
    pub word: Word<'a>,
    /// UTF-16 code-unit span of the node's surface token(s) in `View`.
    pub utf16_span: Utf16Span,
    /// Unicode code-point span of the node's surface token(s) in `View`.
    pub code_point_span: CodePointSpan,
    /// Strongly-typed POS tag label from `A::Pos`, when provided.
    pub pos: Option<&'a str>,
}
impl<A: WordOrTokenAnchor> Copy for ResolvedDependencyNode<'_, A> {}
impl<A: WordOrTokenAnchor> Clone for ResolvedDependencyNode<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A bidirectional, offset-tracked dependency arc linking a resolved dependent
/// to its resolved head (or root) and incoming dependents (`children`).
#[derive(Debug, PartialEq, Eq)]
pub struct ResolvedDependencyArc<'a, A: WordOrTokenAnchor = Tokens> {
    pub dependent: ResolvedDependencyNode<'a, A>,
    pub head: Option<ResolvedDependencyNode<'a, A>>,
    pub children: &'a [Ix<A>],
    pub relation_tag: Tag,
    pub relation: &'a str,
}
impl<A: WordOrTokenAnchor> Copy for ResolvedDependencyArc<'_, A> {}
impl<A: WordOrTokenAnchor> Clone for ResolvedDependencyArc<'_, A> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArcDirection {
    ToHead,
    ToDependent,
}
/// A step along the undirected tree, with its dependency direction intact.
#[derive(Debug, PartialEq, Eq)]
pub struct PathStep<'a, A: WordOrTokenAnchor = Tokens> {
    pub from: Ix<A>,
    pub to: Ix<A>,
    pub direction: ArcDirection,
    pub relation: &'a str,
}

/// Outcome of a bounded shortest-path search in a [`DependencyGraph`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PathSearch {
    /// Whether `u` and `v` connected within the caller's arc bound.
    pub found: bool,
    /// Parent-arc lookups performed during the search and path build.
    pub arc_reads: usize,
}

/// An immutable validated forest, indexed in both directions. `new` requires a
/// single sentence tree; `from_layer` permits a document forest, including empty.
pub struct DependencyGraph<A: WordOrTokenAnchor = Tokens> {
    layer: Dependencies<A>,
    children: Vec<Vec<Ix<A>>>,
    roots: Vec<Ix<A>>,
    /// Depth and root of every node, built on the first path query:
    /// most graphs are never asked for a path (the parser builds one
    /// per sentence), so construction does not pay for the tables.
    tables: OnceLock<(Vec<u32>, Vec<u32>)>,
    pub(crate) label_encoding: crate::LabelEncoding,
}
impl<A: WordOrTokenAnchor> Clone for DependencyGraph<A> {
    fn clone(&self) -> Self {
        Self {
            layer: self.layer.clone(),
            children: self.children.clone(),
            roots: self.roots.clone(),
            tables: self.tables.get().map(|t| OnceLock::from(t.clone())).unwrap_or_default(),
            label_encoding: self.label_encoding,
        }
    }
}
impl<A: WordOrTokenAnchor> std::fmt::Debug for DependencyGraph<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.layer.fmt(f)
    }
}
impl<A: WordOrTokenAnchor> DependencyGraph<A> {
    pub fn new(heads: Vec<Option<Ix<A>>>, relations: Vec<String>) -> Result<Self> {
        if heads.len() != relations.len() {
            return Err(error("one relation per head required"));
        }
        let mut labels: Vec<&str> = Vec::new();
        let mut ids = std::collections::HashMap::new();
        let mut tags = Vec::with_capacity(relations.len());
        for label in &relations {
            if label.chars().all(turbo_unicode::props::white_space) {
                return Err(error("blank relation"));
            }
            let id = match ids.get(label.as_str()) {
                Some(&id) => id,
                None => {
                    let id = u16::try_from(labels.len()).map_err(|_| error("too many relation labels"))?;
                    labels.push(label);
                    ids.insert(label.as_str(), id);
                    id
                }
            };
            tags.push(Tag(id));
        }
        let graph = Self::from_layer(Dependencies {
            head: heads,
            relation: tags,
            relations: TagSet::new("dependency", &labels),
        })?;
        graph.check_tree()?;
        Ok(graph)
    }
    pub fn from_layer(layer: Dependencies<A>) -> Result<Self> {
        layer.check().map_err(error)?;
        layer.check_forest().map_err(error)?;
        let n = layer.head.len();
        if n > u32::MAX as usize {
            return Err(error("too many nodes"));
        }
        let mut children = vec![Vec::new(); n];
        let mut roots = Vec::new();
        for (i, head) in layer.head.iter().enumerate() {
            let node = Ix::new(i as u32);
            if let Some(h) = head {
                children[h.get()].push(node);
            } else {
                roots.push(node);
            }
        }
        Ok(Self { layer, children, roots, tables: OnceLock::new(), label_encoding: crate::LabelEncoding::Plain })
    }

    /// The depth and root tables, built once on first use.
    fn tables(&self) -> &(Vec<u32>, Vec<u32>) {
        let (roots, children) = (&self.roots, &self.children);
        self.tables.get_or_init(|| {
            let n = children.len();
            let mut depth = vec![0u32; n];
            let mut root_of = vec![0u32; n];
            let mut stack = Vec::new();
            for &root in roots {
                let root_id = root.get() as u32;
                root_of[root.get()] = root_id;
                stack.push(root);
                while let Some(u) = stack.pop() {
                    let next_depth = depth[u.get()] + 1;
                    for &child in &children[u.get()] {
                        depth[child.get()] = next_depth;
                        root_of[child.get()] = root_id;
                        stack.push(child);
                    }
                }
            }
            (depth, root_of)
        })
    }
    pub fn to_layer(&self) -> Dependencies<A> {
        self.layer.clone()
    }
    pub fn len(&self) -> usize {
        self.layer.head.len()
    }
    pub fn is_empty(&self) -> bool {
        self.layer.head.is_empty()
    }
    pub fn roots(&self) -> &[Ix<A>] {
        &self.roots
    }
    pub fn root(&self) -> Option<Ix<A>> {
        (self.roots.len() == 1).then(|| self.roots[0])
    }
    pub fn check_tree(&self) -> Result<()> {
        if self.roots.len() != 1 {
            return Err(Error(format!("expected one root, found {}", self.roots.len())));
        }
        Ok(())
    }
    pub fn arc(&self, node: Ix<A>) -> Result<DependencyArc<'_, A>> {
        let head = *self.layer.head.get(node.get()).ok_or_else(|| error("node out of range"))?;
        Ok(DependencyArc {
            dependent: node,
            head,
            relation: self.layer.relations.label(self.layer.relation[node.get()]),
        })
    }
    pub fn children(&self, node: Ix<A>) -> Result<&[Ix<A>]> {
        self.children.get(node.get()).map(Vec::as_slice).ok_or_else(|| error("node out of range"))
    }
    /// Returns the incoming dependency arcs for all dependents whose head is `head`.
    pub fn incoming_arcs(&self, head: Ix<A>) -> Result<Vec<DependencyArc<'_, A>>> {
        let deps = self.children(head)?;
        deps.iter().map(|&d| self.arc(d)).collect()
    }
    /// Returns all nodes in the syntactic subtree rooted at `root`, in surface order.
    pub fn subtree(&self, root: Ix<A>) -> Result<Vec<Ix<A>>> {
        self.children(root)?;
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(node) = stack.pop() {
            out.push(node);
            for &child in self.children[node.get()].iter().rev() {
                stack.push(child);
            }
        }
        out.sort_by_key(|ix| ix.get());
        Ok(out)
    }
    /// Computes the bounding UTF-8 byte [`Span`] in `view` covering the entire
    /// syntactic subtree rooted at `root`.
    pub fn subtree_span(&self, root: Ix<A>, view: &View, tokens: &Tokens, anchor: &A) -> Result<Span> {
        let words = anchor.resolved(view, tokens).map_err(error)?;
        if words.len() != self.len() {
            return Err(error("anchor length does not match dependency graph"));
        }
        let nodes = self.subtree(root)?;
        let mut start = u32::MAX;
        let mut end = 0u32;
        for node in nodes {
            let span = words[node.get()].token_span;
            start = start.min(span.start);
            end = end.max(span.end);
        }
        Ok(Span::new(start, end))
    }
    /// Resolves every node and arc against `view`, `tokens`, `anchor`, and
    /// optional `pos` into bidirectional, offset-tracked [`ResolvedDependencyArc`]s.
    pub fn resolve_arcs<'a>(
        &'a self,
        view: &'a View,
        tokens: &'a Tokens,
        anchor: &'a A,
        pos: Option<&'a A::Pos>,
    ) -> Result<Vec<ResolvedDependencyArc<'a, A>>> {
        let words = anchor.resolved(view, tokens).map_err(error)?;
        if words.len() != self.len() {
            return Err(error("anchor length does not match dependency graph"));
        }
        if let Some(pos) = pos {
            pos.check().map_err(error)?;
            if pos.len() != self.len() {
                return Err(error("POS length does not match dependency graph"));
            }
        }
        let nodes: Vec<ResolvedDependencyNode<'a, A>> = words
            .into_iter()
            .enumerate()
            .map(|(i, word)| {
                let pos_label = match pos {
                    Some(p) => Some(p.label(i).ok_or_else(|| error("invalid POS tag"))?),
                    None => None,
                };
                Ok(ResolvedDependencyNode {
                    index: Ix::new(i as u32),
                    utf16_span: view.span_to_utf16(word.token_span),
                    code_point_span: view.span_to_code_point(word.token_span),
                    word,
                    pos: pos_label,
                })
            })
            .collect::<Result<_>>()?;
        let mut out = Vec::with_capacity(self.len());
        for (i, &dependent) in nodes.iter().enumerate() {
            let head = self.layer.head[i].map(|h| nodes[h.get()]);
            let relation_tag = self.layer.relation[i];
            let relation = self.layer.relations.label(relation_tag);
            out.push(ResolvedDependencyArc { dependent, head, children: &self.children[i], relation_tag, relation });
        }
        Ok(out)
    }
    pub fn is_projective(&self) -> bool {
        // Include the artificial-root arc, as the arc-standard oracle does.
        for i in 0..self.len() {
            let h = self.layer.head[i].map_or(-1, |h| h.get() as i64);
            let (lo, hi) = (h.min(i as i64), h.max(i as i64));
            for (j, head) in self.layer.head.iter().enumerate().take(self.len()).skip(i + 1) {
                let h = head.map_or(-1, |h| h.get() as i64);
                let (a, b) = (h.min(j as i64), h.max(j as i64));
                if (lo < a && a < hi && hi < b) || (a < lo && lo < b && b < hi) {
                    return false;
                }
            }
        }
        true
    }
    fn read_head(&self, node: Ix<A>, arc_reads: &mut usize) -> Option<Ix<A>> {
        *arc_reads += 1;
        self.layer.head[node.get()]
    }

    fn lca_bounded(&self, u: Ix<A>, v: Ix<A>, max_len: Option<usize>, arc_reads: &mut usize) -> Result<Option<Ix<A>>> {
        if u.get() >= self.len() || v.get() >= self.len() {
            return Err(error("node out of range"));
        }
        let (depth, root_of) = self.tables();
        if root_of[u.get()] != root_of[v.get()] {
            return Ok(None);
        }
        let cap = max_len.unwrap_or(usize::MAX);
        let (mut a, mut b) = (u, v);
        let (da, db) = (depth[a.get()], depth[b.get()]);
        let mut steps = da.abs_diff(db) as usize;
        if steps > cap {
            return Ok(None);
        }
        for _ in 0..da.saturating_sub(db) {
            a = self.read_head(a, arc_reads).expect("non-root node has head");
        }
        for _ in 0..db.saturating_sub(da) {
            b = self.read_head(b, arc_reads).expect("non-root node has head");
        }
        while a != b {
            steps = steps.saturating_add(2);
            if steps > cap {
                return Ok(None);
            }
            a = self.read_head(a, arc_reads).expect("connected nodes share an ancestor");
            b = self.read_head(b, arc_reads).expect("connected nodes share an ancestor");
        }
        Ok(Some(a))
    }

    pub fn lowest_common_ancestor(&self, u: Ix<A>, v: Ix<A>) -> Result<Option<Ix<A>>> {
        self.lca_bounded(u, v, None, &mut 0)
    }

    /// Writes the shortest undirected path from `u` to `v` into `out`,
    /// clearing `out` first. `max_len` bounds the maximum arc count when set;
    /// distinct trees or paths longer than `max_len` return `found = false`.
    /// Invalid node indexes return an error.
    pub fn shortest_path_into<'a>(
        &'a self,
        u: Ix<A>,
        v: Ix<A>,
        max_len: Option<usize>,
        out: &mut Vec<PathStep<'a, A>>,
    ) -> Result<PathSearch> {
        out.clear();
        let mut arc_reads = 0;
        let Some(lca) = self.lca_bounded(u, v, max_len, &mut arc_reads)? else {
            return Ok(PathSearch { found: false, arc_reads });
        };
        let (depth, _) = self.tables();
        let up_len = (depth[u.get()] - depth[lca.get()]) as usize;
        let down_len = (depth[v.get()] - depth[lca.get()]) as usize;
        out.reserve(up_len + down_len);
        let mut n = u;
        while n != lca {
            let head = self.read_head(n, &mut arc_reads).expect("ancestor exists");
            let relation = self.layer.relations.label(self.layer.relation[n.get()]);
            out.push(PathStep { from: n, to: head, direction: ArcDirection::ToHead, relation });
            n = head;
        }
        let mut n = v;
        while n != lca {
            let head = self.read_head(n, &mut arc_reads).expect("ancestor exists");
            let relation = self.layer.relations.label(self.layer.relation[n.get()]);
            out.push(PathStep { from: head, to: n, direction: ArcDirection::ToDependent, relation });
            n = head;
        }
        out[up_len..].reverse();
        Ok(PathSearch { found: true, arc_reads })
    }

    /// `None` means distinct trees; invalid node indexes return an error.
    pub fn shortest_path(&self, u: Ix<A>, v: Ix<A>) -> Result<Option<Vec<PathStep<'_, A>>>> {
        let mut out = Vec::new();
        Ok(self.shortest_path_into(u, v, None, &mut out)?.found.then_some(out))
    }
}

impl<A: WordOrTokenAnchor> TryFrom<Dependencies<A>> for DependencyGraph<A> {
    type Error = Error;
    fn try_from(layer: Dependencies<A>) -> Result<Self> {
        Self::from_layer(layer)
    }
}
impl<A: WordOrTokenAnchor> From<DependencyGraph<A>> for Dependencies<A> {
    fn from(graph: DependencyGraph<A>) -> Self {
        graph.layer
    }
}
