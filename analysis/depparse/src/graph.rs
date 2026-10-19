use crate::{Error, Result, error};
use turbo_analysis::{Ix, Layer, Tag, TagSet};
use turbo_analysis_layers::{Dependencies, Tokens, WordOrTokenAnchor};

/// One labeled edge. A root has no head index.
#[derive(Debug)]
pub struct DependencyArc<'a, A: WordOrTokenAnchor = Tokens> {
    pub dependent: Ix<A>,
    pub head: Option<Ix<A>>,
    pub relation: &'a str,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArcDirection {
    ToHead,
    ToDependent,
}
/// A step along the undirected tree, with its dependency direction intact.
#[derive(Debug)]
pub struct PathStep<'a, A: WordOrTokenAnchor = Tokens> {
    pub from: Ix<A>,
    pub to: Ix<A>,
    pub direction: ArcDirection,
    pub relation: &'a str,
}
/// An immutable validated forest, indexed in both directions. `new` requires a
/// single sentence tree; `from_layer` permits a document forest, including empty.
pub struct DependencyGraph<A: WordOrTokenAnchor = Tokens> {
    layer: Dependencies<A>,
    children: Vec<Vec<Ix<A>>>,
    roots: Vec<Ix<A>>,
}
impl<A: WordOrTokenAnchor> Clone for DependencyGraph<A> {
    fn clone(&self) -> Self {
        Self { layer: self.layer.clone(), children: self.children.clone(), roots: self.roots.clone() }
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
        if layer.head.len() > u32::MAX as usize {
            return Err(error("too many nodes"));
        }
        let mut children = vec![Vec::new(); layer.head.len()];
        let mut roots = Vec::new();
        for (i, head) in layer.head.iter().enumerate() {
            let node = Ix::new(i as u32);
            if let Some(h) = head {
                children[h.get()].push(node);
            } else {
                roots.push(node);
            }
        }
        Ok(Self { layer, children, roots })
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
    pub fn is_projective(&self) -> bool {
        // Include the artificial-root arc, as the arc-standard oracle does.
        for i in 0..self.len() {
            let h = self.layer.head[i].map_or(-1, |h| h.get() as i64);
            let (lo, hi) = (h.min(i as i64), h.max(i as i64));
            for j in i + 1..self.len() {
                let h = self.layer.head[j].map_or(-1, |h| h.get() as i64);
                let (a, b) = (h.min(j as i64), h.max(j as i64));
                if (lo < a && a < hi && hi < b) || (a < lo && lo < b && b < hi) {
                    return false;
                }
            }
        }
        true
    }
    fn ancestors(&self, node: Ix<A>) -> Result<Vec<Ix<A>>> {
        self.arc(node)?;
        let mut out = Vec::new();
        let mut at = Some(node);
        while let Some(n) = at {
            out.push(n);
            at = self.layer.head[n.get()];
        }
        Ok(out)
    }
    pub fn lowest_common_ancestor(&self, u: Ix<A>, v: Ix<A>) -> Result<Option<Ix<A>>> {
        let a = self.ancestors(u)?;
        let b = self.ancestors(v)?;
        let mut common = None;
        for (a, b) in a.iter().rev().zip(b.iter().rev()) {
            if a != b {
                break;
            }
            common = Some(*a);
        }
        Ok(common)
    }
    /// `None` means distinct trees; invalid node indexes return an error.
    pub fn shortest_path(&self, u: Ix<A>, v: Ix<A>) -> Result<Option<Vec<PathStep<'_, A>>>> {
        let Some(lca) = self.lowest_common_ancestor(u, v)? else { return Ok(None) };
        let mut out = Vec::new();
        let mut n = u;
        while n != lca {
            let arc = self.arc(n)?;
            let head = arc.head.expect("ancestor exists");
            out.push(PathStep { from: n, to: head, direction: ArcDirection::ToHead, relation: arc.relation });
            n = head;
        }
        let mut tail = Vec::new();
        let mut n = v;
        while n != lca {
            let arc = self.arc(n)?;
            let head = arc.head.expect("ancestor exists");
            tail.push(PathStep { from: head, to: n, direction: ArcDirection::ToDependent, relation: arc.relation });
            n = head;
        }
        out.extend(tail.into_iter().rev());
        Ok(Some(out))
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
