//! Dependency samples and trees: `opennlp.tools.depparse.DependencyArc`,
//! `DependencyGraph` and `DependencySample`.

use std::fmt;

use turbo_analysis::Span;
use turbo_unicode::WhitespaceMode;

use crate::error::FormatError;
use crate::source::Field;

/// Whether `s` is blank under OpenNLP's `StringUtil.isBlank`: empty or
/// every character is whitespace in [`WhitespaceMode::Unicode`].
pub(crate) fn is_blank(s: &str) -> bool {
    s.chars().all(|c| WhitespaceMode::Unicode.is_whitespace(c as u32))
}

/// One labeled edge of a [`DependencyGraph`]: the token at `dependent` is
/// governed by the token at `head` under `relation`
/// (`opennlp.tools.depparse.DependencyArc`).
///
/// Indices are zero-based positions in the sentence's token list. A `head`
/// of [`Self::ROOT_HEAD`] (`-1`) marks the dependent as the sentence root.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DependencyArc<'a> {
    pub head: i32,
    pub dependent: usize,
    pub relation: Field<'a>,
}

impl<'a> DependencyArc<'a> {
    /// The `head` value marking an arc from the artificial root node.
    pub const ROOT_HEAD: i32 = -1;

    /// Validates and builds one dependency arc.
    ///
    /// # Errors
    ///
    /// [`FormatError::InvalidDependencyGraph`] when `head < ROOT_HEAD`,
    /// `head == dependent`, or `relation` is blank.
    pub fn new(head: i32, dependent: usize, relation: Field<'a>) -> Result<DependencyArc<'a>, FormatError> {
        if head < Self::ROOT_HEAD {
            return Err(FormatError::InvalidDependencyGraph {
                detail: format!("head must be a token index or ROOT_HEAD: {head}"),
            });
        }
        if i64::from(head) == dependent as i64 {
            return Err(FormatError::InvalidDependencyGraph { detail: format!("arc must not be a self-loop: {head}") });
        }
        if is_blank(relation.as_str()) {
            return Err(FormatError::InvalidDependencyGraph { detail: "relation must not be null or blank".into() });
        }
        Ok(DependencyArc { head, dependent, relation })
    }
}

/// An immutable dependency tree over one sentence: for every token, the
/// zero-based index of its head (or [`DependencyArc::ROOT_HEAD`] for the
/// single root) and the label of the relation to that head
/// (`opennlp.tools.depparse.DependencyGraph`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyGraph<'a> {
    heads: Vec<i32>,
    relations: Vec<Field<'a>>,
}

impl<'a> DependencyGraph<'a> {
    /// Builds a validated [`DependencyGraph`] from parallel `heads` and
    /// `relations` slices (`DependencyGraph.of`).
    ///
    /// # Errors
    ///
    /// [`FormatError::InvalidDependencyGraph`] when `heads` is empty,
    /// `heads` and `relations` differ in length, a head is out of range or
    /// a self-loop, a relation is blank, there is not exactly one root, or
    /// the head chains contain a cycle.
    pub fn of(heads: Vec<i32>, relations: Vec<Field<'a>>) -> Result<DependencyGraph<'a>, FormatError> {
        if heads.is_empty() {
            return Err(FormatError::InvalidDependencyGraph {
                detail: "a dependency graph needs at least one token".into(),
            });
        }
        if heads.len() != relations.len() {
            return Err(FormatError::InvalidDependencyGraph {
                detail: format!(
                    "heads and relations must have the same length: {} != {}",
                    heads.len(),
                    relations.len()
                ),
            });
        }
        let n = heads.len();
        let mut roots = 0usize;
        for i in 0..n {
            let h = heads[i];
            if h == DependencyArc::ROOT_HEAD {
                roots += 1;
            } else if h < 0 || (h as usize) >= n {
                return Err(FormatError::InvalidDependencyGraph {
                    detail: format!("head of token {i} is out of range: {h}"),
                });
            } else if h as usize == i {
                return Err(FormatError::InvalidDependencyGraph { detail: format!("token {i} must not head itself") });
            }
            if is_blank(relations[i].as_str()) {
                return Err(FormatError::InvalidDependencyGraph {
                    detail: format!("relation of token {i} must not be blank"),
                });
            }
        }
        if roots != 1 {
            return Err(FormatError::InvalidDependencyGraph {
                detail: format!("expected exactly one root, found {roots}"),
            });
        }
        check_acyclic(&heads)?;
        Ok(DependencyGraph { heads, relations })
    }

    /// The number of tokens the graph spans (`DependencyGraph.size`).
    pub fn len(&self) -> usize {
        self.heads.len()
    }

    /// Whether the graph has no tokens (always `false` for a valid graph).
    pub fn is_empty(&self) -> bool {
        self.heads.is_empty()
    }

    /// The zero-based head index per token, or [`DependencyArc::ROOT_HEAD`]
    /// for the root.
    pub fn heads(&self) -> &[i32] {
        &self.heads
    }

    /// The relation label per token.
    pub fn relations(&self) -> &[Field<'a>] {
        &self.relations
    }

    /// The head of the token at `index` (`DependencyGraph.headOf`).
    pub fn head_of(&self, index: usize) -> Option<i32> {
        self.heads.get(index).copied()
    }

    /// The relation label of the token at `index`
    /// (`DependencyGraph.relationOf`).
    pub fn relation_of(&self, index: usize) -> Option<&Field<'a>> {
        self.relations.get(index)
    }

    /// The zero-based index of the sentence root token
    /// (`DependencyGraph.root`).
    pub fn root(&self) -> usize {
        self.heads
            .iter()
            .position(|&h| h == DependencyArc::ROOT_HEAD)
            .expect("graph invariant violated: no root present")
    }

    /// All arcs of the graph in token order, one per token
    /// (`DependencyGraph.arcs`).
    pub fn arcs(&self) -> Vec<DependencyArc<'a>> {
        self.heads
            .iter()
            .zip(&self.relations)
            .enumerate()
            .map(|(i, (&head, relation))| DependencyArc { head, dependent: i, relation: relation.clone() })
            .collect()
    }
}

fn check_acyclic(heads: &[i32]) -> Result<(), FormatError> {
    const UNVISITED: u8 = 0;
    const VISITING: u8 = 1;
    const VISITED: u8 = 2;

    let mut states = vec![UNVISITED; heads.len()];
    for start in 0..heads.len() {
        let mut current = start as i32;
        while current != DependencyArc::ROOT_HEAD && states[current as usize] == UNVISITED {
            states[current as usize] = VISITING;
            current = heads[current as usize];
        }
        if current != DependencyArc::ROOT_HEAD && states[current as usize] == VISITING {
            return Err(FormatError::InvalidDependencyGraph {
                detail: format!("dependency graph contains a cycle at token {current}"),
            });
        }
        let mut current = start as i32;
        while current != DependencyArc::ROOT_HEAD && states[current as usize] == VISITING {
            states[current as usize] = VISITED;
            current = heads[current as usize];
        }
    }
    Ok(())
}

/// `DependencyGraph.toString`: `0<-1:det 1<-2:nsubj 2<--1:root`.
impl fmt::Display for DependencyGraph<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, (&head, relation)) in self.heads.iter().zip(&self.relations).enumerate() {
            if i > 0 {
                f.write_str(" ")?;
            }
            write!(f, "{i}<-{head}:{}", relation.text)?;
        }
        Ok(())
    }
}

/// One dependency-annotated sentence: tokens, their part-of-speech tags,
/// and the gold [`DependencyGraph`] over them
/// (`opennlp.tools.depparse.DependencySample`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencySample<'a> {
    tokens: Vec<Field<'a>>,
    tags: Vec<Field<'a>>,
    graph: DependencyGraph<'a>,
    source: Span,
}

impl<'a> DependencySample<'a> {
    /// A sample of `tokens`, `tags` and `graph`, read from `source`.
    ///
    /// # Errors
    ///
    /// [`FormatError::InvalidDependencyGraph`] when `tokens` is empty, or
    /// [`FormatError::Lengths`] when `tokens`, `tags` and `graph` do not
    /// have the same length.
    pub fn new(
        tokens: Vec<Field<'a>>,
        tags: Vec<Field<'a>>,
        graph: DependencyGraph<'a>,
        source: Span,
    ) -> Result<DependencySample<'a>, FormatError> {
        if tokens.is_empty() {
            return Err(FormatError::InvalidDependencyGraph { detail: "tokens must not be empty".into() });
        }
        if tokens.len() != tags.len() {
            return Err(FormatError::Lengths { what: "tags", expected: tokens.len(), found: tags.len() });
        }
        if tokens.len() != graph.len() {
            return Err(FormatError::Lengths { what: "graph", expected: tokens.len(), found: graph.len() });
        }
        Ok(DependencySample { tokens, tags, graph, source })
    }

    pub fn tokens(&self) -> &[Field<'a>] {
        &self.tokens
    }

    pub fn tags(&self) -> &[Field<'a>] {
        &self.tags
    }

    pub fn graph(&self) -> &DependencyGraph<'a> {
        &self.graph
    }

    /// The bytes of the source the sample was read from.
    pub fn source(&self) -> Span {
        self.source
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// `DependencySample.toString`: one 1-based
/// `id<TAB>token<TAB>tag<TAB>head<TAB>relation` line per token, ending
/// with a line feed.
impl fmt::Display for DependencySample<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.tokens.len() {
            writeln!(
                f,
                "{}\t{}\t{}\t{}\t{}",
                i + 1,
                self.tokens[i].text,
                self.tags[i].text,
                self.graph.heads[i] + 1,
                self.graph.relations[i].text,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(s: &str) -> Field<'_> {
        Field::of(s, Span::new(0, s.len() as u32))
    }

    #[test]
    fn graph_validates_tree_invariants() {
        let g = DependencyGraph::of(vec![1, -1, 3, 1], vec![field("nsubj"), field("root"), field("det"), field("obj")])
            .unwrap();
        assert_eq!(g.len(), 4);
        assert_eq!(g.root(), 1);
        assert_eq!(g.head_of(0), Some(1));
        assert_eq!(g.relation_of(3).unwrap().as_str(), "obj");
        assert_eq!(g.to_string(), "0<-1:nsubj 1<--1:root 2<-3:det 3<-1:obj");
        assert_eq!(g.arcs().len(), 4);

        // Empty graph
        assert!(DependencyGraph::of(Vec::new(), Vec::new()).is_err());
        // Mismatched lengths
        assert!(DependencyGraph::of(vec![-1], vec![field("root"), field("dep")]).is_err());
        // No root
        assert!(DependencyGraph::of(vec![1, 0], vec![field("dep"), field("dep")]).is_err());
        // Multiple roots
        assert!(DependencyGraph::of(vec![-1, -1], vec![field("root"), field("root")]).is_err());
        // Self-loop
        assert!(DependencyGraph::of(vec![-1, 1], vec![field("root"), field("dep")]).is_err());
        // Out-of-range head
        assert!(DependencyGraph::of(vec![-1, 5], vec![field("root"), field("dep")]).is_err());
        // Blank relation (including U+00A0)
        assert!(DependencyGraph::of(vec![-1], vec![field("\u{a0}")]).is_err());
        // Disconnected cycle alongside a single root
        assert!(DependencyGraph::of(vec![-1, 2, 1], vec![field("root"), field("dep"), field("dep")]).is_err());
    }

    #[test]
    fn sample_formats_one_based_tsv() {
        let g = DependencyGraph::of(vec![1, -1], vec![field("nsubj"), field("root")]).unwrap();
        let s = DependencySample::new(
            vec![field("Dogs"), field("bark")],
            vec![field("NNS"), field("VBP")],
            g,
            Span::new(0, 10),
        )
        .unwrap();
        assert_eq!(s.to_string(), "1\tDogs\tNNS\t2\tnsubj\n2\tbark\tVBP\t0\troot\n");
    }
}
