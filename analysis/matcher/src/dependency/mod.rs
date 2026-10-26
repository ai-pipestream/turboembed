//! Dependency patterns share token predicates and borrow the caller's token forest.
mod op;
use crate::input::{Columns, Needs};
use crate::nfa::{TokTests, compile_tests};
use crate::vocab::{Labels, Vocab};
use crate::{Input, Label, MatchError, PatternId, Tok};
pub use op::DependencyOp;
use std::fmt;

/// One node, in output order. Non-root nodes refer to an earlier node's index.
#[derive(Clone, Debug)]
pub struct DependencyNode {
    pub relation: Option<(usize, DependencyOp)>,
    pub token: Tok,
}

/// Pattern/schema errors and refused documents or bounded searches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependencyError {
    Pattern(String),
    Operator(String),
    Input(MatchError),
    Limit(&'static str),
}
impl fmt::Display for DependencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pattern(s) => write!(f, "invalid dependency pattern: {s}"),
            Self::Operator(s) => write!(f, "unsupported dependency operator: {s}"),
            Self::Input(e) => e.fmt(f),
            Self::Limit(s) => write!(f, "dependency matching limit exceeded: {s}"),
        }
    }
}
impl std::error::Error for DependencyError {}
impl From<MatchError> for DependencyError {
    fn from(value: MatchError) -> Self {
        Self::Input(value)
    }
}

/// Search limits. Exceeding either returns an error, never truncated matches.
#[derive(Clone, Copy, Debug)]
pub struct MatchLimits {
    /// Candidate predicate tests, relation tests, and ancestor links traversed.
    pub visits: usize,
    pub matches: usize,
}
impl Default for MatchLimits {
    fn default() -> Self {
        Self { visits: 10_000_000, matches: 100_000 }
    }
}

/// Token indices follow pattern node order, not source order. Each retains its
/// original `Input` span; no synthetic contiguous match span is constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyMatch {
    pub label: Label,
    pub pattern: PatternId,
    pub tokens: Vec<u32>,
}
struct Node {
    relation: Option<(usize, DependencyOp)>,
    token: TokTests,
}
struct Pattern {
    label: Label,
    nodes: Vec<Node>,
    needs: Needs,
}

/// A bounded spaCy-style DependencyMatcher using the existing token predicates.
#[derive(Default)]
pub struct DependencyMatcher {
    vocab: Vocab,
    labels: Labels,
    patterns: Vec<Pattern>,
}
impl DependencyMatcher {
    pub const MAX_NODES: usize = 64;
    pub const MAX_PATTERNS: usize = 4096;
    pub fn new() -> Self {
        Self::default()
    }
    /// Invalid patterns do not change labels, vocabulary, or existing patterns.
    pub fn add(&mut self, label: &str, nodes: &[DependencyNode]) -> Result<PatternId, DependencyError> {
        validate(nodes)?;
        if self.patterns.len() >= Self::MAX_PATTERNS {
            return Err(DependencyError::Limit("patterns"));
        }
        let mut needs = Needs::default();
        let compiled = nodes
            .iter()
            .map(|n| Node { relation: n.relation, token: compile_tests(&n.token, &mut self.vocab, &mut needs) })
            .collect();
        let id = PatternId(self.patterns.len() as u32);
        self.patterns.push(Pattern { label: self.labels.get_or_add(label), nodes: compiled, needs });
        Ok(id)
    }
    pub fn label_name(&self, label: Label) -> &str {
        self.labels.name(label)
    }
    pub fn label(&self, name: &str) -> Option<Label> {
        self.labels.get(name)
    }
    pub fn len(&self) -> usize {
        self.labels.len()
    }
    pub fn is_empty(&self) -> bool {
        self.labels.len() == 0
    }
    pub fn remove(&mut self, name: &str) -> Result<(), MatchError> {
        self.labels.remove(name).map(|_| ())
    }
    pub fn find(&self, input: &Input<'_>) -> Result<Vec<DependencyMatch>, DependencyError> {
        self.find_with_limits(input, MatchLimits::default())
    }
    pub fn find_with_limits(
        &self,
        input: &Input<'_>,
        limits: MatchLimits,
    ) -> Result<Vec<DependencyMatch>, DependencyError> {
        let dependencies = input.dependencies.ok_or(MatchError::Missing("opennlp:dependencies"))?;
        let graph = op::Graph::new(dependencies);
        let mut work = Work { remaining: limits.visits };
        let mut out = Vec::new();
        for (id, pattern) in self.patterns.iter().enumerate().filter(|(_, p)| self.labels.is_active(p.label)) {
            let columns = Columns::build(input, &(0..input.len()), pattern.needs, &self.vocab)?;
            let mut candidates = Vec::with_capacity(pattern.nodes.len());
            for node in &pattern.nodes {
                let mut tokens = Vec::new();
                for i in 0..input.len() {
                    work.visit()?;
                    if node.token.passes(&columns, input, i, i) {
                        tokens.push(i);
                    }
                }
                candidates.push(tokens);
            }
            let mut search = Search {
                pattern,
                id: PatternId(id as u32),
                candidates: &candidates,
                graph: &graph,
                work: &mut work,
                out: &mut out,
                limit: limits.matches,
            };
            search.walk(&mut Vec::with_capacity(pattern.nodes.len()))?;
        }
        Ok(out)
    }
}

pub(crate) fn validate(nodes: &[DependencyNode]) -> Result<(), DependencyError> {
    if nodes.is_empty() || nodes.len() > DependencyMatcher::MAX_NODES {
        return Err(DependencyError::Pattern("expected 1..=64 nodes".into()));
    }
    for (i, node) in nodes.iter().enumerate() {
        if node.token.min != 1 || node.token.max != Some(1) || node.token.negated {
            return Err(DependencyError::Pattern("node quantifiers are not supported".into()));
        }
        if (i == 0 && node.relation.is_some()) || (i > 0 && !node.relation.is_some_and(|(left, _)| left < i)) {
            return Err(DependencyError::Pattern("anchor first; every other node refers to an earlier node".into()));
        }
    }
    Ok(())
}
struct Work {
    remaining: usize,
}
impl Work {
    fn visit(&mut self) -> Result<(), DependencyError> {
        self.remaining = self.remaining.checked_sub(1).ok_or(DependencyError::Limit("visits"))?;
        Ok(())
    }
}
struct Search<'a, 'b> {
    pattern: &'a Pattern,
    id: PatternId,
    candidates: &'a [Vec<usize>],
    graph: &'a op::Graph<'b>,
    work: &'a mut Work,
    out: &'a mut Vec<DependencyMatch>,
    limit: usize,
}
impl Search<'_, '_> {
    fn walk(&mut self, selected: &mut Vec<u32>) -> Result<(), DependencyError> {
        let depth = selected.len();
        if depth == self.pattern.nodes.len() {
            if self.out.len() >= self.limit {
                return Err(DependencyError::Limit("matches"));
            }
            self.out.push(DependencyMatch { label: self.pattern.label, pattern: self.id, tokens: selected.clone() });
            return Ok(());
        }
        for &token in &self.candidates[depth] {
            self.work.visit()?;
            if let Some((left, op)) = self.pattern.nodes[depth].relation
                && !op.holds(selected[left] as usize, token, self.graph, self.work)?
            {
                continue;
            }
            selected.push(token as u32);
            self.walk(selected)?;
            selected.pop();
        }
        Ok(())
    }
}
