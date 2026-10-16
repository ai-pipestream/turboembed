//! Administrative containment chains (`turbo:containment`), ported from
//! OpenNLP's `PlaceAncestor`, `ContainmentChain` and
//! `HierarchyAnnotator.CONTAINMENT` (`opennlp.tools.geo` and `opennlp.geo`).

use std::fmt;

use turbo_analysis::{Ix, Layer, Overlap, Span, Spans, ViewId};

use crate::Strs;

/// Why a [`PlaceAncestor`] or [`ContainmentChain`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ContainmentError {
    /// A required identifier, name, or place type was empty or whitespace.
    Blank(&'static str),
    /// A [`ContainmentChain`] was built with no ancestors.
    EmptyChain,
}

impl fmt::Display for ContainmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ContainmentError::Blank(what) => write!(f, "{what} must not be null or blank"),
            ContainmentError::EmptyChain => f.write_str("ancestors must not be empty"),
        }
    }
}

impl std::error::Error for ContainmentError {}

fn is_blank(s: &str) -> bool {
    s.chars().all(char::is_whitespace)
}

/// One administrative ancestor in a [`ContainmentChain`]
/// (`opennlp.tools.geo.PlaceAncestor`).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaceAncestor {
    id: String,
    name: String,
    kind: String,
}

impl PlaceAncestor {
    /// Creates a checked ancestor record; none of `id`, `name` or `kind` may
    /// be empty or Unicode whitespace only.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        kind: impl Into<String>,
    ) -> Result<PlaceAncestor, ContainmentError> {
        let (id, name, kind) = (id.into(), name.into(), kind.into());
        if is_blank(&id) {
            return Err(ContainmentError::Blank("id"));
        }
        if is_blank(&name) {
            return Err(ContainmentError::Blank("name"));
        }
        if is_blank(&kind) {
            return Err(ContainmentError::Blank("type"));
        }
        Ok(PlaceAncestor { id, name, kind })
    }

    /// The ancestor's identifier in the containment hierarchy.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The ancestor's canonical display name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The ancestor's place type (`neighbourhood`, `borough`, `locality`,
    /// `region`, `country`).
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Alias for [`PlaceAncestor::kind`], matching OpenNLP's `type()` accessor.
    pub fn place_type(&self) -> &str {
        &self.kind
    }
}

/// A borrowed view of one ancestor in a [`Containment`] layer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlaceAncestorRef<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub kind: &'a str,
}

impl<'a> PlaceAncestorRef<'a> {
    /// Converts this borrowed view into an owned [`PlaceAncestor`].
    pub fn to_owned(self) -> PlaceAncestor {
        PlaceAncestor { id: self.id.into(), name: self.name.into(), kind: self.kind.into() }
    }
}

/// An ordered, non-empty chain of administrative ancestors from immediate
/// parent outward (`opennlp.tools.geo.ContainmentChain`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContainmentChain {
    ancestors: Vec<PlaceAncestor>,
}

impl ContainmentChain {
    /// Creates a checked containment chain; `ancestors` must not be empty.
    pub fn new(ancestors: Vec<PlaceAncestor>) -> Result<ContainmentChain, ContainmentError> {
        if ancestors.is_empty() {
            return Err(ContainmentError::EmptyChain);
        }
        Ok(ContainmentChain { ancestors })
    }

    /// The ancestors in outward order (immediate parent first, outermost last).
    pub fn ancestors(&self) -> &[PlaceAncestor] {
        &self.ancestors
    }

    /// Consumes the chain and returns its ancestor vector.
    pub fn into_ancestors(self) -> Vec<PlaceAncestor> {
        self.ancestors
    }
}

/// Administrative containment chains over a text view (`turbo:containment`),
/// stored by column: one span per expanded location mention and flat string
/// columns for the ancestors in each mention's chain.
#[derive(Clone, Debug)]
pub struct Containment {
    pub spans: Spans<Containment>,
    pub ancestor_id: Strs,
    pub ancestor_name: Strs,
    pub ancestor_kind: Strs,
    /// One past the last ancestor index of mention `i` in the `ancestor_*`
    /// columns.
    pub chain_end: Vec<u32>,
}

impl Containment {
    /// An empty containment layer over `view`.
    pub fn new(view: ViewId) -> Containment {
        Containment {
            spans: Spans::new(view),
            ancestor_id: Strs::default(),
            ancestor_name: Strs::default(),
            ancestor_kind: Strs::default(),
            chain_end: Vec::new(),
        }
    }

    /// Appends one mention span and its [`ContainmentChain`].
    pub fn push(&mut self, span: Span, chain: &ContainmentChain) -> Ix<Containment> {
        for a in chain.ancestors() {
            self.ancestor_id.push(a.id());
            self.ancestor_name.push(a.name());
            self.ancestor_kind.push(a.kind());
        }
        self.chain_end.push(self.ancestor_id.len() as u32);
        self.spans.push(span)
    }

    /// Appends one mention span and a non-empty slice of [`PlaceAncestor`]s.
    pub fn push_ancestors(
        &mut self,
        span: Span,
        ancestors: &[PlaceAncestor],
    ) -> Result<Ix<Containment>, ContainmentError> {
        if ancestors.is_empty() {
            return Err(ContainmentError::EmptyChain);
        }
        for a in ancestors {
            self.ancestor_id.push(a.id());
            self.ancestor_name.push(a.name());
            self.ancestor_kind.push(a.kind());
        }
        self.chain_end.push(self.ancestor_id.len() as u32);
        Ok(self.spans.push(span))
    }

    fn range(&self, i: Ix<Containment>) -> std::ops::Range<usize> {
        let idx = i.get();
        let start = if idx == 0 { 0 } else { self.chain_end[idx - 1] as usize };
        let end = self.chain_end[idx] as usize;
        start..end
    }

    /// Iterates over the ancestors of mention `i` without allocating.
    pub fn ancestors(&self, i: Ix<Containment>) -> impl ExactSizeIterator<Item = PlaceAncestorRef<'_>> + '_ {
        self.range(i).map(|k| PlaceAncestorRef {
            id: self.ancestor_id.get(k),
            name: self.ancestor_name.get(k),
            kind: self.ancestor_kind.get(k),
        })
    }

    /// Materializes the [`ContainmentChain`] of mention `i`.
    pub fn chain(&self, i: Ix<Containment>) -> ContainmentChain {
        ContainmentChain { ancestors: self.ancestors(i).map(PlaceAncestorRef::to_owned).collect() }
    }
}

impl Layer for Containment {
    const ID: &'static str = "turbo:containment";
    const OVERLAP: Overlap = Overlap::Any;
    positional!(Containment);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.chain_end.len() != n {
            return Err(format!("{} spans and {} chain_end offsets", n, self.chain_end.len()));
        }
        let total = self.chain_end.last().copied().unwrap_or(0) as usize;
        if self.ancestor_id.len() != total || self.ancestor_name.len() != total || self.ancestor_kind.len() != total {
            return Err("ancestor_id, ancestor_name and ancestor_kind must match the last chain_end".into());
        }
        self.spans.check(Self::OVERLAP)?;
        let mut start = 0usize;
        for (i, span) in self.spans.iter() {
            let idx = i.get();
            if span.is_empty() {
                return Err(format!("containment span {idx} is empty at {}", span.start));
            }
            let end = self.chain_end[idx] as usize;
            if end <= start {
                return Err(format!("containment chain {idx} is empty"));
            }
            for k in start..end {
                let id = self.ancestor_id.get(k);
                let name = self.ancestor_name.get(k);
                let kind = self.ancestor_kind.get(k);
                if is_blank(id) || is_blank(name) || is_blank(kind) {
                    return Err(format!("containment chain {idx} has a blank ancestor field at {k}"));
                }
                for prev in start..k {
                    if self.ancestor_id.get(prev) == id {
                        return Err(format!("containment chain {idx} repeats ancestor id {id}"));
                    }
                }
            }
            start = end;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestors_and_chains_validate_and_round_trip() {
        assert!(PlaceAncestor::new("", "Brooklyn", "borough").is_err());
        assert!(PlaceAncestor::new("1", " ", "borough").is_err());
        assert!(PlaceAncestor::new("1", "Brooklyn", "\u{00A0}").is_err());
        assert!(ContainmentChain::new(Vec::new()).is_err());

        let a1 = PlaceAncestor::new("421205765", "Brooklyn", "borough").unwrap();
        let a2 = PlaceAncestor::new("85977539", "New York", "locality").unwrap();
        let chain = ContainmentChain::new(vec![a1.clone(), a2.clone()]).unwrap();

        let mut layer = Containment::new(ViewId::SOURCE);
        let ix = layer.push(Span::new(17, 27), &chain);
        assert_eq!(layer.check(), Ok(()));
        assert_eq!(layer.chain(ix), chain);
        let names: Vec<&str> = layer.ancestors(ix).map(|a| a.name).collect();
        assert_eq!(names, ["Brooklyn", "New York"]);
    }
}
