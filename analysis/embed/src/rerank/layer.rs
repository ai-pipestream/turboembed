use std::collections::BTreeSet;
use turbo_analysis::{Anchor, Ix, Key, KeyId, Layer, LayerName, ModelRef};
use turbo_vector_index::Hit;

/// A first-stage shortlist, ordered by retrieval rank. Each index points
/// into the explicitly named text layer. Queries may differ per document.
pub struct Shortlist<L: Layer> {
    pub query: String,
    pub source: Key<L>,
    pub items: Vec<Ix<L>>,
}
impl<L: Layer> Shortlist<L> {
    /// Map vector-index hits to document-local items through the metadata
    /// map used when the index was built. Only the first `k` hits are kept.
    pub fn from_hits(query: String, source: Key<L>, hits: &[Hit], k: usize, rows: &[Ix<L>]) -> Result<Self, String> {
        let items = hits
            .iter()
            .take(k)
            .map(|h| {
                if !h.score.is_finite() {
                    return Err("non-finite shortlist retrieval score".into());
                }
                rows.get(h.row as usize).copied().ok_or_else(|| "shortlist hit has no metadata row".into())
            })
            .collect::<Result<Vec<_>, String>>()?;
        let result = Self { query, source, items };
        result.check()?;
        Ok(result)
    }
}
impl<L: Layer> Layer for Shortlist<L> {
    const ID: &'static str = "turbo:rerank-shortlist";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.items.len()
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(self.source.id(), self.items.iter().map(|i| i.get() + 1).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        if self.items.iter().map(|i| i.get()).collect::<BTreeSet<_>>().len() != self.items.len() {
            return Err("shortlist contains duplicate items".into());
        }
        Ok(())
    }
}

/// Relevance order, retaining source identities and the scoring model.
pub struct Reranked<L: Layer> {
    pub source: Key<L>,
    pub items: Vec<Ix<L>>,
    pub scores: Vec<f32>,
    pub model: ModelRef,
}
impl<L: Layer> Layer for Reranked<L> {
    const ID: &'static str = "turbo:reranked";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.items.len()
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(self.source.id(), self.items.iter().map(|i| i.get() + 1).max().unwrap_or(0))]
    }
    fn check(&self) -> Result<(), String> {
        if self.scores.len() != self.items.len() || self.scores.iter().any(|s| !s.is_finite()) {
            return Err("one finite relevance score per item".into());
        }
        if self.scores.windows(2).any(|w| w[0] < w[1]) {
            return Err("reranked scores must be descending".into());
        }
        if self.items.iter().map(|i| i.get()).collect::<BTreeSet<_>>().len() != self.items.len() {
            return Err("reranked items must be unique".into());
        }
        Ok(())
    }
}
