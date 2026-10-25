//! Sparse inverted posting index with MaxScore pruning and the SPLATE
//! token-to-vocabulary sparse adapter (arXiv 2404.13950).

use std::collections::HashMap;

use turbo::sparse::{MaxLogPool, SparseOptions as SparseCut, SparseTerm};

use turbo_analysis::TopK;

use crate::Result;

/// Builder for [`SparsePostingIndex`], numbering documents sequentially
/// from `0` in insertion order.
#[derive(Default)]
pub struct SparsePostingIndexBuilder {
    postings: HashMap<u32, Vec<(u32, f32)>>,
    docs: u32,
}

impl SparsePostingIndexBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one document's sparse terms (sorted or unsorted, positive
    /// finite weights) as the next document row and returns its `doc_id`.
    pub fn add(&mut self, terms: &[SparseTerm]) -> Result<u32> {
        let doc_id = self.docs;
        self.docs = self.docs.checked_add(1).ok_or_else(|| "more documents than u32 numbers".to_owned())?;
        let mut by_term: Vec<(u32, f32)> = Vec::with_capacity(terms.len());
        for t in terms {
            if !t.weight.is_finite() || t.weight < 0.0 {
                return Err(format!("document {doc_id} term {} has invalid weight {}", t.term, t.weight));
            }
            if t.weight == 0.0 {
                continue;
            }
            by_term.push((t.term, t.weight));
        }
        by_term.sort_unstable_by_key(|&(id, _)| id);
        by_term.dedup_by(|a, b| {
            if a.0 == b.0 {
                if a.1 > b.1 {
                    b.1 = a.1;
                }
                true
            } else {
                false
            }
        });
        for (term_id, weight) in by_term {
            self.postings.entry(term_id).or_default().push((doc_id, weight));
        }
        Ok(doc_id)
    }

    pub fn build(self) -> SparsePostingIndex {
        let mut terms: Vec<(u32, SparseList)> = self
            .postings
            .into_iter()
            .map(|(term_id, rows)| {
                let max_weight = rows.iter().fold(0.0f32, |m, &(_, w)| m.max(w));
                (term_id, SparseList { rows, max_weight })
            })
            .collect();
        terms.sort_unstable_by_key(|&(id, _)| id);
        SparsePostingIndex { terms, docs: self.docs as usize }
    }
}

#[derive(Clone, Debug)]
struct SparseList {
    rows: Vec<(u32, f32)>,
    max_weight: f32,
}

/// Immutable inverted index over sparse vocabulary activations (`SparseTerm`).
#[derive(Clone, Debug)]
pub struct SparsePostingIndex {
    terms: Vec<(u32, SparseList)>,
    docs: usize,
}

impl SparsePostingIndex {
    /// Total documents indexed.
    pub fn docs(&self) -> usize {
        self.docs
    }

    /// Number of active vocabulary terms with non-empty posting lists.
    pub fn active_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total `(doc_id, weight)` posting entries across all vocabulary terms.
    pub fn total_postings(&self) -> usize {
        self.terms.iter().map(|(_, l)| l.rows.len()).sum()
    }

    /// Memory footprint of the index structures in bytes.
    pub fn byte_size(&self) -> usize {
        let header_bytes = self.terms.len() * std::mem::size_of::<(u32, SparseList)>();
        let posting_bytes = self.total_postings() * std::mem::size_of::<(u32, f32)>();
        std::mem::size_of::<Self>() + header_bytes + posting_bytes
    }

    fn list(&self, term_id: u32) -> Option<&SparseList> {
        self.terms.binary_search_by_key(&term_id, |&(id, _)| id).ok().map(|i| &self.terms[i].1)
    }

    /// Retrieves the top `k` documents by exact sparse dot product using
    /// MaxScore upper-bound pruning. Ties break by lower `doc_id`.
    pub fn top_k(&self, query: &[SparseTerm], k: usize) -> Vec<(u32, f64)> {
        if k == 0 || query.is_empty() || self.docs == 0 {
            return Vec::new();
        }
        let mut cursors: Vec<SparseCursor<'_>> = query
            .iter()
            .filter(|t| t.weight.is_finite() && t.weight > 0.0)
            .filter_map(|t| {
                let list = self.list(t.term)?;
                let q_w = f64::from(t.weight);
                let bound = q_w * f64::from(list.max_weight);
                Some(SparseCursor { rows: &list.rows, at: 0, q_w, bound })
            })
            .collect();
        if cursors.is_empty() {
            return Vec::new();
        }
        let mut by_bound: Vec<usize> = (0..cursors.len()).collect();
        by_bound.sort_by(|&a, &b| cursors[a].bound.total_cmp(&cursors[b].bound));
        let mut cumulative = Vec::with_capacity(by_bound.len());
        let mut sum = 0.0f64;
        for &c in &by_bound {
            sum += cursors[c].bound;
            cumulative.push(sum);
        }

        let mut top = TopK::new(k);
        let mut essential = 0usize;
        loop {
            if let Some(threshold) = top.threshold() {
                while essential < by_bound.len() && cumulative[essential] * (1.0 + 1e-9) <= threshold {
                    essential += 1;
                }
            }
            let Some(doc_id) = by_bound[essential..].iter().filter_map(|&c| cursors[c].current()).min() else {
                break;
            };
            let mut score = 0.0f64;
            for c in &mut cursors {
                if let Some(w) = c.seek(doc_id) {
                    score += c.q_w * f64::from(w);
                }
            }
            top.offer(doc_id, score);
            for &c in &by_bound[essential..] {
                if cursors[c].current() == Some(doc_id) {
                    cursors[c].at += 1;
                }
            }
        }
        top.into_sorted()
    }

    /// Unpruned reference evaluation of the sparse dot product for testing.
    pub fn exhaustive_top_k(&self, query: &[SparseTerm], k: usize) -> Vec<(u32, f64)> {
        if k == 0 || query.is_empty() || self.docs == 0 {
            return Vec::new();
        }
        let mut scores = vec![0.0f64; self.docs];
        let mut touched = vec![false; self.docs];
        for t in query {
            if !t.weight.is_finite() || t.weight <= 0.0 {
                continue;
            }
            if let Some(list) = self.list(t.term) {
                let q_w = f64::from(t.weight);
                for &(doc_id, w) in &list.rows {
                    let d = doc_id as usize;
                    scores[d] += q_w * f64::from(w);
                    touched[d] = true;
                }
            }
        }
        let mut top = TopK::new(k);
        for (d, (&s, &hit)) in scores.iter().zip(&touched).enumerate() {
            if hit && s > 0.0 {
                top.offer(d as u32, s);
            }
        }
        top.into_sorted()
    }
}

struct SparseCursor<'a> {
    rows: &'a [(u32, f32)],
    at: usize,
    q_w: f64,
    bound: f64,
}

impl SparseCursor<'_> {
    fn current(&self) -> Option<u32> {
        self.rows.get(self.at).map(|&(d, _)| d)
    }

    fn seek(&mut self, doc_id: u32) -> Option<f32> {
        self.at += self.rows[self.at..].partition_point(|&(d, _)| d < doc_id);
        self.rows.get(self.at).filter(|&&(d, _)| d == doc_id).map(|&(_, w)| w)
    }
}

/// SPLATE linear adapter (arXiv 2404.13950) mapping $d$-dimensional
/// late-interaction token vectors onto a sparse vocabulary of size $|V|$
/// via SPLADE max-log pooling (`log(1 + relu(W t + b))`).
#[derive(Clone, Debug)]
pub struct SplateAdapter {
    weights: Vec<f32>,
    bias: Vec<f32>,
    vocab: usize,
    dim: usize,
}

impl SplateAdapter {
    /// Constructs a SPLATE adapter from a `[vocab, dim]` projection basis
    /// and a uniform bias shift (typically negative, e.g. `-0.35`, to
    /// enforce sparsity before ReLU).
    pub fn new(weights: Vec<f32>, bias: Vec<f32>, vocab: usize, dim: usize) -> Result<Self> {
        if vocab == 0 || dim == 0 {
            return Err("SPLATE adapter vocab and dim must be positive".into());
        }
        let expected = vocab
            .checked_mul(dim)
            .ok_or_else(|| format!("SPLATE adapter vocab ({vocab}) * dim ({dim}) overflows usize"))?;
        if weights.len() != expected {
            return Err(format!("SPLATE adapter weights length {} != vocab * dim ({expected})", weights.len()));
        }
        if bias.len() != vocab {
            return Err(format!("SPLATE adapter bias length {} != vocab ({vocab})", bias.len()));
        }
        if weights.iter().any(|x| !x.is_finite()) || bias.iter().any(|x| !x.is_finite()) {
            return Err("SPLATE adapter holds non-finite parameters".into());
        }
        Ok(Self { weights, bias, vocab, dim })
    }

    /// Constructs a SPLATE adapter from a vocabulary embedding table
    /// `[vocab, dim]` with a uniform ReLU threshold bias `bias_shift`
    /// and a positive gain `scale`.
    pub fn from_basis(basis: &[f32], vocab: usize, dim: usize, scale: f32, bias_shift: f32) -> Result<Self> {
        if vocab == 0 || dim == 0 {
            return Err("SPLATE adapter vocab and dim must be positive".into());
        }
        let expected = vocab
            .checked_mul(dim)
            .ok_or_else(|| format!("SPLATE adapter vocab ({vocab}) * dim ({dim}) overflows usize"))?;
        if basis.len() != expected {
            return Err(format!("SPLATE basis length {} != vocab * dim ({expected})", basis.len()));
        }
        let weights: Vec<f32> = basis.iter().map(|&w| w * scale).collect();
        let bias = vec![bias_shift; vocab];
        Self::new(weights, bias, vocab, dim)
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Byte size of the adapter parameters (`weights` + `bias`).
    pub fn byte_size(&self) -> usize {
        (self.weights.len() + self.bias.len()) * std::mem::size_of::<f32>()
    }

    /// Projects a sequence of token vectors (`[n_tokens, dim]`) into
    /// sparse vocabulary terms using `MaxLogPool` and cuts by `cut`.
    pub fn encode_tokens(&self, tokens: &[f32], cut: &SparseCut) -> Result<Vec<SparseTerm>> {
        if !tokens.len().is_multiple_of(self.dim) {
            return Err(format!("token buffer length {} is not a multiple of dim {}", tokens.len(), self.dim));
        }
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let mut pool = MaxLogPool::new(self.vocab);
        let mut logits = vec![0.0f32; self.vocab];
        for tok in tokens.chunks_exact(self.dim) {
            for (v, (row, &b)) in self.weights.chunks_exact(self.dim).zip(&self.bias).enumerate() {
                let mut dot = b;
                for (&w, &x) in row.iter().zip(tok) {
                    dot += w * x;
                }
                logits[v] = dot;
            }
            pool.observe(&logits).map_err(|e| e.to_string())?;
        }
        pool.terms(cut).map_err(|e| e.to_string())
    }
}
