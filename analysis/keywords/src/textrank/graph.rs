//! The co-occurrence graph and its PageRank (Mihalcea and Tarau, "TextRank:
//! Bringing order into texts", EMNLP 2004, section 2): vertices are the
//! content terms, an edge carries how often its ends co-occurred within
//! the window, and the score iterates `PR(v) = (1 - d) + d * sum_u
//! w(u, v) / out(u) * PR(u)` to convergence.

use std::collections::HashMap;

/// The iteration cap, a safety the paper needs no more than a few dozen
/// iterations to stay clear of on real documents.
const MAX_ITER: usize = 100;

/// An undirected weighted graph over folded terms, stored as adjacency
/// lists so PageRank walks edges, not the vertex set.
pub(crate) struct Graph {
    ids: HashMap<String, u32>,
    names: Vec<String>,
    adj: Vec<HashMap<u32, f64>>,
}

impl Graph {
    pub(crate) fn new() -> Graph {
        Graph { ids: HashMap::new(), names: Vec::new(), adj: Vec::new() }
    }

    pub(crate) fn intern(&mut self, folded: &str) -> u32 {
        if let Some(&id) = self.ids.get(folded) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(folded.to_owned(), id);
        self.names.push(folded.to_owned());
        self.adj.push(HashMap::new());
        id
    }

    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn name(&self, id: u32) -> &str {
        &self.names[id as usize]
    }

    /// One co-occurrence of two distinct vertices: both directions gain
    /// a unit of weight. The same word type meeting itself inside a
    /// window ("cats like cats") is not an edge, matching the reference
    /// implementations, which add no self-loops.
    pub(crate) fn cooccur(&mut self, a: u32, b: u32) {
        if a == b {
            return;
        }
        *self.adj[a as usize].entry(b).or_default() += 1.0;
        *self.adj[b as usize].entry(a).or_default() += 1.0;
    }

    /// The power-iteration score of every vertex, `damping` the d of the
    /// paper, `epsilon` the convergence threshold on the L1 delta between
    /// two iterations (1e-4 there). Every vertex starts at 1.0, so a
    /// vertex with no edge converges to `1 - d`; the result is not
    /// normalized, the caller does that.
    pub(crate) fn pagerank(&self, damping: f64, epsilon: f64) -> Vec<f64> {
        let n = self.names.len();
        let mut pr = vec![1.0f64; n];
        for _ in 0..MAX_ITER {
            let mut next = vec![1.0 - damping; n];
            for (u, edges) in self.adj.iter().enumerate() {
                let out: f64 = edges.values().sum();
                if out <= 0.0 {
                    continue;
                }
                for (&v, &w) in edges {
                    next[v as usize] += damping * w / out * pr[u];
                }
            }
            let delta: f64 = next.iter().zip(&pr).map(|(a, b)| (a - b).abs()).sum();
            pr = next;
            if delta < epsilon {
                break;
            }
        }
        pr
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagerank_converges_and_weights_by_cooccurrence() {
        // A star: the hub co-occurs with every leaf, the leaves only
        // with the hub, so the hub dominates and the leaves tie.
        let mut g = Graph::new();
        let hub = g.intern("hub");
        let leaves: Vec<u32> = ["a", "b", "c", "d"].into_iter().map(|l| g.intern(l)).collect();
        for &leaf in &leaves {
            g.cooccur(hub, leaf);
        }
        let pr = g.pagerank(0.85, 1e-4);
        let hub_score = pr[hub as usize];
        assert!(leaves.iter().all(|&l| pr[l as usize] < hub_score));
        assert!(leaves.windows(2).all(|w| (pr[w[0] as usize] - pr[w[1] as usize]).abs() < 1e-12));
        // The d of the paper keeps scores positive.
        assert!(pr.iter().all(|&s| s > 0.0));
        // An isolated vertex still converges to 1 - d.
        let mut lone = Graph::new();
        lone.intern("solo");
        let pr = lone.pagerank(0.85, 1e-4);
        assert!((pr[0] - 0.15).abs() < 1e-9);
    }

    #[test]
    fn epsilon_bounds_the_delta() {
        // A loose epsilon stops early: the result still holds together
        // (no NaN, every score positive) on a connected graph.
        let mut g = Graph::new();
        let a = g.intern("a");
        let b = g.intern("b");
        g.cooccur(a, b);
        let pr = g.pagerank(0.85, 100.0);
        assert_eq!(pr.len(), 2);
        assert!(pr.iter().all(|s| s.is_finite() && *s > 0.0));
    }
}
