//! The TextRank extractor (Mihalcea and Tarau, "TextRank: Bringing
//! order into texts", EMNLP 2004): a co-occurrence graph over the
//! content terms (nouns, proper nouns and adjectives), PageRank as the
//! vertex score, and adjacent top-scoring tokens collapsed into
//! multi-word keyphrases.

pub(crate) mod graph;

use std::collections::HashMap;

use turbo_analysis::{Annotator, Error, Key, Layer, Span};
use turbo_analysis_layers::{PosTags, Sentences, Terms, Tokens};

use crate::ranked::RankedKeyword;

use graph::Graph;

/// The content tags: TextRank keeps tokens the POS filter passes.
const CONTENT_LABELS: [&str; 3] = ["NOUN", "PROPN", "ADJ"];

/// Extracts TextRank keyphrases from one document's layers.
#[derive(Clone, Debug)]
pub struct TextRankExtractor {
    window: usize,
    top_n: usize,
    damping: f64,
    epsilon: f64,
}

impl Default for TextRankExtractor {
    fn default() -> TextRankExtractor {
        // The paper's window, damping and convergence epsilon, with the
        // same top-N default YAKE carries.
        TextRankExtractor { window: 2, top_n: 30, damping: 0.85, epsilon: 1e-4 }
    }
}

impl TextRankExtractor {
    pub fn new() -> TextRankExtractor {
        TextRankExtractor::default()
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn top_n(&self) -> usize {
        self.top_n
    }

    pub fn damping(&self) -> f64 {
        self.damping
    }

    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    pub fn with_window(mut self, window: usize) -> TextRankExtractor {
        self.window = window;
        self
    }

    pub fn with_top_n(mut self, top_n: usize) -> TextRankExtractor {
        self.top_n = top_n;
        self
    }

    /// The d of the paper: the restart probability is `1 - damping`.
    pub fn with_damping(mut self, damping: f64) -> TextRankExtractor {
        self.damping = damping;
        self
    }

    /// The convergence threshold on the L1 delta between iterations.
    pub fn with_epsilon(mut self, epsilon: f64) -> TextRankExtractor {
        self.epsilon = epsilon;
        self
    }

    /// Every ranked keyphrase of the document, best first. Documents
    /// with no content tokens yield an empty ranking, never an error.
    pub fn extract(
        &self,
        sentences: &Sentences,
        tokens: &Tokens,
        pos_tags: &PosTags,
        terms: &Terms,
    ) -> Result<Vec<RankedKeyword>, Error> {
        if self.window == 0 {
            return Err(config("window must be at least 1"));
        }
        if self.top_n == 0 {
            return Err(config("top_n must be at least 1"));
        }
        if !(self.damping > 0.0 && self.damping < 1.0) {
            return Err(config("damping must be in (0, 1)"));
        }
        if self.epsilon.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(config("epsilon must be positive"));
        }
        // The layers are used by index below; their own invariants run
        // first, so malformed tags or ragged columns refuse instead of
        // panicking.
        sentences.check().map_err(invalid::<Sentences>)?;
        tokens.check().map_err(invalid::<Tokens>)?;
        pos_tags.check().map_err(invalid::<PosTags>)?;
        terms.check().map_err(invalid::<Terms>)?;
        let n = tokens.spans.len();
        if pos_tags.tags.len() != n {
            return Err(Error::Annotator {
                annotator: crate::TextRankAnnotator::ID,
                detail: format!("{} tags and {} tokens", pos_tags.tags.len(), n),
            });
        }
        if terms.len() != n {
            return Err(Error::Annotator {
                annotator: crate::TextRankAnnotator::ID,
                detail: format!("{} terms and {} tokens", terms.len(), n),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let content: Vec<bool> = {
            let mut set = vec![false; pos_tags.tagset.len()];
            for label in CONTENT_LABELS {
                if let Some(tag) = pos_tags.tagset.tag(label) {
                    set[tag.0 as usize] = true;
                }
            }
            pos_tags.tags.iter().map(|t| set[t.0 as usize]).collect()
        };
        let sentence_of = crate::fold::sentence_of(sentences, tokens, crate::TextRankAnnotator::ID)?;
        let (tok_start, tok_end) = tokens.spans.columns();

        // The content runs, per sentence: (token index, vertex id).
        let mut graph = Graph::new();
        let mut runs: Vec<Vec<(usize, u32)>> = vec![Vec::new(); sentences.spans.len()];
        for i in 0..n {
            if !content[i] {
                continue;
            }
            let id = graph.intern(terms.folded.get(i));
            runs[sentence_of[i] as usize].push((i, id));
        }
        for run in &runs {
            for i in 0..run.len() {
                let end = i.saturating_add(self.window).saturating_add(1).min(run.len());
                for j in i + 1..end {
                    graph.cooccur(run[i].1, run[j].1);
                }
            }
        }
        let mut pr = graph.pagerank(self.damping, self.epsilon);
        let total: f64 = pr.iter().sum();
        if total > 0.0 {
            for s in &mut pr {
                *s /= total;
            }
        }

        // The top-N vertices by centrality, marked for the collapse.
        let mut order: Vec<u32> = (0..graph.len() as u32).collect();
        order.sort_by(|&a, &b| {
            pr[b as usize].partial_cmp(&pr[a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b))
        });
        order.truncate(self.top_n.min(order.len()));
        let marked: std::collections::HashSet<u32> = order.into_iter().collect();

        // Adjacent marked content tokens collapse into one keyphrase,
        // scored by the sum of its distinct members' centralities, per
        // occurrence: a phrase repeating a term ("cats cats") counts its
        // vertex once, so the score stays inside the total centrality
        // mass of 1. Adjacency is in the token stream: a token that did
        // not pass the POS filter breaks a run as a word a keyphrase
        // would not span.
        let mut phrases: HashMap<Vec<u32>, (f64, Vec<Span>)> = HashMap::new();
        for run in &runs {
            let mut i = 0;
            while i < run.len() {
                if !marked.contains(&run[i].1) {
                    i += 1;
                    continue;
                }
                let mut j = i + 1;
                while j < run.len() && marked.contains(&run[j].1) && run[j].0 == run[j - 1].0 + 1 {
                    j += 1;
                }
                let ids: Vec<u32> = run[i..j].iter().map(|&(_, id)| id).collect();
                let mut distinct = ids.clone();
                distinct.sort_unstable();
                distinct.dedup();
                let score: f64 = distinct.iter().map(|&id| pr[id as usize]).sum();
                let span = Span::new(tok_start[run[i].0], tok_end[run[j - 1].0]);
                let entry = phrases.entry(ids).or_insert((score, Vec::new()));
                debug_assert!((entry.0 - score).abs() < 1e-12, "one score per phrase");
                entry.1.push(span);
                i = j;
            }
        }
        let mut ranked: Vec<RankedKeyword> = phrases
            .into_iter()
            .map(|(ids, (score, occurrences))| {
                let phrase = ids.iter().map(|&id| graph.name(id)).collect::<Vec<_>>().join(" ");
                RankedKeyword { phrase, terms: ids.len() as u16, salience: score, occurrences, rank: 0 }
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.salience
                .partial_cmp(&a.salience)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.occurrences[0].start.cmp(&b.occurrences[0].start))
                .then_with(|| a.phrase.cmp(&b.phrase))
        });
        for (rank, kw) in ranked.iter_mut().enumerate() {
            kw.rank = rank;
        }
        Ok(ranked)
    }

    pub(crate) fn config_hash(&self) -> u64 {
        let mut h = 0x9E37_79B9_7F4A_7C15u64;
        for v in [self.window as u64, self.top_n as u64, self.damping.to_bits(), self.epsilon.to_bits()] {
            h = (h ^ v).wrapping_mul(0x100_0000_01b3);
        }
        h
    }
}

fn config(detail: &'static str) -> Error {
    Error::Annotator { annotator: crate::TextRankAnnotator::ID, detail: format!("invalid configuration: {detail}") }
}

/// A layer whose own `check` refused it, as the step's error.
fn invalid<L: Layer>(detail: String) -> Error {
    Error::Invalid { layer: Key::<L>::DEFAULT.id(), detail }
}
