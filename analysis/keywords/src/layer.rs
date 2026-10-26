//! The `turbo:keywords` layer: ranked keyphrases as positional spans.
//!
//! One item is one occurrence of one keyphrase, so a phrase that appears
//! twice is two items with different spans carrying the same phrase and
//! score. Items are in text order and may overlap: the deduplication pass
//! removes near-duplicates, not every phrase that shares a word with
//! another ("machine learning" and "learning competitions" can both
//! survive).

use std::sync::{Arc, OnceLock};

use turbo_analysis::{Document, Error, Layer, Overlap, ProjectableLayer, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::Strs;

/// The `yake` entry of the algorithm tag set: items Campos et al.'s YAKE
/// scored, `S(kw) = prod(S(w)) / (KF * (1 + sum(S(w))))`, lower raw score
/// more salient, normalized here into `score`.
pub const YAKE: &str = "yake";
/// The `textrank` entry: items Mihalcea and Tarau's TextRank scored by
/// co-occurrence graph centrality.
pub const TEXTRANK: &str = "textrank";

/// The two-label tag set the `algorithm` column indexes. Every writer of
/// the layer shares this one instance.
pub fn algorithm_tagset() -> Arc<TagSet> {
    static SET: OnceLock<Arc<TagSet>> = OnceLock::new();
    Arc::clone(SET.get_or_init(|| TagSet::new("keyword-algorithm", &[YAKE, TEXTRANK])))
}

/// A ranked keyphrase occurrence (`turbo:keywords`). Columns beside the
/// span: `phrase`, the normalized (folded) keyphrase; `score`, its
/// salience in `[0, 1]` where higher is more salient (the YAKE raw score
/// is lower-is-better; the annotator normalizes survivors so one column
/// has one direction); `algorithm`, which extractor produced it; and
/// `terms`, how many terms the keyphrase holds.
#[derive(Clone, Debug)]
pub struct Keywords {
    pub spans: Spans<Keywords>,
    pub phrase: Strs,
    pub score: Vec<f64>,
    pub algorithm: Vec<Tag>,
    pub algorithms: Arc<TagSet>,
    pub terms: Vec<u16>,
}

impl Keywords {
    /// An empty layer in `view`, carrying the shared algorithm tag set.
    pub fn new(view: ViewId) -> Keywords {
        Keywords {
            spans: Spans::new(view),
            phrase: Strs::default(),
            score: Vec::new(),
            algorithm: Vec::new(),
            algorithms: algorithm_tagset(),
            terms: Vec::new(),
        }
    }

    /// Adds one occurrence: its span, normalized phrase, salience, the
    /// algorithm's tag and its term count.
    pub fn push(&mut self, span: Span, phrase: &str, score: f64, algorithm: Tag, terms: u16) {
        self.spans.push(span);
        self.phrase.push(phrase);
        self.score.push(score);
        self.algorithm.push(algorithm);
        self.terms.push(terms);
    }

    /// The span of item `i`.
    pub fn span(&self, i: usize) -> Span {
        self.spans.get(turbo_analysis::Ix::new(i as u32))
    }

    /// The phrase, score, algorithm label and term count of item `i`.
    pub fn get(&self, i: usize) -> Keyword<'_> {
        Keyword {
            phrase: self.phrase.get(i),
            score: self.score[i],
            algorithm: self.algorithms.label(self.algorithm[i]),
            terms: self.terms[i],
        }
    }

    /// Every item with its span, in text order.
    pub fn iter(&self) -> impl Iterator<Item = (Span, Keyword<'_>)> + '_ {
        (0..self.spans.len()).map(|i| (self.span(i), self.get(i)))
    }
}

/// One [`Keywords`] item, borrowed.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Keyword<'k> {
    pub phrase: &'k str,
    pub score: f64,
    pub algorithm: &'k str,
    pub terms: u16,
}

impl Layer for Keywords {
    const ID: &'static str = "turbo:keywords";
    /// Surviving keyphrases may still overlap: dedup drops
    /// near-duplicates, not every phrase sharing a word.
    const OVERLAP: Overlap = Overlap::Any;

    fn len(&self) -> usize {
        self.spans.len()
    }

    fn text_spans(&self) -> Option<turbo_analysis::TextSpans<'_>> {
        Some(self.spans.text_spans())
    }

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.phrase.len() != n || self.score.len() != n || self.algorithm.len() != n || self.terms.len() != n {
            return Err("a phrase, a score, an algorithm and a term count for every keyword".into());
        }
        if self.phrase.iter().any(str::is_empty) {
            return Err("an empty keyphrase".into());
        }
        if self.algorithm.iter().any(|t| t.0 as usize >= self.algorithms.len()) {
            return Err(format!("an algorithm outside {}", self.algorithms.name));
        }
        if let Some(s) = self.score.iter().find(|s| !s.is_finite() || **s < 0.0 || **s > 1.0) {
            return Err(format!("a keyword score outside [0, 1]: {s}"));
        }
        if self.terms.contains(&0) {
            return Err("a keyphrase with a term count of zero".into());
        }
        self.spans.check(Self::OVERLAP)?;
        Ok(())
    }
}

impl ProjectableLayer for Keywords {
    fn project_view(&self, doc: &Document, to: ViewId) -> Result<Self, Error> {
        Ok(Keywords {
            spans: doc.project_spans_monotonic(&self.spans, to)?,
            phrase: self.phrase.clone(),
            score: self.score.clone(),
            algorithm: self.algorithm.clone(),
            algorithms: Arc::clone(&self.algorithms),
            terms: self.terms.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_occurrences_with_columns() {
        let mut k = Keywords::new(ViewId::SOURCE);
        let yake = algorithm_tagset().tag(YAKE).unwrap();
        assert!(k.is_empty());
        k.push(Span::new(0, 11), "machine learning", 0.9, yake, 2);
        k.push(Span::new(30, 38), "learning", 0.4, yake, 1);
        assert_eq!(k.len(), 2);
        let (s0, item) = k.iter().next().unwrap();
        assert_eq!(s0, Span::new(0, 11));
        assert_eq!(item, Keyword { phrase: "machine learning", score: 0.9, algorithm: "yake", terms: 2 });
        assert_eq!(Keywords::ID, "turbo:keywords");
        assert!(k.check().is_ok());
    }

    #[test]
    fn check_refuses_bad_columns_scores_and_tags() {
        let yake = algorithm_tagset().tag(YAKE).unwrap();
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "abc", f64::NAN, yake, 1);
        assert!(k.check().unwrap_err().starts_with("a keyword score outside"));
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "abc", 1.5, yake, 1);
        assert!(k.check().unwrap_err().starts_with("a keyword score outside"));
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "", 0.5, yake, 1);
        assert_eq!(k.check().unwrap_err(), "an empty keyphrase");
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "abc", 0.5, Tag(9), 1);
        assert!(k.check().unwrap_err().starts_with("an algorithm outside"));
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "abc", 0.5, yake, 0);
        assert_eq!(k.check().unwrap_err(), "a keyphrase with a term count of zero");
        let mut k = Keywords::new(ViewId::SOURCE);
        k.push(Span::new(0, 3), "abc", 0.5, yake, 1);
        k.terms.pop();
        assert!(k.check().unwrap_err().starts_with("a phrase, a score"));
    }

    #[test]
    fn projects_keywords_across_views() {
        use turbo_analysis::{Ix, Key, Provenance, View, ViewBuilder};

        let mut doc = Document::new(">>> machine learning");
        let v0 = doc.get::<View>().unwrap().clone();
        let mut b = ViewBuilder::new(v0.text());
        b.replace(4, "");
        let (t1, a1) = b.finish();
        doc.add(&Key::<View>::named("clean"), View::derived(&v0, t1, a1), Provenance::by("test")).unwrap();
        let v1_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();

        let yake = algorithm_tagset().tag(YAKE).unwrap();
        let mut k = Keywords::new(v1_id);
        k.push(Span::new(0, 16), "machine learning", 0.9, yake, 2);
        let projected = doc.project_layer(&k, ViewId::SOURCE).unwrap();
        assert_eq!(projected.spans.view(), ViewId::SOURCE);
        assert_eq!(projected.spans.get(Ix::new(0)), Span::new(4, 20));
    }
}
