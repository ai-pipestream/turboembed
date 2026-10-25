//! Fractional term counts: no rounding of information-density contributions.

use std::marker::PhantomData;

use turbo_analysis::{Anchor, Annotator, Error, Ix, Key, KeyId, Layer, LayerName, Refs};
use turbo_analysis_layers::{Boilerplate, SpanQuality, Structure, Tokens};

use super::content::content_terms;
use super::count::unit_ranges;
use super::{Query, Terms, Vocabulary};

/// Sparse term frequencies weighted by quality information density, aligned
/// to L. Zero-density occurrences are absent, including from document frequency.
/// This is a separate layer because integer TermStats cannot preserve fractions.
pub struct DensityTermStats<L> {
    pub row: Vec<u32>,
    pub term: Vec<Ix<Vocabulary>>,
    pub tf: Vec<f64>,
    pub length: Vec<f64>,
    pub df: Vec<u32>,
    pub _of: PhantomData<fn() -> L>,
}

impl<L: Layer> DensityTermStats<L> {
    /// Score one unit with the existing BM25 query's idf and parameters,
    /// retaining fractional frequencies and effective unit lengths.
    pub fn score(&self, query: &Query, unit: Ix<L>) -> Result<f64, String> {
        if unit.get() >= self.length.len()
            || !query.model.k1.is_finite()
            || query.model.k1 < 0.0
            || !(0.0..=1.0).contains(&query.model.b)
            || !query.mean_length.is_finite()
            || query.mean_length < 0.0
        {
            return Err("invalid density BM25 unit or parameters".into());
        }
        if self.row.len() != self.length.len() + 1 || self.term.len() != self.tf.len() {
            return Err("invalid density BM25 columns".into());
        }
        let a = self.row[unit.get()] as usize;
        let b = self.row[unit.get() + 1] as usize;
        if a > b || b > self.term.len() {
            return Err("invalid density BM25 row".into());
        }
        let terms = &self.term[a..b];
        let tf = &self.tf[a..b];
        let sum: f64 = tf.iter().sum();
        let length = self.length[unit.get()];
        if !terms.windows(2).all(|w| w[0] < w[1])
            || terms.iter().any(|t| t.get() >= self.df.len())
            || tf.iter().any(|v| !v.is_finite() || *v <= 0.0)
            || !sum.is_finite()
            || !length.is_finite()
            || length < 0.0
            || (sum - length).abs() > 1e-10 * sum.max(1.0)
        {
            return Err("invalid density BM25 row values".into());
        }
        let relative = if query.mean_length > 0.0 { self.length[unit.get()] / query.mean_length } else { 1.0 };
        let norm = query.model.k1 * (1.0 - query.model.b + query.model.b * relative);
        let mut score = 0.0;
        for &(term, idf) in &query.terms {
            if term.get() >= self.df.len() || !idf.is_finite() || idf < 0.0 {
                return Err("invalid density BM25 query term or idf".into());
            }
            if let Ok(at) = terms.binary_search(&term) {
                let tf = self.tf[a + at];
                // Scale before adding, so two finite large values cannot
                // overflow the denominator and silently produce zero.
                let ratio = if norm <= tf {
                    1.0 / (1.0 + norm / tf)
                } else {
                    let scaled = tf / norm;
                    scaled / (1.0 + scaled)
                };
                score += idf * ratio;
            }
        }
        if !score.is_finite() || !norm.is_finite() {
            return Err("density BM25 arithmetic is not finite".into());
        }
        Ok(score)
    }
}

impl<L: Layer> Layer for DensityTermStats<L> {
    const ID: &'static str = "turbo:density-term-stats";
    const OF: Option<&'static LayerName> = Some(L::NAME);
    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }
    fn len(&self) -> usize {
        self.length.len()
    }
    fn indexes(&self) -> Vec<(KeyId, usize)> {
        vec![(Key::<Vocabulary>::DEFAULT.id(), self.df.len())]
    }
    fn check_document(&self, doc: &turbo_analysis::Document, on: &[KeyId]) -> Result<(), String> {
        let vocab = on
            .iter()
            .find(|k| k.ty == std::any::TypeId::of::<Vocabulary>())
            .and_then(|k| doc.get_key(&Key::<Vocabulary>::named(k.name)))
            .ok_or("density vocabulary must resolve on the document")?;
        if self.df.len() != vocab.len() {
            return Err("density df must cover the full resolved vocabulary".into());
        }
        Ok(())
    }
    fn check(&self) -> Result<(), String> {
        if self.row.len() != self.length.len() + 1
            || self.row.first() != Some(&0)
            || self.row.last().map(|&x| x as usize) != Some(self.term.len())
            || self.tf.len() != self.term.len()
            || !self.row.windows(2).all(|w| w[0] <= w[1])
        {
            return Err("density term columns and sparse rows must agree".into());
        }
        let mut df = vec![0u32; self.df.len()];
        for i in 0..self.length.len() {
            let range = self.row[i] as usize..self.row[i + 1] as usize;
            if !self.term[range.clone()].windows(2).all(|w| w[0] < w[1]) {
                return Err("density terms must be unique and sorted in each row".into());
            }
            let mut sum = 0.0;
            for j in range {
                let t = self.term[j].get();
                if t >= df.len() || !self.tf[j].is_finite() || self.tf[j] <= 0.0 {
                    return Err("positive finite density counts inside the vocabulary".into());
                }
                df[t] = df[t].checked_add(1).ok_or("density document frequency overflow")?;
                sum += self.tf[j];
            }
            let length = self.length[i];
            if !length.is_finite() || length < 0.0 || !sum.is_finite() || (length - sum).abs() > 1e-10 * sum.max(1.0) {
                return Err("density length must equal its finite sum of counts".into());
            }
        }
        if df != self.df {
            return Err("density document frequencies must count positive rows".into());
        }
        Ok(())
    }
}

/// The opt-in mode returned by ContentTermCounter::with_quality().
pub struct QualityContentTermCounter<L, B = Structure> {
    pub(super) _of: PhantomData<fn() -> (L, B)>,
}

impl<L: Layer, B: Layer> Annotator for QualityContentTermCounter<L, B> {
    const ID: &'static str = "turbo:quality-content-term-counter";
    type Reads = (Tokens, Terms, Vocabulary, L, B, Boilerplate<B>, SpanQuality);
    type Writes = (DensityTermStats<L>,);

    fn annotate(
        &self,
        (tokens, terms, vocabulary, units, blocks, flags, quality): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail| Error::Annotator { annotator: Self::ID, detail };
        let ranges = unit_ranges(tokens, units).map_err(fail)?;
        let terms = content_terms(tokens, terms, blocks, flags).map_err(fail)?;
        quality.check().map_err(fail)?;
        if quality.spans.view() != tokens.spans.view() {
            return Err(fail("quality and tokens must be in the same view".into()));
        }
        let mut weights = Vec::with_capacity(terms.len());
        let mut q = 0;
        let (start, _) = tokens.spans.columns();
        for &s in start {
            while q < quality.spans.len() && quality.spans.get(Ix::new(q as u32)).end <= s {
                q += 1;
            }
            let weight = if q < quality.spans.len() && quality.spans.get(Ix::new(q as u32)).start <= s {
                f64::from(quality.metrics[q].information_density)
            } else {
                1.0
            };
            weights.push(weight);
        }
        let mut out = DensityTermStats {
            row: vec![0],
            term: Vec::new(),
            tf: Vec::new(),
            length: Vec::new(),
            df: vec![0; vocabulary.len()],
            _of: PhantomData,
        };
        let mut entries = Vec::new();
        for unit in ranges {
            entries.clear();
            for range in unit {
                for i in range {
                    if let Some(term) = terms[i] {
                        if term.get() >= vocabulary.len() {
                            return Err(fail("density term lies outside the vocabulary".into()));
                        }
                        if weights[i] > 0.0 {
                            entries.push((term, weights[i]));
                        }
                    }
                }
            }
            entries.sort_unstable_by_key(|&(term, _)| term);
            let mut at = 0;
            let mut length = 0.0;
            while at < entries.len() {
                let term = entries[at].0;
                let mut tf = 0.0;
                while at < entries.len() && entries[at].0 == term {
                    tf += entries[at].1;
                    at += 1;
                }
                length += tf;
                out.term.push(term);
                out.tf.push(tf);
                out.df[term.get()] =
                    out.df[term.get()].checked_add(1).ok_or_else(|| fail("density df overflow".into()))?;
            }
            out.length.push(length);
            out.row.push(u32::try_from(out.term.len()).map_err(|_| fail("density rows exceed u32".into()))?);
        }
        out.check().map_err(fail)?;
        Ok((out,))
    }
}
