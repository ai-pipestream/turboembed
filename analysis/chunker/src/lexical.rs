//! Lexical TextTiling over the term analyzer's folded or stemmed tokens,
//! with optional collection-IDF (`TF * IDF`) weighting (`#2346`).

use std::collections::BTreeMap;
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, CorpusIdf, Error, Fnv1a, Layer, Overlap, Refs, View};
use turbo_analysis_layers::{Boundaries, Sentences, Terms, Tokens};
use turbo_unicode::category::general_category;

use crate::drift::{Drift, DriftBoundaries, MICRO, score_cosines};

/// Named boundary instance written by [`LexicalBoundaries`] and [`IdfLexicalBoundaries`].
pub const LEXICAL: &str = "lexical";

/// Term column used for lexical cohesion. There is no silent fallback.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LexicalForm {
    #[default]
    Folded,
    Stem,
    Lemma,
}

/// Adjacent sentence windows compared by cosine of term frequencies,
/// followed by the same smoothing/depth scoring as embedding drift.
/// Punctuation-only forms are omitted; no stop list or IDF is applied.
/// An empty window has no cosine, rather than a manufactured topic shift.
#[derive(Clone, Copy, Debug, Default)]
pub struct LexicalBoundaries {
    pub drift: Drift,
    pub form: LexicalForm,
}

impl LexicalBoundaries {
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, LEXICAL)
    }

    /// Upgrades this lexical boundary scorer with collection-wide BM25 IDF
    /// weights (`#2346`), comparing adjacent sentence windows by `TF * IDF`
    /// cosine so shared ubiquitous terms do not mask genuine topic shifts.
    pub fn with_corpus_idf(self, idf: Arc<CorpusIdf>) -> IdfLexicalBoundaries {
        IdfLexicalBoundaries { drift: self.drift, form: self.form, idf }
    }
}

/// Collection-IDF-weighted lexical TextTiling boundary scorer (`#2346`).
#[derive(Clone, Debug)]
pub struct IdfLexicalBoundaries {
    pub drift: Drift,
    pub form: LexicalForm,
    pub idf: Arc<CorpusIdf>,
}

impl IdfLexicalBoundaries {
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, LEXICAL)
    }
}

fn invalid(annotator: &'static str, detail: impl Into<String>) -> Error {
    Error::Annotator { annotator, detail: detail.into() }
}

type Counts<'a> = BTreeMap<&'a str, u32>;

fn change<'a>(counts: &mut Counts<'a>, row: &[&'a str], add: bool) {
    for &term in row {
        if add {
            *counts.entry(term).or_default() += 1;
        } else {
            let n = counts.get_mut(term).expect("a term leaving the window was counted");
            *n -= 1;
            if *n == 0 {
                counts.remove(term);
            }
        }
    }
}

fn cosine(a: &Counts<'_>, b: &Counts<'_>, idf: Option<&CorpusIdf>) -> Option<i64> {
    match idf {
        None => {
            let square = |counts: &Counts<'_>| counts.values().map(|&n| u128::from(n).pow(2)).sum::<u128>();
            let (na, nb) = (square(a), square(b));
            if na == 0 || nb == 0 {
                return None;
            }
            let dot: u128 = a.iter().map(|(t, &n)| u128::from(n) * u128::from(*b.get(t).unwrap_or(&0))).sum();
            Some(((dot as f64 / (na as f64 * nb as f64).sqrt()).clamp(0.0, 1.0) * MICRO as f64).round() as i64)
        }
        Some(table) => {
            let weight = |term: &str, tf: u32| (tf as f64) * table.idf(term);
            let sq = |counts: &Counts<'_>| -> f64 {
                counts
                    .iter()
                    .map(|(t, &n)| {
                        let w = weight(t, n);
                        w * w
                    })
                    .sum()
            };
            let (na, nb) = (sq(a), sq(b));
            if na <= 0.0 || nb <= 0.0 {
                return None;
            }
            let dot: f64 = a
                .iter()
                .filter_map(|(t, &na_tf)| {
                    let nb_tf = *b.get(t)?;
                    let w_idf = table.idf(t);
                    Some((na_tf as f64) * (nb_tf as f64) * w_idf * w_idf)
                })
                .sum();
            let denom = (na * nb).sqrt();
            if denom <= 0.0 || !denom.is_finite() {
                return None;
            }
            Some(((dot / denom).clamp(0.0, 1.0) * MICRO as f64).round() as i64)
        }
    }
}

fn annotate_lexical(
    annotator: &'static str,
    drift: &Drift,
    form: LexicalForm,
    idf: Option<&CorpusIdf>,
    (view, sentences, tokens, terms): (&View, &Sentences, &Tokens, &Terms),
) -> Result<(Boundaries,), Error> {
    let bad = |d: String| invalid(annotator, d);
    sentences.check().map_err(bad)?;
    tokens.check().map_err(bad)?;
    tokens.spans.check(Overlap::None).map_err(bad)?;
    terms.check().map_err(bad)?;
    if sentences.spans.view() != view.id() || tokens.spans.view() != view.id() {
        return Err(invalid(annotator, "sentences, tokens and text must share a view"));
    }
    if terms.len() != tokens.spans.len() {
        return Err(invalid(annotator, "one term per token is required"));
    }
    let forms = match form {
        LexicalForm::Folded => &terms.folded,
        LexicalForm::Stem => terms.stem.as_ref().ok_or_else(|| invalid(annotator, "missing stem column"))?,
        LexicalForm::Lemma => terms.lemma.as_ref().ok_or_else(|| invalid(annotator, "missing lemma column"))?,
    };
    for (_, span) in sentences.spans.iter() {
        if view.text().get(span.range()).is_none() {
            return Err(invalid(annotator, "sentence outside text or UTF-8 boundaries"));
        }
    }
    let n = sentences.spans.len();
    let mut rows = vec![Vec::new(); n];
    let (starts, ends) = sentences.spans.columns();
    for (i, span) in tokens.spans.iter() {
        if view.text().get(span.range()).is_none() {
            return Err(invalid(annotator, "token outside text or UTF-8 boundaries"));
        }
        let s = ends.partition_point(|&end| end <= span.start);
        if s >= n || span.start < starts[s] || span.end > ends[s] || span.start == span.end {
            return Err(invalid(annotator, "each token must be contained in one sentence"));
        }
        if tokens.sentence.as_ref().is_some_and(|col| col[i.get()].get() != s) {
            return Err(invalid(annotator, "token sentence index differs from its span"));
        }
        let term = forms.get(i.get());
        if term.chars().any(|c| {
            let cat = general_category(c);
            cat.is_letter() || cat.is_number()
        }) {
            rows[s].push(term);
        }
    }
    let w = drift.window.get() as usize;
    let mut cos = vec![None; n];
    if w <= n / 2 {
        let (mut left, mut right) = (Counts::new(), Counts::new());
        for row in &rows[..w] {
            change(&mut left, row, true);
        }
        for row in &rows[w..2 * w] {
            change(&mut right, row, true);
        }
        for split in w..=n - w {
            cos[split - 1] = cosine(&left, &right, idf);
            if split < n - w {
                change(&mut left, &rows[split - w], false);
                change(&mut left, &rows[split], true);
                change(&mut right, &rows[split], false);
                change(&mut right, &rows[split + w], true);
            }
        }
    }
    let (cost, reason) = score_cosines(&cos, drift);
    Ok((Boundaries { cost, reason },))
}

impl Annotator for LexicalBoundaries {
    const ID: &'static str = "turbo:lexical-boundaries";
    type Reads = (View, Sentences, Tokens, Terms);
    type Writes = (Boundaries,);

    fn annotate(&self, reads: Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        annotate_lexical(Self::ID, &self.drift, self.form, None, reads)
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::with_seed(DriftBoundaries { drift: self.drift }.config_hash()).fold_u64(self.form as u64).finish()
    }
}

impl Annotator for IdfLexicalBoundaries {
    const ID: &'static str = "turbo:idf-lexical-boundaries";
    type Reads = (View, Sentences, Tokens, Terms);
    type Writes = (Boundaries,);

    fn annotate(&self, reads: Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        annotate_lexical(Self::ID, &self.drift, self.form, Some(&self.idf), reads)
    }

    fn config_hash(&self) -> u64 {
        Fnv1a::with_seed(DriftBoundaries { drift: self.drift }.config_hash())
            .fold_u64(self.form as u64)
            .fold_u64(self.idf.config_hash())
            .finish()
    }
}

#[cfg(test)]
mod idf_tests {
    use super::*;

    #[test]
    fn corpus_idf_downweights_shared_boilerplate_cosine() {
        let mut a = Counts::new();
        let mut b = Counts::new();
        a.insert("boilerplate", 3);
        a.insert("quantum", 1);
        b.insert("boilerplate", 3);
        b.insert("gardening", 1);

        let raw = cosine(&a, &b, None).unwrap();
        let idf = CorpusIdf::from_entries(10_000, [("boilerplate", 9_990), ("quantum", 2), ("gardening", 2)]).unwrap();
        let weighted = cosine(&a, &b, Some(&idf)).unwrap();
        assert!(weighted < raw, "expected IDF-weighted cosine {weighted} < raw {raw}");
    }
}
