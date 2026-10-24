//! Lexical TextTiling over the term analyzer's folded or stemmed tokens.

use std::collections::BTreeMap;

use turbo_analysis::{Annotator, Bound, Error, Layer, Overlap, Refs, View};
use turbo_analysis_layers::{Boundaries, Sentences, Terms, Tokens};
use turbo_unicode::category::general_category;

use crate::drift::{Drift, DriftBoundaries, MICRO, score_cosines};

/// Named boundary instance written by [`LexicalBoundaries`].
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
}

fn invalid(detail: impl Into<String>) -> Error {
    Error::Annotator { annotator: LexicalBoundaries::ID, detail: detail.into() }
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

fn cosine(a: &Counts<'_>, b: &Counts<'_>) -> Option<i64> {
    let square = |counts: &Counts<'_>| counts.values().map(|&n| u128::from(n).pow(2)).sum::<u128>();
    let (na, nb) = (square(a), square(b));
    if na == 0 || nb == 0 {
        return None;
    }
    let dot: u128 = a.iter().map(|(t, &n)| u128::from(n) * u128::from(*b.get(t).unwrap_or(&0))).sum();
    Some(((dot as f64 / (na as f64 * nb as f64).sqrt()).clamp(0.0, 1.0) * MICRO as f64).round() as i64)
}

impl Annotator for LexicalBoundaries {
    const ID: &'static str = "turbo:lexical-boundaries";
    type Reads = (View, Sentences, Tokens, Terms);
    type Writes = (Boundaries,);

    fn annotate(&self, (view, sentences, tokens, terms): Refs<'_, Self>) -> Result<(Boundaries,), Error> {
        sentences.check().map_err(invalid)?;
        tokens.check().map_err(invalid)?;
        // Window movement needs tokens in text order and counted once.
        tokens.spans.check(Overlap::None).map_err(invalid)?;
        terms.check().map_err(invalid)?;
        if sentences.spans.view() != view.id() || tokens.spans.view() != view.id() {
            return Err(invalid("sentences, tokens and text must share a view"));
        }
        if terms.len() != tokens.spans.len() {
            return Err(invalid("one term per token is required"));
        }
        let forms = match self.form {
            LexicalForm::Folded => &terms.folded,
            LexicalForm::Stem => terms.stem.as_ref().ok_or_else(|| invalid("missing stem column"))?,
            LexicalForm::Lemma => terms.lemma.as_ref().ok_or_else(|| invalid("missing lemma column"))?,
        };
        for (_, span) in sentences.spans.iter() {
            if view.text().get(span.range()).is_none() {
                return Err(invalid("sentence outside text or UTF-8 boundaries"));
            }
        }
        let n = sentences.spans.len();
        let mut rows = vec![Vec::new(); n];
        let (starts, ends) = sentences.spans.columns();
        for (i, span) in tokens.spans.iter() {
            if view.text().get(span.range()).is_none() {
                return Err(invalid("token outside text or UTF-8 boundaries"));
            }
            let s = ends.partition_point(|&end| end <= span.start);
            if s >= n || span.start < starts[s] || span.end > ends[s] || span.start == span.end {
                return Err(invalid("each token must be contained in one sentence"));
            }
            if tokens.sentence.as_ref().is_some_and(|col| col[i.get()].get() != s) {
                return Err(invalid("token sentence index differs from its span"));
            }
            let term = forms.get(i.get());
            if term.chars().any(|c| {
                let cat = general_category(c);
                cat.is_letter() || cat.is_number()
            }) {
                rows[s].push(term);
            }
        }
        let w = self.drift.window.get() as usize;
        let mut cos = vec![None; n];
        // Subtraction avoids overflow on 32-bit platforms for a huge window.
        if w <= n / 2 {
            let (mut left, mut right) = (Counts::new(), Counts::new());
            for row in &rows[..w] {
                change(&mut left, row, true);
            }
            for row in &rows[w..2 * w] {
                change(&mut right, row, true);
            }
            for split in w..=n - w {
                cos[split - 1] = cosine(&left, &right);
                if split < n - w {
                    change(&mut left, &rows[split - w], false);
                    change(&mut left, &rows[split], true);
                    change(&mut right, &rows[split], false);
                    change(&mut right, &rows[split + w], true);
                }
            }
        }
        let (cost, reason) = score_cosines(&cos, &self.drift);
        Ok((Boundaries { cost, reason },))
    }

    fn config_hash(&self) -> u64 {
        (DriftBoundaries { drift: self.drift }.config_hash() ^ (self.form as u64)).wrapping_mul(0x0100_0000_01b3)
    }
}
