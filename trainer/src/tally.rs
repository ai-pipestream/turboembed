//! The dev scores of the feature heads through the core's own inference
//! path: accuracy or exact span F1 of a tagger, and the accuracies of
//! the sentence boundary and token split heads.

use turbo::head::boundary::LAST;
use turbo::head::crf::{WordSpan, spans};
use turbo::head::tagger::{NO_WORD, Work as CoreWork};
use turbo::head::{SentenceBoundary, TokenSplitter, TokenTagger};
use turbo::manifest::{HeadSpec, Scheme};
use turbo::status::Result;

use crate::fit::{Example, check_examples};

/// Accuracy (SCHEME_TAGS) or exact span F1 (BIO, BILOU) of `tagger` on
/// `examples`, decoded by the core exactly as at inference. Refuses an
/// example without exactly one gold label in range per feature row.
pub fn score(tagger: &TokenTagger, examples: &[Example]) -> Result<f64> {
    let spec = tagger.spec();
    let hidden = tagger.hidden();
    check_examples(examples, hidden, spec.labels.len(), tagger.reads_forms())?;
    let mut tally = Tally::new(spec);
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.features.len() / hidden;
        let words: Vec<u32> = (0..n as u32).collect();
        debug_assert!(!words.contains(&NO_WORD));
        tagger.scores_forms(&e.features, &words, e.forms().as_deref(), &mut scores, &mut work)?;
        tagger.decode(&scores, n, &mut tags)?;
        tally.add(&tags, &e.labels)?;
    }
    Ok(tally.score())
}

/// The running dev score of a tagger head: right words of all under
/// SCHEME_TAGS, found and gold spans and the right ones under BIO and
/// BILOU.
pub(crate) struct Tally<'a> {
    spec: &'a HeadSpec,
    right: usize,
    total: usize,
    found: usize,
    gold: usize,
}

impl<'a> Tally<'a> {
    pub(crate) fn new(spec: &'a HeadSpec) -> Tally<'a> {
        Tally { spec, right: 0, total: 0, found: 0, gold: 0 }
    }

    /// One sentence's decoded `tags` against its `labels`.
    pub(crate) fn add(&mut self, tags: &[u32], labels: &[u32]) -> Result<()> {
        let spec = self.spec;
        if spec.scheme == Scheme::Tags {
            self.right += tags.iter().zip(labels).filter(|(a, b)| a == b).count();
            self.total += labels.len();
        } else {
            let want: Vec<WordSpan> = spans(spec.scheme, &spec.labels, labels)?;
            let got = spans(spec.scheme, &spec.labels, tags)?;
            self.right += got.iter().filter(|s| want.contains(s)).count();
            self.found += got.len();
            self.gold += want.len();
        }
        Ok(())
    }

    /// Accuracy, or exact span F1; 1 when there is nothing to find and
    /// nothing was found.
    pub(crate) fn score(&self) -> f64 {
        if self.spec.scheme == Scheme::Tags {
            if self.total == 0 { 0.0 } else { self.right as f64 / self.total as f64 }
        } else if self.found + self.gold == 0 {
            1.0
        } else {
            2.0 * self.right as f64 / (self.found + self.gold) as f64
        }
    }
}

/// Accuracy of `head` on `examples`: whether each word ends a sentence as
/// the core decodes it (the last word always does) against its gold label
/// being `last`.
pub fn score_boundary(head: &SentenceBoundary, examples: &[Example]) -> Result<f64> {
    let hidden = head.hidden();
    check_examples(examples, hidden, head.spec().labels.len(), head.reads_forms())?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.labels.len();
        let words: Vec<u32> = (0..n as u32).collect();
        head.scores(&e.features, &words, e.forms().as_deref(), &mut scores, &mut work)?;
        head.decode(&scores, n, &mut ends)?;
        right += ends.iter().zip(&e.labels).filter(|&(&end, &gold)| end == (gold == LAST)).count();
        total += n;
    }
    Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 })
}

/// Accuracy of `head` on `examples`: each token's label as the core
/// decodes it, which skips a split that does not apply to the token's
/// form. Refuses an example without a form per token.
pub fn score_split(head: &TokenSplitter, examples: &[Example]) -> Result<f64> {
    let hidden = head.hidden();
    check_examples(examples, hidden, head.spec().labels.len(), true)?;
    let (mut right, mut total) = (0usize, 0usize);
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in examples {
        let n = e.labels.len();
        let words: Vec<u32> = (0..n as u32).collect();
        let forms: Vec<&str> = e.forms.iter().map(String::as_str).collect();
        let read = head.reads_forms().then_some(forms.as_slice());
        head.scores(&e.features, &words, read, &mut scores, &mut work)?;
        head.decode(&scores, &forms, &mut labels)?;
        right += labels.iter().zip(&e.labels).filter(|(a, b)| a == b).count();
        total += n;
    }
    Ok(if total == 0 { 0.0 } else { right as f64 / total as f64 })
}
