//! A model-free relation scorer: a pair scores under a label by the
//! share of the label's words found, lower-cased, among the words
//! between its two mentions. "founded" between "Jobs" and "Apple" gives
//! "founded by" 0.5; both words give 1. It reads no order, so the two
//! orders of a pair score alike. It is a baseline and a way to run the
//! step without a model, not a substitute for one.

use std::collections::HashSet;

use turbo_analysis::ModelRef;

use super::{Mention, RelationError, RelationScorer};

/// Scores pairs by the label words between their mentions.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct LexicalRelationScorer;

/// The case-folded words of `text` with their byte ranges: runs of
/// letters and digits.
fn words(text: &str) -> Vec<(u32, u32, String)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        match (c.is_alphanumeric(), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                let mut w = String::new();
                turbo_unicode::casefold::fold(&text[s..i], &mut w);
                out.push((s as u32, i as u32, w));
                start = None;
            }
            _ => {}
        }
    }
    out
}

impl RelationScorer for LexicalRelationScorer {
    fn model_ref(&self) -> Option<ModelRef> {
        None
    }

    fn max_labels(&self) -> usize {
        usize::MAX
    }

    fn max_mentions(&self) -> usize {
        usize::MAX
    }

    fn score(
        &self,
        text: &str,
        mentions: &[Mention<'_>],
        pairs: &[(u32, u32)],
        labels: &[&str],
        out: &mut Vec<f32>,
    ) -> Result<(), RelationError> {
        let words = words(text);
        let label_words: Vec<Vec<String>> = labels
            .iter()
            .map(|l| {
                let mut w: Vec<String> = self::words(l).into_iter().map(|w| w.2).collect();
                w.sort_unstable();
                w.dedup();
                w
            })
            .collect();
        let starts: Vec<u32> = words.iter().map(|w| w.0).collect();
        let mut between: HashSet<&str> = HashSet::new();
        for &(h, t) in pairs {
            let (a, b) = (mentions[h as usize].span, mentions[t as usize].span);
            let (from, to) = (a.end.min(b.end), a.start.max(b.start));
            between.clear();
            for w in &words[starts.partition_point(|&s| s < from)..] {
                if w.1 > to {
                    break;
                }
                between.insert(&w.2);
            }
            for lw in &label_words {
                let hit = lw.iter().filter(|w| between.contains(w.as_str())).count();
                out.push(if lw.is_empty() { 0.0 } else { hit as f32 / lw.len() as f32 });
            }
        }
        Ok(())
    }
}
