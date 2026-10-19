use std::collections::{BTreeMap, HashSet};

use turbo_analysis::{Tag, TagSet};
use turbo_eval::model_gate::Counts;
use turbo_formats::evaluate::PosEvaluator;
use turbo_formats::sample::PosSample;
use turbo_maxent::ModelFile;
use turbo_postag::{PosModel, PosTagFormat, PosTagger};
use turbo_zoo::Sha256;

use super::{MISMATCHES_LISTED, ReferenceMapping, Scored};
use crate::card::{Mismatch, TagCount};
use crate::error::{Error, Result};

/// The tags a reference tagger can give, and the mappings its tags are
/// compared through, which decide the held-out tokens the gate scores:
/// those whose tag is one of them, as written or through a mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceTags {
    tags: HashSet<String>,
    pub(super) mappings: &'static [ReferenceMapping],
}

impl ReferenceTags {
    /// The tags of the reference's tag set, as it gives them, compared
    /// through `mappings` ([`super::REFERENCE_MAPPINGS`] for the MASC tagger).
    pub fn of(tagset: &TagSet, mappings: &'static [ReferenceMapping]) -> ReferenceTags {
        let tags = (0..tagset.len()).map(|i| tagset.label(Tag(i as u16)).to_owned()).collect();
        ReferenceTags { tags, mappings }
    }

    /// Whether a held-out token tagged `tag` is scored.
    pub fn scores(&self, tag: &str) -> bool {
        self.tags.contains(tag) || self.mappings.iter().any(|m| m.held_out == tag && self.tags.contains(m.reference))
    }

    /// The held-out tokens left out of the scores, by their tag as
    /// written, sorted by tag.
    pub fn left_out(&self, held_out: &HeldOutTags<'_>) -> Vec<TagCount> {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for sample in held_out.samples() {
            for tag in sample.tags() {
                if !self.scores(tag.as_str()) {
                    *counts.entry(tag.as_str()).or_default() += 1;
                }
            }
        }
        counts.into_iter().map(|(tag, tokens)| TagCount { tag: tag.to_owned(), tokens }).collect()
    }
}

/// The held-out part-of-speech samples, borrowed from the held-out corpus.
#[derive(Debug, Clone)]
pub struct HeldOutTags<'c> {
    pub(super) samples: Vec<PosSample<'c>>,
    pub(super) skipped: usize,
}

impl<'c> HeldOutTags<'c> {
    /// The held-out samples `samples`, read with `skipped` sentences
    /// skipped.
    pub fn new(samples: Vec<PosSample<'c>>, skipped: usize) -> HeldOutTags<'c> {
        HeldOutTags { samples, skipped }
    }

    /// The samples scored.
    pub fn samples(&self) -> &[PosSample<'c>] {
        &self.samples
    }

    /// How many held-out sentences [`super::read_samples`] skipped.
    pub fn skipped(&self) -> usize {
        self.skipped
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// The tokens of the samples.
    pub fn tokens(&self) -> u64 {
        self.samples.iter().map(|s| s.len() as u64).sum()
    }

    /// The SHA-256 of the samples as scored with a reference of the tags
    /// `reference`: for each sample, its number of tokens as a little
    /// endian `u64`, then for each token its text's length in bytes as a
    /// little endian `u64`, the text, its tag's length and the tag, and a
    /// byte, 1 when the token is scored ([`ReferenceTags::scores`]) and 0
    /// when it is left out; after the samples, the number of the
    /// reference's mappings as a `u64` (0 for none) and each one's held-out
    /// tag and reference tag, each as its length and its bytes. Two gates
    /// whose references leave out other tokens, or that map other tags,
    /// have other digests.
    pub fn sha256(&self, reference: &ReferenceTags) -> String {
        let mut bytes = Vec::new();
        let put = |bytes: &mut Vec<u8>, s: &str| {
            bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
            bytes.extend_from_slice(s.as_bytes());
        };
        for sample in &self.samples {
            bytes.extend_from_slice(&(sample.len() as u64).to_le_bytes());
            for (token, tag) in sample.tokens().iter().zip(sample.tags()) {
                put(&mut bytes, token.as_str());
                put(&mut bytes, tag.as_str());
                bytes.push(u8::from(reference.scores(tag.as_str())));
            }
        }
        bytes.extend_from_slice(&(reference.mappings.len() as u64).to_le_bytes());
        for m in reference.mappings {
            put(&mut bytes, m.held_out);
            put(&mut bytes, m.reference);
        }
        Sha256::of(&bytes).to_string()
    }
}

/// The tagger in `bytes`, its tags asked for in `format`.
pub(super) fn open(bytes: &[u8], format: PosTagFormat) -> Result<PosTagger> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    let model = PosModel::open_with_format(file, format).map_err(|e| Error::Model(e.to_string()))?;
    Ok(PosTagger::new(model))
}

/// A tagger's counts on the held-out samples, and its most frequent
/// mismatches on the scored tokens.
pub(super) struct Scores {
    pub(super) units: Vec<Counts>,
    /// At most [`MISMATCHES_LISTED`], most frequent first.
    pub(super) mismatches: Vec<Mismatch>,
}

/// [`super::PosRecipe::score`] with a tagger already open.
pub(super) fn score_with(
    tagger: &PosTagger,
    scored_as: Scored,
    held_out: &HeldOutTags<'_>,
    reference: &ReferenceTags,
) -> Result<Scores> {
    let tagset = tagger.pos_model().tagset().clone();
    let mut units = Vec::with_capacity(held_out.len());
    let mut mismatches: BTreeMap<(&str, &str), u64> = BTreeMap::new();
    let (mut tags, mut probs) = (Vec::new(), Vec::new());
    let mut words: Vec<&str> = Vec::new();
    let mut labels: Vec<&str> = Vec::new();
    for sample in held_out.samples() {
        words.clear();
        words.extend(sample.tokens().iter().map(|t| t.as_str()));
        tags.clear();
        probs.clear();
        let found = tagger.tag(&words, &mut tags, &mut probs).map_err(|e| Error::Model(e.to_string()))?;
        if !found {
            return Err(Error::Model(format!("no tag sequence scored above the minimum for: {sample}")));
        }
        labels.clear();
        labels.extend(tags.iter().map(|&t| tagset.label(t)));
        let mut evaluator = PosEvaluator::default();
        evaluator.evaluate(sample, &labels)?;
        let (mut scored, mut right, mut right_as_written) = (0u64, 0u64, 0u64);
        for (expected, &given) in sample.tags().iter().zip(&labels) {
            right_as_written += u64::from(expected.as_str() == given);
            if reference.scores(expected.as_str()) {
                scored += 1;
                if scored_as.is_right(reference.mappings, expected.as_str(), given) {
                    right += 1;
                } else {
                    *mismatches.entry((expected.as_str(), given)).or_default() += 1;
                }
            }
        }
        // The tags as written, counted over every token, are what
        // `POSEvaluator` counts.
        debug_assert_eq!(evaluator.word_count(), sample.len() as u64);
        let expected_accuracy = if sample.is_empty() { 0.0 } else { right_as_written as f64 / sample.len() as f64 };
        debug_assert_eq!(expected_accuracy, evaluator.word_accuracy());
        units.push(Counts::accuracy(right, scored));
    }
    Ok(Scores { units, mismatches: most_frequent(mismatches) })
}

/// The [`MISMATCHES_LISTED`] most frequent mismatches, most frequent
/// first, then by held-out value and given value: tags, or a lemmatizer's
/// lemmas ([`crate::ud`]).
pub(crate) fn most_frequent<H: AsRef<str> + Ord, G: AsRef<str> + Ord>(
    mismatches: BTreeMap<(H, G), u64>,
) -> Vec<Mismatch> {
    let mut all: Vec<((H, G), u64)> = mismatches.into_iter().collect();
    // Stable, so equal counts keep the map's order of the values.
    all.sort_by_key(|a| std::cmp::Reverse(a.1));
    all.into_iter()
        .take(MISMATCHES_LISTED)
        .map(|((held_out, given), tokens)| Mismatch {
            held_out: held_out.as_ref().to_owned(),
            given: given.as_ref().to_owned(),
            tokens,
        })
        .collect()
}
