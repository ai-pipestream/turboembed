use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use turbo_eval::model_gate::Counts;
use turbo_formats::evaluate::LemmatizerEvaluator;
use turbo_formats::sample::LemmaSample;
use turbo_lemmatizer::{LemmatizerModel, MaxentLemmatizer, TokenTag, edit_script::decode_lemma};
use turbo_maxent::{JavaString, ModelFile};
use turbo_train_components::lemmatize;
use turbo_zoo::Sha256;

use crate::card::{Mismatch, TagCount};
use crate::error::{Error, Result};
use crate::postag::most_frequent;

/// The lemma classes a reference lemmatizer has outcomes for, which decide
/// the held-out tokens a lemmatizer's gate scores. A token's gold class is
/// the class training gives the token and its held-out lemma
/// ([`lemmatize::lemma_class`], `LemmatizerME.encodeLemmas`). A token is
/// left out of both models' scores when its gold class gives its lemma
/// back, so a lemmatizer trained on the treebank can give it, and no
/// outcome of the reference gives that lemma for the token, so the
/// reference cannot get it right, as a token whose tag a reference tagger
/// cannot give ([`crate::postag::ReferenceTags`]). Lacking the gold class
/// is not enough: `decodeShortestEditScript` deletes a character without
/// reading the letter the class names for it, so classes that differ only
/// in those letters give the same lemma, and a reference without a token's
/// gold class may give its lemma through another class.
///
/// A token whose gold class gives another string is scored, as
/// `LemmatizerEvaluator` counts it: training teaches the candidate the
/// gold classes, so the rule leaves out only tokens it can be taught.
/// `decodeLemmas` applies a class to the lowercased token, and every class
/// is made from a lowercased lemma, so no model gives a lemma with a
/// capital letter (MarkIT writes `[PUNCT]` as the lemma of most
/// punctuation); and it reads a class's index as one digit, so a gold
/// class with an index of ten or more gives another string
/// (`turbo_lemmatizer`'s documentation). Making classes from the lemma as
/// written and applying them to the token as written would let a model
/// give capitalised lemmas, an improvement over OpenNLP that the recipes
/// leave out so that they train and score as OpenNLP does.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceClasses {
    classes: HashSet<JavaString>,
}

impl ReferenceClasses {
    /// The classes `outcomes`.
    pub fn new(outcomes: impl IntoIterator<Item = JavaString>) -> ReferenceClasses {
        ReferenceClasses { classes: outcomes.into_iter().collect() }
    }

    /// The outcomes of the lemmatizer in `bytes`.
    ///
    /// # Errors
    ///
    /// A model file that is not a lemmatizer model.
    pub fn of_model(bytes: &[u8]) -> Result<ReferenceClasses> {
        Ok(ReferenceClasses::new(open_lemmatizer(bytes)?.maxent().outcomes().cloned()))
    }

    /// Whether a held-out token `token` of the lemma `lemma` is scored:
    /// unless its gold class gives the lemma and no class of the reference
    /// does.
    pub fn scores(&self, token: &str, lemma: &str) -> bool {
        let gives = |class: &JavaString| decode_lemma(token, class).is_ok_and(|given| given == lemma);
        let class = lemmatize::lemma_class(token, lemma);
        self.classes.contains(&class) || !gives(&class) || self.classes.iter().any(gives)
    }

    /// The held-out tokens left out of the scores, by their tag as
    /// written, sorted by tag.
    pub fn left_out(&self, samples: &[LemmaSample<'_>]) -> Vec<TagCount> {
        let mut counts: BTreeMap<&str, u64> = BTreeMap::new();
        for s in samples {
            for ((token, tag), lemma) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()) {
                if !self.scores(token.as_str(), lemma.as_str()) {
                    *counts.entry(tag.as_str()).or_default() += 1;
                }
            }
        }
        counts.into_iter().map(|(tag, tokens)| TagCount { tag: tag.to_owned(), tokens }).collect()
    }
}

/// The tokens of lemma samples.
pub(super) fn lemma_tokens(samples: &[LemmaSample<'_>]) -> u64 {
    samples.iter().map(|s| s.len() as u64).sum()
}

/// The SHA-256 of lemma samples as scored with a reference of the classes
/// `reference`: for each sample, its number of tokens as a little endian
/// `u64`, then for each token its text, its tag and its lemma, each as its
/// length in bytes as a little endian `u64` and the bytes, and a byte, 1
/// when the token is scored ([`ReferenceClasses::scores`]) and 0 when it
/// is left out. Two gates whose references leave out other tokens have
/// other digests.
pub fn lemma_samples_sha256(samples: &[LemmaSample<'_>], reference: &ReferenceClasses) -> String {
    let mut bytes = Vec::new();
    let put = |bytes: &mut Vec<u8>, s: &str| {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
    };
    for s in samples {
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        for ((token, tag), lemma) in s.tokens().iter().zip(s.tags()).zip(s.lemmas()) {
            put(&mut bytes, token.as_str());
            put(&mut bytes, tag.as_str());
            put(&mut bytes, lemma.as_str());
            bytes.push(u8::from(reference.scores(token.as_str(), lemma.as_str())));
        }
    }
    Sha256::of(&bytes).to_string()
}

/// The lemmatizer in `bytes`.
fn open_lemmatizer(bytes: &[u8]) -> Result<LemmatizerModel> {
    let file = ModelFile::open(bytes.to_vec()).map_err(|e| Error::Model(e.to_string()))?;
    LemmatizerModel::open(file).map_err(|e| Error::Model(e.to_string()))
}

/// Each sample's counts for the lemmatizer in `bytes`, given the
/// sample's tags: the tokens given the sample's lemma, of its tokens that
/// `classes` scores ([`ReferenceClasses::scores`]). The
/// lemmas are those `LemmatizerME.lemmatize` gives, which
/// `LemmatizerEvaluator` compares with the sample's: `decodeLemmas` applies
/// each class to the lowercased token, so a capitalised held-out lemma is
/// never given.
///
/// # Errors
///
/// A model file that is not a lemmatizer model, a sentence no class
/// sequence scores above the search's minimum for, a class the decoder
/// cannot read, or what the evaluator throws on.
pub fn score_lemmas(bytes: &[u8], samples: &[LemmaSample<'_>], classes: &ReferenceClasses) -> Result<Vec<Counts>> {
    Ok(lemma_scores(bytes, samples, classes)?.units)
}

/// A lemmatizer's counts on the held-out samples, and its most frequent
/// mismatches on the scored tokens: the held-out lemma and the lemma
/// given.
pub(super) struct LemmaScores {
    pub(super) units: Vec<Counts>,
    /// At most [`crate::postag::MISMATCHES_LISTED`], most frequent first.
    pub(super) mismatches: Vec<Mismatch>,
}

/// [`score_lemmas`], with the mismatches.
pub(super) fn lemma_scores<'s>(
    bytes: &[u8],
    samples: &'s [LemmaSample<'_>],
    classes: &ReferenceClasses,
) -> Result<LemmaScores> {
    let lemmatizer = MaxentLemmatizer::new(Arc::new(open_lemmatizer(bytes)?));
    let mut units = Vec::with_capacity(samples.len());
    let mut mismatches: BTreeMap<(&'s str, String), u64> = BTreeMap::new();
    let (mut predicted_classes, mut probs) = (Vec::new(), Vec::new());
    let mut words: Vec<TokenTag<'_>> = Vec::new();
    let mut lemmas: Vec<String> = Vec::new();
    for sample in samples {
        words.clear();
        words.extend(
            sample.tokens().iter().zip(sample.tags()).map(|(t, g)| TokenTag { token: t.as_str(), tag: g.as_str() }),
        );
        predicted_classes.clear();
        probs.clear();
        if !lemmatizer.classes(&words, &mut predicted_classes, &mut probs) {
            return Err(Error::Model(format!("no lemma class sequence scored above the minimum for: {sample:?}")));
        }
        lemmas.clear();
        for (w, class) in words.iter().zip(&predicted_classes) {
            lemmas.push(decode_lemma(w.token, class).map_err(|e| Error::Model(format!("{:?}: {e}", w.token)))?);
        }
        let predicted: Vec<&str> = lemmas.iter().map(String::as_str).collect();
        let mut evaluator = LemmatizerEvaluator::default();
        evaluator.evaluate(sample, &predicted)?;
        let (mut scored, mut right, mut right_as_written) = (0u64, 0u64, 0u64);
        for ((token, expected), &given) in sample.tokens().iter().zip(sample.lemmas()).zip(&predicted) {
            let expected: &'s str = expected.as_str();
            right_as_written += u64::from(expected == given);
            if classes.scores(token.as_str(), expected) {
                scored += 1;
                if expected == given {
                    right += 1;
                } else {
                    *mismatches.entry((expected, given.to_owned())).or_default() += 1;
                }
            }
        }
        // The lemmas counted over every token are what
        // `LemmatizerEvaluator` counts.
        debug_assert_eq!(evaluator.word_count(), sample.len() as u64);
        let expected_accuracy = if sample.is_empty() { 0.0 } else { right_as_written as f64 / sample.len() as f64 };
        debug_assert_eq!(expected_accuracy, evaluator.word_accuracy());
        units.push(Counts::accuracy(right, scored));
    }
    Ok(LemmaScores { units, mismatches: most_frequent(mismatches) })
}
