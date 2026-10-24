//! Noise scoring: stretches of a view that look like OCR damage,
//! gibberish or encoded binary data, reported as a `Noise` layer with a
//! severity and a score. The text is not changed.
//!
//! The scorer is a port of OpenNLP's `StructuralNoiseScorer` (the noise
//! add-on). It splits the text at `White_Space`, trims surrounding
//! punctuation from each token and scores the ASCII cores of three or
//! more characters: binary-ish when the core looks like base64, gibberish
//! or damaged by how many structural signals it raises, misspelled when a
//! dictionary is given and one OCR confusion turns the core into a word.
//! Findings separated only by whitespace are joined. Placeholder spans
//! (URLs, e-mail addresses, encoded assets a normalizer found) are
//! excluded when the layer is present. `NoiseStats` measures a whole
//! document in the manner of the Gopher and C4 quality filters.

mod exclusions;
mod stats;
mod token;

use std::collections::{BTreeSet, HashSet};
use std::sync::Arc;

use turbo_analysis::{Annotator, Error, Refs, Span, View, ViewId};
use turbo_normalize_layers::{Noise, NoiseSeverity, Placeholders};
use turbo_unicode::props;

use exclusions::Exclusions;
pub use stats::NoiseStats;

/// A word list the scorer asks about lowercase ASCII words. None ships
/// with this crate.
///
/// Every implementation provides a content-sensitive [`Dictionary::config_hash`]
/// so distinct lexicons produce distinct provenance and normalizer cache keys.
/// Stateful closure predicates pair their closure with an explicit identity key
/// (`(id, |w: &str| ...)` or [`Dehyphenate::with_identified_lexicon`](crate::Dehyphenate::with_identified_lexicon) /
/// [`NoiseScorer::with_identified_dictionary`]).
pub trait Dictionary: Send + Sync {
    /// Whether `word`, in lowercase, is a known word.
    fn contains(&self, word: &str) -> bool;

    /// A content fingerprint of this dictionary for annotator and normalizer cache keys.
    fn config_hash(&self) -> u64;
}

impl<I: std::hash::Hash + Send + Sync, F: Fn(&str) -> bool + Send + Sync> Dictionary for (I, F) {
    fn contains(&self, word: &str) -> bool {
        (self.1)(word)
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(&(&self.0, std::any::type_name::<F>()))
    }
}

impl Dictionary for HashSet<String> {
    fn contains(&self, word: &str) -> bool {
        HashSet::contains(self, word)
    }

    fn config_hash(&self) -> u64 {
        let mut words: Vec<&str> = self.iter().map(String::as_str).collect();
        words.sort_unstable();
        crate::hash_config(&words)
    }
}

impl Dictionary for HashSet<&'static str> {
    fn contains(&self, word: &str) -> bool {
        HashSet::contains(self, word)
    }

    fn config_hash(&self) -> u64 {
        let mut words: Vec<&str> = self.iter().copied().collect();
        words.sort_unstable();
        crate::hash_config(&words)
    }
}

impl Dictionary for BTreeSet<String> {
    fn contains(&self, word: &str) -> bool {
        BTreeSet::contains(self, word)
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(self)
    }
}

impl Dictionary for BTreeSet<&'static str> {
    fn contains(&self, word: &str) -> bool {
        BTreeSet::contains(self, word)
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(self)
    }
}

impl<D: Dictionary + ?Sized> Dictionary for Arc<D> {
    fn contains(&self, word: &str) -> bool {
        (**self).contains(word)
    }

    fn config_hash(&self) -> u64 {
        (**self).config_hash()
    }
}

impl Dictionary for turbo_hunspell::Dictionary {
    fn contains(&self, word: &str) -> bool {
        !turbo_hunspell::Stemmer::new(self).stems(word).is_empty()
    }

    fn config_hash(&self) -> u64 {
        self.fingerprint()
    }
}

impl Dictionary for turbo_morfologik::MorfologikDictionary {
    fn contains(&self, word: &str) -> bool {
        !self.lookup(word).is_empty()
    }

    fn config_hash(&self) -> u64 {
        let m = self.metadata();
        crate::hash_config(&(self.model_ref().sha256.as_str(), m.separator, format!("{m:?}")))
    }
}

impl Dictionary for turbo_morfologik::MorfologikTagDictionary {
    fn contains(&self, word: &str) -> bool {
        self.tags(word).is_some()
    }

    fn config_hash(&self) -> u64 {
        crate::hash_config(&(self.dictionary().config_hash(), self.is_case_sensitive()))
    }
}

/// One finding: where, how bad, and how strong within its severity.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NoiseFinding {
    pub span: Span,
    pub severity: NoiseSeverity,
    /// In (0, 1]. Scores of different severities are not comparable.
    pub score: f64,
}

/// An exclusion that does not fit the text it is for.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExclusionOutOfBounds {
    pub span: Span,
    pub text_len: usize,
}

impl std::fmt::Display for ExclusionOutOfBounds {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "exclusion {}..{} past the end of a {}-byte text", self.span.start, self.span.end, self.text_len)
    }
}

impl std::error::Error for ExclusionOutOfBounds {}

/// Scores noise in a text. Without a dictionary the misspelled tier is
/// never reported, as in OpenNLP. It holds its dictionary and nothing
/// else, and is safe to share between threads.
#[derive(Clone, Default)]
pub struct NoiseScorer {
    dictionary: Option<Arc<dyn Dictionary>>,
}

impl std::fmt::Debug for NoiseScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NoiseScorer").field("dictionary", &self.dictionary.is_some()).finish()
    }
}

impl NoiseScorer {
    pub fn new() -> NoiseScorer {
        NoiseScorer::default()
    }

    /// A scorer that leaves dictionary words alone and reports cores one
    /// OCR confusion away from a dictionary word as misspelled.
    pub fn with_dictionary(dictionary: impl Dictionary + 'static) -> NoiseScorer {
        NoiseScorer { dictionary: Some(Arc::new(dictionary)) }
    }

    /// A scorer backed by a closure or custom lookup paired with an explicit
    /// content identity key `id` for [`Annotator::config_hash`].
    pub fn with_identified_dictionary(
        id: impl std::hash::Hash,
        lookup: impl Fn(&str) -> bool + Send + Sync + 'static,
    ) -> NoiseScorer {
        Self::with_dictionary((crate::hash_config(&id), lookup))
    }

    /// The findings in `text`, leaving out every token that overlaps a
    /// non-empty span of `exclude` (spans in any order, overlapping or
    /// not). Findings are in text order and do not overlap an exclusion.
    pub fn score(&self, text: &str, exclude: &[Span]) -> Result<Vec<NoiseFinding>, ExclusionOutOfBounds> {
        if let Some(&span) = exclude.iter().find(|s| s.end as usize > text.len()) {
            return Err(ExclusionOutOfBounds { span, text_len: text.len() });
        }
        let mut found = Vec::new();
        self.scan(text, &Exclusions::new(exclude.iter().copied()), &mut |f| found.push(f));
        Ok(found)
    }

    /// Document statistics of `text`.
    pub fn stats(&self, text: &str) -> NoiseStats {
        NoiseStats::of(text)
    }

    /// The findings as a layer over `view`.
    fn layer(&self, text: &str, view: ViewId, exclude: &Exclusions) -> Noise {
        let mut layer = Noise::new(view);
        self.scan(text, exclude, &mut |f| {
            layer.spans.push(f.span);
            layer.severity.push(f.severity);
            layer.score.push(f.score);
        });
        layer
    }

    /// Scores each token and passes the findings on, joining a finding to
    /// the one before when only whitespace that is not excluded lies
    /// between them.
    fn scan(&self, text: &str, exclude: &Exclusions, emit: &mut dyn FnMut(NoiseFinding)) {
        let dictionary = self.dictionary.as_deref();
        let mut lower = String::new();
        let mut pending: Option<NoiseFinding> = None;
        for (start, end) in tokens(text) {
            if exclude.overlaps(start, end) {
                continue;
            }
            let (from, to) = token::core(&text[start..end]);
            let Some((severity, score)) = token::score(&text[start + from..start + to], dictionary, &mut lower) else {
                continue;
            };
            let next = NoiseFinding { span: Span::new((start + from) as u32, (start + to) as u32), severity, score };
            pending = Some(match pending {
                Some(previous) if joins(text, exclude, previous.span, next.span) => join(previous, next),
                Some(previous) => {
                    emit(previous);
                    next
                }
                None => next,
            });
        }
        if let Some(last) = pending {
            emit(last);
        }
    }
}

/// The `White_Space`-delimited tokens of `text` as byte ranges.
fn tokens(text: &str) -> impl Iterator<Item = (usize, usize)> + '_ {
    text.split(props::white_space).filter(|t| !t.is_empty()).map(move |t| {
        let start = t.as_ptr() as usize - text.as_ptr() as usize;
        (start, start + t.len())
    })
}

/// Whether two findings are separated only by whitespace and nothing
/// between them is excluded.
fn joins(text: &str, exclude: &Exclusions, a: Span, b: Span) -> bool {
    let gap = a.end as usize..b.start as usize;
    text[gap.clone()].chars().all(props::white_space) && !exclude.overlaps(gap.start, gap.end)
}

/// Two joined findings: the worse severity, with the larger score when
/// both have it and the worse one's own score otherwise.
fn join(a: NoiseFinding, b: NoiseFinding) -> NoiseFinding {
    let span = Span::new(a.span.start, b.span.end);
    let (severity, score) = if a.severity == b.severity {
        (a.severity, a.score.max(b.score))
    } else if rank(a.severity) >= rank(b.severity) {
        (a.severity, a.score)
    } else {
        (b.severity, b.score)
    };
    NoiseFinding { span, severity, score }
}

/// OpenNLP's order of the tiers, higher is worse.
fn rank(s: NoiseSeverity) -> u8 {
    match s {
        NoiseSeverity::Misspelled => 1,
        NoiseSeverity::Damaged => 2,
        NoiseSeverity::Gibberish => 3,
        NoiseSeverity::Binaryish => 4,
    }
}

impl Annotator for NoiseScorer {
    const ID: &'static str = "turbo:noise";
    type Reads = (View, Option<Placeholders>);
    type Writes = (Noise,);

    /// Scores the view, leaving out the placeholder spans when the
    /// document has them. The placeholders must be in the view read.
    fn annotate(&self, (view, placeholders): Refs<'_, Self>) -> Result<(Noise,), Error> {
        let exclude = match placeholders {
            Some(p) if p.spans.view() != view.id() => {
                return Err(Error::Annotator {
                    annotator: Self::ID,
                    detail: format!(
                        "the placeholders are in view {}, the text read is view {}",
                        p.spans.view().0,
                        view.id().0
                    ),
                });
            }
            Some(p) => Exclusions::new(p.spans.iter().map(|(_, s)| s)),
            None => Exclusions::new([]),
        };
        Ok((self.layer(view.text(), view.id(), &exclude),))
    }

    /// Records the dictionary's configuration fingerprint when one is set.
    fn config_hash(&self) -> u64 {
        crate::hash_config(&self.dictionary.as_ref().map(|d| d.config_hash()))
    }
}
