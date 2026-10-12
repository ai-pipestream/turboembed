//! Text normalizers: steps that rewrite a document's text into a new view
//! while recording where every byte of the new text came from.
//!
//! A normalizer never changes the source text. It writes a `View` through
//! a `ViewBuilder`, which records an `Alignment` as the text is written,
//! so every span found in the normalized view maps back to exact source
//! bytes and every source span maps forward. The normalizers port what
//! Apache OpenNLP 3.x has in `opennlp.tools.util.normalizer` and in its
//! add-ons, and keep the same output text where OpenNLP has a reference
//! implementation, with the differences stated in `docs/normalizers.md`.
//!
//! Each normalizer is a small type that implements [`Normalizer`]. It runs
//! three ways:
//!
//! - as a pipeline step, through [`NormalizeStep`], which is an
//!   `Annotator` reading one `View` and writing another;
//! - as a plain function over a string, through [`normalize`] or
//!   [`apply`], for callers that need the text alone (the language
//!   detector runs five of them before n-gramming);
//! - inside a [`Chain`], which runs several normalizers in order and
//!   composes their alignments into one.
//!
//! The crate depends on `turbo-unicode` for Unicode 17.0 character data
//! and on nothing else. docs/normalizers.md describes every normalizer,
//! its OpenNLP counterpart and the tests that hold them together.

use std::borrow::Cow;
use std::hash::{DefaultHasher, Hash, Hasher};

use turbo_analysis::{Alignment, Annotator, Error, Refs, Span, View, ViewBuilder};

pub mod chain;

pub use chain::Chain;

/// One text normalizer. Its only job is to copy `text` into `out`,
/// keeping what it leaves alone and replacing what it changes, so the
/// alignment is recorded as a side effect of writing the text.
///
/// `ID` names the normalizer in the provenance of every view it writes.
/// Ids are `turbo:normalize:<name>`. `config_hash` distinguishes two
/// configurations of one normalizer (whitespace with and without
/// paragraph breaks); a normalizer with no options leaves it at zero.
pub trait Normalizer: Send + Sync + 'static {
    const ID: &'static str;
    const VERSION: &'static str = "0";

    /// Writes the normalized form of `text` through `out`. The builder
    /// starts at offset 0 of `text` and `finish` copies whatever is left,
    /// so a normalizer only has to call `keep` and `replace` around the
    /// stretches it changes.
    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>);

    fn config_hash(&self) -> u64 {
        0
    }
}

/// Normalizes `text` on its own, with the alignment.
pub fn normalize<N: Normalizer>(n: &N, text: &str) -> (String, Alignment) {
    let mut out = ViewBuilder::new(text);
    n.normalize(text, &mut out);
    out.finish()
}

/// Normalizes `text` and returns only the text: the input itself when the
/// normalizer changed nothing, a new string otherwise.
pub fn apply<'a, N: Normalizer>(n: &N, text: &'a str) -> Cow<'a, str> {
    let (out, alignment) = normalize(n, text);
    if alignment.edits().is_empty() { Cow::Borrowed(text) } else { Cow::Owned(out) }
}

/// A normalizer as a pipeline step: reads a `View`, writes a `View`. Which
/// instances it reads and writes is bound where the step is added
/// (`Bound::new(NormalizeStep::new(Nfkc)).write(0, "nfkc")`).
pub struct NormalizeStep<N> {
    normalizer: N,
}

impl<N: Normalizer> NormalizeStep<N> {
    pub fn new(normalizer: N) -> NormalizeStep<N> {
        NormalizeStep { normalizer }
    }

    pub fn normalizer(&self) -> &N {
        &self.normalizer
    }
}

impl<N: Normalizer> Annotator for NormalizeStep<N> {
    const ID: &'static str = N::ID;
    const VERSION: &'static str = N::VERSION;
    type Reads = (View,);
    type Writes = (View,);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View,), Error> {
        let (text, alignment) = normalize(&self.normalizer, src.text());
        Ok((View::derived(src, text, alignment),))
    }

    fn config_hash(&self) -> u64 {
        self.normalizer.config_hash()
    }
}

/// A normalizer whose type is erased, for chains assembled at run time.
/// Every `Normalizer` is one.
pub trait DynNormalizer: Send + Sync {
    fn id(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn normalize_dyn(&self, text: &str, out: &mut ViewBuilder<'_>);
    fn config_hash_dyn(&self) -> u64;
}

impl<N: Normalizer> DynNormalizer for N {
    fn id(&self) -> &'static str {
        N::ID
    }

    fn version(&self) -> &'static str {
        N::VERSION
    }

    fn normalize_dyn(&self, text: &str, out: &mut ViewBuilder<'_>) {
        self.normalize(text, out)
    }

    fn config_hash_dyn(&self) -> u64 {
        self.config_hash()
    }
}

/// Hashes a normalizer's options for `config_hash`, the same way every
/// normalizer in this crate does it.
pub fn hash_config<T: Hash>(options: &T) -> u64 {
    let mut h = DefaultHasher::new();
    options.hash(&mut h);
    h.finish()
}

/// Writes `child` into `out` as the result of two stages, `parent` to a
/// middle text by `first` and the middle text to `child` by `second`, so
/// that the alignment `out` records maps `child` straight to `parent`.
///
/// Every stretch of the parent that either stage touched becomes one
/// edit. The stretches are the parent spans of the first stage's edits
/// and the parent spans the second stage's edits map to through the
/// first; overlapping stretches merge. The edges of a merged stretch lie
/// outside every edit of both stages, so mapping an edge through both
/// alignments is exact, and the child text between two edges is what the
/// stretch became.
pub fn compose(parent: &str, first: &Alignment, child: &str, second: &Alignment, out: &mut ViewBuilder<'_>) {
    let mut stretches: Vec<Span> = Vec::with_capacity(first.edits().len() + second.edits().len());
    let mut a = first.edits().iter().map(|e| e.parent).peekable();
    let mut b = second.edits().iter().map(|e| first.span_to_parent(e.parent)).peekable();
    // Merge the two sorted lists of parent spans into one sorted list.
    loop {
        let next = match (a.peek(), b.peek()) {
            (Some(x), Some(y)) => {
                if x.start <= y.start {
                    a.next()
                } else {
                    b.next()
                }
            }
            (Some(_), None) => a.next(),
            (None, Some(_)) => b.next(),
            (None, None) => None,
        };
        let Some(s) = next else { break };
        match stretches.last_mut() {
            // Overlapping stretches merge. Touching ones merge too: an
            // insertion at the edge of an edit belongs with that edit.
            Some(last) if s.start <= last.end => last.end = last.end.max(s.end),
            _ => stretches.push(s),
        }
    }
    for s in stretches {
        let mid = first.span_to_child(s);
        let c = second.span_to_child(mid);
        out.keep(s.start as usize);
        out.replace(s.end as usize, &child[c.range()]);
    }
    debug_assert!(out.position() <= parent.len());
}
