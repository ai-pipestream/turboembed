//! Reduces text to its confusable skeleton (UTS #39).

use turbo_analysis::ViewBuilder;
use turbo_unicode::confusables::prototype;
use turbo_unicode::normalize::{Form, QuickCheck, canonical_combining_class, normalize, quick_check};

use crate::Normalizer;
use crate::fold_runs::{RunFold, Scratch, fold_runs, nfd_first};

/// Writes the skeleton of the text as UTS #39 section 4 defines it: the
/// text in NFD, each character replaced by its prototype from
/// confusables.txt (Unicode 18.0), and the result in NFD again. Text that
/// looks alike has the same skeleton: Cyrillic "раураl" and Latin
/// "paypal" both become "paypal", "m" becomes "rn", the digit 1 and the
/// capital I become "l", U+2026 becomes "...". The skeleton is meant for
/// comparing and indexing, not for reading.
///
/// Each source character whose skeleton differs from it is one edit, so a
/// span of the skeleton maps back to the characters that produced it.
/// Where normalization reorders combining marks across source characters,
/// the base and its marks form one edit; see `fold_runs`. Characters with
/// no prototype that are already in NFD (all of ASCII apart from the eight
/// characters confusables.txt maps) are kept without any work.
///
/// OpenNLP counterpart: `ConfusableSkeletonCharSequenceNormalizer` and
/// `Confusables.skeleton`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ConfusableSkeleton;

impl ConfusableSkeleton {
    /// NFD of `text`, every character replaced by its prototype, NFD of
    /// that.
    fn skeleton_into(text: &str, out: &mut String, s: &mut Scratch) {
        s.decomposed.clear();
        normalize(Form::Nfd, text, &mut s.decomposed);
        s.mapped.clear();
        for c in s.decomposed.chars() {
            match prototype(c) {
                Some(target) => s.mapped.push_str(target),
                None => s.mapped.push(c),
            }
        }
        normalize(Form::Nfd, &s.mapped, out);
    }
}

/// Whether the canonical decomposition of `c` starts with a character of
/// combining class 0, so canonical ordering cannot move anything from
/// before `c` past the start of its decomposition.
fn decomposes_to_starter(c: char) -> bool {
    canonical_combining_class(nfd_first(c)) == 0
}

impl RunFold for ConfusableSkeleton {
    type State = ();

    /// A character whose decomposition starts with a starter that has no
    /// prototype, or whose prototype also decomposes to a starter first.
    /// Both NFD passes then split the text before it.
    fn is_boundary(&self, c: char) -> bool {
        let first = nfd_first(c);
        if canonical_combining_class(first) != 0 {
            return false;
        }
        match prototype(first) {
            None => true,
            Some(target) => target.chars().next().is_some_and(decomposes_to_starter),
        }
    }

    /// A starter in NFD with no prototype.
    fn is_inert(&self, c: char) -> bool {
        if c.is_ascii() {
            return prototype(c).is_none();
        }
        canonical_combining_class(c) == 0 && quick_check(Form::Nfd, c) == QuickCheck::Yes && prototype(c).is_none()
    }

    fn fold_run(&self, run: &str, out: &mut String, s: &mut Scratch) {
        ConfusableSkeleton::skeleton_into(run, out, s);
    }

    fn fold_char(&self, c: char, _: &mut (), out: &mut String, s: &mut Scratch) {
        ConfusableSkeleton::skeleton_into(c.encode_utf8(&mut [0; 4]), out, s);
    }
}

impl Normalizer for ConfusableSkeleton {
    const ID: &'static str = "turbo:normalize:confusable-skeleton";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        fold_runs(self, text, out);
    }
}
