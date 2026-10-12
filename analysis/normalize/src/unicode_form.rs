//! The Unicode normalization forms (UAX #15) as a view: NFC, NFD, NFKC and
//! NFKD.

use turbo_analysis::ViewBuilder;
use turbo_unicode::normalize::{canonical_combining_class, for_each_run};

use crate::{Normalizer, hash_config};

/// The normalization forms, re-exported so callers of [`UnicodeForm::new`]
/// need not depend on `turbo-unicode`.
pub use turbo_unicode::normalize::Form;

/// Puts the text in one of the four Unicode normalization forms.
///
/// The text is split where `turbo_unicode::normalize::for_each_run` splits
/// it: before each character with combining class 0 whose quick check
/// value for the form is Yes. Nothing before such a point reorders or
/// composes with anything after it, so each run normalizes on its own.
/// A run the form leaves alone is kept; a run it changes is replaced by
/// its normalized text in one `replace`, except that a starter at the
/// start of the run that comes out unchanged is kept (see
/// `skip_kept_starter`). The alignment therefore maps each changed run
/// to its exact source bytes, which is as fine as the
/// forms allow: "e" U+0301 composes to U+00E9, and no part of U+00E9 comes
/// from only one of the two source characters.
///
/// Text already in the form (all ASCII text, for every form) gives no
/// edit at all and costs one scan with the quick check.
///
/// OpenNLP counterparts: `NfcCharSequenceNormalizer` and
/// `NfkcCharSequenceNormalizer`, which call `java.text.Normalizer` and do
/// not report an alignment. NFD and NFKD have no OpenNLP normalizer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct UnicodeForm {
    form: Form,
}

impl UnicodeForm {
    pub fn new(form: Form) -> UnicodeForm {
        UnicodeForm { form }
    }

    /// Canonical decomposition followed by canonical composition.
    pub fn nfc() -> UnicodeForm {
        UnicodeForm::new(Form::Nfc)
    }

    /// Compatibility decomposition followed by canonical composition.
    pub fn nfkc() -> UnicodeForm {
        UnicodeForm::new(Form::Nfkc)
    }

    /// Canonical decomposition.
    pub fn nfd() -> UnicodeForm {
        UnicodeForm::new(Form::Nfd)
    }

    /// Compatibility decomposition.
    pub fn nfkd() -> UnicodeForm {
        UnicodeForm::new(Form::Nfkd)
    }

    pub fn form(&self) -> Form {
        self.form
    }
}

impl Normalizer for UnicodeForm {
    const ID: &'static str = "turbo:normalize:unicode-form";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        // Unchanged ranges need no call: the next replace (or `finish`)
        // copies them.
        for_each_run(self.form, text, |range, normalized| {
            if let Some(normalized) = normalized {
                let (start, normalized) = skip_kept_starter(text, range.start, normalized);
                out.keep(start);
                out.replace(range.end, normalized);
            }
        });
    }

    fn config_hash(&self) -> u64 {
        hash_config(&self.form)
    }
}

/// Leaves the first character of a changed run out of its edit when the
/// form keeps it as it is.
///
/// A run starts at the starter before the characters that fail the quick
/// check, because those may compose with it. Often they do not: "x"
/// before a Hangul syllable under NFD, or before U+2126 OHM SIGN under
/// NFC, comes out unchanged. When the normalized run begins with the same
/// starter, nothing composed with it, and the rest of the run normalizes
/// to the rest of the output on its own (the starter has combining class
/// 0, so no mark moves across it), so the starter is kept and only the
/// rest is replaced. Returns where the edit starts and what it writes.
fn skip_kept_starter<'n>(text: &str, start: usize, normalized: &'n str) -> (usize, &'n str) {
    let Some(first) = text[start..].chars().next() else { return (start, normalized) };
    let kept = (first.is_ascii() || canonical_combining_class(first) == 0)
        && normalized.starts_with(first)
        && normalized.len() > first.len_utf8();
    if kept { (start + first.len_utf8(), &normalized[first.len_utf8()..]) } else { (start, normalized) }
}
