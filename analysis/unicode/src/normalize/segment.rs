//! Splitting text into runs and the public normalization functions.

use std::ops::Range;

use super::run::RunNormalizer;
use super::{Form, QuickCheck, data, quick_check_of};
use crate::ascii::ascii_prefix_len;
use crate::tables::normalize::CCC_MASK;

/// Appends `text` in normalization form `form` to `out`.
pub fn normalize(form: Form, text: &str, out: &mut String) {
    out.reserve(text.len());
    for_each_run(form, text, |range, normalized| match normalized {
        Some(normalized) => out.push_str(normalized),
        None => out.push_str(&text[range]),
    });
}

/// Whether `text` is in normalization form `form`. Uses the quick check
/// algorithm of UAX #15 and normalizes only when the quick check answers
/// Maybe.
pub fn is_normalized(form: Form, text: &str) -> bool {
    match quick_check_text(form, text) {
        QuickCheck::Yes => true,
        QuickCheck::No => false,
        QuickCheck::Maybe => {
            let mut changed = false;
            for_each_run(form, text, |_, normalized| changed |= normalized.is_some());
            !changed
        }
    }
}

/// The quick check of a whole text: No when a character has quick check
/// No or a combining mark is out of canonical order, otherwise Maybe when a
/// character has quick check Maybe, otherwise Yes.
fn quick_check_text(form: Form, text: &str) -> QuickCheck {
    let mut result = QuickCheck::Yes;
    let mut last_class = 0;
    for c in text.chars() {
        if c.is_ascii() {
            last_class = 0;
            continue;
        }
        let packed = data(c);
        let class = (packed & CCC_MASK) as u8;
        if class != 0 && last_class > class {
            return QuickCheck::No;
        }
        match quick_check_of(form, packed) {
            QuickCheck::Yes => {}
            QuickCheck::No => return QuickCheck::No,
            QuickCheck::Maybe => result = QuickCheck::Maybe,
        }
        last_class = class;
    }
    result
}

/// Calls `f` for consecutive ranges of `text` that cover it from start to
/// end. A range with `None` is already in form `form`; a range with
/// `Some(normalized)` is one run that normalizes to a different string.
///
/// A run starts at each character with combining class 0 and quick check
/// Yes for the form (and at the start of the text), and normalizes
/// independently of the text around it. Consecutive runs that are already
/// normalized are reported as one range, so text that is already in the
/// form gives a single call. Empty text gives no call.
pub fn for_each_run(form: Form, text: &str, mut f: impl FnMut(Range<usize>, Option<&str>)) {
    let mut runs = Runs { form, text, unchanged_from: 0, normalizer: RunNormalizer::default() };
    let bytes = text.as_bytes();
    let mut run_start = 0;
    let mut needs_work = false;
    let mut last_class = 0u8;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii() {
            // An ASCII character is a starter with quick check Yes in every
            // form, so each one starts a run that needs no work. Only the
            // last one of a stretch can start a run that continues with
            // combining marks.
            if needs_work {
                runs.finish(run_start..i, &mut f);
                needs_work = false;
            }
            let stretch = ascii_prefix_len(&bytes[i..]);
            i += stretch;
            run_start = i - 1;
            last_class = 0;
            continue;
        }
        let c = text[i..].chars().next().expect("i is at a character boundary");
        let packed = data(c);
        let class = (packed & CCC_MASK) as u8;
        let check = quick_check_of(form, packed);
        if class == 0 && check == QuickCheck::Yes {
            if needs_work {
                runs.finish(run_start..i, &mut f);
                needs_work = false;
            }
            run_start = i;
        } else if check != QuickCheck::Yes || (class != 0 && last_class > class) {
            needs_work = true;
        }
        last_class = class;
        i += c.len_utf8();
    }
    if needs_work {
        runs.finish(run_start..bytes.len(), &mut f);
    }
    if runs.unchanged_from < bytes.len() {
        f(runs.unchanged_from..bytes.len(), None);
    }
}

/// The state of [`for_each_run`] between runs.
struct Runs<'t> {
    form: Form,
    text: &'t str,
    /// The start of the text already in the form that has not been reported.
    unchanged_from: usize,
    normalizer: RunNormalizer,
}

impl Runs<'_> {
    /// Normalizes a run that failed the quick check and reports it when
    /// normalizing changes it, after the unchanged text before it.
    fn finish(&mut self, run: Range<usize>, f: &mut impl FnMut(Range<usize>, Option<&str>)) {
        let original = &self.text[run.clone()];
        self.normalizer.normalize(self.form, original);
        if self.normalizer.out == original {
            return;
        }
        if self.unchanged_from < run.start {
            f(self.unchanged_from..run.start, None);
        }
        self.unchanged_from = run.end;
        f(run, Some(&self.normalizer.out));
    }
}
