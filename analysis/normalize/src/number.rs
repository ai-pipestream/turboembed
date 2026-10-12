//! `Numbers`: OpenNLP's `NumberCharSequenceNormalizer`.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_normalize_layers::{PlaceholderKind, Placeholders};

use crate::Normalizer;
use crate::placeholder::{self, Sink, span};

/// Replaces each maximal run of ASCII digits `0-9` with one U+0020:
/// `12.34` becomes `" . "`. Digits of other scripts (Arabic-Indic,
/// Bengali, mathematical bold) and other numerals (circled digits) are
/// left alone; [`Digits`](crate::digits::Digits) is the normalizer that
/// folds those to ASCII.
///
/// OpenNLP counterpart: `NumberCharSequenceNormalizer`, whose digit class
/// is ASCII only.
///
/// As an `Annotator` it reads a `View` and writes the normalized `View`
/// and a `Placeholders` layer with one `Number` row per run.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Numbers;

impl Numbers {
    /// Writes `text` with every run of ASCII digits replaced by one space,
    /// and reports each run to `sink`.
    pub fn scan(text: &str, out: &mut ViewBuilder<'_>, sink: &mut Sink<'_>) {
        let bytes = text.as_bytes();
        let mut i = 0;
        while let Some(k) = bytes[i..].iter().position(u8::is_ascii_digit) {
            let start = i + k;
            let end = start + bytes[start..].iter().take_while(|b| b.is_ascii_digit()).count();
            out.keep(start);
            out.replace(end, " ");
            sink(span(start, end), PlaceholderKind::Number);
            i = end;
        }
    }
}

impl Normalizer for Numbers {
    const ID: &'static str = "turbo:normalize:numbers";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        Numbers::scan(text, out, &mut placeholder::discard);
    }
}

impl Annotator for Numbers {
    const ID: &'static str = <Numbers as Normalizer>::ID;
    type Reads = (View,);
    type Writes = (View, Placeholders);

    fn annotate(&self, (src,): Refs<'_, Self>) -> Result<(View, Placeholders), Error> {
        Ok(placeholder::annotate(src, Numbers::scan))
    }
}
