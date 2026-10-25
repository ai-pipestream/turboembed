//! Calendar mentions in text and the date a document is dated with,
//! ported from the temporal package of the OpenNLP numeric add-on.
//!
//! - [`TemporalExtractor`] finds dates (`2026-07-14`, `July 14, 2026`,
//!   `14th July 2026`), months (`July 2026`) and quarters (`Q3 2024`), and,
//!   against a reference date, relative expressions (`yesterday`,
//!   `next month`, `3 weeks ago`), and writes them as the `turbo:temporal`
//!   layer ([`Temporals`]): `CursorTemporalExtractor` and
//!   `TemporalAnnotator`.
//! - [`DocumentDateSelector`] picks the first absolute day as the
//!   document's date, the `turbo:document-date` layer ([`DocumentDate`]):
//!   `DocumentDateAnnotator`.
//!
//! docs/temporal.md describes the rules.

mod clock;
mod select;
mod shapes;
mod words;

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Date, Granularity, Temporals};

pub use select::{DocumentDateSelector, TimesDocumentDateSelector};
pub use turbo_analysis_layers::{DocumentDate, Origin, TemporalMention, TemporalValue};

use crate::scan;

/// Writes `turbo:temporal` over a text view.
///
/// Relative expressions resolve against a reference date. By default it is
/// the first absolute day in the text (its dateline), as
/// `TemporalAnnotator(extractor)` takes it; a text without one gets its
/// absolute mentions only, never a date from the clock.
/// [`TemporalExtractor::with_reference`] fixes the date instead, as
/// `TemporalAnnotator(extractor, reference)` does, for a document whose
/// date is known from metadata; a dateline in the text then does not
/// count.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct TemporalExtractor {
    reference: Option<Date>,
    opennlp_parity: bool,
}

impl TemporalExtractor {
    /// An extractor that takes each text's dateline as its reference.
    pub fn new() -> TemporalExtractor {
        TemporalExtractor::default()
    }

    /// An extractor that resolves relative expressions against `reference`
    /// in every text.
    pub fn with_reference(reference: Date) -> TemporalExtractor {
        TemporalExtractor { reference: Some(reference), opennlp_parity: false }
    }

    /// Configures OpenNLP-compatible parity mode (`false` by default).
    ///
    /// When `false` (default), absolute mentions accept four-digit years from
    /// `0001` to `9999`, relative counts accept up to nine digits (`in 100 days`,
    /// `365 days ago`), `last`/`this`/`next` accept plural units (`next weeks`),
    /// and supplementary-plane letters are treated as letters. When `true`,
    /// matches OpenNLP's `CursorTemporalExtractor` behavior for JVM golden
    /// verification.
    pub fn with_opennlp_parity(mut self, enabled: bool) -> Self {
        self.opennlp_parity = enabled;
        self
    }

    /// Whether OpenNLP-compatible parity mode is enabled.
    pub fn opennlp_parity(&self) -> bool {
        self.opennlp_parity
    }

    /// The fixed reference date, or `None` for the dateline.
    pub fn reference(&self) -> Option<Date> {
        self.reference
    }

    /// Appends the mentions of `text` to `out`, in text order, resolving
    /// relative expressions against the fixed reference or the text's
    /// dateline.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn extract(&self, text: &str, out: &mut Temporals) {
        let reference = self.reference.or_else(|| dateline_with_parity(text, self.opennlp_parity));
        scan_with_parity(text, reference, self.opennlp_parity, out);
    }
}

/// `CursorTemporalExtractor.extract`: appends every mention in `text` to
/// `out`, in text order, with byte offsets into `text`. With a reference
/// date, relative expressions are read and resolved against it; without
/// one they are not read at all. A relative expression whose date falls
/// outside the `LocalDate` range is left out, and the scan goes on.
///
/// After a mention the scan goes on at its end, so mentions never
/// overlap; after a failed candidate, at the next character.
///
/// # Panics
///
/// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
pub fn scan(text: &str, reference: Option<Date>, out: &mut Temporals) {
    scan_with_parity(text, reference, false, out);
}

/// Like [`scan`], with OpenNLP-compatible parity mode configured.
pub fn scan_with_parity(text: &str, reference: Option<Date>, opennlp_parity: bool, out: &mut Temporals) {
    scan_with(text, reference, opennlp_parity, |m| {
        out.push(m);
        true
    });
}

/// The date of the first absolute day mention in `text`: the reference
/// `TemporalAnnotator` elects when it has none of its own. The scan stops
/// there, since the absolute pass is used for nothing else.
pub fn dateline(text: &str) -> Option<Date> {
    dateline_with_parity(text, false)
}

/// Like [`dateline`], with OpenNLP-compatible parity mode configured.
pub fn dateline_with_parity(text: &str, opennlp_parity: bool) -> Option<Date> {
    let mut found = None;
    scan_with(text, None, opennlp_parity, |m| match m.value {
        TemporalValue::Day(date) => {
            found = Some(date);
            false
        }
        _ => true,
    });
    found
}

/// The scan loop, handing each mention to `found` until it returns false.
fn scan_with(
    text: &str,
    reference: Option<Date>,
    opennlp_parity: bool,
    mut found: impl FnMut(TemporalMention) -> bool,
) {
    assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        // Only these bytes can start a mention. Each is ASCII or the first
        // byte of a character, so where one is found is a character
        // boundary, as the Java's step by code points would reach.
        match b[i..].iter().position(|&x| STARTS[usize::from(x)]) {
            Some(skip) => i += skip,
            None => break,
        }
        match shapes::match_at(text, i, reference, opennlp_parity) {
            Some(m) => {
                i = m.span.end as usize;
                if !found(m) {
                    return;
                }
            }
            None => i += scan::char_at(text, i).map_or(1, char::len_utf8),
        }
    }
}

/// The bytes a mention can start with: an ASCII digit, `Q`, the first
/// letter of a month or a relative keyword in either case, and 0xC4, the
/// first byte of U+0130, which lowercases to the `i` of `in` (see
/// words.rs). A word starting with any other letter lowercases to none of
/// those words.
const STARTS: [bool; 256] = {
    let mut t = [false; 256];
    let mut d = b'0';
    while d <= b'9' {
        t[d as usize] = true;
        d += 1;
    }
    let letters = b"adfijlmnoqsty";
    let mut k = 0;
    while k < letters.len() {
        t[letters[k] as usize] = true;
        t[letters[k].to_ascii_uppercase() as usize] = true;
        k += 1;
    }
    t[0xC4] = true;
    t
};

impl Annotator for TemporalExtractor {
    const ID: &'static str = "turbo-numeric:temporal";
    type Reads = (View,);
    type Writes = (Temporals,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Temporals,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let mut temporals = Temporals::new(view.id());
        self.extract(text, &mut temporals);
        Ok((temporals,))
    }

    /// FNV-1a over the reference date and parity flag, fixed so that stored
    /// provenance stays comparable across builds.
    fn config_hash(&self) -> u64 {
        let reference = match self.reference {
            None => [0, 0],
            Some(d) => [1, d.epoch_day() as u64],
        };
        let mut h = reference.iter().fold(turbo_analysis::Fnv1a::new(), |mut h, &x| {
            h.fold_u64(x);
            h
        });
        if self.opennlp_parity {
            h.fold_u64(1);
        }
        h.finish()
    }
}

/// Whether a mention can date a document: an absolute day.
fn is_absolute_day(granularity: Granularity, origin: Origin) -> bool {
    granularity == Granularity::Day && origin == Origin::Absolute
}
