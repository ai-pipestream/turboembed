//! Finds every number in a text: the candidate loop of OpenNLP's
//! `CursorQuantityExtractor.extract` and `matchAt`, without the money
//! exclusion and without the unit after the number.

use turbo_analysis::{Annotator, Error, Fnv1a, Refs, Span, View};
use turbo_analysis_layers::{NumberMention, NumberNotation, Numbers};

use crate::scan::{boundary_before, continues_number, parse, sign_boundary_before};

/// Writes `turbo:numbers` over a text view.
///
/// The notation can be chosen directly, resolved from an ISO 3166-1 alpha-2
/// [`RegionCode`](turbo_analysis_layers::RegionCode) via
/// [`NumberScanner::for_region`] or a BCP-47 locale tag via
/// [`NumberScanner::for_locale`], or selected per document from the
/// [`Regions`](turbo_analysis_layers::Regions) ballot with
/// [`RegionAwareNumberAnnotator`](crate::RegionAwareNumberAnnotator).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumberScanner {
    /// The convention the text groups digits and marks fractions in.
    pub notation: NumberNotation,
    /// Whether scale suffixes (`12k`) and scale words (`3 million`) are
    /// read and multiply the value. When they are, a letter directly after
    /// the digits that is no scale suffix rejects the number (`5x`).
    pub apply_scale: bool,
}

impl Default for NumberScanner {
    fn default() -> NumberScanner {
        NumberScanner { notation: NumberNotation::LatinUs, apply_scale: true }
    }
}

impl NumberScanner {
    /// A scanner for `notation` that reads scale markers.
    pub fn new(notation: NumberNotation) -> NumberScanner {
        NumberScanner { notation, ..NumberScanner::default() }
    }

    /// Appends every number in `text` to `out`, in text order, with byte
    /// offsets into `text`.
    ///
    /// At each character the scan tries a number: an optional minus that
    /// may open a negative number (see [`sign_boundary_before`]), then an
    /// ASCII digit that does not go on with a number before it (see
    /// [`continues_number`]) and a word boundary before the mention. A
    /// number found is added and the scan goes on after it; otherwise it
    /// goes on at the next character.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes, since offsets are `u32`.
    pub fn scan(&self, text: &str, out: &mut Numbers) {
        self.scan_with_fallback(text, None, out);
    }

    /// Scans `text` in `self.notation`, falling back to `fallback` at
    /// candidate positions that fail in `self.notation` when a document's
    /// region ballot carries both conventions.
    pub(crate) fn scan_with_fallback(&self, text: &str, fallback: Option<NumberNotation>, out: &mut Numbers) {
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        let b = text.as_bytes();
        let fallback = fallback.filter(|&n| n != self.notation);
        // One buffer for the digits of every number in the text.
        let mut digits = String::new();
        let mut i = 0;
        while i < b.len() {
            // Only a minus or a digit can open a number, so the characters
            // before the next one would each fail. Neither byte occurs inside
            // a multibyte character, so where it is found is a character
            // boundary, as the Java's step by code points would reach.
            match b[i..].iter().position(|&x| x == b'-' || x.is_ascii_digit()) {
                Some(skip) => i += skip,
                None => break,
            }
            let matched = self
                .match_at(text, i, self.notation, &mut digits, out)
                .or_else(|| fallback.and_then(|alt| self.match_at(text, i, alt, &mut digits, out)));
            match matched {
                Some(end) => i = end,
                // A minus or a digit is one byte.
                None => i += 1,
            }
        }
    }

    /// `CursorQuantityExtractor.matchAt` without the unit: adds the
    /// number that starts at byte `start` in `notation` to `out` and
    /// returns its end.
    fn match_at(
        &self,
        text: &str,
        start: usize,
        notation: NumberNotation,
        digits: &mut String,
        out: &mut Numbers,
    ) -> Option<usize> {
        let b = text.as_bytes();
        let negative = b[start] == b'-';
        let first = start + usize::from(negative);
        if !b.get(first).is_some_and(u8::is_ascii_digit) || continues_number(text, first, notation) {
            return None;
        }
        let boundary = if negative { sign_boundary_before(text, start) } else { boundary_before(text, start) };
        if !boundary {
            return None;
        }
        let r = parse(text, first, self.apply_scale, notation, digits)?;
        out.push(NumberMention {
            span: Span::new(start as u32, r.end as u32),
            numeral_end: r.numeral_end as u32,
            negative,
            coefficient: digits,
            scale: r.scale,
            multiplier: r.multiplier,
            marker: r.marker,
            notation,
        });
        Some(r.end)
    }
}

impl Annotator for NumberScanner {
    const ID: &'static str = "turbo-numeric:numbers";
    type Reads = (View,);
    type Writes = (Numbers,);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Numbers,), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let mut numbers = Numbers::new(view.id());
        self.scan(text, &mut numbers);
        Ok((numbers,))
    }

    /// FNV-1a over the notation and the scale option, fixed so that stored
    /// provenance stays comparable across builds.
    fn config_hash(&self) -> u64 {
        let notation: u64 = match self.notation {
            NumberNotation::LatinUs => 1,
            NumberNotation::LatinEu => 2,
        };
        [notation, u64::from(self.apply_scale)]
            .iter()
            .fold(Fnv1a::new(), |mut h, &x| {
                h.u64(x);
                h
            })
            .finish()
    }
}
