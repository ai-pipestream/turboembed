//! Finds every number in a text: the candidate loop of OpenNLP's
//! `CursorQuantityExtractor.extract` and `matchAt`, without the money
//! exclusion and without the unit after the number.

use turbo_analysis::{Annotator, Error, Refs, Span, View};
use turbo_analysis_layers::{NumberMention, NumberNotation, Numbers};

use crate::scan::{boundary_before, continues_number, parse, sign_boundary_before};

/// Writes `turbo:numbers` over a text view.
///
/// The notation is the caller's choice. OpenNLP can also pick it from a
/// `java.util.Locale` (`NumberNotation.forLocale`, which reads the JDK's
/// locale data); that lookup is not here, so a caller that knows the
/// document's locale maps it to a notation itself.
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
        assert!(u32::try_from(text.len()).is_ok(), "a text of at most u32::MAX bytes");
        let b = text.as_bytes();
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
            match self.match_at(text, i, &mut digits, out) {
                Some(end) => i = end,
                // A minus or a digit is one byte.
                None => i += 1,
            }
        }
    }

    /// `CursorQuantityExtractor.matchAt` without the unit: adds the
    /// number that starts at byte `start` to `out` and returns its end.
    fn match_at(&self, text: &str, start: usize, digits: &mut String, out: &mut Numbers) -> Option<usize> {
        let b = text.as_bytes();
        let negative = b[start] == b'-';
        let first = start + usize::from(negative);
        if !b.get(first).is_some_and(u8::is_ascii_digit) || continues_number(text, first, self.notation) {
            return None;
        }
        let boundary = if negative { sign_boundary_before(text, start) } else { boundary_before(text, start) };
        if !boundary {
            return None;
        }
        let r = parse(text, first, self.apply_scale, self.notation, digits)?;
        out.push(NumberMention {
            span: Span::new(start as u32, r.end as u32),
            numeral_end: r.numeral_end as u32,
            negative,
            coefficient: digits,
            scale: r.scale,
            multiplier: r.multiplier,
            marker: r.marker,
            notation: self.notation,
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
            .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
