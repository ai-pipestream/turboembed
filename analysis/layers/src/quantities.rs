//! Measured quantities and percentages written in a text view: the value
//! as an exact decimal and the unit it is in.
//!
//! The layer holds what OpenNLP's `CursorQuantityExtractor` finds (`80 kg`,
//! `2.5km`, `45%`, `50 percent`). No unit conversion is done: `2.5km` keeps
//! the value 2.5 and the unit `km`.

use turbo_analysis::{Ix, Layer, Span, Spans, ViewId};

use crate::{DecimalRef, Strs};

/// The unit a percentage is written with, whether the text has `%` or the
/// word `percent`.
pub const PERCENT: &str = "%";

/// Every quantity mention in a view, one item per mention.
///
/// The value is a decimal stored as in [`crate::Money`]: `coefficient`
/// times ten to the power of minus `scale`, negated when `negative`. These
/// are the unscaled value and the scale of the Java `BigDecimal` OpenNLP
/// computes, with grouping removed: `1,250.5 GB` has the coefficient
/// `12505` and the scale 1. A value read from text has a scale of zero or
/// more.
///
/// The unit is a column of strings, not an enum: OpenNLP's extractor takes
/// a caller's unit set as well as its own, so the set of units is open.
/// A percentage has the unit [`PERCENT`]; any other unit is the token as
/// written, matched case-sensitively.
pub struct Quantities {
    pub spans: Spans<Quantities>,
    /// Whether the value is below zero. A zero value is never negative,
    /// as in `BigDecimal`, though the text may write `-0%`.
    pub negative: Vec<bool>,
    /// The value's digits without sign, without a decimal point and
    /// without leading zeros; `0` for zero.
    pub coefficient: Strs,
    /// `BigDecimal.scale()` of the value.
    pub scale: Vec<i32>,
    /// The unit token, or `%` for a percentage.
    pub unit: Strs,
}

/// One mention, as handed to [`Quantities::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QuantityMention<'a> {
    pub span: Span,
    pub negative: bool,
    pub coefficient: &'a str,
    pub scale: i32,
    pub unit: &'a str,
}

impl Quantities {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Quantities {
        Quantities {
            spans: Spans::new(view),
            negative: Vec::new(),
            coefficient: Strs::default(),
            scale: Vec::new(),
            unit: Strs::default(),
        }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: QuantityMention<'_>) -> Ix<Quantities> {
        self.negative.push(m.negative);
        self.coefficient.push(m.coefficient);
        self.scale.push(m.scale);
        self.unit.push(m.unit);
        self.spans.push(m.span)
    }

    /// The value of mention `i`.
    pub fn value(&self, i: Ix<Quantities>) -> DecimalRef<'_> {
        DecimalRef {
            negative: self.negative[i.get()],
            digits: self.coefficient.get(i.get()),
            scale: self.scale[i.get()].into(),
        }
    }

    /// The unit of mention `i`: the token, or [`PERCENT`].
    pub fn unit(&self, i: Ix<Quantities>) -> &str {
        self.unit.get(i.get())
    }

    /// Whether mention `i` is a percentage.
    pub fn is_percent(&self, i: Ix<Quantities>) -> bool {
        self.unit(i) == PERCENT
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Quantities>) -> QuantityMention<'_> {
        QuantityMention {
            span: self.spans.get(i),
            negative: self.negative[i.get()],
            coefficient: self.coefficient.get(i.get()),
            scale: self.scale[i.get()],
            unit: self.unit.get(i.get()),
        }
    }
}

impl Layer for Quantities {
    const ID: &'static str = "turbo:quantities";
    positional!(Quantities);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [self.negative.len(), self.coefficient.len(), self.scale.len(), self.unit.len()];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        if !self.spans.is_ordered() {
            return Err("quantity mentions in text order, not overlapping".into());
        }
        for (i, span) in self.spans.iter() {
            let i = i.get();
            if span.is_empty() {
                return Err(format!("quantity mention {i} is empty at {}", span.start));
            }
            let digits = self.coefficient.get(i);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("quantity mention {i} has a coefficient that is not all ASCII digits: {digits:?}"));
            }
            if digits.len() > 1 && digits.starts_with('0') {
                return Err(format!("quantity mention {i} has a coefficient with a leading zero: {digits:?}"));
            }
            if digits == "0" && self.negative[i] {
                return Err(format!("quantity mention {i} is a negative zero"));
            }
            // OpenNLP's `Quantity` refuses a unit that is empty or only
            // whitespace.
            if self.unit.get(i).chars().all(char::is_whitespace) {
                return Err(format!("quantity mention {i} has a blank unit: {:?}", self.unit.get(i)));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mention<'a>(start: u32, end: u32, coefficient: &'a str, unit: &'a str) -> QuantityMention<'a> {
        QuantityMention { span: Span::new(start, end), negative: false, coefficient, scale: 0, unit }
    }

    #[test]
    fn values_and_units_come_from_the_columns() {
        let mut q = Quantities::new(ViewId::SOURCE);
        // "-2.5km and 45%"
        q.push(QuantityMention { negative: true, scale: 1, ..mention(0, 6, "25", "km") });
        q.push(mention(11, 14, "45", PERCENT));
        assert_eq!(q.check(), Ok(()));
        assert_eq!(q.len(), 2);
        assert_eq!(q.value(Ix::new(0)).to_plain_string(), "-2.5");
        assert_eq!(q.unit(Ix::new(0)), "km");
        assert!(!q.is_percent(Ix::new(0)));
        assert!(q.is_percent(Ix::new(1)));
        assert_eq!(q.get(Ix::new(1)), mention(11, 14, "45", "%"));
        let (view, start, end) = q.text_spans().unwrap();
        assert_eq!((view, start, end), (ViewId::SOURCE, &[0, 11][..], &[6, 14][..]));
    }

    #[test]
    fn check_accepts_an_empty_layer() {
        assert_eq!(Quantities::new(ViewId::SOURCE).check(), Ok(()));
    }

    #[test]
    fn check_wants_every_column() {
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(mention(0, 4, "5", "kg"));
        q.scale.pop();
        assert_eq!(q.check().unwrap_err(), "every column as long as the spans");
    }

    #[test]
    fn check_wants_text_order() {
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(mention(5, 7, "1", "%"));
        q.push(mention(0, 2, "1", "%"));
        assert!(q.check().unwrap_err().contains("text order"));
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(mention(0, 3, "1", "m"));
        q.push(mention(2, 4, "1", "m"));
        assert!(q.check().unwrap_err().contains("overlapping"));
    }

    #[test]
    fn check_wants_a_canonical_value() {
        for bad in ["", "1a", "-1", "1.5"] {
            let mut q = Quantities::new(ViewId::SOURCE);
            q.push(mention(0, 2, bad, "%"));
            assert!(q.check().unwrap_err().contains("not all ASCII digits"), "{bad:?}");
        }
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(mention(0, 3, "05", "%"));
        assert!(q.check().unwrap_err().contains("leading zero"));
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(QuantityMention { negative: true, ..mention(0, 3, "0", "%") });
        assert!(q.check().unwrap_err().contains("negative zero"));
        let mut q = Quantities::new(ViewId::SOURCE);
        q.push(mention(2, 2, "1", "%"));
        assert!(q.check().unwrap_err().contains("empty"));
    }

    /// `testQuantityValidation` and `testQuantityRejectsUnicodeBlankUnit`
    /// of `CursorQuantityExtractorTest`: a unit that is empty or only
    /// whitespace, ASCII or not.
    #[test]
    fn check_wants_a_unit() {
        for blank in ["", " ", "\u{00A0}", "\u{3000}", "\u{2028}"] {
            let mut q = Quantities::new(ViewId::SOURCE);
            q.push(mention(0, 2, "1", blank));
            assert!(q.check().unwrap_err().contains("blank unit"), "{blank:?}");
        }
    }

    #[test]
    fn the_layer_is_named() {
        assert_eq!(Quantities::ID, "turbo:quantities");
    }
}
