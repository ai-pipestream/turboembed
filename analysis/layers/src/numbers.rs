//! Numbers written in a text view, with the exact value each one states.
//!
//! The layer holds what OpenNLP's `NumberScan` reads: ASCII digits with
//! strict grouping, an optional fraction, an optional leading minus and an
//! optional scale marker (`12k`, `3 million`). The value is kept as a
//! decimal, the way Java's `BigDecimal` holds it, so `0.1` stays `0.1`
//! and a reader that needs a float asks for one.

use turbo_analysis::{Ix, Layer, Span, Spans, ViewId};

use crate::Strs;

/// The written convention a number groups digits and marks its fraction
/// in. A text such as `1.234` is a valid number in both conventions with
/// two different values, so the caller says which one the text is in;
/// nothing in this crate guesses it from a locale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum NumberNotation {
    /// Commas group digits and a dot marks the fraction: `1,234.56`.
    /// OpenNLP's numeric extractors read this one unless told otherwise.
    #[default]
    LatinUs,
    /// Dots group digits and a comma marks the fraction: `1.234,56`.
    LatinEu,
}

impl NumberNotation {
    /// The character between groups of three digits.
    pub fn group_separator(self) -> char {
        match self {
            NumberNotation::LatinUs => ',',
            NumberNotation::LatinEu => '.',
        }
    }

    /// The character between the integer part and the fraction.
    pub fn decimal_separator(self) -> char {
        match self {
            NumberNotation::LatinUs => '.',
            NumberNotation::LatinEu => ',',
        }
    }
}

/// The power of ten a scale marker multiplies a number by.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Multiplier {
    #[default]
    One,
    Thousand,
    Million,
    Billion,
    Trillion,
}

impl Multiplier {
    /// The factor itself: 1, 1 000, 1 000 000 and so on.
    pub fn factor(self) -> u64 {
        10u64.pow(self.zeros())
    }

    /// How many zeros the factor has.
    pub fn zeros(self) -> u32 {
        match self {
            Multiplier::One => 0,
            Multiplier::Thousand => 3,
            Multiplier::Million => 6,
            Multiplier::Billion => 9,
            Multiplier::Trillion => 12,
        }
    }
}

/// How a number's scale was written.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ScaleMarker {
    /// No marker: the multiplier is one.
    #[default]
    None,
    /// A letter directly after the digits: `k`, `m`, `b` or `bn`, in
    /// either case.
    Suffix,
    /// A word after one whitespace character: `thousand`, `million`,
    /// `billion` or `trillion`, in any case.
    Word,
}

/// Every number in a view, one item per mention.
///
/// A mention's span covers all of it: the minus sign, the digits and the
/// scale marker. The numeral (digits, group separators and fraction)
/// starts at the span's start, or one byte later when the mention is
/// negative, and ends at `numeral_end`.
///
/// The value is a decimal: `coefficient` times ten to the power of minus
/// `scale`, negated when `negative`, and it includes the multiplier, so
/// `1.5k` has the coefficient `15000` and the scale 1 (1500.0). These are
/// the unscaled value and the scale of the Java `BigDecimal` OpenNLP
/// computes for the same mention.
pub struct Numbers {
    pub spans: Spans<Numbers>,
    /// Where the numeral ends: before the scale marker, or at the span's
    /// end when there is none.
    pub numeral_end: Vec<u32>,
    /// Whether the mention starts with a minus sign.
    pub negative: Vec<bool>,
    /// The value's digits without sign, without a decimal point and
    /// without leading zeros; `0` for zero.
    pub coefficient: Strs,
    /// How many of the coefficient's digits are the fraction.
    pub scale: Vec<u32>,
    pub multiplier: Vec<Multiplier>,
    pub marker: Vec<ScaleMarker>,
    /// The convention the mention was read in.
    pub notation: Vec<NumberNotation>,
}

/// One mention, as handed to [`Numbers::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumberMention<'a> {
    pub span: Span,
    pub numeral_end: u32,
    pub negative: bool,
    pub coefficient: &'a str,
    pub scale: u32,
    pub multiplier: Multiplier,
    pub marker: ScaleMarker,
    pub notation: NumberNotation,
}

impl Numbers {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Numbers {
        Numbers {
            spans: Spans::new(view),
            numeral_end: Vec::new(),
            negative: Vec::new(),
            coefficient: Strs::default(),
            scale: Vec::new(),
            multiplier: Vec::new(),
            marker: Vec::new(),
            notation: Vec::new(),
        }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: NumberMention<'_>) -> Ix<Numbers> {
        self.numeral_end.push(m.numeral_end);
        self.negative.push(m.negative);
        self.coefficient.push(m.coefficient);
        self.scale.push(m.scale);
        self.multiplier.push(m.multiplier);
        self.marker.push(m.marker);
        self.notation.push(m.notation);
        self.spans.push(m.span)
    }

    /// The span of the numeral alone, without the sign and the scale
    /// marker.
    pub fn numeral(&self, i: Ix<Numbers>) -> Span {
        let span = self.spans.get(i);
        Span::new(span.start.saturating_add(u32::from(self.negative[i.get()])), self.numeral_end[i.get()])
    }

    /// The value of mention `i`.
    pub fn value(&self, i: Ix<Numbers>) -> DecimalRef<'_> {
        let digits = self.coefficient.get(i.get());
        DecimalRef { negative: self.negative[i.get()] && digits != "0", digits, scale: self.scale[i.get()].into() }
    }
}

impl Layer for Numbers {
    const ID: &'static str = "turbo:numbers";
    positional!(Numbers);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [
            self.numeral_end.len(),
            self.negative.len(),
            self.coefficient.len(),
            self.scale.len(),
            self.multiplier.len(),
            self.marker.len(),
            self.notation.len(),
        ];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "numbers in text order, not overlapping")?;
        for (i, span) in self.spans.iter() {
            let i = i.get();
            let end = self.numeral_end[i];
            let Some(numeral_start) = span.start.checked_add(u32::from(self.negative[i])) else {
                return Err(format!("number {i} has its numeral end {end} outside {}..{}", span.start, span.end));
            };
            if end <= numeral_start || end > span.end {
                return Err(format!("number {i} has its numeral end {end} outside {}..{}", span.start, span.end));
            }
            let digits = self.coefficient.get(i);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("number {i} has a coefficient that is not all ASCII digits: {digits:?}"));
            }
            if digits.len() > 1 && digits.starts_with('0') {
                return Err(format!("number {i} has a coefficient with a leading zero: {digits:?}"));
            }
            if (self.marker[i] == ScaleMarker::None) != (self.multiplier[i] == Multiplier::One) {
                return Err(format!("number {i} has a scale marker without a multiplier, or the other way round"));
            }
            if (self.marker[i] == ScaleMarker::None) != (end == span.end) {
                return Err(format!("number {i} has text after its numeral that is not a scale marker"));
            }
        }
        Ok(())
    }
}

/// A decimal value borrowed from a layer: `digits` times ten to the
/// power of minus `scale`, negated when `negative`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DecimalRef<'a> {
    /// Whether the value is below zero. Never true for zero: a negated
    /// zero is zero, as in `BigDecimal`.
    pub negative: bool,
    /// ASCII digits without leading zeros, `0` for zero.
    pub digits: &'a str,
    /// How many of the digits are the fraction, as `BigDecimal.scale()`
    /// counts: below zero the value is the digits followed by that many
    /// zeros. A number read from text has a scale of zero or more; a
    /// computed amount (a currency conversion) can have one below zero.
    /// `i64` holds both the `u32` of a text's fraction and Java's `int`.
    pub scale: i64,
}

impl DecimalRef<'_> {
    fn is_zero(&self) -> bool {
        !self.digits.is_empty() && self.digits.bytes().all(|b| b == b'0')
    }

    /// Appends the value written out without an exponent, as
    /// `BigDecimal.toPlainString` writes it: `-1500.0`, `0.05`, `0.00`,
    /// and `1000` for the digits `1` at the scale -3. A zero with a scale
    /// below zero is `0`.
    pub fn write_plain(&self, out: &mut String) {
        let zero = self.is_zero();
        if self.scale < 0 {
            if zero {
                out.push('0');
                return;
            }
            if self.negative {
                out.push('-');
            }
            out.push_str(self.digits);
            out.extend(std::iter::repeat_n('0', self.scale.unsigned_abs() as usize));
            return;
        }
        if self.negative && !zero {
            out.push('-');
        }
        let scale = self.scale as usize;
        let len = self.digits.len();
        if scale == 0 {
            out.push_str(if zero { "0" } else { self.digits });
        } else if len > scale {
            out.push_str(&self.digits[..len - scale]);
            out.push('.');
            out.push_str(&self.digits[len - scale..]);
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', scale - len));
            out.push_str(self.digits);
        }
    }

    /// The value written out without an exponent; see [`Self::write_plain`].
    pub fn to_plain_string(&self) -> String {
        let mut out = String::with_capacity(self.digits.len() + 3);
        self.write_plain(&mut out);
        out
    }

    /// The nearest `f64`, rounded once from the exact decimal. A value
    /// beyond the `f64` range is infinite, as `BigDecimal.doubleValue`
    /// gives.
    pub fn to_f64(&self) -> f64 {
        if self.is_zero() {
            return 0.0;
        }
        // The digits with an exponent, so a scale far from zero does not
        // write out its zeros.
        let mut s = String::with_capacity(self.digits.len() + 24);
        if self.negative {
            s.push('-');
        }
        s.push_str(self.digits);
        s.push('e');
        s.push_str(&self.scale.saturating_neg().to_string());
        s.parse().expect("digits with an exponent parse as f64")
    }
}

impl std::fmt::Display for DecimalRef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        self.write_plain(&mut s);
        f.write_str(&s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mention(start: u32, end: u32, coefficient: &str) -> NumberMention<'_> {
        NumberMention {
            span: Span::new(start, end),
            numeral_end: end,
            negative: false,
            coefficient,
            scale: 0,
            multiplier: Multiplier::One,
            marker: ScaleMarker::None,
            notation: NumberNotation::LatinUs,
        }
    }

    fn plain(negative: bool, digits: &str, scale: i64) -> String {
        DecimalRef { negative, digits, scale }.to_plain_string()
    }

    #[test]
    fn notation_separators_mirror_each_other() {
        assert_eq!(NumberNotation::LatinUs.group_separator(), ',');
        assert_eq!(NumberNotation::LatinUs.decimal_separator(), '.');
        assert_eq!(NumberNotation::LatinEu.group_separator(), '.');
        assert_eq!(NumberNotation::LatinEu.decimal_separator(), ',');
        assert_eq!(NumberNotation::default(), NumberNotation::LatinUs);
    }

    #[test]
    fn multiplier_factors() {
        let all =
            [Multiplier::One, Multiplier::Thousand, Multiplier::Million, Multiplier::Billion, Multiplier::Trillion];
        let factors: Vec<u64> = all.iter().map(|m| m.factor()).collect();
        assert_eq!(factors, [1, 1_000, 1_000_000, 1_000_000_000, 1_000_000_000_000]);
    }

    #[test]
    fn plain_strings_match_big_decimal() {
        assert_eq!(plain(false, "0", 0), "0");
        assert_eq!(plain(false, "1234", 0), "1234");
        assert_eq!(plain(true, "1234", 0), "-1234");
        assert_eq!(plain(false, "123456789", 2), "1234567.89");
        assert_eq!(plain(true, "15000", 1), "-1500.0");
        assert_eq!(plain(false, "5", 2), "0.05");
        assert_eq!(plain(false, "5", 1), "0.5");
        assert_eq!(plain(false, "15", 2), "0.15");
        assert_eq!(plain(true, "5", 3), "-0.005");
        assert_eq!(plain(false, "0", 2), "0.00");
        assert_eq!(plain(true, "0", 0), "0");
        assert_eq!(plain(true, "0", 2), "0.00");
        assert_eq!(plain(false, "0", 1), "0.0");
        assert_eq!(DecimalRef { negative: false, digits: "120", scale: 1 }.to_string(), "12.0");
        // A scale below zero appends zeros; a zero stays `0`.
        assert_eq!(plain(false, "12", -3), "12000");
        assert_eq!(plain(true, "12", -1), "-120");
        assert_eq!(plain(false, "0", -5), "0");
        assert_eq!(plain(true, "0", -5), "0");
    }

    #[test]
    fn floats_are_rounded_from_the_exact_decimal() {
        assert_eq!(DecimalRef { negative: false, digits: "1", scale: 1 }.to_f64(), 0.1);
        assert_eq!(DecimalRef { negative: true, digits: "15000", scale: 1 }.to_f64(), -1500.0);
        assert_eq!(DecimalRef { negative: false, digits: "0", scale: 2 }.to_f64(), 0.0);
        assert!(DecimalRef { negative: false, digits: "0", scale: 2 }.to_f64().is_sign_positive());
        assert!(DecimalRef { negative: true, digits: "0", scale: 2 }.to_f64().is_sign_positive());
        // 2^53 + 1 is not a double; it rounds to even, 2^53.
        assert_eq!(DecimalRef { negative: false, digits: "9007199254740993", scale: 0 }.to_f64(), 9007199254740992.0);
        let huge = "9".repeat(400);
        assert_eq!(DecimalRef { negative: false, digits: &huge, scale: 0 }.to_f64(), f64::INFINITY);
        assert_eq!(DecimalRef { negative: false, digits: "1", scale: 400 }.to_f64(), 0.0);
        assert_eq!(DecimalRef { negative: true, digits: "15", scale: -2 }.to_f64(), -1500.0);
        let far = DecimalRef { negative: false, digits: "1", scale: i64::from(i32::MIN) };
        assert_eq!(far.to_f64(), f64::INFINITY);
    }

    #[test]
    fn values_come_from_the_columns() {
        let mut n = Numbers::new(ViewId::SOURCE);
        // "-1.5k and -0.00"
        n.push(NumberMention {
            span: Span::new(0, 5),
            numeral_end: 4,
            negative: true,
            coefficient: "15000",
            scale: 1,
            multiplier: Multiplier::Thousand,
            marker: ScaleMarker::Suffix,
            notation: NumberNotation::LatinUs,
        });
        n.push(NumberMention { negative: true, scale: 2, ..mention(10, 15, "0") });
        assert_eq!(n.check(), Ok(()));
        assert_eq!(n.len(), 2);
        assert_eq!(n.numeral(Ix::new(0)), Span::new(1, 4));
        assert_eq!(n.value(Ix::new(0)), DecimalRef { negative: true, digits: "15000", scale: 1 });
        assert_eq!(n.value(Ix::new(0)).to_plain_string(), "-1500.0");
        // A negated zero is zero, though the text has the sign.
        assert!(n.negative[1]);
        assert_eq!(n.value(Ix::new(1)).to_plain_string(), "0.00");
        assert_eq!(n.numeral(Ix::new(1)), Span::new(11, 15));
    }

    #[test]
    fn check_accepts_an_empty_layer() {
        assert_eq!(Numbers::new(ViewId::SOURCE).check(), Ok(()));
    }

    #[test]
    fn check_wants_every_column() {
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(0, 2, "12"));
        n.scale.pop();
        assert_eq!(n.check().unwrap_err(), "every column as long as the spans");
    }

    #[test]
    fn check_wants_text_order() {
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(4, 6, "12"));
        n.push(mention(0, 2, "12"));
        assert!(n.check().unwrap_err().contains("text order"));
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(0, 3, "123"));
        n.push(mention(2, 4, "34"));
        assert!(n.check().unwrap_err().contains("overlapping"));
    }

    #[test]
    fn check_wants_the_numeral_inside_the_span() {
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { numeral_end: 3, ..mention(0, 2, "12") });
        assert!(n.check().unwrap_err().contains("numeral end 3 outside 0..2"));
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { numeral_end: 0, ..mention(0, 2, "12") });
        assert!(n.check().unwrap_err().contains("outside"));
        // A negative numeral needs a digit after the sign.
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { negative: true, numeral_end: 1, ..mention(0, 1, "1") });
        assert!(n.check().unwrap_err().contains("outside"));
        let mut max_span = Numbers::new(ViewId::SOURCE);
        max_span.push(NumberMention { negative: true, numeral_end: u32::MAX, ..mention(u32::MAX, u32::MAX, "1") });
        assert!(max_span.check().unwrap_err().contains("outside"));
    }

    #[test]
    fn check_wants_a_canonical_coefficient() {
        for bad in ["", "1a", "-1", "1.5", "\u{0661}"] {
            let mut n = Numbers::new(ViewId::SOURCE);
            n.push(mention(0, 2, bad));
            assert!(n.check().unwrap_err().contains("not all ASCII digits"), "{bad:?}");
        }
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(0, 2, "05"));
        assert!(n.check().unwrap_err().contains("leading zero"));
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(0, 1, "0"));
        assert_eq!(n.check(), Ok(()));
    }

    #[test]
    fn check_pairs_markers_with_multipliers() {
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { numeral_end: 2, marker: ScaleMarker::Suffix, ..mention(0, 3, "12000") });
        assert!(n.check().unwrap_err().contains("without a multiplier"));
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { multiplier: Multiplier::Million, ..mention(0, 2, "12000000") });
        assert!(n.check().unwrap_err().contains("without a multiplier"));
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(NumberMention { numeral_end: 1, ..mention(0, 2, "12") });
        assert!(n.check().unwrap_err().contains("not a scale marker"));
        let mut n = Numbers::new(ViewId::SOURCE);
        let m = mention(0, 9, "3000000");
        n.push(NumberMention { multiplier: Multiplier::Million, marker: ScaleMarker::Word, numeral_end: 1, ..m });
        assert_eq!(n.check(), Ok(()));
    }

    #[test]
    fn the_layer_is_named_and_positional() {
        assert_eq!(Numbers::ID, "turbo:numbers");
        let mut n = Numbers::new(ViewId::SOURCE);
        n.push(mention(3, 5, "12"));
        let ts = n.text_spans().unwrap();
        assert_eq!((ts.view, ts.start, ts.end), (ViewId::SOURCE, &[3][..], &[5][..]));
    }
}
