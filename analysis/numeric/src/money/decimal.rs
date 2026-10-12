//! An owned exact decimal, as Java's `BigDecimal` holds one: a sign, the
//! digits of the unscaled value and a scale.

use std::cmp::Ordering;
use std::fmt;

use turbo_analysis_layers::DecimalRef;
use turbo_unicode::java;

use super::ScaleRangeError;
use super::natural::Natural;

/// A decimal value: `digits` times ten to the power of minus `scale`,
/// negated when `negative`. Two decimals with the same value and different
/// scales (`1.20` and `1.2`) are different, as in `BigDecimal.equals`;
/// [`Decimal::cmp_value`] compares values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decimal {
    /// Whether the value is below zero; never true for zero.
    pub negative: bool,
    /// ASCII digits without leading zeros, `0` for zero.
    pub digits: String,
    /// `BigDecimal.scale()`.
    pub scale: i32,
}

/// Why a string is not a decimal `new BigDecimal(String)` accepts.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DecimalParseError {
    /// No digit in the significand.
    NoDigits,
    /// A second decimal point.
    TwoPoints,
    /// A character that is no digit, point or exponent mark.
    Character(char),
    /// An exponent mark without exponent digits, or a character in the
    /// exponent that is no digit.
    Exponent,
    /// More than ten exponent digits after its leading zeros, or a scale
    /// outside the `int` range.
    ScaleRange,
}

impl fmt::Display for DecimalParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecimalParseError::NoDigits => f.write_str("no digits found"),
            DecimalParseError::TwoPoints => f.write_str("more than one decimal point"),
            DecimalParseError::Character(c) => {
                write!(f, "character {c:?} is neither a decimal digit, a decimal point nor an exponent mark")
            }
            DecimalParseError::Exponent => f.write_str("a malformed exponent"),
            DecimalParseError::ScaleRange => f.write_str("the scale is out of the int range"),
        }
    }
}

impl std::error::Error for DecimalParseError {}

impl Default for Decimal {
    fn default() -> Decimal {
        Decimal::zero(0)
    }
}

impl Decimal {
    /// Zero at `scale`.
    pub fn zero(scale: i32) -> Decimal {
        Decimal { negative: false, digits: "0".into(), scale }
    }

    /// One, at the scale 0: `BigDecimal.ONE`.
    pub fn one() -> Decimal {
        Decimal { negative: false, digits: "1".into(), scale: 0 }
    }

    /// Reads a decimal as `new BigDecimal(String)` does: an optional `+`
    /// or `-`, digits with at most one `.` among them, and an optional
    /// exponent, `e` or `E` with an optional sign and digits. The digits
    /// may be any Unicode decimal digits (category Nd) of the Basic
    /// Multilingual Plane, as `Character.isDigit(char)` allows; nothing
    /// else, not even whitespace, may appear.
    pub fn parse(s: &str) -> Result<Decimal, DecimalParseError> {
        let mut chars = s.chars().peekable();
        let negative = match chars.peek() {
            Some('-') => {
                chars.next();
                true
            }
            Some('+') => {
                chars.next();
                false
            }
            _ => false,
        };
        let mut digits = String::new();
        let mut any_digit = false;
        let mut point = false;
        let mut fraction: i64 = 0;
        let mut exponent: i64 = 0;
        while let Some(c) = chars.next() {
            if c == '.' {
                if point {
                    return Err(DecimalParseError::TwoPoints);
                }
                point = true;
            } else if c == 'e' || c == 'E' {
                exponent = parse_exponent(chars.by_ref())?;
                break;
            } else if let Some(d) = java_digit(c) {
                any_digit = true;
                if d != 0 || !digits.is_empty() {
                    digits.push(char::from(b'0' + d));
                }
                if point {
                    fraction += 1;
                }
            } else {
                return Err(DecimalParseError::Character(c));
            }
        }
        if !any_digit {
            return Err(DecimalParseError::NoDigits);
        }
        let scale = i32::try_from(fraction - exponent).map_err(|_| DecimalParseError::ScaleRange)?;
        if digits.is_empty() {
            return Ok(Decimal::zero(scale));
        }
        Ok(Decimal { negative, digits, scale })
    }

    /// The value as a [`DecimalRef`].
    pub fn as_ref(&self) -> DecimalRef<'_> {
        DecimalRef { negative: self.negative, digits: &self.digits, scale: self.scale.into() }
    }

    /// Sets this to `value`, reusing the digit buffer.
    pub fn set(&mut self, value: DecimalRef<'_>) -> Result<(), ScaleRangeError> {
        self.scale = i32::try_from(value.scale).map_err(|_| ScaleRangeError)?;
        self.negative = value.negative;
        self.digits.clear();
        self.digits.push_str(value.digits);
        Ok(())
    }

    pub fn is_zero(&self) -> bool {
        self.digits == "0"
    }

    /// How many digits the unscaled value has: `BigDecimal.precision()`.
    pub fn precision(&self) -> usize {
        self.digits.len()
    }

    /// The value written out without an exponent, as
    /// `BigDecimal.toPlainString` writes it.
    pub fn to_plain_string(&self) -> String {
        self.as_ref().to_plain_string()
    }

    /// Compares the values, ignoring the scale: `BigDecimal.compareTo`.
    pub fn cmp_value(&self, other: &Decimal) -> Ordering {
        cmp_values(self.as_ref(), other.as_ref())
    }

    /// The exact product, as `BigDecimal.multiply` computes it. The
    /// product's scale is the sum of the scales; when that is outside the
    /// `int` range the product is refused, unless `self` is zero, whose
    /// scale is then clamped to the range.
    pub fn multiply(a: DecimalRef<'_>, b: DecimalRef<'_>, out: &mut Decimal) -> Result<(), ScaleRangeError> {
        let scale = a.scale + b.scale;
        let a_zero = a.digits == "0";
        let scale = match i32::try_from(scale) {
            Ok(s) => s,
            Err(_) if a_zero => saturate(scale),
            Err(_) => return Err(ScaleRangeError),
        };
        let product = Natural::from_digits(a.digits).mul(&Natural::from_digits(b.digits));
        out.digits.clear();
        product.write_digits(&mut out.digits);
        out.negative = !product.is_zero() && (a.negative != b.negative);
        out.scale = scale;
        Ok(())
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_ref(), f)
    }
}

/// A scale clamped to the `int` range: `BigDecimal.saturateLong`.
pub(crate) fn saturate(scale: i64) -> i32 {
    scale.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
}

/// `Character.digit(c, 10)` for the characters `Character.isDigit(char)`
/// accepts: a decimal digit of the Basic Multilingual Plane. A digit
/// outside it is a surrogate pair to the Java, and neither half is a
/// digit.
fn java_digit(c: char) -> Option<u8> {
    if java::is_supplementary(c) { None } else { java::digit(c) }
}

/// `BigDecimal.parseExp`, after the mark: an optional sign, then at most
/// ten digits after any leading zeros. The exponent itself may be outside
/// the `int` range (`2e2147483648` is read); the scale it gives may not.
fn parse_exponent(chars: impl Iterator<Item = char>) -> Result<i64, DecimalParseError> {
    let mut chars = chars.peekable();
    let negative = match chars.peek() {
        Some('-') => {
            chars.next();
            true
        }
        Some('+') => {
            chars.next();
            false
        }
        _ => false,
    };
    let mut value: i64 = 0;
    let mut count = 0;
    let mut significant = 0;
    for c in chars {
        let d = java_digit(c).ok_or(DecimalParseError::Exponent)?;
        count += 1;
        if significant > 0 || d != 0 {
            significant += 1;
            if significant > 10 {
                return Err(DecimalParseError::ScaleRange);
            }
            value = value * 10 + i64::from(d);
        }
    }
    if count == 0 {
        return Err(DecimalParseError::Exponent);
    }
    Ok(if negative { -value } else { value })
}

/// Compares two values, ignoring their scales: `BigDecimal.compareTo`.
pub fn cmp_values(a: DecimalRef<'_>, b: DecimalRef<'_>) -> Ordering {
    let sign = |d: &DecimalRef<'_>| {
        if d.digits == "0" {
            0
        } else if d.negative {
            -1
        } else {
            1
        }
    };
    let (sa, sb) = (sign(&a), sign(&b));
    if sa != sb || sa == 0 {
        return sa.cmp(&sb);
    }
    let magnitude = cmp_magnitudes(a, b);
    if sa < 0 { magnitude.reverse() } else { magnitude }
}

/// Compares the absolute values of two non-zero decimals: first by the
/// position of the leading digit, then digit by digit.
fn cmp_magnitudes(a: DecimalRef<'_>, b: DecimalRef<'_>) -> Ordering {
    let lead = |d: &DecimalRef<'_>| d.digits.len() as i64 - d.scale;
    lead(&a).cmp(&lead(&b)).then_with(|| {
        let (x, y) = (a.digits.trim_end_matches('0'), b.digits.trim_end_matches('0'));
        x.cmp(y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(s: &str) -> (bool, String, i32) {
        let d = Decimal::parse(s).unwrap_or_else(|e| panic!("{s:?}: {e}"));
        (d.negative, d.digits, d.scale)
    }

    #[test]
    fn big_decimal_strings() {
        assert_eq!(parsed("1.2000"), (false, "12000".into(), 4));
        assert_eq!(parsed("+1.25"), (false, "125".into(), 2));
        assert_eq!(parsed("-0.001"), (true, "1".into(), 3));
        assert_eq!(parsed(".125"), (false, "125".into(), 3));
        assert_eq!(parsed("1."), (false, "1".into(), 0));
        assert_eq!(parsed("007"), (false, "7".into(), 0));
        assert_eq!(parsed("0.0000"), (false, "0".into(), 4));
        assert_eq!(parsed("-0"), (false, "0".into(), 0));
        assert_eq!(parsed("1E-30"), (false, "1".into(), 30));
        assert_eq!(parsed("1E+30"), (false, "1".into(), -30));
        assert_eq!(parsed("1.5e3"), (false, "15".into(), -2));
        assert_eq!(parsed("0E+20"), (false, "0".into(), -20));
        assert_eq!(parsed("1E-2147483647"), (false, "1".into(), i32::MAX));
        assert_eq!(parsed("1E+2147483647"), (false, "1".into(), -i32::MAX));
        assert_eq!(parsed("1e00000000000000005"), (false, "1".into(), -5));
        // The exponent leaves the int range, the scale does not.
        assert_eq!(parsed("2e2147483648"), (false, "2".into(), i32::MIN));
        assert_eq!(parsed("1.5e2147483649"), (false, "15".into(), i32::MIN));
        // Unicode decimal digits, Arabic-Indic and fullwidth.
        assert_eq!(parsed("\u{0661}.\u{0665}"), (false, "15".into(), 1));
        assert_eq!(parsed("\u{FF13}e\u{FF12}"), (false, "3".into(), -2));
    }

    #[test]
    fn rejected_strings() {
        use DecimalParseError::*;
        let cases = [
            ("", NoDigits),
            ("+", NoDigits),
            ("-", NoDigits),
            (".", NoDigits),
            ("e5", NoDigits),
            ("1.2.3", TwoPoints),
            ("1.2x", Character('x')),
            ("NaN", Character('N')),
            ("Infinity", Character('I')),
            (" 1", Character(' ')),
            ("++1", Character('+')),
            ("1e", Exponent),
            ("1e+", Exponent),
            ("1e+-5", Exponent),
            ("1e5.0", Exponent),
            ("1E-2147483648", ScaleRange),
            ("1.5e-2147483647", ScaleRange),
            ("0.1E-2147483647", ScaleRange),
            ("1e12345678901", ScaleRange),
            // A mathematical digit is outside the Basic Multilingual Plane.
            ("\u{1D7D9}", Character('\u{1D7D9}')),
        ];
        for (s, want) in cases {
            assert_eq!(Decimal::parse(s), Err(want), "{s:?}");
        }
    }

    #[test]
    fn values_compare_without_their_scales() {
        let d = |s: &str| Decimal::parse(s).unwrap();
        assert_eq!(d("1.20").cmp_value(&d("1.2")), Ordering::Equal);
        assert_eq!(d("1E+3").cmp_value(&d("1000.000")), Ordering::Equal);
        assert_eq!(d("0.00").cmp_value(&d("-0")), Ordering::Equal);
        assert_eq!(d("-5").cmp_value(&d("-4")), Ordering::Less);
        assert_eq!(d("0.1").cmp_value(&d("0.09")), Ordering::Greater);
        assert_eq!(d("-0.1").cmp_value(&d("0")), Ordering::Less);
        assert_eq!(d("99").cmp_value(&d("100")), Ordering::Less);
        assert_eq!(d("1E-2147483647").cmp_value(&d("1E+2147483647")), Ordering::Less);
    }

    #[test]
    fn products() {
        let d = |s: &str| Decimal::parse(s).unwrap();
        let mut out = Decimal::default();
        Decimal::multiply(d("-10.125").as_ref(), d("1.25").as_ref(), &mut out).unwrap();
        assert_eq!((out.to_plain_string(), out.scale), ("-12.65625".into(), 5));
        Decimal::multiply(d("-3").as_ref(), d("0.00").as_ref(), &mut out).unwrap();
        assert_eq!((out.negative, out.digits.as_str(), out.scale), (false, "0", 2));
        let far = Decimal { negative: false, digits: "1".into(), scale: i32::MAX };
        assert_eq!(Decimal::multiply(far.as_ref(), d("0.1").as_ref(), &mut out), Err(ScaleRangeError));
        Decimal::multiply(Decimal::zero(i32::MAX).as_ref(), d("0.1").as_ref(), &mut out).unwrap();
        assert_eq!((out.digits.as_str(), out.scale), ("0", i32::MAX));
    }
}
