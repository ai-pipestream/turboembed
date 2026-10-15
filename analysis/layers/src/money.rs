//! Amounts of money written in a text view: the amount as an exact
//! decimal and the currency it is in.
//!
//! The layer holds what OpenNLP's `CursorMoneyExtractor` finds (`$1.2M`,
//! `100 USD`, `3 billion pounds`), and the same mentions restated in one
//! currency by a conversion step, which writes a second instance of the
//! layer under [`CONVERTED_MONEY`].

use std::fmt;

use turbo_analysis::{Ix, Key, Layer, Span, Spans, ViewId};

use crate::{DecimalRef, Strs};

/// A currency code: three ASCII capital letters, the way ISO 4217 writes
/// alphabetic codes (`USD`, `EUR`). The type checks the form only; whether
/// a code is one ISO 4217 assigns is for the code that makes it to decide.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Currency([u8; 3]);

impl Currency {
    /// The code, when all three bytes are ASCII capital letters.
    pub const fn new(code: [u8; 3]) -> Option<Currency> {
        if code[0].is_ascii_uppercase() && code[1].is_ascii_uppercase() && code[2].is_ascii_uppercase() {
            Some(Currency(code))
        } else {
            None
        }
    }

    /// The code for a string of exactly three ASCII capital letters.
    pub fn parse(code: &str) -> Option<Currency> {
        Currency::new(code.as_bytes().try_into().ok()?)
    }

    /// The code as a `&str`: always three ASCII capital letters.
    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("ASCII capital letters")
    }

    /// The three bytes of the code.
    pub const fn code(self) -> [u8; 3] {
        self.0
    }
}

impl fmt::Display for Currency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl fmt::Debug for Currency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The instance a conversion step writes: the mentions of the default
/// instance that it could restate in its target currency, on the same
/// spans. OpenNLP calls this layer `money.converted`.
pub const CONVERTED_MONEY: Key<Money> = Key::named("converted");

/// Every money mention in a view, one item per mention.
///
/// The amount is a decimal: `coefficient` times ten to the power of minus
/// `scale`, negated when `negative`. These are the unscaled value and the
/// scale of the Java `BigDecimal` OpenNLP computes, with grouping removed
/// and any scale marker applied: `$1.2M` has the coefficient `12000000`
/// and the scale 1 (1200000.0). An extracted amount has a scale of zero
/// or more; a converted one can have a scale below zero, as `BigDecimal`
/// does after rounding to 16 digits.
pub struct Money {
    pub spans: Spans<Money>,
    /// Whether the amount is below zero. A zero amount is never negative,
    /// as in `BigDecimal`, though the text may write `-$0`.
    pub negative: Vec<bool>,
    /// The amount's digits without sign, without a decimal point and
    /// without leading zeros; `0` for zero.
    pub coefficient: Strs,
    /// `BigDecimal.scale()` of the amount.
    pub scale: Vec<i32>,
    pub currency: Vec<Currency>,
}

/// One mention, as handed to [`Money::push`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MoneyMention<'a> {
    pub span: Span,
    pub negative: bool,
    pub coefficient: &'a str,
    pub scale: i32,
    pub currency: Currency,
}

impl Money {
    /// An empty layer over `view`.
    pub fn new(view: ViewId) -> Money {
        Money {
            spans: Spans::new(view),
            negative: Vec::new(),
            coefficient: Strs::default(),
            scale: Vec::new(),
            currency: Vec::new(),
        }
    }

    /// Appends one mention to every column.
    pub fn push(&mut self, m: MoneyMention<'_>) -> Ix<Money> {
        self.negative.push(m.negative);
        self.coefficient.push(m.coefficient);
        self.scale.push(m.scale);
        self.currency.push(m.currency);
        self.spans.push(m.span)
    }

    /// The amount of mention `i`.
    pub fn amount(&self, i: Ix<Money>) -> DecimalRef<'_> {
        DecimalRef {
            negative: self.negative[i.get()],
            digits: self.coefficient.get(i.get()),
            scale: self.scale[i.get()].into(),
        }
    }

    /// Mention `i`, column by column.
    pub fn get(&self, i: Ix<Money>) -> MoneyMention<'_> {
        MoneyMention {
            span: self.spans.get(i),
            negative: self.negative[i.get()],
            coefficient: self.coefficient.get(i.get()),
            scale: self.scale[i.get()],
            currency: self.currency[i.get()],
        }
    }
}

impl Layer for Money {
    const ID: &'static str = "turbo:money";
    positional!(Money);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [self.negative.len(), self.coefficient.len(), self.scale.len(), self.currency.len()];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "money mentions in text order, not overlapping")?;
        for (i, span) in self.spans.iter() {
            let i = i.get();
            if span.is_empty() {
                return Err(format!("money mention {i} is empty at {}", span.start));
            }
            let digits = self.coefficient.get(i);
            if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
                return Err(format!("money mention {i} has a coefficient that is not all ASCII digits: {digits:?}"));
            }
            if digits.len() > 1 && digits.starts_with('0') {
                return Err(format!("money mention {i} has a coefficient with a leading zero: {digits:?}"));
            }
            if digits == "0" && self.negative[i] {
                return Err(format!("money mention {i} is a negative zero"));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const USD: Currency = match Currency::new(*b"USD") {
        Some(c) => c,
        None => panic!(),
    };

    fn mention(start: u32, end: u32, coefficient: &str) -> MoneyMention<'_> {
        MoneyMention { span: Span::new(start, end), negative: false, coefficient, scale: 0, currency: USD }
    }

    #[test]
    fn currency_codes_are_three_capitals() {
        assert_eq!(Currency::parse("EUR").map(|c| c.code()), Some(*b"EUR"));
        assert_eq!(Currency::parse("EUR").unwrap().to_string(), "EUR");
        assert_eq!(format!("{:?}", USD), "USD");
        for bad in ["", "EU", "EURO", "eur", "EuR", "E1R", "ÉUR", " EU"] {
            assert_eq!(Currency::parse(bad), None, "{bad:?}");
        }
        assert!(Currency::parse("ABC") < Currency::parse("ABD"));
    }

    #[test]
    fn amounts_come_from_the_columns() {
        let mut m = Money::new(ViewId::SOURCE);
        // "-$1.2M and 50 EUR"
        m.push(MoneyMention { negative: true, scale: 1, ..mention(0, 6, "12000000") });
        m.push(MoneyMention { currency: Currency::parse("EUR").unwrap(), ..mention(11, 17, "50") });
        assert_eq!(m.check(), Ok(()));
        assert_eq!(m.len(), 2);
        assert_eq!(m.amount(Ix::new(0)).to_plain_string(), "-1200000.0");
        assert_eq!(m.get(Ix::new(1)).currency.as_str(), "EUR");
        assert_eq!(m.get(Ix::new(1)).span, Span::new(11, 17));
        let ts = m.text_spans().unwrap();
        assert_eq!((ts.view, ts.start, ts.end), (ViewId::SOURCE, &[0, 11][..], &[6, 17][..]));
    }

    #[test]
    fn a_converted_amount_can_have_a_scale_below_zero() {
        let mut m = Money::new(ViewId::SOURCE);
        m.push(MoneyMention { scale: -3, ..mention(0, 4, "1234567890123456") });
        assert_eq!(m.check(), Ok(()));
        assert_eq!(m.amount(Ix::new(0)).to_plain_string(), "1234567890123456000");
    }

    #[test]
    fn check_wants_every_column() {
        let mut m = Money::new(ViewId::SOURCE);
        m.push(mention(0, 2, "12"));
        m.currency.pop();
        assert_eq!(m.check().unwrap_err(), "every column as long as the spans");
    }

    #[test]
    fn check_wants_text_order() {
        let mut m = Money::new(ViewId::SOURCE);
        m.push(mention(4, 6, "1"));
        m.push(mention(0, 2, "1"));
        assert!(m.check().unwrap_err().contains("text order"));
        let mut m = Money::new(ViewId::SOURCE);
        m.push(mention(0, 3, "1"));
        m.push(mention(2, 4, "1"));
        assert!(m.check().unwrap_err().contains("overlapping"));
    }

    #[test]
    fn check_wants_a_canonical_amount() {
        for bad in ["", "1a", "-1", "1.5"] {
            let mut m = Money::new(ViewId::SOURCE);
            m.push(mention(0, 2, bad));
            assert!(m.check().unwrap_err().contains("not all ASCII digits"), "{bad:?}");
        }
        let mut m = Money::new(ViewId::SOURCE);
        m.push(mention(0, 2, "05"));
        assert!(m.check().unwrap_err().contains("leading zero"));
        let mut m = Money::new(ViewId::SOURCE);
        m.push(MoneyMention { negative: true, ..mention(0, 3, "0") });
        assert!(m.check().unwrap_err().contains("negative zero"));
        let mut m = Money::new(ViewId::SOURCE);
        m.push(mention(2, 2, "1"));
        assert!(m.check().unwrap_err().contains("empty"));
    }

    #[test]
    fn the_layer_is_named_and_has_a_converted_instance() {
        assert_eq!(Money::ID, "turbo:money");
        assert_eq!(CONVERTED_MONEY.name, "converted");
        assert!(CONVERTED_MONEY.id() != Key::<Money>::DEFAULT.id());
    }
}
