//! Measured quantities and percentages written in a text view: the written
//! value and unit alongside the normalized value and SI / stated base unit.
//!
//! The layer holds what OpenNLP's `CursorQuantityExtractor` or `turbo-grammar`
//! finds (`80 kg`, `2.5km`, `45%`, `50 percent`, `100 °C`, `36 km/h`).
//! Alongside the written value (`2.5` and `km`), [`Quantities::push`] stores
//! the normalized value and base unit (`2500` and `m`) from [`crate::unit`],
//! so quantities can be compared or filtered across units while keeping the
//! written columns unchanged.

use turbo_analysis::{Ix, Layer, Span, Spans, ViewId};

use crate::dec::Dec;
use crate::unit::Unit;
use crate::{DecimalRef, Strs};

/// The unit a percentage is written with, whether the text has `%` or the
/// word `percent`.
pub const PERCENT: &str = "%";

/// Every quantity mention in a view, one item per mention.
///
/// The written value is a decimal stored as in [`crate::Money`]: `coefficient`
/// times ten to the power of minus `scale`, negated when `negative`. These
/// are the unscaled value and the scale of the Java `BigDecimal` OpenNLP
/// computes, with grouping removed: `1,250.5 GB` has the coefficient
/// `12505` and the scale 1.
///
/// Beside the written value and `unit`, `normalized_negative`,
/// `normalized_coefficient`, `normalized_scale`, and `base_unit` hold the
/// quantity normalized to its SI or stated base unit (`2.5km` -> `2500 m`,
/// `500 g` -> `0.5 kg`, `100 °C` -> `373.15 K`, `36 km/h` -> `10 m/s`). When
/// a unit is not in the shared unit table (such as `%` or a caller's custom
/// token), the normalized columns hold the canonicalized written value and
/// unit. When the written value or its converted form lies outside [`Dec`]'s
/// `i128` mantissa or `i16` scale range, `base_unit` is empty (`""`) and
/// [`Quantities::normalized_dec`] returns `None`.
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
    /// Whether the normalized value is below zero; never true for zero.
    pub normalized_negative: Vec<bool>,
    /// The normalized value's digits in `base_unit` without sign, decimal
    /// point or leading zeros; `0` for zero.
    pub normalized_coefficient: Strs,
    /// `BigDecimal.scale()` of the normalized value.
    pub normalized_scale: Vec<i32>,
    /// The SI or stated base unit symbol (`m`, `kg`, `m³`, `K`, `m/s`, etc.),
    /// `unit` when no base conversion applies, or `""` when the value lies
    /// outside [`Dec`]'s range.
    pub base_unit: Strs,
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
            normalized_negative: Vec::new(),
            normalized_coefficient: Strs::default(),
            normalized_scale: Vec::new(),
            base_unit: Strs::default(),
        }
    }

    /// Appends one mention to every column, normalizing through the shared
    /// [`Unit`] table when `m.unit` maps to a convertible unit.
    pub fn push(&mut self, m: QuantityMention<'_>) -> Ix<Quantities> {
        let parsed = Unit::from_symbol(m.unit);
        self.push_inner(m, parsed)
    }

    /// Appends one mention with its known [`Unit`], avoiding symbol re-parsing.
    pub fn push_with_unit(&mut self, m: QuantityMention<'_>, unit: Unit) -> Ix<Quantities> {
        self.push_inner(m, Some(unit))
    }

    fn push_inner(&mut self, m: QuantityMention<'_>, parsed: Option<Unit>) -> Ix<Quantities> {
        self.negative.push(m.negative);
        self.coefficient.push(m.coefficient);
        self.scale.push(m.scale);
        self.unit.push(m.unit);

        let normalized = Dec::from_unscaled(m.negative, m.coefficient, m.scale).and_then(|dec| match parsed {
            Some(u) => u.normalize(dec).map(|(norm_dec, base)| {
                let mut base_sym = String::new();
                base.write_symbol(&mut base_sym);
                (norm_dec, base_sym)
            }),
            None => Some((dec, m.unit.to_owned())),
        });

        if let Some((norm_dec, base_sym)) = normalized {
            let mut digits = String::new();
            // Dec's scale is an i16, so write_coefficient's u32 is at most 32767 and always fits in i32.
            let norm_scale = norm_dec.write_coefficient(&mut digits) as i32;
            self.normalized_negative.push(norm_dec.is_negative());
            self.normalized_coefficient.push(&digits);
            self.normalized_scale.push(norm_scale);
            self.base_unit.push(&base_sym);
        } else {
            self.normalized_negative.push(m.negative);
            self.normalized_coefficient.push(m.coefficient);
            self.normalized_scale.push(m.scale);
            self.base_unit.push("");
        }

        self.spans.push(m.span)
    }

    /// The written value of mention `i`.
    pub fn value(&self, i: Ix<Quantities>) -> DecimalRef<'_> {
        DecimalRef {
            negative: self.negative[i.get()],
            digits: self.coefficient.get(i.get()),
            scale: self.scale[i.get()].into(),
        }
    }

    /// The written unit of mention `i`: the token, or [`PERCENT`].
    pub fn unit(&self, i: Ix<Quantities>) -> &str {
        self.unit.get(i.get())
    }

    /// The normalized value of mention `i` in [`Quantities::base_unit`].
    pub fn normalized_value(&self, i: Ix<Quantities>) -> DecimalRef<'_> {
        DecimalRef {
            negative: self.normalized_negative[i.get()],
            digits: self.normalized_coefficient.get(i.get()),
            scale: self.normalized_scale[i.get()].into(),
        }
    }

    /// The normalized value of mention `i` as an exact [`Dec`] for ordering
    /// and comparison across units, or `None` when the mention could not be
    /// normalized within [`Dec`]'s range (`base_unit` is empty).
    pub fn normalized_dec(&self, i: Ix<Quantities>) -> Option<Dec> {
        let k = i.get();
        if self.base_unit.get(k).is_empty() {
            return None;
        }
        Dec::from_unscaled(self.normalized_negative[k], self.normalized_coefficient.get(k), self.normalized_scale[k])
    }

    /// The SI or stated base unit symbol of mention `i` (`m`, `kg`, `m³`,
    /// `K`, `m/s`, etc.), or `""` when the mention lies outside [`Dec`]'s range.
    pub fn base_unit(&self, i: Ix<Quantities>) -> &str {
        self.base_unit.get(i.get())
    }

    /// The typed [`Unit`] of mention `i` from the shared unit table, when the
    /// written unit is a known measure unit.
    pub fn parsed_unit(&self, i: Ix<Quantities>) -> Option<Unit> {
        Unit::from_symbol(self.unit(i))
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

fn check_decimal(i: usize, label: &str, negative: bool, digits: &str) -> Result<(), String> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("quantity mention {i} has a {label} that is not all ASCII digits: {digits:?}"));
    }
    if digits.len() > 1 && digits.starts_with('0') {
        return Err(format!("quantity mention {i} has a {label} with a leading zero: {digits:?}"));
    }
    if digits == "0" && negative {
        return Err(format!("quantity mention {i} is a negative zero"));
    }
    Ok(())
}

impl Layer for Quantities {
    const ID: &'static str = "turbo:quantities";
    const SCHEMA: u16 = 2;
    positional!(Quantities);

    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        let columns = [
            self.negative.len(),
            self.coefficient.len(),
            self.scale.len(),
            self.unit.len(),
            self.normalized_negative.len(),
            self.normalized_coefficient.len(),
            self.normalized_scale.len(),
            self.base_unit.len(),
        ];
        if columns.iter().any(|&l| l != n) {
            return Err("every column as long as the spans".into());
        }
        self.spans.check(Self::OVERLAP).map_err(|_| "quantity mentions in text order, not overlapping")?;
        for (i, span) in self.spans.iter() {
            let i = i.get();
            if span.is_empty() {
                return Err(format!("quantity mention {i} is empty at {}", span.start));
            }
            check_decimal(i, "coefficient", self.negative[i], self.coefficient.get(i))?;
            check_decimal(
                i,
                "normalized coefficient",
                self.normalized_negative[i],
                self.normalized_coefficient.get(i),
            )?;
            if self.unit.get(i).chars().all(char::is_whitespace) {
                return Err(format!("quantity mention {i} has a blank unit: {:?}", self.unit.get(i)));
            }
            let base = self.base_unit.get(i);
            if !base.is_empty() && base.chars().all(char::is_whitespace) {
                return Err(format!("quantity mention {i} has a blank base unit: {base:?}"));
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
        assert_eq!(q.normalized_value(Ix::new(0)).to_plain_string(), "-2500");
        assert_eq!(q.base_unit(Ix::new(0)), "m");
        assert_eq!(q.normalized_dec(Ix::new(0)), Some(Dec::from_i64(-2500)));
        assert!(!q.is_percent(Ix::new(0)));
        assert!(q.is_percent(Ix::new(1)));
        assert_eq!(q.normalized_value(Ix::new(1)).to_plain_string(), "45");
        assert_eq!(q.base_unit(Ix::new(1)), "%");
        assert_eq!(q.normalized_dec(Ix::new(1)), Some(Dec::from_i64(45)));
        assert_eq!(q.get(Ix::new(1)), mention(11, 14, "45", "%"));
        let ts = q.text_spans().unwrap();
        assert_eq!((ts.view, ts.start, ts.end), (ViewId::SOURCE, &[0, 11][..], &[6, 14][..]));
    }

    #[test]
    fn values_outside_dec_range_mark_base_unit_empty_and_normalized_dec_none() {
        let mut q = Quantities::new(ViewId::SOURCE);
        // 1e-40000 g: scale 40000 exceeds Dec's i16 scale range.
        q.push(QuantityMention { scale: 40_000, ..mention(0, 6, "1", "g") });
        // 0.001 g: fits in Dec and converts to 1e-6 kg.
        q.push(QuantityMention { scale: 3, ..mention(7, 14, "1", "g") });
        // 1e32767 km: fits in Dec before conversion, but multiplying by 1000 overflows i16 scale.
        q.push(QuantityMention { scale: -32_767, ..mention(15, 22, "1", "km") });
        assert_eq!(q.check(), Ok(()));

        assert_eq!(q.base_unit(Ix::new(0)), "");
        assert_eq!(q.normalized_dec(Ix::new(0)), None);

        assert_eq!(q.base_unit(Ix::new(1)), "kg");
        assert_eq!(q.normalized_dec(Ix::new(1)), Dec::new(1, 6));

        assert_eq!(q.base_unit(Ix::new(2)), "");
        assert_eq!(q.normalized_dec(Ix::new(2)), None);
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

    /// `testQuantityValidation`, `testQuantityRejectsUnicodeBlankUnit`.
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
