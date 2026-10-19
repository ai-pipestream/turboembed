//! `GoldenCompare` for extraction layers: numbers, money, quantities, temporals, PII, and document dates.

use turbo_analysis::{Ix, Layer, ViewId};
use turbo_analysis_layers::{DocumentDate, Money, Numbers, Pii, Quantities, Strs, Temporals};

use super::super::{Cx, GoldenCompare, Items, Labels, Mismatch, Names, Tolerance, compare_items};
use super::{SPAN, check_bool_u32, golden_spans, golden_spans_only, spans_only, u32s};
use crate::format::ColumnType::{Str, U32};
use crate::format::{ColumnSpec, GoldenLayer};

impl GoldenCompare for Numbers {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("numeral_end", U32),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
    ];

    /// The span first, then the rest of each row as one label: the value as
    /// `BigDecimal.toPlainString` writes it, where the numeral ends, the sign
    /// and the coefficient and scale. A report shows both rows whole.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let (numeral_end, negative, scale) =
            (u32s(golden, "numeral_end"), u32s(golden, "negative"), u32s(golden, "scale"));
        check_bool_u32(cx, negative, "negative")?;
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let numeral = self.numeral(ix);
            let num_end = if self.spans.view() == ViewId::SOURCE || numeral.start > numeral.end {
                self.numeral_end[i]
            } else {
                cx.doc.to_source(self.spans.view(), numeral).end
            };
            line.clear();
            self.value(ix).write_plain(&mut line);
            let rest = format!(
                " numeral_end {} negative {} coefficient {} scale {}",
                num_end,
                u32::from(self.negative[i]),
                self.coefficient.get(i),
                self.scale[i]
            );
            line.push_str(&rest);
            ours.push(&line);
        }
        let (coefficient, plain) = (golden.strs("coefficient"), golden.strs("plain"));
        let (coefficient, plain) = (coefficient.expect("a str column"), plain.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} numeral_end {} negative {} coefficient {} scale {}",
                plain.get(i),
                numeral_end[i],
                negative[i],
                coefficient.get(i),
                scale[i]
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Money {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
        ColumnSpec::new("currency", Str),
    ];

    /// The span first, then the rest of each row as one label: the amount
    /// as `BigDecimal.toPlainString` writes it, the sign, the coefficient
    /// and scale, and the currency. The goldens write an amount with a
    /// scale below zero (`2E+1`) with scale 0, so ours is written the same
    /// way: the coefficient followed by that many zeros.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let (negative, scale) = (u32s(golden, "negative"), u32s(golden, "scale"));
        check_bool_u32(cx, negative, "negative")?;
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let amount = self.amount(ix);
            line.clear();
            amount.write_plain(&mut line);
            line.push_str(" negative ");
            line.push_str(if amount.negative { "1" } else { "0" });
            line.push_str(" coefficient ");
            line.push_str(amount.digits);
            let scale = if amount.scale < 0 {
                if amount.digits != "0" {
                    line.extend(std::iter::repeat_n('0', amount.scale.unsigned_abs() as usize));
                }
                0
            } else {
                amount.scale
            };
            line.push_str(&format!(" scale {scale} currency {}", self.currency[i]));
            ours.push(&line);
        }
        let (coefficient, plain, currency) =
            (golden.strs("coefficient"), golden.strs("plain"), golden.strs("currency"));
        let (coefficient, plain, currency) =
            (coefficient.expect("a str column"), plain.expect("a str column"), currency.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} negative {} coefficient {} scale {} currency {}",
                plain.get(i),
                negative[i],
                coefficient.get(i),
                scale[i],
                currency.get(i)
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Quantities {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("coefficient", Str),
        ColumnSpec::new("negative", U32),
        ColumnSpec::new("scale", U32),
        ColumnSpec::new("plain", Str),
        ColumnSpec::new("unit", Str),
    ];

    /// The span first, then the rest of each row as one label: the value
    /// as `BigDecimal.toPlainString` writes it, the sign, the coefficient
    /// and scale, and the unit. As for money, a value with a scale below
    /// zero is written with scale 0, the coefficient followed by that many
    /// zeros; a value read from text has none.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let (negative, scale) = (u32s(golden, "negative"), u32s(golden, "scale"));
        check_bool_u32(cx, negative, "negative")?;
        let mut ours = Strs::default();
        let mut line = String::new();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            let value = self.value(ix);
            line.clear();
            value.write_plain(&mut line);
            line.push_str(" negative ");
            line.push_str(if value.negative { "1" } else { "0" });
            line.push_str(" coefficient ");
            line.push_str(value.digits);
            let scale = if value.scale < 0 {
                if value.digits != "0" {
                    line.extend(std::iter::repeat_n('0', value.scale.unsigned_abs() as usize));
                }
                0
            } else {
                value.scale
            };
            line.push_str(&format!(" scale {scale} unit {}", self.unit(ix)));
            ours.push(&line);
        }
        let (coefficient, plain, unit) = (golden.strs("coefficient"), golden.strs("plain"), golden.strs("unit"));
        let (coefficient, plain, unit) =
            (coefficient.expect("a str column"), plain.expect("a str column"), unit.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!(
                "{} negative {} coefficient {} scale {} unit {}",
                plain.get(i),
                negative[i],
                coefficient.get(i),
                scale[i],
                unit.get(i)
            ));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let golden = Items {
            len: golden.len(),
            spans: Some(golden_spans(golden)),
            context: None,
            tokens: None,
            units: None,
            labels: Some(Labels::Strs(&theirs)),
            values: None,
        };
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Temporals {
    const COLUMNS: &'static [ColumnSpec] = &[
        SPAN[0],
        SPAN[1],
        ColumnSpec::new("value", Str),
        ColumnSpec::new("granularity", Str),
        ColumnSpec::new("origin", Str),
    ];

    /// The span first, then the rest of each row as one label: the value
    /// as OpenNLP's `TemporalExpression.value()` writes it, written here
    /// from the layer's calendar fields, then the granularity and origin
    /// by their Java names.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for i in 0..self.len() {
            let ix = Ix::new(i as u32);
            ours.push(&format!("{} {} {}", self.value(ix), self.granularity[i].name(), self.origin[i].name()));
        }
        let (value, granularity, origin) = (golden.strs("value"), golden.strs("granularity"), golden.strs("origin"));
        let (value, granularity, origin) =
            (value.expect("a str column"), granularity.expect("a str column"), origin.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("{} {} {}", value.get(i), granularity.get(i), origin.get(i)));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let mut golden = golden_spans_only(golden);
        golden.labels = Some(Labels::Strs(&theirs));
        compare_items(cx, &rust, &golden, Names { label: Some("value"), value: None }, tol)
    }
}

impl GoldenCompare for Pii {
    const COLUMNS: &'static [ColumnSpec] =
        &[SPAN[0], SPAN[1], ColumnSpec::new("type", Str), ColumnSpec::new("normalized", Str)];

    /// The span first, then the kind by its OpenNLP type name and the
    /// normalized form as one label. OpenNLP has no confidence, so the
    /// goldens hold none.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        for m in self.mentions() {
            ours.push(&format!("{} {}", m.kind.name(), m.normalized));
        }
        let (kind, normalized) = (golden.strs("type"), golden.strs("normalized"));
        let (kind, normalized) = (kind.expect("a str column"), normalized.expect("a str column"));
        let mut theirs = Strs::default();
        for i in 0..golden.len() {
            theirs.push(&format!("{} {}", kind.get(i), normalized.get(i)));
        }
        let mut rust = spans_only(self);
        rust.labels = Some(Labels::Strs(&ours));
        let mut golden = golden_spans_only(golden);
        golden.labels = Some(Labels::Strs(&theirs));
        compare_items(cx, &rust, &golden, Names { label: Some("mention"), value: None }, tol)
    }
}

impl GoldenCompare for DocumentDate {
    const COLUMNS: &'static [ColumnSpec] = &[ColumnSpec::new("date", Str), ColumnSpec::new("mention", U32)];

    /// No row, or one: the date as `LocalDate.toString` writes it and the
    /// index of the temporal mention it was read from in the file's
    /// `turbo:temporal` layer, compared as one label. A date with no
    /// mention behind it, which only a caller can give, has none to write
    /// and is compared with `-`.
    fn compare(&self, cx: &Cx<'_>, golden: &GoldenLayer, tol: Tolerance) -> Result<(), Mismatch> {
        let mut ours = Strs::default();
        if let Some(date) = self.date {
            ours.push(&format!("{date} mention {}", self.mention.map_or("-".into(), |m| m.get().to_string())));
        }
        let (date, mention) = (golden.strs("date").expect("a str column"), u32s(golden, "mention"));
        let mut theirs = Strs::default();
        for (i, m) in mention.iter().enumerate().take(golden.len()) {
            theirs.push(&format!("{} mention {m}", date.get(i)));
        }
        let items =
            |len, labels| Items { len, spans: None, context: None, tokens: None, units: None, labels, values: None };
        let rust = items(self.len(), Some(Labels::Strs(&ours)));
        let golden = items(golden.len(), Some(Labels::Strs(&theirs)));
        compare_items(cx, &rust, &golden, Names { label: Some("date"), value: None }, tol)
    }
}
