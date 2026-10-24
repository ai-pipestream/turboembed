//! Writes what a grammar finds as layers.
//!
//! One parse of the view gives four outputs, each chosen on its own (see
//! [`crate::Parser::select`]), so a percentage's number is also in the
//! numbers and `forty two and a third` holds the ordinal `third` as well
//! as the number:
//!
//! - numbers go to a [`Numbers`] instance under [`GRAMMAR_NUMBERS`], never
//!   the default one, which `turbo-numeric`'s scanner writes. A mention's
//!   numeral ends where its span ends, with the multiplier `One` and no
//!   scale marker; `negative` says whether the value is below zero, as a
//!   spelled `minus five` has no minus sign to start with.
//! - percentages go to a [`Quantities`] instance under
//!   [`GRAMMAR_PERCENTS`], each with the unit [`PERCENT`]; with a unit
//!   grammar, quantities with other units go to the same instance, each
//!   with its unit's symbol ([`crate::unit::Unit::write_symbol`]).
//! - money goes to a [`Money`] instance under [`GRAMMAR_MONEY`], stated
//!   in the currency its unit denotes ([`CurrencyUnit::currency`]): `50
//!   cents` is 0.50 USD. Money in a unit with no such currency (`20
//!   francs`) is left out.
//! - ordinals go to the default [`Ordinals`] instance.

use std::sync::Mutex;

use turbo_analysis::{Annotator, Bound, Error, Ix, Key, Layer, Refs, Span, View};
use turbo_analysis_layers::{
    Money, MoneyMention, Multiplier, NumberMention, NumberNotation, Numbers, OrdinalBase, OrdinalMention, Ordinals,
    PERCENT, Quantities, QuantityMention, ScaleMarker,
};

use crate::dec::Dec;
use crate::engine::{Grammar, Mention, Output, Parser};
use crate::lang::en::unit::ENGLISH_UNITS;
use crate::unit::{CurrencyUnit, Unit};
use crate::value::Value;

/// The instance name the grammar's numbers and percentages are written
/// under.
pub const GRAMMAR: &str = "grammar";
pub const GRAMMAR_NUMBERS: Key<Numbers> = Key::named(GRAMMAR);
pub const GRAMMAR_PERCENTS: Key<Quantities> = Key::named(GRAMMAR);
pub const GRAMMAR_MONEY: Key<Money> = Key::named(GRAMMAR);

/// Runs a [`Grammar`] over a text view and writes numbers, percentages,
/// quantities, money and ordinals. Parsers are pooled, so their buffers are reused between
/// documents.
pub struct GrammarAnnotator {
    grammar: &'static dyn Grammar,
    pool: Mutex<Vec<Parser>>,
}

/// What one text gives, layer by layer.
pub struct Found {
    pub numbers: Numbers,
    pub percents: Quantities,
    pub ordinals: Ordinals,
    /// Quantities with a unit other than a percentage.
    pub quantities: Quantities,
    pub money: Money,
}

impl GrammarAnnotator {
    pub fn new(grammar: &'static dyn Grammar) -> GrammarAnnotator {
        GrammarAnnotator { grammar, pool: Mutex::new(Vec::new()) }
    }

    /// English numbers and units: [`ENGLISH_UNITS`] holds the number
    /// rules as well.
    pub fn english() -> GrammarAnnotator {
        GrammarAnnotator::new(&ENGLISH_UNITS)
    }

    pub fn spanish() -> GrammarAnnotator {
        GrammarAnnotator::new(&crate::lang::es::SPANISH)
    }

    /// The step for a pipeline: writes numbers under [`GRAMMAR_NUMBERS`],
    /// percentages and quantities under [`GRAMMAR_PERCENTS`], money under
    /// [`GRAMMAR_MONEY`] and ordinals to the default instance.
    pub fn bound(self) -> Bound<GrammarAnnotator> {
        Bound::new(self).write(0, GRAMMAR).write(1, GRAMMAR).write(3, GRAMMAR)
    }

    /// Finds everything in `text`, with spans in view `view`.
    ///
    /// # Panics
    ///
    /// When `text` is longer than `u32::MAX` bytes.
    pub fn find(&self, text: &str, view: turbo_analysis::ViewId) -> Found {
        let mut parser = self.pool.lock().ok().and_then(|mut p| p.pop()).unwrap_or_else(|| Parser::new(self.grammar));
        parser.parse(text);
        let mut mentions = Vec::new();
        let mut found = Found {
            numbers: Numbers::new(view),
            percents: Quantities::new(view),
            ordinals: Ordinals::new(view),
            quantities: Quantities::new(view),
            money: Money::new(view),
        };
        let mut digits = String::new();
        parser.select(text, Output::Number, &mut mentions);
        for m in &mentions {
            if let Value::Num(n) = m.value {
                push_number(&mut found.numbers, m, n.value, self.grammar.notation(), &mut digits);
            }
        }
        mentions.clear();
        parser.select(text, Output::Percent, &mut mentions);
        for m in &mentions {
            if let Value::Percent(v) = m.value {
                digits.clear();
                let scale = v.write_coefficient(&mut digits);
                found.percents.push(QuantityMention {
                    span: Span::new(m.start, m.end),
                    negative: v.is_negative(),
                    coefficient: &digits,
                    scale: i32::try_from(scale).unwrap_or(i32::MAX),
                    unit: PERCENT,
                });
            }
        }
        mentions.clear();
        parser.select(text, Output::Ordinal, &mut mentions);
        for m in &mentions {
            let (base, offset) = match m.value {
                Value::Ordinal(o) => (OrdinalBase::Start, o.value),
                Value::Relative(r) => (r.to, i64::from(r.offset)),
                _ => continue,
            };
            found.ordinals.push(OrdinalMention { span: Span::new(m.start, m.end), base, offset });
        }
        mentions.clear();
        parser.select(text, Output::Quantity, &mut mentions);
        let mut symbol = String::new();
        for m in &mentions {
            if let Value::Quantity(q) = m.value {
                push_quantity(&mut found, m, q.value(), q.unit(), &mut digits, &mut symbol);
            }
        }
        if let Ok(mut pool) = self.pool.lock() {
            pool.push(parser);
        }
        found
    }
}

/// Writes a quantity to the money or the quantities.
fn push_quantity(found: &mut Found, m: &Mention, value: Dec, unit: Unit, digits: &mut String, symbol: &mut String) {
    digits.clear();
    let span = Span::new(m.start, m.end);
    match unit {
        Unit::Currency(c) => {
            let Some((currency, e)) = CurrencyUnit::currency(c) else { return };
            let Some(amount) = value.shift(-i32::from(e)) else { return };
            let scale = amount.write_coefficient(digits);
            found.money.push(MoneyMention {
                span,
                negative: amount.is_negative(),
                coefficient: digits,
                scale: i32::try_from(scale).unwrap_or(i32::MAX),
                currency,
            });
        }
        _ => {
            symbol.clear();
            unit.write_symbol(symbol);
            let scale = value.write_coefficient(digits);
            found.quantities.push(QuantityMention {
                span,
                negative: value.is_negative(),
                coefficient: digits,
                scale: i32::try_from(scale).unwrap_or(i32::MAX),
                unit: symbol,
            });
        }
    }
}

/// The percentages and the other quantities in one layer, in text order.
/// The two are chosen apart and each is free of overlaps; where one of
/// each overlaps, the longer is kept (the earlier on a tie).
fn merge(percents: &Quantities, quantities: &Quantities) -> Quantities {
    let mut all: Vec<(&Quantities, Ix<Quantities>)> = percents
        .spans
        .iter()
        .map(|(i, _)| (percents, i))
        .chain(quantities.spans.iter().map(|(i, _)| (quantities, i)))
        .collect();
    all.sort_by_key(|&(q, i)| q.spans.get(i).start);
    let mut kept: Vec<(&Quantities, Ix<Quantities>)> = Vec::with_capacity(all.len());
    for (q, i) in all {
        let s = q.spans.get(i);
        if let Some(&(lq, li)) = kept.last() {
            let l = lq.spans.get(li);
            if s.start < l.end {
                if s.end - s.start > l.end - l.start {
                    kept.pop();
                    kept.push((q, i));
                }
                continue;
            }
        }
        kept.push((q, i));
    }
    let mut out = Quantities::new(percents.spans.view());
    for (q, i) in kept {
        out.push(q.get(i));
    }
    out
}

fn push_number(numbers: &mut Numbers, m: &Mention, value: Dec, notation: NumberNotation, digits: &mut String) {
    digits.clear();
    let scale = value.write_coefficient(digits);
    numbers.push(NumberMention {
        span: Span::new(m.start, m.end),
        numeral_end: m.end,
        negative: value.is_negative(),
        coefficient: digits,
        scale,
        multiplier: Multiplier::One,
        marker: ScaleMarker::None,
        notation,
    });
}

impl Annotator for GrammarAnnotator {
    const VERSION: &'static str = "2";
    const ID: &'static str = "turbo-grammar:numbers";
    type Reads = (View,);
    type Writes = (Numbers, Quantities, Ordinals, Money);

    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<(Numbers, Quantities, Ordinals, Money), Error> {
        let text = view.text();
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let f = self.find(text, view.id());
        let quantities = if f.quantities.is_empty() { f.percents } else { merge(&f.percents, &f.quantities) };
        Ok((f.numbers, quantities, f.ordinals, f.money))
    }

    /// FNV-1a over the grammar's name, fixed so that stored provenance
    /// stays comparable across builds.
    fn config_hash(&self) -> u64 {
        turbo_analysis::fnv1a(self.grammar.name().as_bytes())
    }
}

#[cfg(test)]
mod tests {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, ViewId};

    use super::*;

    #[test]
    fn layers_hold_what_the_grammar_finds() {
        let text = "minus five percent of the second batch, twenty-one and a half, 1.5k";
        let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        assert_eq!(f.numbers.check(), Ok(()));
        assert_eq!(f.percents.check(), Ok(()));
        assert_eq!(f.ordinals.check(), Ok(()));
        let numbers: Vec<(String, String)> = (0..f.numbers.len())
            .map(|i| {
                let s = f.numbers.spans.get(Ix::new(i as u32));
                (text[s.range()].to_string(), f.numbers.value(Ix::new(i as u32)).to_plain_string())
            })
            .collect();
        assert_eq!(
            numbers,
            [
                ("minus five".to_string(), "-5".to_string()),
                ("twenty-one and a half".into(), "21.5".into()),
                ("1.5k".into(), "1500".into()),
            ]
        );
        assert!(f.numbers.negative[0]);
        assert_eq!(f.percents.len(), 1);
        assert_eq!(f.percents.value(Ix::new(0)).to_plain_string(), "-5");
        assert!(f.percents.is_percent(Ix::new(0)));
        assert_eq!(text[f.percents.spans.get(Ix::new(0)).range()].to_string(), "minus five percent");
        assert_eq!(f.ordinals.len(), 1);
        assert_eq!(f.ordinals.get(Ix::new(0)).base, OrdinalBase::Start);
        assert_eq!(f.ordinals.get(Ix::new(0)).offset, 2);
    }

    #[test]
    fn large_and_fractional_values_fill_the_columns() {
        let text = "1e10 and two thirds";
        let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        assert_eq!(f.numbers.check(), Ok(()));
        assert_eq!(f.numbers.coefficient.get(0), "10000000000");
        assert_eq!(f.numbers.scale[0], 0);
        assert_eq!(f.numbers.scale[1], 34);
        assert!(f.numbers.value(Ix::new(1)).to_plain_string().starts_with("0.6666"));
    }

    #[test]
    fn a_pipeline_writes_the_named_instances() {
        let analyzer = Analyzer::builder().step(GrammarAnnotator::english().bound()).build().unwrap();
        let mut docs = vec![Document::new("the next three hundred and five percent, 2 GB for $5")];
        let results = analyzer.run(&mut docs);
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        let doc = &docs[0];
        assert!(doc.get::<Numbers>().is_none(), "the default numbers instance is turbo-numeric's");
        assert_eq!(doc.get_key(&GRAMMAR_NUMBERS).unwrap().len(), 3);
        assert_eq!(doc.get_key(&GRAMMAR_PERCENTS).unwrap().len(), 2);
        assert!(doc.get::<Money>().is_none(), "the default money instance is turbo-numeric's");
        assert_eq!(doc.get_key(&GRAMMAR_MONEY).unwrap().len(), 1);
        let ordinals = doc.get::<Ordinals>().unwrap();
        assert_eq!(ordinals.get(Ix::new(0)).base, OrdinalBase::Current);
    }

    #[test]
    fn spanish_numbers_take_the_decimal_comma() {
        let text = "1.234,56 y veintiún";
        let f = GrammarAnnotator::spanish().find(text, ViewId::SOURCE);
        assert_eq!(f.numbers.check(), Ok(()));
        assert_eq!(f.numbers.len(), 2);
        assert_eq!(f.numbers.value(Ix::new(0)).to_plain_string(), "1234.56");
        assert_eq!(f.numbers.notation[0], NumberNotation::LatinEu);
        assert_eq!(&text[f.numbers.spans.get(Ix::new(1)).range()], "veintiún");
        assert_eq!(f.numbers.value(Ix::new(1)).to_plain_string(), "21");
    }

    #[test]
    fn quantities_carry_their_unit_symbols() {
        let text = "a 10 km run at minus 40 degrees, 5% of it uphill";
        let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        assert_eq!(f.quantities.check(), Ok(()));
        let got: Vec<(String, String, String)> = (0..f.quantities.len())
            .map(|i| {
                let i = Ix::new(i as u32);
                let s = f.quantities.spans.get(i);
                (text[s.range()].to_string(), f.quantities.value(i).to_plain_string(), f.quantities.unit(i).to_string())
            })
            .collect();
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0].0, "10 km");
        assert_eq!(got[0].1, "10");
        assert_eq!(got[1].0, "minus 40 degrees");
        assert_eq!(got[1].1, "-40");
        let merged = merge(&f.percents, &f.quantities);
        assert_eq!(merged.len(), 3);
        assert!(merged.is_percent(Ix::new(2)));
    }

    #[test]
    fn tonnes_and_tons_have_their_own_symbols() {
        let text = "3.5 tonnes, 2 tons and 1 long ton";
        let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        let units: Vec<&str> = (0..f.quantities.len()).map(|i| f.quantities.unit(Ix::new(i as u32))).collect();
        assert_eq!(units, ["t", "sh tn", "long tn"]);
    }

    #[test]
    fn money_is_stated_in_its_currency() {
        let text = "it costs $3.5m, 50 cents or 20 francs";
        let f = GrammarAnnotator::english().find(text, ViewId::SOURCE);
        assert_eq!(f.money.check(), Ok(()));
        assert_eq!(f.money.len(), 2, "francs name no one currency");
        let first = f.money.get(Ix::new(0));
        assert_eq!(first.currency.as_str(), "USD");
        assert_eq!(f.money.amount(Ix::new(0)).to_plain_string(), "3500000");
        assert_eq!(f.money.amount(Ix::new(1)).to_plain_string(), "0.5");
        assert_eq!(text[f.money.spans.get(Ix::new(1)).range()].to_string(), "50 cents");
    }

    #[test]
    fn overlapping_percents_and_quantities_keep_the_longer() {
        let mut p = Quantities::new(ViewId::SOURCE);
        let mut q = Quantities::new(ViewId::SOURCE);
        let m = |start, end, unit| QuantityMention {
            span: Span::new(start, end),
            negative: false,
            coefficient: "1",
            scale: 0,
            unit,
        };
        p.push(m(0, 4, PERCENT));
        p.push(m(10, 12, PERCENT));
        q.push(m(2, 8, "km"));
        q.push(m(10, 11, "m"));
        let all = merge(&p, &q);
        let spans: Vec<(u32, u32)> =
            (0..all.len()).map(|i| all.spans.get(Ix::new(i as u32))).map(|s| (s.start, s.end)).collect();
        assert_eq!(spans, [(2, 8), (10, 12)]);
    }

    #[test]
    fn the_config_hash_names_the_grammar() {
        assert_eq!(GrammarAnnotator::english().config_hash(), GrammarAnnotator::english().config_hash());
        assert_ne!(GrammarAnnotator::english().config_hash(), 0);
    }
}
