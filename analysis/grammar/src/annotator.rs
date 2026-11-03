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
//!   cents` is 0.50 USD. Currency rates (`$5 per hour`, `$5/day`) record
//!   their denominator symbol (`"h"`, `"d"`) in [`Money::per`]. Money in
//!   a unit with no single ISO currency (`20 francs`) is left out.
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
        GrammarAnnotator::new(&crate::SPANISH_UNITS)
    }

    pub fn dutch() -> GrammarAnnotator {
        GrammarAnnotator::new(&crate::lang::nl::DUTCH_UNITS)
    }

    pub fn italian() -> GrammarAnnotator {
        GrammarAnnotator::new(&crate::ITALIAN_UNITS)
    }

    pub fn portuguese() -> GrammarAnnotator {
        GrammarAnnotator::new(&crate::PORTUGUESE_UNITS)
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
                push_number(&mut found.numbers, text, m, n.value, self.grammar.notation(), &mut digits);
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
    match unit.currency() {
        Some(c) => {
            let Some((currency, e)) = CurrencyUnit::currency(c) else { return };
            let Some(amount) = value.shift(-i32::from(e)) else { return };
            let scale = amount.write_coefficient(digits);
            let mention = MoneyMention {
                span,
                negative: amount.is_negative(),
                coefficient: digits,
                scale: i32::try_from(scale).unwrap_or(i32::MAX),
                currency,
            };
            let per = unit.per().map_or("", |p| p.symbol());
            found.money.push_with_per(mention, per);
        }
        _ => {
            symbol.clear();
            unit.write_symbol(symbol);
            let scale = value.write_coefficient(digits);
            found.quantities.push_with_unit(
                QuantityMention {
                    span,
                    negative: value.is_negative(),
                    coefficient: digits,
                    scale: i32::try_from(scale).unwrap_or(i32::MAX),
                    unit: symbol,
                },
                unit,
            );
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

fn push_number(
    numbers: &mut Numbers,
    text: &str,
    m: &Mention,
    value: Dec,
    notation: NumberNotation,
    digits: &mut String,
) {
    digits.clear();
    let scale = value.write_coefficient(digits);
    let slice = &text[m.start as usize..m.end as usize];
    let sign_bytes = if value.is_negative() {
        if let Some(rest) = slice.strip_prefix('-').or_else(|| slice.strip_prefix('\u{2212}')) {
            let trimmed = rest.trim_start();
            if trimmed.is_empty() { 0 } else { slice.len() - trimmed.len() }
        } else {
            0
        }
    } else {
        0
    };
    numbers.push_with_numeral_start(
        m.start + sign_bytes as u32,
        NumberMention {
            span: Span::new(m.start, m.end),
            numeral_end: m.end,
            negative: value.is_negative(),
            coefficient: digits,
            scale,
            multiplier: Multiplier::One,
            marker: ScaleMarker::None,
            notation,
        },
    );
}

impl Annotator for GrammarAnnotator {
    const VERSION: &'static str = "3";
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
#[path = "annotator_tests.rs"]
mod tests;
