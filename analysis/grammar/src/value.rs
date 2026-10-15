//! What a chart node stands for: a number, an ordinal, a percentage, or
//! the payload of a word from a grammar's lexicon.
//!
//! The types are shared by every language. The parts a grammar needs only
//! while it builds (the grain of a spelled number, its flags) ride along in
//! the same values, so a node is one `Copy` struct with no side tables.

use turbo_analysis_layers::OrdinalBase;

use crate::dec::Dec;

/// Flag bits on a [`Num`] or an [`Ordinal`]. The engine sets none of them
/// itself except [`flag::DIGITS`] on a numeral; a grammar's rules set and
/// test them. Bits from [`flag::GRAMMAR`] up are the grammar's own.
pub mod flag {
    /// Built from at least one word.
    pub const SPELLED: u8 = 1;
    /// Built from at least one numeral.
    pub const DIGITS: u8 = 1 << 1;
    /// Written in the plural: `fifths`, `hundreds`.
    pub const PLURAL: u8 = 1 << 2;
    /// The first bit a grammar may give its own meaning.
    pub const GRAMMAR: u8 = 1 << 3;
}

/// The Recognizers-Text subtype of a number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NumKind {
    Integer,
    Decimal,
    Fraction,
    /// Written with an exponent: `1.3e6`, `2.5^-1`, `6.23x10^-4`.
    Power,
}

impl NumKind {
    /// The subtype name Recognizers-Text uses: `integer`, `decimal`,
    /// `fraction` or `power`.
    pub fn name(self) -> &'static str {
        match self {
            NumKind::Integer => "integer",
            NumKind::Decimal => "decimal",
            NumKind::Fraction => "fraction",
            NumKind::Power => "power",
        }
    }

    /// `Integer` for a value without a fraction, `Decimal` otherwise: the
    /// subtype of a plain numeral (`5.00` is an integer, `4.8` is not).
    pub fn of(value: Dec) -> NumKind {
        if value.is_integer() { NumKind::Integer } else { NumKind::Decimal }
    }
}

/// A number.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Num {
    pub value: Dec,
    pub kind: NumKind,
    /// The power of ten of the last multiplier a spelled number ends in:
    /// 2 for `three hundred`, 6 for `two million`, 0 for none. A sum such
    /// as `two thousand five` may only add a part below it.
    pub grain: i8,
    pub flags: u8,
}

impl Num {
    pub fn new(value: Dec, kind: NumKind, flags: u8) -> Num {
        Num { value, kind, grain: 0, flags }
    }

    pub fn has(&self, f: u8) -> bool {
        self.flags & f != 0
    }
}

/// A position counted from the start: `third`, `21st`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Ordinal {
    pub value: i64,
    pub flags: u8,
}

impl Ordinal {
    pub fn has(&self, f: u8) -> bool {
        self.flags & f != 0
    }
}

/// A position counted from the end or from the current item: `last` is
/// the end plus 0, `penultimate` the end minus 1, `next` the current
/// item plus 1.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Relative {
    pub offset: i32,
    pub to: OrdinalBase,
}

/// A node's value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Value {
    /// A word that carries nothing beyond its class.
    #[default]
    Empty,
    /// A lexicon word's payload: its number and flags, read by the rules
    /// that use the word's class.
    Term {
        value: i64,
        flags: u8,
    },
    Num(Num),
    Ordinal(Ordinal),
    Relative(Relative),
    Percent(Dec),
}

impl Value {
    pub fn num(&self) -> Option<&Num> {
        match self {
            Value::Num(n) => Some(n),
            _ => None,
        }
    }

    pub fn ordinal(&self) -> Option<&Ordinal> {
        match self {
            Value::Ordinal(o) => Some(o),
            _ => None,
        }
    }

    /// A lexicon payload as `(value, flags)`.
    pub fn term(&self) -> Option<(i64, u8)> {
        match *self {
            Value::Term { value, flags } => Some((value, flags)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_follow_the_value() {
        assert_eq!(NumKind::of(Dec::from_digits(b"5", b"00")), NumKind::Integer);
        assert_eq!(NumKind::of(Dec::from_digits(b"4", b"8")), NumKind::Decimal);
        assert_eq!(NumKind::Fraction.name(), "fraction");
        assert_eq!(NumKind::Power.name(), "power");
    }

    #[test]
    fn accessors_pick_their_variant() {
        let n = Value::Num(Num::new(Dec::ONE, NumKind::Integer, flag::SPELLED));
        assert!(n.num().unwrap().has(flag::SPELLED));
        assert!(!n.num().unwrap().has(flag::DIGITS));
        assert_eq!(n.ordinal(), None);
        assert_eq!(Value::Term { value: 7, flags: 0 }.term(), Some((7, 0)));
        assert_eq!(Value::Empty.term(), None);
        let o = Value::Ordinal(Ordinal { value: 3, flags: flag::PLURAL });
        assert!(o.ordinal().unwrap().has(flag::PLURAL));
    }

    #[test]
    fn a_value_fits_in_a_few_words() {
        assert!(std::mem::size_of::<Value>() <= 48);
    }
}
