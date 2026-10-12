//! The General_Category property.

use crate::tables::category::{BLOCKS, INDEX};
use crate::trie::lookup;

/// A General_Category value (Unicode chapter 4.5). The variants carry
/// the two-letter abbreviations the UCD files use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum GeneralCategory {
    /// Uppercase_Letter.
    Lu,
    /// Lowercase_Letter.
    Ll,
    /// Titlecase_Letter.
    Lt,
    /// Modifier_Letter.
    Lm,
    /// Other_Letter.
    Lo,
    /// Nonspacing_Mark.
    Mn,
    /// Spacing_Mark.
    Mc,
    /// Enclosing_Mark.
    Me,
    /// Decimal_Number.
    Nd,
    /// Letter_Number.
    Nl,
    /// Other_Number.
    No,
    /// Connector_Punctuation.
    Pc,
    /// Dash_Punctuation.
    Pd,
    /// Open_Punctuation.
    Ps,
    /// Close_Punctuation.
    Pe,
    /// Initial_Punctuation.
    Pi,
    /// Final_Punctuation.
    Pf,
    /// Other_Punctuation.
    Po,
    /// Math_Symbol.
    Sm,
    /// Currency_Symbol.
    Sc,
    /// Modifier_Symbol.
    Sk,
    /// Other_Symbol.
    So,
    /// Space_Separator.
    Zs,
    /// Line_Separator.
    Zl,
    /// Paragraph_Separator.
    Zp,
    /// Control.
    Cc,
    /// Format.
    Cf,
    /// Surrogate. No `char` has it; it is listed so the enum is complete.
    Cs,
    /// Private_Use.
    Co,
    /// Unassigned.
    Cn,
}

use GeneralCategory::*;

/// Every value, in the order of the numbers the generated table stores.
const ALL: [GeneralCategory; 30] = [
    Lu, Ll, Lt, Lm, Lo, Mn, Mc, Me, Nd, Nl, No, Pc, Pd, Ps, Pe, Pi, Pf, Po, Sm, Sc, Sk, So, Zs, Zl, Zp, Cc, Cf, Cs, Co,
    Cn,
];

impl GeneralCategory {
    /// The two-letter abbreviation, for example `"Lu"`.
    pub fn abbreviation(self) -> &'static str {
        const NAMES: [&str; 30] = [
            "Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl", "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po",
            "Sm", "Sc", "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn",
        ];
        NAMES[self as usize]
    }

    /// A letter: Lu, Ll, Lt, Lm or Lo.
    pub fn is_letter(self) -> bool {
        matches!(self, Lu | Ll | Lt | Lm | Lo)
    }

    /// A mark: Mn, Mc or Me.
    pub fn is_mark(self) -> bool {
        matches!(self, Mn | Mc | Me)
    }

    /// A number: Nd, Nl or No.
    pub fn is_number(self) -> bool {
        matches!(self, Nd | Nl | No)
    }

    /// Punctuation: Pc, Pd, Ps, Pe, Pi, Pf or Po.
    pub fn is_punctuation(self) -> bool {
        matches!(self, Pc | Pd | Ps | Pe | Pi | Pf | Po)
    }

    /// A symbol: Sm, Sc, Sk or So.
    pub fn is_symbol(self) -> bool {
        matches!(self, Sm | Sc | Sk | So)
    }

    /// A separator: Zs, Zl or Zp.
    pub fn is_separator(self) -> bool {
        matches!(self, Zs | Zl | Zp)
    }

    /// Other: Cc, Cf, Cs, Co or Cn.
    pub fn is_other(self) -> bool {
        matches!(self, Cc | Cf | Cs | Co | Cn)
    }
}

/// The general category of `c`; unassigned code points are [`GeneralCategory::Cn`].
#[inline]
pub fn general_category(c: char) -> GeneralCategory {
    ALL[lookup(&INDEX, &BLOCKS, c) as usize]
}
