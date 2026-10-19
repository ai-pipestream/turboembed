//! The English rules and the functions that build their values.
//!
//! Children arrive in item order with word items left out, so a rule
//! `[INT, opt "and", INT]` builds from two values.

use turbo_analysis_layers::OrdinalBase;
use turbo_unicode::category::{GeneralCategory, general_category};

use crate::dec::Dec;
use crate::engine::{
    Ctx, Grammar, Item, Kind, Rule, exact, glued, glued_if, glued_word, node, node_if, opt_word, word,
};
use crate::value::flag::{DIGITS, PLURAL, SPELLED};
use crate::value::{Num, NumKind, Ordinal, Relative, Value};

use crate::lang::en::{
    ARTICLE, DEN, DIGSEQ, DOZEN, ENGLISH, FRAC_WORD, HUNDRED, INT, NEG_WORD, NUM, NUMERAL, ORD, ORD_SCALE, ORD_SUFFIX,
    ORD_TENS, ORD_WORD, PCT, PCT_WORD, REL, REL_WORD, SCALE, SIGN, SIMPLE, TEEN, TENS, TO_CURRENT, TO_END, UNIT,
    VULGAR,
};

// Reading children. A rule's items fix each child's category, and every
// category holds one kind of value, so these cannot fail on a rule's own
// children.

pub(in crate::lang) fn num(v: &Value) -> Num {
    *v.num().expect("a number child")
}

mod predicates;
use predicates::*;
pub(in crate::lang) use predicates::{
    after_currency, digit_proper_fraction, digits_non_negative, digits_only, fraction, int, integer, is_num, is_ten,
    minus_opens, non_negative, non_negative_integer, not_integer, number, ord, positive, positive_fraction,
    positive_integer, pow10, proper_fraction, slash_alone, spelled_only, spelled_positive_fraction, term,
};

mod integers;
use integers::*;
pub(in crate::lang) use integers::{b_dozen, b_neg_sign, b_neg_word, b_numeral_int, b_numeral_scale, b_same, suffix};

mod fractions;
use fractions::*;
pub(in crate::lang) use fractions::{
    b_caret, b_digit_fraction, b_digit_word, b_fraction_of_dozen, b_fraction_of_scale, b_minus, b_mixed, b_more_digits,
    b_over, b_part_of, b_plus, b_point, b_scale_and_fraction, b_scale_plus_fraction, b_sign_minus, b_sign_plus,
    b_ten_power, b_times_ten_power, b_vulgar,
};

mod ordinals;
use ordinals::*;
pub(in crate::lang) use ordinals::{b_end_minus_one, b_percent_after, b_percent_before, relative};

mod table;

pub use table::RULES;
