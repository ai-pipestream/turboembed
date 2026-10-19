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

pub(crate) fn num(v: &Value) -> Num {
    *v.num().expect("a number child")
}

mod predicates;
pub(crate) use predicates::*;

mod integers;
pub(crate) use integers::*;

mod fractions;
pub(crate) use fractions::*;

mod ordinals;
pub(crate) use ordinals::*;

mod table;

pub use table::RULES;
