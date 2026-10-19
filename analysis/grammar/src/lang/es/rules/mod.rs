//! The Spanish rules and the functions that build their values. Builders
//! that read only values are shared with English
//! (`crate::lang::en::rules`).
//!
//! Children arrive in item order with word items left out.

use turbo_analysis_layers::OrdinalBase;

use crate::dec::Dec;
use crate::engine::{Ctx, Grammar, Item, Kind, Rule, glued, glued_if, glued_word, node, node_if, opt_word, word};
use crate::lang::en::rules as en;
use crate::lang::en::rules::{
    digit_proper_fraction, integer, is_num, is_ten, non_negative, non_negative_integer, not_integer, num, ord,
    positive, positive_fraction, positive_integer, pow10, proper_fraction, spelled_only, spelled_positive_fraction,
    term,
};
use crate::value::flag::{DIGITS, PLURAL, SPELLED};
use crate::value::{Num, NumKind, Ordinal, Relative, Value};

use crate::lang::es::lexicon::{ANY_DIGIT, CIEN, ELEVEN, NOUN, TWELVE};
use crate::lang::es::{
    AVO, CLOSED, DEN, DIGSEQ, DOZEN, FRAC_WORD, HUNDREDS, INT, NEG_WORD, NUM, NUMERAL, ONE_PART, ORD, ORD_HUND,
    ORD_SCALE, ORD_SUFFIX, ORD_TENS, ORD_WORD, PCT, REL, REL_WORD, SCALE, SIGN, SIMPLE, SPANISH, TEEN, TENS, TO_END,
    UNIT, VULGAR, counts,
};

mod builders;
pub(crate) use builders::*;

mod table;

pub use table::RULES;
