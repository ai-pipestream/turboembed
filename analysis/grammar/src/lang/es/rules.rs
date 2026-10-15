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

use super::lexicon::{ANY_DIGIT, CIEN, ELEVEN, NOUN, TWELVE};
use super::{
    AVO, CLOSED, DEN, DIGSEQ, DOZEN, FRAC_WORD, HUNDREDS, INT, NEG_WORD, NUM, NUMERAL, ONE_PART, ORD, ORD_HUND,
    ORD_SCALE, ORD_SUFFIX, ORD_TENS, ORD_WORD, PCT, REL, REL_WORD, SCALE, SIGN, SIMPLE, SPANISH, TEEN, TENS, TO_END,
    UNIT, VULGAR, counts,
};

fn int(value: Dec, grain: i8, flags: u8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain, flags }))
}

fn fraction(value: Dec, flags: u8) -> Option<Value> {
    Some(Value::Num(Num::new(value, NumKind::Fraction, flags)))
}

fn ordinal(value: i64, flags: u8) -> Option<Value> {
    Some(Value::Ordinal(Ordinal { value, flags }))
}

// Tests on children.

fn unit_1_to_9(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| (1..=9).contains(&t))
}

fn grain_from_2(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 2 && !n.has(CLOSED))
}

fn grain_from_3(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 3)
}

fn grain_from_6(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 6)
}

fn spelled_positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && positive(n))
}

fn spelled_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n))
}

/// What `docena` multiplies: a number below a hundred with no scale, or
/// any positive digit integer.
fn small_multiplicand(v: &Value) -> bool {
    is_num(v, |n| n.grain == 0 && positive(n) && (n.has(DIGITS) || n.value < Dec::from_i64(100)))
}

fn singular(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PLURAL == 0)
}

/// `medio` or `media`, said of one half.
fn one_half(v: &Value) -> bool {
    v.term().is_some_and(|(d, f)| d == 2 && f & (PLURAL | NOUN) == 0)
}

fn denominator_word(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & NOUN == 0)
}

fn noun_half(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & NOUN != 0)
}

fn spelled_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED))
}

fn denominator_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(SIMPLE))
}

// Context checks.

/// Whether a word follows the match that multiplies the number before it
/// but could not multiply a sum ending in a part below `grain`: in `tres
/// mil dos mil` the second number is `dos mil`.
fn before_multiplier(ctx: &Ctx<'_>, grain: i8) -> bool {
    ctx.after().is_some_and(|a| {
        SPANISH.lexicon().lookup(ctx.word(a)).iter().any(|e| match e.dim {
            SCALE => e.value <= i64::from(grain),
            DOZEN => true,
            _ => false,
        })
    })
}

/// Whether the match is one written word: every lexeme after the first
/// is glued to the one before (`cientocincoavos`).
fn one_written_word(ctx: &Ctx<'_>) -> bool {
    (ctx.start + 1..ctx.end).all(|i| ctx.lexeme(i).glued && ctx.lexeme(i).kind == Kind::Word)
}

/// Whether the match starts a written word: `cinco` in `cientocincoavos`
/// does not.
fn opens_word(ctx: &Ctx<'_>) -> bool {
    let first = ctx.lexeme(ctx.start);
    !(first.glued && ctx.before().is_some_and(|p| ctx.lexeme(p).kind == Kind::Word))
}

/// Whether `menos` before a number makes it negative: not after another
/// number (`siete menos cuarto`), nor in `al menos`, `lo menos`, `por lo
/// menos`.
fn menos_opens(ctx: &Ctx<'_>) -> bool {
    let Some(p) = ctx.before() else { return true };
    !(counts(ctx, p) || matches!(ctx.word(p), "al" | "lo"))
}

// Integers.

fn b_word_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0), 0, SPELLED)
}

fn b_tens_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0 + term(&v[1]).0), 0, SPELLED)
}

/// `cien` takes no part after it; `ciento` and `doscientos` do.
fn b_hundreds(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, f) = term(&v[0]);
    int(Dec::from_i64(value), 2, SPELLED | if f & CIEN != 0 { CLOSED } else { 0 })
}

/// `dos mil`, `tres millones`, `mil millones`, `322 millones`. A plain
/// digit integer takes any scale; a spelled one is below it, and from a
/// million up one takes the singular and more the plural.
fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    let digits = m.grain == 0 && m.has(DIGITS) && !m.has(SPELLED);
    if !digits && m.value >= pow10(s) {
        return None;
    }
    if s >= 6 && (m.value == Dec::ONE) == (f & PLURAL != 0) {
        return None;
    }
    int(m.value.checked_mul(pow10(s))?, s as i8, m.flags | SPELLED)
}

/// `mil`, `millón` alone.
fn b_bare_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let s = term(&v[0]).0;
    int(pow10(s), s as i8, SPELLED)
}

fn b_sum(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (l, r) = (num(&v[0]), num(&v[1]));
    if r.grain >= l.grain || r.value >= pow10(i64::from(l.grain)) || before_multiplier(ctx, l.grain) {
        return None;
    }
    int(l.value.checked_add(r.value)?, r.grain, l.flags | r.flags)
}

fn b_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(Dec::from_i64(12))?, 0, m.flags | SPELLED)
}

/// `media docena` is six.
fn b_word_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = Dec::from_i64(12).checked_div(Dec::from_i64(term(&v[0]).0))?;
    let kind = if value.is_integer() { NumKind::Integer } else { NumKind::Fraction };
    Some(Value::Num(Num::new(value, kind, SPELLED)))
}

/// `10k`, `2,33 k`: the subtype is the numeral's.
fn b_suffix(v: &[Value], e: i64) -> Option<Value> {
    let n = num(&v[0]);
    Some(Value::Num(Num::new(n.value.checked_mul(pow10(e))?, n.kind, DIGITS)))
}

fn b_neg_word(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !menos_opens(ctx) {
        return None;
    }
    en::b_neg_word(v, ctx)
}

// Fractions.

/// A denominator is one written word: `tres dosmilésimos` is 3/2000, but
/// `cien mil billonésimas` is 100000/10^12.
fn b_den_ordinal(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    (opens_word(ctx) && one_written_word(ctx)).then_some(v[0])
}

fn b_den_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, f) = term(&v[0]);
    ordinal(d, (f & PLURAL) | SPELLED)
}

/// `veintiunavo`, `doscientosavos`: a number and `avo` in one word.
fn b_den_avo(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[0]);
    if !opens_word(ctx) || !one_written_word(ctx) || n.value < Dec::from_i64(3) {
        return None;
    }
    ordinal(n.value.to_i64()?, (term(&v[1]).1 & PLURAL) | SPELLED)
}

/// One takes the singular (`un quinto`), anything else the plural.
fn b_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]).value, ord(&v[1]));
    if (n == Dec::ONE) == d.has(PLURAL) {
        return None;
    }
    let one = if n == Dec::ONE { ONE_PART } else { 0 };
    fraction(n.checked_div(Dec::from_i64(d.value))?, SPELLED | one)
}

/// `la mitad`.
fn b_half_noun(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(2))?, SPELLED)
}

fn half() -> Option<Dec> {
    Dec::ONE.checked_div(Dec::from_i64(2))
}

/// `cinco y medio`.
fn b_and_half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let w = num(&v[0]);
    fraction(w.value.checked_add(half()?)?, w.flags | SPELLED)
}

/// `tres millones y medio`: half of the scale.
fn b_scale_and_half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let w = num(&v[0]);
    fraction(w.value.checked_add(half()?.checked_mul(pow10(i64::from(w.grain)))?)?, w.flags | SPELLED)
}

/// `cinco y un cuarto`: after `y`, a fraction of one part.
fn b_and_one_part(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    num(&v[1]).has(ONE_PART).then(|| en::b_mixed(v, ctx))?
}

/// `medio millón`, `un tercio de millón`.
fn b_word_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, s) = (term(&v[0]).0, term(&v[1]).0);
    fraction(pow10(s).checked_div(Dec::from_i64(d))?, SPELLED)
}

// Spoken decimals: `siete con cincuenta`, `doscientos coma cero tres`.

/// The decimals said as one number: `cincuenta` is .50, `doscientos` .200.
fn b_int_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[0]);
    let count = n.value.to_i64()?.checked_ilog10()? + 1;
    Some(Value::Num(Num { value: n.value, kind: NumKind::Integer, grain: count as i8, flags: SPELLED }))
}

// Ordinals.

fn b_ordinal_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, flags) = term(&v[0]);
    ordinal(value, flags | SPELLED)
}

/// `dos milésimo`, `tres mil millonésimo`.
fn b_ordinal_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    if m.value >= pow10(s) {
        return None;
    }
    ordinal(m.value.checked_mul(pow10(s))?.to_i64()?, (f & PLURAL) | SPELLED)
}

fn b_ordinal_scale_bare(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (s, f) = term(&v[0]);
    ordinal(pow10(s).to_i64()?, (f & PLURAL) | SPELLED)
}

/// `tres millones dos milésimo`: a number ending in a scale and an
/// ordinal below it.
fn b_ordinal_sum(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, o) = (num(&v[0]), ord(&v[1]));
    if Dec::from_i64(o.value) >= pow10(i64::from(l.grain)) {
        return None;
    }
    ordinal(l.value.to_i64()?.checked_add(o.value)?, (o.flags & PLURAL) | SPELLED)
}

/// `vigésimo primero`, `centésimo vigésimo quinto`, `milésimo
/// quingentésimo`: an ordinal and one below its last nonzero digit.
fn b_ordinal_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, r) = (ord(&v[0]), ord(&v[1]));
    let zeros = (0..18).take_while(|&k| l.value % 10i64.pow(k + 1) == 0).count() as u32;
    if r.value <= 0 || zeros == 0 || r.value >= 10i64.pow(zeros) {
        return None;
    }
    ordinal(l.value.checked_add(r.value)?, (r.flags & PLURAL) | SPELLED)
}

/// `1ro`, `2da`, `11vo`, `30ma`, `5ª`: the suffix has to fit the number's
/// last digits.
fn b_ordinal_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = num(&v[0]).value.to_i64()?;
    let bits = term(&v[1]).0;
    let last = match value % 10 {
        0 => 1 << 10,
        d => 1 << (d - 1),
    };
    let fits =
        bits & last != 0 || (value % 100 == 11 && bits & ELEVEN != 0) || (value % 100 == 12 && bits & TWELVE != 0);
    fits.then(|| ordinal(value, DIGITS))?
}

/// `1.º`, `3.ª`: a dot before the indicator.
fn b_ordinal_digits_dot(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    (term(&v[1]).0 == ANY_DIGIT).then(|| b_ordinal_digits(v, ctx))?
}

fn b_relative_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (offset, f) = term(&v[0]);
    let to = if f == TO_END { OrdinalBase::End } else { OrdinalBase::Current };
    Some(Value::Relative(Relative { offset: i32::try_from(offset).ok()?, to }))
}

const fn r(
    name: &'static str,
    dim: crate::engine::Dim,
    weight: i16,
    items: &'static [Item],
    build: crate::engine::Build,
) -> Rule {
    Rule { name, dim, items, weight, build }
}

/// Every rule. A weight above zero makes a reading win over others of the
/// same text: a fraction of a large scale over a mixed number, a mixed
/// number over a spoken decimal.
pub static RULES: &[Rule] = &[
    // Integers in words and digits.
    r("int_unit", INT, 0, &[node(UNIT)], b_word_int),
    r("int_teen", INT, 0, &[node(TEEN)], b_word_int),
    r("int_tens", INT, 0, &[node(TENS)], b_word_int),
    r("int_tens_unit", INT, 0, &[node(TENS), word("y"), node_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_tens_i_unit", INT, 0, &[node(TENS), glued_word("i"), glued_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_hundreds", INT, 0, &[node(HUNDREDS)], b_hundreds),
    r("int_numeral", INT, 0, &[node_if(NUMERAL, integer)], en::b_numeral_int),
    r("int_scale", INT, 0, &[node_if(INT, positive_integer), node(SCALE)], b_scale),
    r("int_bare_scale", INT, 0, &[node_if(SCALE, singular)], b_bare_scale),
    r("int_sum", INT, 0, &[node_if(INT, grain_from_2), node_if(INT, positive_integer)], b_sum),
    r("int_sum_y", INT, 0, &[node_if(INT, grain_from_3), word("y"), node_if(INT, positive_integer)], b_sum),
    r("int_dozen", INT, 0, &[node_if(INT, small_multiplicand), node(DOZEN)], b_dozen),
    // Numbers.
    r("num_int", NUM, 0, &[node(INT)], en::b_same),
    r("num_numeral", NUM, 0, &[node_if(NUMERAL, not_integer)], en::b_same),
    r("num_numeral_scale", NUM, 0, &[node_if(NUMERAL, not_integer), node(SCALE)], en::b_numeral_scale),
    r("num_suffix_k", NUM, 0, &[node(NUMERAL), word("k")], |v, _| b_suffix(v, 3)),
    r("num_suffix_m", NUM, 0, &[node(NUMERAL), crate::engine::exact("M")], |v, _| b_suffix(v, 6)),
    r("num_suffix_mm", NUM, 0, &[node(NUMERAL), crate::engine::exact("MM")], |v, _| b_suffix(v, 6)),
    r("num_suffix_g", NUM, 0, &[node(NUMERAL), crate::engine::exact("G")], |v, _| b_suffix(v, 9)),
    r("num_suffix_t", NUM, 0, &[node(NUMERAL), crate::engine::exact("T")], |v, _| b_suffix(v, 12)),
    r("num_neg_word", NUM, 0, &[node(NEG_WORD), node_if(NUM, non_negative)], b_neg_word),
    r("num_neg_sign", NUM, 0, &[word("-"), node_if(NUM, en_digits_non_negative)], en::b_neg_sign),
    r("num_neg_minus_sign", NUM, 0, &[word("\u{2212}"), node_if(NUM, en_digits_non_negative)], en::b_neg_sign),
    // Fractions.
    r("den_ordinal", DEN, 0, &[node_if(ORD, denominator_ordinal)], b_den_ordinal),
    r("den_word", DEN, 0, &[node_if(FRAC_WORD, denominator_word)], b_den_word),
    r("den_avo", DEN, 0, &[node_if(INT, spelled_positive_integer), glued(AVO)], b_den_avo),
    r("frac", NUM, 0, &[node_if(INT, spelled_positive_integer), node(DEN)], b_fraction),
    r("frac_half_noun", NUM, 0, &[node_if(FRAC_WORD, noun_half)], b_half_noun),
    r(
        "frac_mixed_con",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("con"), node_if(NUM, positive_fraction)],
        en::b_mixed,
    ),
    r(
        "frac_and_half",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("y"), node_if(FRAC_WORD, one_half)],
        b_and_half,
    ),
    r(
        "frac_and_one_part",
        NUM,
        1,
        &[node_if(NUM, non_negative_integer), word("y"), node_if(NUM, proper_fraction)],
        b_and_one_part,
    ),
    r(
        "frac_scale_and_half",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("y"), node_if(FRAC_WORD, one_half)],
        b_scale_and_half,
    ),
    r(
        "frac_scale_con_fraction",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("con"), node_if(NUM, proper_fraction)],
        en::b_scale_and_fraction,
    ),
    r(
        "frac_scale_plus_fraction",
        NUM,
        1,
        &[node_if(INT, grain_from_3), word("y"), node_if(NUM, positive_fraction)],
        en::b_scale_plus_fraction,
    ),
    r(
        "frac_of_scale",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), opt_word("de"), node(SCALE)],
        en::b_fraction_of_scale,
    ),
    r("frac_word_of_scale", NUM, 0, &[node_if(FRAC_WORD, one_half), node_if(SCALE, singular)], b_word_of_scale),
    r("frac_word_of_dozen", NUM, 0, &[node_if(FRAC_WORD, one_half), node_if(DOZEN, singular)], b_word_of_dozen),
    r(
        "frac_digits",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), glued_word("/"), glued_if(NUMERAL, positive_integer)],
        en::b_digit_fraction,
    ),
    r(
        "frac_digits_mixed",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), node_if(NUM, digit_proper_fraction)],
        en::b_mixed,
    ),
    r("frac_vulgar", NUM, 0, &[node(VULGAR)], en::b_vulgar),
    r("frac_over", NUM, 0, &[node_if(INT, integer), word("sobre"), node_if(INT, positive_integer)], en::b_over),
    // Spoken decimals.
    r("digits_unit", DIGSEQ, 0, &[node(UNIT)], en::b_digit_word),
    r("digits_more", DIGSEQ, 0, &[node(DIGSEQ), node(UNIT)], en::b_more_digits),
    r("digits_int", DIGSEQ, 0, &[node_if(INT, spelled_positive_integer)], b_int_digits),
    r("num_coma", NUM, 0, &[node_if(INT, spelled_integer), word("coma"), node(DIGSEQ)], en::b_point),
    r("num_con", NUM, 0, &[node_if(INT, spelled_integer), word("con"), node(DIGSEQ)], en::b_point),
    // Exponents.
    r("sign_plus", SIGN, 0, &[word("+")], en::b_plus),
    r("sign_minus", SIGN, 0, &[word("-")], en::b_minus),
    r("sign_more_plus", SIGN, 0, &[node(SIGN), glued_word("+")], en::b_sign_plus),
    r("sign_more_minus", SIGN, 0, &[node(SIGN), glued_word("-")], en::b_sign_minus),
    r("pow_e", NUM, 0, &[node(NUMERAL), glued_word("e"), glued_if(NUMERAL, integer)], en::b_ten_power),
    r(
        "pow_e_signed",
        NUM,
        0,
        &[node(NUMERAL), glued_word("e"), glued(SIGN), glued_if(NUMERAL, integer)],
        en::b_ten_power,
    ),
    r("pow_caret", NUM, 0, &[node(NUMERAL), glued_word("^"), glued_if(NUMERAL, integer)], en::b_caret),
    r(
        "pow_caret_signed",
        NUM,
        0,
        &[node(NUMERAL), glued_word("^"), glued(SIGN), glued_if(NUMERAL, integer)],
        en::b_caret,
    ),
    r(
        "pow_times_ten",
        NUM,
        0,
        &[node(NUMERAL), glued_word("x"), glued_if(NUMERAL, is_ten), glued_word("^"), glued_if(NUMERAL, integer)],
        en::b_times_ten_power,
    ),
    // Ordinals.
    r("ord_word", ORD, 0, &[node(ORD_WORD)], b_ordinal_word),
    r("ord_tens", ORD, 0, &[node(ORD_TENS)], b_ordinal_word),
    r("ord_hund", ORD, 0, &[node(ORD_HUND)], b_ordinal_word),
    r("ord_scale", ORD, 0, &[node_if(INT, spelled_positive_integer), node(ORD_SCALE)], b_ordinal_scale),
    r("ord_scale_bare", ORD, 0, &[node(ORD_SCALE)], b_ordinal_scale_bare),
    r("ord_sum", ORD, 0, &[node_if(INT, spelled_grain_from_3), node_if(ORD, spelled_ordinal)], b_ordinal_sum),
    r("ord_ord", ORD, 0, &[node_if(ORD, spelled_ordinal), node_if(ORD, spelled_ordinal)], b_ordinal_ordinal),
    r("ord_digits", ORD, 0, &[node_if(NUMERAL, positive_integer), glued(ORD_SUFFIX)], b_ordinal_digits),
    r(
        "ord_digits_dot",
        ORD,
        0,
        &[node_if(NUMERAL, positive_integer), glued_word("."), glued(ORD_SUFFIX)],
        b_ordinal_digits_dot,
    ),
    // Relative ordinals.
    r("rel_word", REL, 0, &[node(REL_WORD)], b_relative_word),
    r("rel_antes_del_ultimo", REL, 0, &[word("antes"), word("del"), word("ultimo")], en::b_end_minus_one),
    r("rel_antes_de_la_ultima", REL, 0, &[word("antes"), word("de"), word("la"), word("ultima")], en::b_end_minus_one),
    r("rel_anterior_al_ultimo", REL, 0, &[word("anterior"), word("al"), word("ultimo")], en::b_end_minus_one),
    r(
        "rel_anterior_a_la_ultima",
        REL,
        0,
        &[word("anterior"), word("a"), word("la"), word("ultima")],
        en::b_end_minus_one,
    ),
    // Percentages.
    r("pct_sign", PCT, 0, &[node(NUM), word("%")], en::b_percent_after),
    r("pct_por_ciento", PCT, 0, &[node(NUM), word("por"), word("ciento")], en::b_percent_after),
    r("pct_por_cien", PCT, 0, &[node(NUM), word("por"), word("cien")], en::b_percent_after),
];

fn en_digits_non_negative(v: &Value) -> bool {
    is_num(v, |n| n.has(DIGITS) && !n.has(SPELLED) && !n.value.is_negative())
}

fn spelled_grain_from_3(v: &Value) -> bool {
    spelled_integer(v) && grain_from_3(v)
}
