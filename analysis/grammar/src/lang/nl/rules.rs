//! The Dutch rules and the functions that build their values. Builders
//! that read the same in English come from [`crate::lang::en::rules`].
//!
//! The pieces of a compound are glued words, so `tweeentwintig` meets the
//! rule `[UNIT, glued "en", glued TENS]` and `honderdzes` the sum rule.

use turbo_analysis_layers::OrdinalBase;

use crate::dec::Dec;
use crate::engine::{
    Ctx, Grammar, Item, Rule, exact, glued, glued_if, glued_word, node, node_if, opt_word, word,
};
use crate::lang::en::rules::{
    after_currency, b_caret, b_digit_fraction, b_digit_word, b_end_minus_one, b_fraction_of_scale, b_minus, b_mixed,
    b_more_digits, b_neg_sign, b_neg_word, b_numeral_int, b_numeral_scale, b_percent_after, b_percent_before, b_plus,
    b_point, b_same, b_scale_and_fraction, b_scale_plus_fraction, b_sign_minus, b_sign_plus, b_ten_power,
    b_times_ten_power, b_vulgar, digit_proper_fraction, digits_non_negative, digits_only, fraction, int, integer,
    is_num, is_ten, non_negative, non_negative_integer, not_integer, num, ord, positive, positive_fraction,
    positive_integer, pow10, proper_fraction, relative, spelled_only, spelled_positive_fraction, suffix, term,
};
use crate::value::flag::{DIGITS, PLURAL, SPELLED};
use crate::value::{Num, NumKind, Ordinal, Value};

use super::{
    DEN, DIGSEQ, DOZEN, DUTCH, FRAC_WORD, GLUED, RATIO, HUNDRED, INT, NEG_WORD, NUM, NUMERAL, ONE_HALF, ORD, ORD_SCALE,
    ORD_SUFFIX, ORD_TENS, ORD_WORD, PCT, PCT_WORD, REL, REL_WORD, SCALE, SIGN, SIMPLE, TEEN, TENS, TO_END, UNIT,
    VULGAR,
};

// Tests on children.

fn unit_1_to_9(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| (1..=9).contains(&t))
}

fn singular_word(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PLURAL == 0)
}

fn thousand(v: &Value) -> bool {
    v.term() == Some((3, 0))
}

fn dozen_of_twelve(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| t == 12)
}

/// What `honderd` or `dozijn` multiplies: a spelled number below a
/// hundred that ends in no scale, or any positive digit integer.
fn small_multiplicand(v: &Value) -> bool {
    is_num(v, |n| n.grain == 0 && positive(n) && (digits_only(n) || n.value < Dec::from_i64(100)))
}

fn grain_from_2(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 2)
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

fn spelled_integer_from_grain_2(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && n.grain >= 2)
}

fn spelled_non_negative_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && !n.value.is_negative())
}

fn spelled_two_digits(v: &Value) -> bool {
    is_num(v, |n| spelled_only(n) && n.grain == 0 && n.value >= Dec::from_i64(10) && n.value < Dec::from_i64(100))
}

/// A fraction below one that is no ratio such as `vier op de tien`: the
/// part after `en` in a mixed number.
fn part_fraction(v: &Value) -> bool {
    proper_fraction(v) && !is_num(v, |n| n.has(RATIO))
}

fn spelled_proper_fraction(v: &Value) -> bool {
    proper_fraction(v) && is_num(v, spelled_only)
}

/// A spelled ordinal that can name a denominator: not `eerste`, and not
/// glued to the word before it (`honderdachtste` is the 108th).
fn denominator_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(SIMPLE) && !o.has(GLUED))
}

fn spelled_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED))
}

// Context checks.

/// Whether the match's first lexeme is glued to a word or numeral before
/// it.
fn glued_to_word(ctx: &Ctx<'_>) -> bool {
    ctx.lexeme(ctx.start).glued && ctx.before().is_some_and(|p| ctx.lexeme(p).is_alnum())
}

/// Whether a word follows the match that multiplies the number before it
/// but could not multiply a sum ending in a part below `grain`: in `drie
/// honderd en twee dozijn` the second part is `twee dozijn`.
fn before_multiplier(ctx: &Ctx<'_>, grain: i8) -> bool {
    ctx.after().is_some_and(|a| {
        DUTCH.lexicon().lookup(ctx.word(a)).iter().any(|e| match e.dim {
            HUNDRED | SCALE => e.value <= i64::from(grain),
            DOZEN => true,
            _ => false,
        })
    })
}

// Integers.

fn b_word_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0), 0, SPELLED)
}

fn b_unit_tens(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0 + term(&v[1]).0), 0, SPELLED)
}

fn b_hundred(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(pow10(2))?, 2, m.flags | SPELLED)
}

fn b_bare_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let s = term(&v[0]).0;
    int(pow10(s), s as i8, SPELLED)
}

fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, s) = (num(&v[0]), term(&v[1]).0);
    if !(m.grain == 0 && digits_only(&m)) && m.value >= pow10(s) {
        return None;
    }
    int(m.value.checked_mul(pow10(s))?, s as i8, m.flags | SPELLED)
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
    int(m.value.checked_mul(Dec::from_i64(term(&v[1]).0))?, 0, m.flags | SPELLED)
}

// Fractions.

fn b_den_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, f) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value: d, flags: f | SPELLED }))
}

/// A spelled fraction keeps the grain of its numerator, so a mixed number
/// can tell `honderd duizend biljoensten` (one fraction) from `honderd
/// drie vijfde`.
fn b_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]), ord(&v[1]));
    let value = n.value.checked_div(Dec::from_i64(d.value))?;
    Some(Value::Num(Num { value, kind: NumKind::Fraction, grain: n.grain, flags: SPELLED }))
}

/// `twintig drie vijfde`, `honderd driekwart`: a whole and a fraction
/// whose numerator the whole could not take as a multiplier.
fn b_mixed_bare(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    if f.grain > 0 && f.grain >= w.grain {
        return None;
    }
    b_mixed(v, ctx)
}

fn b_one_half(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::from_digits(b"1", b"5"), SPELLED)
}

/// `tweeenhalf`: a whole and one part.
fn b_and_part(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, d) = (num(&v[0]), term(&v[1]).0);
    fraction(w.value.checked_add(Dec::ONE.checked_div(Dec::from_i64(d))?)?, SPELLED)
}

/// `een half dozijn` is six.
fn b_fraction_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, size) = (num(&v[0]), term(&v[1]).0);
    let value = f.value.checked_mul(Dec::from_i64(size))?;
    let kind = if value.is_integer() { NumKind::Integer } else { NumKind::Fraction };
    Some(Value::Num(Num::new(value, kind, f.flags)))
}

/// `een op de drie`, `1 uit 21`: only a part below the whole, and not a
/// year after `van` or `op` (`3 van 2009`).
fn b_part_of(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]), num(&v[1]));
    let year = digits_only(&d) && d.value >= Dec::from_i64(1000) && d.value < Dec::from_i64(3000);
    if n.value >= d.value || year || after_currency(ctx) {
        return None;
    }
    fraction(n.value.checked_div(d.value)?, n.flags | d.flags | RATIO)
}

fn b_two_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Num(Num { value: num(&v[0]).value, kind: NumKind::Integer, grain: 2, flags: SPELLED }))
}

// Ordinals. Each records whether it is glued to the word before it.

fn ordinal(value: i64, flags: u8, ctx: &Ctx<'_>) -> Option<Value> {
    let glued = if glued_to_word(ctx) { GLUED } else { 0 };
    Some(Value::Ordinal(Ordinal { value, flags: flags | glued }))
}

fn b_ordinal_word(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (value, flags) = term(&v[0]);
    ordinal(value, flags | SPELLED, ctx)
}

fn b_ordinal_compound(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (u, (t, f)) = (term(&v[0]).0, term(&v[1]));
    ordinal(t + u, (f & PLURAL) | SPELLED, ctx)
}

fn b_ordinal_scale(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    if m.value >= pow10(s) {
        return None;
    }
    ordinal(m.value.checked_mul(pow10(s))?.to_i64()?, (f & PLURAL) | SPELLED, ctx)
}

fn b_ordinal_scale_bare(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (s, f) = term(&v[0]);
    ordinal(pow10(s).to_i64()?, (f & PLURAL) | SPELLED, ctx)
}

fn b_ordinal_sum(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (l, o) = (num(&v[0]), ord(&v[1]));
    if Dec::from_i64(o.value) >= pow10(i64::from(l.grain)) {
        return None;
    }
    ordinal(l.value.to_i64()?.checked_add(o.value)?, (o.flags & PLURAL) | SPELLED, ctx)
}

/// The suffix Dutch gives a digit ordinal: `ste` after what is spelled
/// with `-ste` (`1ste`, `8ste`, `20ste`, `100ste`), `de` after the rest
/// (`2de`, `11de`, `102de`); `e` goes with any (`11e`).
fn suffix_fits(n: i64, suffix: i64) -> bool {
    const STE: i64 = 1;
    let want = match n % 100 {
        _ if n == 0 => 2,
        0 | 1 | 8 | 20..=99 => STE,
        _ => 2,
    };
    suffix == 0 || suffix == want
}

fn b_ordinal_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = num(&v[0]).value.to_i64()?;
    suffix_fits(value, term(&v[1]).0).then_some(Value::Ordinal(Ordinal { value, flags: DIGITS }))
}

fn b_relative_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (offset, f) = term(&v[0]);
    relative(offset, if f == TO_END { OrdinalBase::End } else { OrdinalBase::Current })
}

/// `twee na laatste`, `de op een na laatste`: the end minus the count.
fn b_count_before_last(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    relative(-num(&v[0]).value.to_i64()?, OrdinalBase::End)
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
/// same text: a mixed number over an improper fraction, a fraction of a
/// large scale over a sum.
pub static RULES: &[Rule] = &[
    // Integers in words and digits.
    r("int_unit", INT, 0, &[node(UNIT)], b_word_int),
    r("int_teen", INT, 0, &[node(TEEN)], b_word_int),
    r("int_tens", INT, 0, &[node(TENS)], b_word_int),
    r("int_unit_tens", INT, 0, &[node_if(UNIT, unit_1_to_9), glued_word("en"), glued(TENS)], b_unit_tens),
    r("int_numeral", INT, 0, &[node_if(NUMERAL, integer)], b_numeral_int),
    r("int_hundred", INT, 0, &[node_if(INT, small_multiplicand), node_if(HUNDRED, singular_word)], b_hundred),
    r("int_bare_hundred", INT, 0, &[node_if(HUNDRED, singular_word)], b_bare_scale),
    r("int_bare_thousand", INT, 0, &[node_if(SCALE, thousand)], b_bare_scale),
    r("int_scale", INT, 0, &[node_if(INT, positive_integer), node_if(SCALE, singular_word)], b_scale),
    r("int_sum", INT, 0, &[node_if(INT, grain_from_2), opt_word("en"), node_if(INT, positive_integer)], b_sum),
    r("int_dozen", INT, 0, &[node_if(INT, small_multiplicand), node(DOZEN)], b_dozen),
    // Numbers.
    r("num_int", NUM, 0, &[node(INT)], b_same),
    r("num_numeral", NUM, 0, &[node_if(NUMERAL, not_integer)], b_same),
    r("num_numeral_hundred", NUM, 0, &[node_if(NUMERAL, not_integer), node(HUNDRED)], b_numeral_scale),
    r("num_numeral_scale", NUM, 0, &[node_if(NUMERAL, not_integer), node(SCALE)], b_numeral_scale),
    r("num_suffix_k", NUM, 0, &[node(NUMERAL), word("k")], |v, _| suffix(v, 3)),
    r("num_suffix_m", NUM, 0, &[node(NUMERAL), exact("M")], |v, _| suffix(v, 6)),
    r("num_suffix_g", NUM, 0, &[node(NUMERAL), word("g")], |v, _| suffix(v, 9)),
    r("num_suffix_b", NUM, 0, &[node(NUMERAL), word("b")], |v, _| suffix(v, 9)),
    r("num_suffix_t", NUM, 0, &[node(NUMERAL), exact("T")], |v, _| suffix(v, 12)),
    r("num_neg_word", NUM, 0, &[node(NEG_WORD), node_if(NUM, non_negative)], b_neg_word),
    r("num_neg_sign", NUM, 0, &[word("-"), node_if(NUM, digits_non_negative)], b_neg_sign),
    r("num_neg_minus_sign", NUM, 0, &[word("\u{2212}"), node_if(NUM, digits_non_negative)], b_neg_sign),
    // Fractions.
    r("den_ordinal", DEN, 0, &[node_if(ORD, denominator_ordinal)], b_same),
    r("den_word", DEN, 0, &[node(FRAC_WORD)], b_den_word),
    r("frac", NUM, 0, &[node_if(INT, spelled_positive_integer), opt_word("-"), node(DEN)], b_fraction),
    r("frac_slash", NUM, 0, &[node_if(INT, spelled_positive_integer), word("/"), node(DEN)], b_fraction),
    r("frac_one_half", NUM, 0, &[node(ONE_HALF)], b_one_half),
    r("frac_mixed", NUM, 1, &[node_if(NUM, non_negative_integer), word("en"), node_if(NUM, part_fraction)], b_mixed),
    r(
        "frac_mixed_bare",
        NUM,
        1,
        &[node_if(NUM, spelled_non_negative_integer), node_if(NUM, spelled_proper_fraction)],
        b_mixed_bare,
    ),
    r(
        "frac_and_part",
        NUM,
        1,
        &[node_if(INT, spelled_positive_integer), glued_word("en"), glued_if(FRAC_WORD, singular_word)],
        b_and_part,
    ),
    r(
        "frac_scale_and_fraction",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("en"), node_if(NUM, proper_fraction)],
        b_scale_and_fraction,
    ),
    r(
        "frac_scale_plus_fraction",
        NUM,
        1,
        &[node_if(INT, grain_from_3), word("en"), node_if(NUM, positive_fraction)],
        b_scale_plus_fraction,
    ),
    r(
        "frac_of_scale",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), opt_word("van"), node_if(SCALE, singular_word)],
        b_fraction_of_scale,
    ),
    r(
        "frac_of_dozen",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), node_if(DOZEN, dozen_of_twelve)],
        b_fraction_of_dozen,
    ),
    r(
        "frac_digits",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), glued_word("/"), glued_if(NUMERAL, positive_integer)],
        b_digit_fraction,
    ),
    r(
        "frac_digits_mixed",
        NUM,
        0,
        &[node_if(NUMERAL, non_negative_integer), node_if(NUM, digit_proper_fraction)],
        b_mixed,
    ),
    r("frac_vulgar", NUM, 0, &[node(VULGAR)], b_vulgar),
    r("frac_op", NUM, 0, &[node_if(INT, integer), word("op"), opt_word("de"), node_if(INT, positive_integer)], b_part_of),
    r(
        "frac_uit",
        NUM,
        0,
        &[node_if(INT, integer), word("uit"), opt_word("de"), node_if(INT, positive_integer)],
        b_part_of,
    ),
    r(
        "frac_van_de",
        NUM,
        0,
        &[node_if(INT, integer), word("van"), word("de"), node_if(INT, positive_integer)],
        b_part_of,
    ),
    // Spoken decimals.
    r("digits_unit", DIGSEQ, 0, &[node(UNIT)], b_digit_word),
    r("digits_more", DIGSEQ, 0, &[node(DIGSEQ), node(UNIT)], b_more_digits),
    r("digits_two", DIGSEQ, 0, &[node_if(INT, spelled_two_digits)], b_two_digits),
    r("num_komma", NUM, 0, &[node_if(INT, spelled_non_negative_integer), word("komma"), node(DIGSEQ)], b_point),
    // Exponents.
    r("sign_plus", SIGN, 0, &[word("+")], b_plus),
    r("sign_minus", SIGN, 0, &[word("-")], b_minus),
    r("sign_more_plus", SIGN, 0, &[node(SIGN), glued_word("+")], b_sign_plus),
    r("sign_more_minus", SIGN, 0, &[node(SIGN), glued_word("-")], b_sign_minus),
    r("pow_e", NUM, 0, &[node(NUMERAL), glued_word("e"), glued_if(NUMERAL, integer)], b_ten_power),
    r("pow_e_signed", NUM, 0, &[node(NUMERAL), glued_word("e"), glued(SIGN), glued_if(NUMERAL, integer)], b_ten_power),
    r("pow_caret", NUM, 0, &[node(NUMERAL), glued_word("^"), glued_if(NUMERAL, integer)], b_caret),
    r("pow_caret_signed", NUM, 0, &[node(NUMERAL), glued_word("^"), glued(SIGN), glued_if(NUMERAL, integer)], b_caret),
    r(
        "pow_times_ten",
        NUM,
        0,
        &[node(NUMERAL), glued_word("x"), glued_if(NUMERAL, is_ten), glued_word("^"), glued_if(NUMERAL, integer)],
        b_times_ten_power,
    ),
    // Ordinals.
    r("ord_word", ORD, 0, &[node(ORD_WORD)], b_ordinal_word),
    r("ord_tens", ORD, 0, &[node(ORD_TENS)], b_ordinal_word),
    r(
        "ord_compound",
        ORD,
        1,
        &[node_if(UNIT, unit_1_to_9), glued_word("en"), glued(ORD_TENS)],
        b_ordinal_compound,
    ),
    r("ord_scale", ORD, 0, &[node_if(INT, spelled_positive_integer), node(ORD_SCALE)], b_ordinal_scale),
    r("ord_scale_bare", ORD, 0, &[node(ORD_SCALE)], b_ordinal_scale_bare),
    r(
        "ord_sum",
        ORD,
        0,
        &[node_if(INT, spelled_integer_from_grain_2), opt_word("en"), node_if(ORD, spelled_ordinal)],
        b_ordinal_sum,
    ),
    r("ord_digits", ORD, 0, &[node_if(NUMERAL, non_negative_integer), glued(ORD_SUFFIX)], b_ordinal_digits),
    // Relative ordinals.
    r("rel_word", REL, 0, &[node(REL_WORD)], b_relative_word),
    r(
        "rel_count_before_last",
        REL,
        0,
        &[node_if(INT, spelled_positive_integer), word("na"), word("laatste")],
        b_count_before_last,
    ),
    r(
        "rel_op_count_before_last",
        REL,
        0,
        &[word("op"), node_if(INT, spelled_positive_integer), word("na"), word("laatste")],
        b_count_before_last,
    ),
    r(
        "rel_de_op_count_before_last",
        REL,
        0,
        &[word("de"), word("op"), node_if(INT, spelled_positive_integer), word("na"), word("laatste")],
        b_count_before_last,
    ),
    r(
        "rel_one_before_last",
        REL,
        0,
        &[word("de"), word("een"), word("voor"), word("de"), word("laatste")],
        b_end_minus_one,
    ),
    r("rel_that_before_last", REL, 0, &[word("die"), word("voor"), word("de"), word("laatste")], b_end_minus_one),
    // Percentages.
    r("pct_sign", PCT, 0, &[node(NUM), word("%")], b_percent_after),
    r("pct_word", PCT, 0, &[node(NUM), node(PCT_WORD)], b_percent_after),
    r("pct_word_of", PCT, 0, &[node(PCT_WORD), word("van"), node(NUM)], b_percent_before),
];
