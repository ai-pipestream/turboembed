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

use super::{
    ARTICLE, DEN, DIGSEQ, DOZEN, ENGLISH, FRAC_WORD, HUNDRED, INT, NEG_WORD, NUM, NUMERAL, ORD, ORD_SCALE, ORD_SUFFIX,
    ORD_TENS, ORD_WORD, PCT, PCT_WORD, REL, REL_WORD, SCALE, SIGN, SIMPLE, TEEN, TENS, TO_CURRENT, TO_END, UNIT,
    VULGAR,
};

// Reading children. A rule's items fix each child's category, and every
// category holds one kind of value, so these cannot fail on a rule's own
// children.

fn num(v: &Value) -> Num {
    *v.num().expect("a number child")
}

fn term(v: &Value) -> (i64, u8) {
    v.term().expect("a lexicon child")
}

fn ord(v: &Value) -> Ordinal {
    *v.ordinal().expect("an ordinal child")
}

fn pow10(e: i64) -> Dec {
    Dec::ONE.shift(e as i32).expect("a power of ten of a scale word")
}

fn int(value: Dec, grain: i8, flags: u8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain, flags }))
}

fn number(value: Dec, kind: NumKind, flags: u8) -> Option<Value> {
    Some(Value::Num(Num::new(value, kind, flags)))
}

fn fraction(value: Dec, flags: u8) -> Option<Value> {
    number(value, NumKind::Fraction, flags)
}

// Tests on children.

fn is_num(v: &Value, f: impl Fn(&Num) -> bool) -> bool {
    v.num().is_some_and(f)
}

fn digits_only(n: &Num) -> bool {
    n.has(DIGITS) && !n.has(SPELLED)
}

fn spelled_only(n: &Num) -> bool {
    n.has(SPELLED) && !n.has(DIGITS)
}

fn positive(n: &Num) -> bool {
    n.value > Dec::ZERO
}

fn integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer)
}

fn not_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind != NumKind::Integer)
}

fn positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && positive(n))
}

fn non_negative(v: &Value) -> bool {
    is_num(v, |n| !n.value.is_negative())
}

fn non_negative_integer(v: &Value) -> bool {
    integer(v) && non_negative(v)
}

/// What `hundred` or `dozen` multiplies: a spelled number below a hundred
/// that ends in no scale, or any positive digit integer (`322 hundred`).
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

fn digits_non_negative(v: &Value) -> bool {
    is_num(v, |n| digits_only(n) && !n.value.is_negative())
}

fn spelled_positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && positive(n))
}

fn spelled_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n))
}

fn spelled_integer_from_grain_2(v: &Value) -> bool {
    spelled_integer(v) && grain_from_2(v)
}

/// A spelled number of two digits with no scale: `seventy-one`, `eleven`.
fn spelled_two_digits(v: &Value) -> bool {
    is_num(v, |n| spelled_only(n) && n.grain == 0 && n.value >= Dec::from_i64(10) && n.value < Dec::from_i64(100))
}

fn proper_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n) && n.value < Dec::ONE)
}

fn positive_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n))
}

fn spelled_positive_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n) && !n.has(DIGITS))
}

fn digit_proper_fraction(v: &Value) -> bool {
    proper_fraction(v) && is_num(v, digits_only)
}

fn is_ten(v: &Value) -> bool {
    is_num(v, |n| n.value == Dec::from_i64(10))
}

fn unit_1_to_9(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| (1..=9).contains(&t))
}

fn singular_word(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PLURAL == 0)
}

fn half(v: &Value) -> bool {
    v.term() == Some((2, 0))
}

fn spelled_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED))
}

fn denominator_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(SIMPLE))
}

fn singular_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| !o.has(PLURAL))
}

/// `second` to `tenth`, singular, before `to last`.
fn counted_from_last(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(PLURAL) && (2..=10).contains(&o.value))
}

fn counts_from_current(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f == TO_CURRENT)
}

// Context checks.

/// Whether a minus at the match's first lexeme opens a negative number.
/// It does not right after a number (`5 - 10`, `5k -10`: a range), nor
/// glued to what comes before unless that is an opening bracket or quote
/// (`SCP-173`, `+1-222`).
fn minus_opens(ctx: &Ctx<'_>) -> bool {
    let Some(p) = ctx.before() else { return true };
    let prev = ctx.lexeme(p);
    let after_number = prev.kind == Kind::Numeral
        || (prev.kind == Kind::Word && prev.glued && p > 0 && ctx.lexeme(p - 1).kind == Kind::Numeral);
    if after_number {
        return false;
    }
    !ctx.lexeme(ctx.start).glued || matches!(ctx.punct(p), Some('(' | '[' | '{' | '"' | '\'' | '“' | '‘' | '«' | '='))
}

/// Whether a digit fraction stands alone, not inside a run such as a date
/// (`5/17/2018`) or a phone number (`222-2222/2222`).
fn slash_alone(ctx: &Ctx<'_>) -> bool {
    if ctx.lexeme(ctx.start).glued
        && let Some(p) = ctx.before()
    {
        match ctx.punct(p) {
            Some('/') => return false,
            Some('-' | '.' | ',' | ':') if p > 0 && ctx.lexeme(p).glued && ctx.lexeme(p - 1).is_alnum() => {
                return false;
            }
            _ => {}
        }
    }
    !ctx.after().is_some_and(|a| ctx.lexeme(a).glued && ctx.punct(a) == Some('/'))
}

/// Whether a currency symbol comes just before the match: `$2000 over 3
/// years` is money and a duration, not a fraction.
fn after_currency(ctx: &Ctx<'_>) -> bool {
    ctx.before().and_then(|p| ctx.punct(p)).is_some_and(|c| general_category(c) == GeneralCategory::Sc)
}

/// Whether a word follows the match that multiplies the number before it
/// but could not multiply a sum ending in a part below `grain`. A sum
/// that stops there would take that number from its multiplier: in `3
/// hundred and 2 hundred` the second number is `2 hundred`, not the `2`
/// of `302`, while `two hundred and two thousand` is 202 thousands.
fn before_multiplier(ctx: &Ctx<'_>, grain: i8) -> bool {
    ctx.after().is_some_and(|a| {
        ENGLISH.lexicon().lookup(ctx.word(a)).iter().any(|e| match e.dim {
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

fn b_tens_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0 + term(&v[1]).0), 0, SPELLED)
}

fn b_numeral_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

fn b_hundred(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(pow10(2))?, 2, m.flags | SPELLED)
}

fn b_a_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let s = term(&v[1]).0;
    int(pow10(s), s as i8, SPELLED)
}

fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, s) = (num(&v[0]), term(&v[1]).0);
    // A plain digit integer takes any scale (`322 thousand`); otherwise
    // the multiplicand is below the scale, so `two hundred thousand`
    // reads and `two million thousand` does not.
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
    int(m.value.checked_mul(Dec::from_i64(12))?, 0, m.flags | SPELLED)
}

fn b_a_dozen(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(12), 0, SPELLED)
}

/// A number from its parts' sole child, unchanged.
fn b_same(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

fn b_numeral_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, s) = (num(&v[0]), term(&v[1]).0);
    let value = m.value.checked_mul(pow10(s))?;
    number(value, NumKind::of(value), DIGITS | SPELLED)
}

fn suffix(v: &[Value], e: i64) -> Option<Value> {
    let value = num(&v[0]).value.checked_mul(pow10(e))?;
    number(value, NumKind::of(value), DIGITS)
}

fn b_neg_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[1]);
    Some(Value::Num(Num { value: n.value.negated(), flags: n.flags | SPELLED, ..n }))
}

fn b_neg_sign(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !minus_opens(ctx) {
        return None;
    }
    let n = num(&v[0]);
    Some(Value::Num(Num { value: n.value.negated(), ..n }))
}

// Fractions.

fn b_den_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

fn b_den_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, f) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value: d, flags: f | SPELLED }))
}

fn b_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]).value, ord(&v[1]));
    // One takes the singular (`one fifth`), anything else the plural.
    if (n == Dec::ONE) == d.has(PLURAL) {
        return None;
    }
    fraction(n.checked_div(Dec::from_i64(d.value))?, SPELLED)
}

fn b_article_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(ord(&v[1]).value))?, SPELLED)
}

fn b_half(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(2))?, SPELLED)
}

fn b_mixed(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    fraction(w.value.checked_add(f.value)?, w.flags | f.flags)
}

/// `one million and a half`: the fraction is of the scale.
fn b_scale_and_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    fraction(w.value.checked_add(f.value.checked_mul(pow10(i64::from(w.grain)))?)?, w.flags | f.flags)
}

/// `one billion and three quarters million`: a fractional part below the
/// whole's last scale.
fn b_scale_plus_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, f) = (num(&v[0]), num(&v[1]));
    if f.value >= pow10(i64::from(w.grain)) {
        return None;
    }
    fraction(w.value.checked_add(f.value)?, w.flags | f.flags)
}

/// `two thirds of a million`, `five and a half million`.
fn b_fraction_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (f, s) = (num(&v[0]), term(&v[1]).0);
    if f.value >= pow10(s) {
        return None;
    }
    fraction(f.value.checked_mul(pow10(s))?, f.flags)
}

/// `half million`, `quarter of a billion`.
fn b_word_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, s) = (term(&v[0]).0, term(&v[1]).0);
    fraction(pow10(s).checked_div(Dec::from_i64(d))?, SPELLED)
}

/// `half a dozen` is six, an integer.
fn of_dozen(value: Dec, flags: u8) -> Option<Value> {
    let value = value.checked_mul(Dec::from_i64(12))?;
    let kind = if value.is_integer() { NumKind::Integer } else { NumKind::Fraction };
    number(value, kind, flags)
}

fn b_fraction_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let f = num(&v[0]);
    of_dozen(f.value, f.flags)
}

fn b_word_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    of_dozen(Dec::ONE.checked_div(Dec::from_i64(term(&v[0]).0))?, SPELLED)
}

fn b_digit_fraction(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !slash_alone(ctx) {
        return None;
    }
    fraction(num(&v[0]).value.checked_div(num(&v[1]).value)?, DIGITS)
}

fn b_vulgar(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let t = term(&v[0]).0;
    fraction(Dec::from_i64(t / 1000).checked_div(Dec::from_i64(t % 1000))?, DIGITS)
}

fn b_over(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if after_currency(ctx) {
        return None;
    }
    let (n, d) = (num(&v[0]), num(&v[1]));
    fraction(n.value.checked_div(d.value)?, n.flags | d.flags)
}

/// `one in three`, `two out of five`: only a part below the whole reads
/// as a fraction, so `12 in six boxes` is two numbers.
fn b_part_of(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]), num(&v[1]));
    if n.value >= d.value {
        return None;
    }
    b_over(v, ctx)
}

/// `1 in 5`; but `3 in 2009` is a number and a year.
fn b_one_in(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let d = num(&v[1]);
    let year = digits_only(&d) && d.value >= Dec::from_i64(1000) && d.value < Dec::from_i64(3000);
    if year {
        return None;
    }
    b_part_of(v, ctx)
}

// Spoken decimals: `two point zero three`.

fn digits_value(value: Dec, count: i8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain: count, flags: SPELLED }))
}

fn b_digit_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    digits_value(Dec::from_i64(term(&v[0]).0), 1)
}

fn b_more_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let d = num(&v[0]);
    if d.grain >= 30 {
        return None;
    }
    digits_value(d.value.checked_mul(Dec::from_i64(10))?.checked_add(Dec::from_i64(term(&v[1]).0))?, d.grain + 1)
}

fn b_two_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    digits_value(num(&v[0]).value, 2)
}

fn b_point(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (w, d) = (num(&v[0]), num(&v[1]));
    let value = w.value.checked_add(d.value.shift(-i32::from(d.grain))?)?;
    number(value, NumKind::Decimal, SPELLED)
}

// Exponents.

fn b_plus(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: 1, flags: 0 })
}

fn b_minus(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: -1, flags: 0 })
}

fn b_sign_plus(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(v[0])
}

fn b_sign_minus(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: -term(&v[0]).0, flags: 0 })
}

/// The exponent written last, with the sign run before it if any.
fn exponent(v: &[Value]) -> Option<i32> {
    let e = num(v.last().expect("an exponent")).value.to_i64()?;
    let sign = if v.len() > 2 { term(&v[v.len() - 2]).0 } else { 1 };
    i32::try_from(e * sign).ok()
}

fn power(value: Dec) -> Option<Value> {
    number(value, NumKind::Power, DIGITS)
}

fn b_ten_power(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    power(num(&v[0]).value.shift(exponent(v)?)?)
}

/// `1.3x10^6`: the ten is a child too, before the optional sign.
fn b_times_ten_power(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let e = exponent(&v[1..])?;
    power(num(&v[0]).value.shift(e)?)
}

fn b_caret(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let e = exponent(v)?;
    if e.unsigned_abs() > 64 {
        return None;
    }
    power(num(&v[0]).value.pow(e)?)
}

// Ordinals.

fn b_ordinal_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, flags) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value, flags: flags | SPELLED }))
}

fn b_ordinal_compound(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (t, (u, f)) = (term(&v[0]).0, term(&v[1]));
    Some(Value::Ordinal(Ordinal { value: t + u, flags: (f & PLURAL) | SPELLED }))
}

fn b_ordinal_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    if m.value >= pow10(s) {
        return None;
    }
    let value = m.value.checked_mul(pow10(s))?.to_i64()?;
    Some(Value::Ordinal(Ordinal { value, flags: (f & PLURAL) | SPELLED }))
}

fn b_ordinal_scale_bare(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (s, f) = term(&v[0]);
    Some(Value::Ordinal(Ordinal { value: pow10(s).to_i64()?, flags: (f & PLURAL) | SPELLED }))
}

fn b_ordinal_sum(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, o) = (num(&v[0]), ord(&v[1]));
    if Dec::from_i64(o.value) >= pow10(i64::from(l.grain)) {
        return None;
    }
    let value = l.value.to_i64()?.checked_add(o.value)?;
    Some(Value::Ordinal(Ordinal { value, flags: (o.flags & PLURAL) | SPELLED }))
}

/// `1st`, `22nd`, `13th`: the suffix has to be the one English gives the
/// number.
fn b_ordinal_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = num(&v[0]).value.to_i64()?;
    let want = match (value % 100, value % 10) {
        (11..=13, _) => 0,
        (_, d @ 1..=3) => d,
        _ => 0,
    };
    (term(&v[1]).0 == want).then_some(Value::Ordinal(Ordinal { value, flags: DIGITS }))
}

fn relative(offset: i64, to: OrdinalBase) -> Option<Value> {
    Some(Value::Relative(Relative { offset: i32::try_from(offset).ok()?, to }))
}

fn b_relative_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (offset, f) = term(&v[0]);
    relative(offset, if f == TO_END { OrdinalBase::End } else { OrdinalBase::Current })
}

/// `second to last` is the end minus one.
fn b_to_last(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    relative(1 - ord(&v[0]).value, OrdinalBase::End)
}

fn b_end_minus_one(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    relative(-1, OrdinalBase::End)
}

// Percentages.

fn b_percent_after(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Percent(num(&v[0]).value))
}

fn b_percent_before(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Percent(num(v.last().expect("a number")).value))
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

/// Every rule. A weight above zero makes a reading win over others of
/// the same text: a mixed number over an improper fraction, a fraction of
/// a large scale over a sum, a compound ordinal denominator over a
/// longer numerator.
pub static RULES: &[Rule] = &[
    // Integers in words and digits.
    r("int_unit", INT, 0, &[node(UNIT)], b_word_int),
    r("int_teen", INT, 0, &[node(TEEN)], b_word_int),
    r("int_tens", INT, 0, &[node(TENS)], b_word_int),
    r("int_tens_unit", INT, 0, &[node(TENS), opt_word("-"), node_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_tens_and_unit", INT, -1, &[node(TENS), word("and"), node_if(UNIT, unit_1_to_9)], b_tens_unit),
    r("int_numeral", INT, 0, &[node_if(NUMERAL, integer)], b_numeral_int),
    r("int_hundred", INT, 0, &[node_if(INT, small_multiplicand), node(HUNDRED)], b_hundred),
    r("int_a_hundred", INT, 0, &[node(ARTICLE), node(HUNDRED)], b_a_scale),
    r("int_scale", INT, 0, &[node_if(INT, positive_integer), node(SCALE)], b_scale),
    r("int_a_scale", INT, 0, &[node(ARTICLE), node(SCALE)], b_a_scale),
    r("int_sum", INT, 0, &[node_if(INT, grain_from_2), opt_word("and"), node_if(INT, positive_integer)], b_sum),
    r("int_dozen", INT, 0, &[node_if(INT, small_multiplicand), node(DOZEN)], b_dozen),
    r("int_a_dozen", INT, 0, &[node(ARTICLE), node(DOZEN)], b_a_dozen),
    // Numbers.
    r("num_int", NUM, 0, &[node(INT)], b_same),
    r("num_numeral", NUM, 0, &[node_if(NUMERAL, not_integer)], b_same),
    r("num_numeral_hundred", NUM, 0, &[node_if(NUMERAL, not_integer), node(HUNDRED)], b_numeral_scale),
    r("num_numeral_scale", NUM, 0, &[node_if(NUMERAL, not_integer), node(SCALE)], b_numeral_scale),
    r("num_suffix_k", NUM, 0, &[node(NUMERAL), word("k")], |v, _| suffix(v, 3)),
    r("num_suffix_m", NUM, 0, &[node(NUMERAL), exact("M")], |v, _| suffix(v, 6)),
    r("num_suffix_mm", NUM, 0, &[node(NUMERAL), exact("MM")], |v, _| suffix(v, 6)),
    r("num_suffix_mil", NUM, 0, &[node(NUMERAL), word("mil")], |v, _| suffix(v, 6)),
    r("num_suffix_g", NUM, 0, &[node(NUMERAL), exact("G")], |v, _| suffix(v, 9)),
    r("num_suffix_b", NUM, 0, &[node(NUMERAL), word("b")], |v, _| suffix(v, 9)),
    r("num_suffix_t", NUM, 0, &[node(NUMERAL), exact("T")], |v, _| suffix(v, 12)),
    r("num_neg_word", NUM, 0, &[node(NEG_WORD), node_if(NUM, non_negative)], b_neg_word),
    r("num_neg_sign", NUM, 0, &[word("-"), node_if(NUM, digits_non_negative)], b_neg_sign),
    r("num_neg_minus_sign", NUM, 0, &[word("\u{2212}"), node_if(NUM, digits_non_negative)], b_neg_sign),
    // Fractions.
    r("den_ordinal", DEN, 0, &[node_if(ORD, denominator_ordinal)], b_den_ordinal),
    r("den_word", DEN, 0, &[node(FRAC_WORD)], b_den_word),
    r("frac", NUM, 0, &[node_if(INT, spelled_positive_integer), opt_word("-"), node(DEN)], b_fraction),
    r("frac_article", NUM, 0, &[node(ARTICLE), node_if(DEN, singular_ordinal)], b_article_fraction),
    r("frac_half", NUM, 0, &[node_if(FRAC_WORD, half)], b_half),
    r("frac_mixed", NUM, 1, &[node_if(NUM, non_negative_integer), word("and"), node_if(NUM, proper_fraction)], b_mixed),
    r(
        "frac_scale_and_fraction",
        NUM,
        2,
        &[node_if(INT, grain_from_6), word("and"), node_if(NUM, proper_fraction)],
        b_scale_and_fraction,
    ),
    r(
        "frac_scale_plus_fraction",
        NUM,
        1,
        &[node_if(INT, grain_from_3), word("and"), node_if(NUM, positive_fraction)],
        b_scale_plus_fraction,
    ),
    r(
        "frac_of_scale",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), opt_word("of"), opt_word("a"), node(SCALE)],
        b_fraction_of_scale,
    ),
    r(
        "frac_word_of_scale",
        NUM,
        0,
        &[node_if(FRAC_WORD, singular_word), opt_word("of"), opt_word("a"), node(SCALE)],
        b_word_of_scale,
    ),
    r(
        "frac_of_dozen",
        NUM,
        0,
        &[node_if(NUM, spelled_positive_fraction), opt_word("of"), opt_word("a"), node(DOZEN)],
        b_fraction_of_dozen,
    ),
    r(
        "frac_word_of_dozen",
        NUM,
        0,
        &[node_if(FRAC_WORD, singular_word), opt_word("of"), opt_word("a"), node(DOZEN)],
        b_word_of_dozen,
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
    r("frac_over", NUM, 0, &[node_if(INT, integer), word("over"), node_if(INT, positive_integer)], b_over),
    r("frac_in", NUM, 0, &[node_if(INT, integer), word("in"), node_if(INT, positive_integer)], b_one_in),
    r(
        "frac_out_of",
        NUM,
        0,
        &[node_if(INT, integer), word("out"), word("of"), node_if(INT, positive_integer)],
        b_part_of,
    ),
    // Spoken decimals.
    r("digits_unit", DIGSEQ, 0, &[node(UNIT)], b_digit_word),
    r("digits_more", DIGSEQ, 0, &[node(DIGSEQ), node(UNIT)], b_more_digits),
    r("digits_two", DIGSEQ, 0, &[node_if(INT, spelled_two_digits)], b_two_digits),
    r("num_point", NUM, 0, &[node_if(INT, spelled_integer), word("point"), node(DIGSEQ)], b_point),
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
    r(
        "pow_times_ten_signed",
        NUM,
        0,
        &[
            node(NUMERAL),
            glued_word("x"),
            glued_if(NUMERAL, is_ten),
            glued_word("^"),
            glued(SIGN),
            glued_if(NUMERAL, integer),
        ],
        b_times_ten_power,
    ),
    // Ordinals.
    r("ord_word", ORD, 0, &[node(ORD_WORD)], b_ordinal_word),
    r("ord_tens", ORD, 0, &[node(ORD_TENS)], b_ordinal_word),
    r("ord_compound", ORD, 1, &[node(TENS), opt_word("-"), node_if(ORD_WORD, unit_1_to_9)], b_ordinal_compound),
    r("ord_scale", ORD, 0, &[node_if(INT, spelled_positive_integer), node(ORD_SCALE)], b_ordinal_scale),
    r("ord_scale_bare", ORD, 0, &[node(ORD_SCALE)], b_ordinal_scale_bare),
    r(
        "ord_sum",
        ORD,
        0,
        &[node_if(INT, spelled_integer_from_grain_2), opt_word("and"), node_if(ORD, spelled_ordinal)],
        b_ordinal_sum,
    ),
    r("ord_digits", ORD, 0, &[node_if(NUMERAL, positive_integer), glued(ORD_SUFFIX)], b_ordinal_digits),
    // Relative ordinals.
    r("rel_word", REL, 0, &[node(REL_WORD)], b_relative_word),
    r("rel_one", REL, 0, &[node_if(REL_WORD, counts_from_current), word("one")], b_relative_word),
    r("rel_to_last", REL, 0, &[node_if(ORD, counted_from_last), word("to"), word("last")], b_to_last),
    r("rel_the_to_last", REL, 0, &[word("the"), node_if(ORD, counted_from_last), word("to"), word("last")], b_to_last),
    r("rel_next_to_last", REL, 0, &[word("next"), word("to"), word("last")], b_end_minus_one),
    r("rel_last_but_one", REL, 0, &[word("last"), word("but"), word("one")], b_end_minus_one),
    r("rel_the_last_but_one", REL, 0, &[word("the"), word("last"), word("but"), word("one")], b_end_minus_one),
    r(
        "rel_one_before_last",
        REL,
        0,
        &[word("the"), word("one"), word("before"), word("the"), word("last"), opt_word("one")],
        b_end_minus_one,
    ),
    // Percentages.
    r("pct_sign", PCT, 0, &[node(NUM), word("%")], b_percent_after),
    r("pct_word", PCT, 0, &[node(NUM), node(PCT_WORD)], b_percent_after),
    r("pct_per_cent", PCT, 0, &[node(NUM), word("per"), word("cent")], b_percent_after),
    r("pct_per_cents", PCT, 0, &[node(NUM), word("per"), word("cents")], b_percent_after),
    r("pct_word_of", PCT, 0, &[node(PCT_WORD), word("of"), node(NUM)], b_percent_before),
    r("pct_per_cent_of", PCT, 0, &[word("per"), word("cent"), word("of"), node(NUM)], b_percent_before),
    r("pct_per_cents_of", PCT, 0, &[word("per"), word("cents"), word("of"), node(NUM)], b_percent_before),
];
