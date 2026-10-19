use super::*;

pub(in crate::lang) fn term(v: &Value) -> (i64, u8) {
    v.term().expect("a lexicon child")
}

pub(in crate::lang) fn ord(v: &Value) -> Ordinal {
    *v.ordinal().expect("an ordinal child")
}

pub(in crate::lang) fn pow10(e: i64) -> Dec {
    Dec::ONE.shift(e as i32).expect("a power of ten of a scale word")
}

pub(in crate::lang) fn int(value: Dec, grain: i8, flags: u8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain, flags }))
}

pub(in crate::lang) fn number(value: Dec, kind: NumKind, flags: u8) -> Option<Value> {
    Some(Value::Num(Num::new(value, kind, flags)))
}

pub(in crate::lang) fn fraction(value: Dec, flags: u8) -> Option<Value> {
    number(value, NumKind::Fraction, flags)
}

// Tests on children.

pub(in crate::lang) fn is_num(v: &Value, f: impl Fn(&Num) -> bool) -> bool {
    v.num().is_some_and(f)
}

pub(in crate::lang) fn digits_only(n: &Num) -> bool {
    n.has(DIGITS) && !n.has(SPELLED)
}

pub(in crate::lang) fn spelled_only(n: &Num) -> bool {
    n.has(SPELLED) && !n.has(DIGITS)
}

pub(in crate::lang) fn positive(n: &Num) -> bool {
    n.value > Dec::ZERO
}

pub(in crate::lang) fn integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer)
}

pub(in crate::lang) fn not_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind != NumKind::Integer)
}

pub(in crate::lang) fn positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && positive(n))
}

pub(in crate::lang) fn non_negative(v: &Value) -> bool {
    is_num(v, |n| !n.value.is_negative())
}

pub(in crate::lang) fn non_negative_integer(v: &Value) -> bool {
    integer(v) && non_negative(v)
}

/// What `hundred` or `dozen` multiplies: a spelled number below a hundred
/// that ends in no scale, or any positive digit integer (`322 hundred`).
pub(super) fn small_multiplicand(v: &Value) -> bool {
    is_num(v, |n| n.grain == 0 && positive(n) && (digits_only(n) || n.value < Dec::from_i64(100)))
}

pub(super) fn grain_from_2(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 2)
}

pub(super) fn grain_from_3(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 3)
}

pub(super) fn grain_from_6(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 6)
}

pub(in crate::lang) fn digits_non_negative(v: &Value) -> bool {
    is_num(v, |n| digits_only(n) && !n.value.is_negative())
}

pub(super) fn spelled_positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && positive(n))
}

pub(super) fn spelled_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n))
}

pub(super) fn spelled_integer_from_grain_2(v: &Value) -> bool {
    spelled_integer(v) && grain_from_2(v)
}

/// A spelled number of two digits with no scale: `seventy-one`, `eleven`.
pub(super) fn spelled_two_digits(v: &Value) -> bool {
    is_num(v, |n| spelled_only(n) && n.grain == 0 && n.value >= Dec::from_i64(10) && n.value < Dec::from_i64(100))
}

pub(in crate::lang) fn proper_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n) && n.value < Dec::ONE)
}

pub(in crate::lang) fn positive_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n))
}

pub(in crate::lang) fn spelled_positive_fraction(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Fraction && positive(n) && !n.has(DIGITS))
}

pub(in crate::lang) fn digit_proper_fraction(v: &Value) -> bool {
    proper_fraction(v) && is_num(v, digits_only)
}

pub(in crate::lang) fn is_ten(v: &Value) -> bool {
    is_num(v, |n| n.value == Dec::from_i64(10))
}

pub(super) fn unit_1_to_9(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| (1..=9).contains(&t))
}

pub(super) fn singular_word(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PLURAL == 0)
}

pub(super) fn half(v: &Value) -> bool {
    v.term() == Some((2, 0))
}

pub(super) fn spelled_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED))
}

pub(super) fn denominator_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(SIMPLE))
}

pub(super) fn singular_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| !o.has(PLURAL))
}

/// `second` to `tenth`, singular, before `to last`.
pub(super) fn counted_from_last(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(PLURAL) && (2..=10).contains(&o.value))
}

pub(super) fn counts_from_current(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f == TO_CURRENT)
}

// Context checks.

/// Whether a minus at the match's first lexeme opens a negative number.
/// It does not right after a number (`5 - 10`, `5k -10`: a range), nor
/// glued to what comes before unless that is an opening bracket or quote
/// (`SCP-173`, `+1-222`).
pub(in crate::lang) fn minus_opens(ctx: &Ctx<'_>) -> bool {
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
pub(in crate::lang) fn slash_alone(ctx: &Ctx<'_>) -> bool {
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
pub(in crate::lang) fn after_currency(ctx: &Ctx<'_>) -> bool {
    ctx.before().and_then(|p| ctx.punct(p)).is_some_and(|c| general_category(c) == GeneralCategory::Sc)
}

/// Whether a word follows the match that multiplies the number before it
/// but could not multiply a sum ending in a part below `grain`. A sum
/// that stops there would take that number from its multiplier: in `3
/// hundred and 2 hundred` the second number is `2 hundred`, not the `2`
/// of `302`, while `two hundred and two thousand` is 202 thousands.
pub(super) fn before_multiplier(ctx: &Ctx<'_>, grain: i8) -> bool {
    ctx.after().is_some_and(|a| {
        ENGLISH.lexicon().lookup(ctx.word(a)).iter().any(|e| match e.dim {
            HUNDRED | SCALE => e.value <= i64::from(grain),
            DOZEN => true,
            _ => false,
        })
    })
}
