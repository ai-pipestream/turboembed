use super::*;

pub(crate) fn int(value: Dec, grain: i8, flags: u8) -> Option<Value> {
    Some(Value::Num(Num { value, kind: NumKind::Integer, grain, flags }))
}

pub(crate) fn fraction(value: Dec, flags: u8) -> Option<Value> {
    Some(Value::Num(Num::new(value, NumKind::Fraction, flags)))
}

pub(crate) fn ordinal(value: i64, flags: u8) -> Option<Value> {
    Some(Value::Ordinal(Ordinal { value, flags }))
}

// Tests on children.

pub(crate) fn unit_1_to_9(v: &Value) -> bool {
    v.term().is_some_and(|(t, _)| (1..=9).contains(&t))
}

pub(crate) fn grain_from_2(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 2 && !n.has(CLOSED))
}

pub(crate) fn grain_from_3(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 3)
}

pub(crate) fn grain_from_6(v: &Value) -> bool {
    is_num(v, |n| n.grain >= 6)
}

pub(crate) fn spelled_positive_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n) && positive(n))
}

pub(crate) fn spelled_integer(v: &Value) -> bool {
    is_num(v, |n| n.kind == NumKind::Integer && spelled_only(n))
}

/// What `docena` multiplies: a number below a hundred with no scale, or
/// any positive digit integer.
pub(crate) fn small_multiplicand(v: &Value) -> bool {
    is_num(v, |n| n.grain == 0 && positive(n) && (n.has(DIGITS) || n.value < Dec::from_i64(100)))
}

pub(crate) fn singular(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & PLURAL == 0)
}

/// `medio` or `media`, said of one half.
pub(crate) fn one_half(v: &Value) -> bool {
    v.term().is_some_and(|(d, f)| d == 2 && f & (PLURAL | NOUN) == 0)
}

pub(crate) fn denominator_word(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & NOUN == 0)
}

pub(crate) fn noun_half(v: &Value) -> bool {
    v.term().is_some_and(|(_, f)| f & NOUN != 0)
}

pub(crate) fn spelled_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED))
}

pub(crate) fn denominator_ordinal(v: &Value) -> bool {
    v.ordinal().is_some_and(|o| o.has(SPELLED) && !o.has(SIMPLE))
}

// Context checks.

/// Whether a word follows the match that multiplies the number before it
/// but could not multiply a sum ending in a part below `grain`: in `tres
/// mil dos mil` the second number is `dos mil`.
pub(crate) fn before_multiplier(ctx: &Ctx<'_>, grain: i8) -> bool {
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
pub(crate) fn one_written_word(ctx: &Ctx<'_>) -> bool {
    (ctx.start + 1..ctx.end).all(|i| ctx.lexeme(i).glued && ctx.lexeme(i).kind == Kind::Word)
}

/// Whether the match starts a written word: `cinco` in `cientocincoavos`
/// does not.
pub(crate) fn opens_word(ctx: &Ctx<'_>) -> bool {
    let first = ctx.lexeme(ctx.start);
    !(first.glued && ctx.before().is_some_and(|p| ctx.lexeme(p).kind == Kind::Word))
}

/// Whether `menos` before a number makes it negative: not after another
/// number (`siete menos cuarto`), nor in `al menos`, `lo menos`, `por lo
/// menos`.
pub(crate) fn menos_opens(ctx: &Ctx<'_>) -> bool {
    let Some(p) = ctx.before() else { return true };
    !(counts(ctx, p) || matches!(ctx.word(p), "al" | "lo"))
}

// Integers.

pub(crate) fn b_word_int(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0), 0, SPELLED)
}

pub(crate) fn b_tens_unit(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    int(Dec::from_i64(term(&v[0]).0 + term(&v[1]).0), 0, SPELLED)
}

/// `cien` takes no part after it; `ciento` and `doscientos` do.
pub(crate) fn b_hundreds(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, f) = term(&v[0]);
    int(Dec::from_i64(value), 2, SPELLED | if f & CIEN != 0 { CLOSED } else { 0 })
}

/// `dos mil`, `tres millones`, `mil millones`, `322 millones`. A plain
/// digit integer takes any scale; a spelled one is below it, and from a
/// million up one takes the singular and more the plural.
pub(crate) fn b_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
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
pub(crate) fn b_bare_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let s = term(&v[0]).0;
    int(pow10(s), s as i8, SPELLED)
}

pub(crate) fn b_sum(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (l, r) = (num(&v[0]), num(&v[1]));
    if r.grain >= l.grain || r.value >= pow10(i64::from(l.grain)) || before_multiplier(ctx, l.grain) {
        return None;
    }
    int(l.value.checked_add(r.value)?, r.grain, l.flags | r.flags)
}

pub(crate) fn b_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let m = num(&v[0]);
    int(m.value.checked_mul(Dec::from_i64(12))?, 0, m.flags | SPELLED)
}

/// `media docena` is six.
pub(crate) fn b_word_of_dozen(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let value = Dec::from_i64(12).checked_div(Dec::from_i64(term(&v[0]).0))?;
    let kind = if value.is_integer() { NumKind::Integer } else { NumKind::Fraction };
    Some(Value::Num(Num::new(value, kind, SPELLED)))
}

/// `10k`, `2,33 k`: the subtype is the numeral's.
pub(crate) fn b_suffix(v: &[Value], e: i64) -> Option<Value> {
    let n = num(&v[0]);
    Some(Value::Num(Num::new(n.value.checked_mul(pow10(e))?, n.kind, DIGITS)))
}

pub(crate) fn b_neg_word(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    if !menos_opens(ctx) {
        return None;
    }
    en::b_neg_word(v, ctx)
}

// Fractions.

/// A denominator is one written word: `tres dosmilésimos` is 3/2000, but
/// `cien mil billonésimas` is 100000/10^12.
pub(crate) fn b_den_ordinal(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    (opens_word(ctx) && one_written_word(ctx)).then_some(v[0])
}

pub(crate) fn b_den_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, f) = term(&v[0]);
    ordinal(d, (f & PLURAL) | SPELLED)
}

/// `veintiunavo`, `doscientosavos`: a number and `avo` in one word.
pub(crate) fn b_den_avo(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[0]);
    if !opens_word(ctx) || !one_written_word(ctx) || n.value < Dec::from_i64(3) {
        return None;
    }
    ordinal(n.value.to_i64()?, (term(&v[1]).1 & PLURAL) | SPELLED)
}

/// One takes the singular (`un quinto`), anything else the plural.
pub(crate) fn b_fraction(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (n, d) = (num(&v[0]).value, ord(&v[1]));
    if (n == Dec::ONE) == d.has(PLURAL) {
        return None;
    }
    let one = if n == Dec::ONE { ONE_PART } else { 0 };
    fraction(n.checked_div(Dec::from_i64(d.value))?, SPELLED | one)
}

/// `la mitad`.
pub(crate) fn b_half_noun(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    fraction(Dec::ONE.checked_div(Dec::from_i64(2))?, SPELLED)
}

pub(crate) fn half() -> Option<Dec> {
    Dec::ONE.checked_div(Dec::from_i64(2))
}

/// `cinco y medio`.
pub(crate) fn b_and_half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let w = num(&v[0]);
    fraction(w.value.checked_add(half()?)?, w.flags | SPELLED)
}

/// `tres millones y medio`: half of the scale.
pub(crate) fn b_scale_and_half(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let w = num(&v[0]);
    fraction(w.value.checked_add(half()?.checked_mul(pow10(i64::from(w.grain)))?)?, w.flags | SPELLED)
}

/// `cinco y un cuarto`: after `y`, a fraction of one part.
pub(crate) fn b_and_one_part(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    num(&v[1]).has(ONE_PART).then(|| en::b_mixed(v, ctx))?
}

/// `medio millón`, `un tercio de millón`.
pub(crate) fn b_word_of_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (d, s) = (term(&v[0]).0, term(&v[1]).0);
    fraction(pow10(s).checked_div(Dec::from_i64(d))?, SPELLED)
}

// Spoken decimals: `siete con cincuenta`, `doscientos coma cero tres`.

/// The decimals said as one number: `cincuenta` is .50, `doscientos` .200.
pub(crate) fn b_int_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let n = num(&v[0]);
    let count = n.value.to_i64()?.checked_ilog10()? + 1;
    Some(Value::Num(Num { value: n.value, kind: NumKind::Integer, grain: count as i8, flags: SPELLED }))
}

// Ordinals.

pub(crate) fn b_ordinal_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (value, flags) = term(&v[0]);
    ordinal(value, flags | SPELLED)
}

/// `dos milésimo`, `tres mil millonésimo`.
pub(crate) fn b_ordinal_scale(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (m, (s, f)) = (num(&v[0]), term(&v[1]));
    if m.value >= pow10(s) {
        return None;
    }
    ordinal(m.value.checked_mul(pow10(s))?.to_i64()?, (f & PLURAL) | SPELLED)
}

pub(crate) fn b_ordinal_scale_bare(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (s, f) = term(&v[0]);
    ordinal(pow10(s).to_i64()?, (f & PLURAL) | SPELLED)
}

/// `tres millones dos milésimo`: a number ending in a scale and an
/// ordinal below it.
pub(crate) fn b_ordinal_sum(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, o) = (num(&v[0]), ord(&v[1]));
    if Dec::from_i64(o.value) >= pow10(i64::from(l.grain)) {
        return None;
    }
    ordinal(l.value.to_i64()?.checked_add(o.value)?, (o.flags & PLURAL) | SPELLED)
}

/// `vigésimo primero`, `centésimo vigésimo quinto`, `milésimo
/// quingentésimo`: an ordinal and one below its last nonzero digit.
pub(crate) fn b_ordinal_ordinal(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (l, r) = (ord(&v[0]), ord(&v[1]));
    let zeros = (0..18).take_while(|&k| l.value % 10i64.pow(k + 1) == 0).count() as u32;
    if r.value <= 0 || zeros == 0 || r.value >= 10i64.pow(zeros) {
        return None;
    }
    ordinal(l.value.checked_add(r.value)?, (r.flags & PLURAL) | SPELLED)
}

/// `1ro`, `2da`, `11vo`, `30ma`, `5ª`: the suffix has to fit the number's
/// last digits.
pub(crate) fn b_ordinal_digits(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
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
pub(crate) fn b_ordinal_digits_dot(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    (term(&v[1]).0 == ANY_DIGIT).then(|| b_ordinal_digits(v, ctx))?
}

pub(crate) fn b_relative_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (offset, f) = term(&v[0]);
    let to = if f == TO_END { OrdinalBase::End } else { OrdinalBase::Current };
    Some(Value::Relative(Relative { offset: i32::try_from(offset).ok()?, to }))
}
