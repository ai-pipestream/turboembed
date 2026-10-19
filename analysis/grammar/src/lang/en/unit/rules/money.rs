use super::*;

// Currencies.

pub(super) fn currency(c: CurrencyUnit) -> Option<Value> {
    Some(Value::Unit(Unit::Currency(c)))
}

pub(super) fn coded(lower: &str) -> Option<Value> {
    currency(CurrencyUnit::Code(code(lower)?))
}

pub(super) fn b_money_word(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let (i, f) = term(&v[0]);
    if f & NATION_ONLY != 0 {
        return None;
    }
    Some(Value::Unit(MONEY_WORDS[i as usize].unit))
}

/// A code; one that is also an English word only in capitals.
pub(super) fn b_code(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let (i, f) = term(&v[0]);
    if f & AMBIG != 0 && has_lower(raw(ctx, ctx.start)) {
        return None;
    }
    coded(CODES[i as usize].0)
}

/// A country word followed by the name of its currency.
pub(super) fn b_nation_money(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    let rows = nation_rows(term(&v[0]).0 as usize);
    let named = MONEY_WORDS[term(&v[1]).0 as usize].unit;
    rows.iter().filter(|r| r.flags & ONLY_DOLLAR == 0).find_map(|r| {
        let c = code(r.code)?;
        let fits = match (r.generic, named) {
            (Some(g), Unit::Currency(CurrencyUnit::Generic(h))) => g == h,
            (None, Unit::Currency(CurrencyUnit::Code(x))) => x == c,
            _ => false,
        };
        fits.then_some(Value::Unit(Unit::Currency(CurrencyUnit::Code(c))))
    })
}

/// Words before which a lowercase `us` is the country: `a us $ 150
/// million`, not `give us $5`.
pub(super) const US_CUES: &[&str] = &["a", "an", "the", "s", "its", "his", "her", "their", "our", "your", "my"];

/// A country before `$`: `C$`, `hk $`, `australian $`.
pub(super) fn b_nation_dollar(v: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let rows = nation_rows(term(&v[0]).0 as usize);
    let r = rows.iter().find(|r| r.generic == Some(Generic::Dollar))?;
    let first = ctx.lexeme(ctx.start);
    let short = r.flags & ONLY_DOLLAR != 0 || r.word == "us";
    // A short form glued to a word before it, or after an apostrophe, is
    // no country: `it's $5`, `music ' s $ 500 million man`.
    let after_apostrophe = ctx.before().is_some_and(|p| matches!(ctx.punct(p), Some('\'' | '’')));
    if short && (after_apostrophe || first.glued && ctx.before().is_some_and(|p| ctx.lexeme(p).kind != Kind::Punct)) {
        return None;
    }
    if r.word == "us"
        && has_lower(raw(ctx, ctx.start))
        && ctx.before().is_some_and(|p| ctx.lexeme(p).kind == Kind::Word && !US_CUES.contains(&ctx.word(p)))
    {
        return None;
    }
    coded(r.code)
}

pub(super) fn nation_term(word: &str) -> Option<Value> {
    Some(Value::Term { value: nation(word)? as i64, flags: 0 })
}

/// A code glued to `$` is a currency in any case: `cad$ 15`.
pub(super) fn b_code_dollar(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    coded(CODES[term(&v[0]).0 as usize].0)
}
