//! Shared source-preserving lexemes and exact numerals for CJK unit grammars.
mod number;

use crate::engine::{Ctx, Dim, Entry, Lexicon, Rule, node};
use crate::value::{Num, NumKind, Value, flag};
pub use number::parse_number;

pub const NUMERAL: Dim = Dim(0);
pub const ATOM: Dim = Dim(1);
pub const NUM: Dim = Dim(2);
pub const DIMS: u16 = 3;

/// Numeric terms only; each language adds its units before constructing a lexicon.
pub fn entries() -> Vec<Entry> {
    number::WORDS.iter().map(|&(word, _)| Entry { word, dim: ATOM, value: 0, flags: 0 }).collect()
}

/// Cut a folded word without changing its bytes. Longest unit terms take priority
/// over their numeric prefixes (`千米`, `兆字节`); unknown scalars stay separate.
pub fn split(word: &str, cuts: &mut Vec<u32>, lexicon: &Lexicon) {
    let mut start = 0;
    while start < word.len() {
        let mut end = start + word[start..].chars().next().expect("nonempty tail").len_utf8();
        for (at, ch) in word[start..].char_indices() {
            let candidate = start + at + ch.len_utf8();
            let piece = &word[start..candidate];
            if !lexicon.has_prefix(piece) {
                break;
            }
            if !lexicon.lookup(piece).is_empty() {
                end = candidate;
            }
        }
        start = end;
        if start < word.len() {
            cuts.push(start as u32);
        }
    }
}

fn atoms(_: &[Value], _: &Ctx<'_>) -> Option<Value> {
    Some(Value::Empty)
}
fn number(_: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    let start = ctx.lexeme(ctx.start).start as usize;
    let end = ctx.lexeme(ctx.end - 1).end as usize;
    let source = &ctx.text[start..end];
    if source.starts_with(['-', '－', '−', '‐']) && ctx.before().is_some_and(|p| parse_number(ctx.raw(p)).is_some())
    {
        return None;
    }
    let value = parse_number(source)?;
    Some(Value::Num(Num::new(value, NumKind::of(value), flag::SPELLED)))
}

/// Intermediate atom spans also admit incomplete decimal/sign prefixes. Only
/// the final numeric rule accepts a span after checking its complete syntax.
pub const RULES: &[Rule] = &[
    Rule { name: "cjk_digits", dim: ATOM, items: &[node(NUMERAL)], weight: 0, build: atoms },
    Rule { name: "cjk_atoms", dim: ATOM, items: &[node(ATOM), node(ATOM)], weight: 0, build: atoms },
    Rule { name: "cjk_number", dim: NUM, items: &[node(ATOM)], weight: 1, build: number },
];

/// Script boundaries used by CJK grammars after lexicon segmentation. Latin
/// identifiers remain whole; Han and kana do not require spaces around units.
pub fn word_boundary(left: &str, right: &str) -> bool {
    use turbo_unicode::script::{Script, script};
    left.chars()
        .next_back()
        .into_iter()
        .chain(right.chars().next())
        .any(|c| matches!(script(c), Script::Han | Script::Hiragana | Script::Katakana))
}
