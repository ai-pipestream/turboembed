//! Japanese measures, ages, temperatures and money with original source spans.
mod lexicon;
mod rules;

use crate::engine::{Ctx, Dim, Grammar, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::lang::cjk;
use crate::unit::{Age, Unit};
use crate::value::Value;
use std::sync::OnceLock;

const RAW: Dim = Dim(cjk::DIMS);
const UNIT: Dim = Dim(cjk::DIMS + 1);
const QTY: Dim = Dim(cjk::DIMS + 2);
const MARKER: Dim = Dim(cjk::DIMS + 3);

pub struct JapaneseUnits {
    lexicon: OnceLock<Lexicon>,
}
pub static JAPANESE_UNITS: JapaneseUnits = JapaneseUnits { lexicon: OnceLock::new() };
static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
impl WordForms for JapaneseUnits {
    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        cjk::split(word, cuts, self.lexicon());
    }
}
impl Grammar for JapaneseUnits {
    fn name(&self) -> &'static str {
        "ja-units"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| {
            let mut entries = cjk::entries();
            entries.extend(lexicon::entries());
            Lexicon::new(&entries)
        })
    }
    fn rules(&self) -> &'static [Rule] {
        RULES.get_or_init(|| {
            let mut r = cjk::RULES.to_vec();
            r.extend_from_slice(rules::RULES);
            r
        })
    }
    fn dims(&self) -> u16 {
        cjk::DIMS + 4
    }
    fn numeral(&self) -> Dim {
        cjk::NUMERAL
    }
    fn style(&self) -> NumeralStyle {
        NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false }
    }
    fn words(&self) -> &dyn WordForms {
        self
    }
    fn word_boundary(&self) -> Option<fn(&str, &str) -> bool> {
        Some(cjk::word_boundary)
    }
    fn output(&self, dim: Dim) -> Option<Output> {
        matches!(dim, UNIT | QTY).then_some(Output::Quantity)
    }
    fn keep(&self, _: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        let unit = match node.value {
            Value::Unit(u) => {
                return lexicon::WORDS.iter().any(|w| w.unit == u && w.bare && w.text == ctx.word(ctx.start))
                    && !matches!(ctx.word(ctx.start), "円" | "元" | "$" | "€" | "¥" | "￥" | "\\" | "£");
            }
            Value::Quantity(q) => q.unit(),
            _ => return false,
        };
        if matches!(unit, Unit::Currency(crate::unit::CurrencyUnit::Minor(_))) {
            return false;
        }
        if matches!(unit, Unit::Age(_))
            && ctx.before().is_some_and(|p| {
                matches!(ctx.word(p), "マイナス" | "负" | "負")
                    || matches!(ctx.word(p), "-" | "－" | "−" | "‐")
                        && (p == 0 || cjk::parse_number(ctx.raw(p - 1)).is_none())
            })
        {
            return false;
        }
        if matches!(unit, Unit::Age(Age::Day | Age::Week)) && matches!(ctx.word(ctx.end - 1), "日" | "週") {
            let start = ctx.lexeme(ctx.start).start as usize;
            // A leading period may be a sentence delimiter absorbed by the
            // number reader. It cannot carry birth context into bare days/weeks.
            if ctx.text[start..].starts_with(['.', '．']) {
                return false;
            }
            let before = &ctx.text[..start];
            // A birth cue applies only within the current sentence or clause.
            let clause = before
                .rsplit([
                    '。', '！', '？', '!', '?', ';', '；', '、', ',', '.', '．', '\n', '\r', '\u{2028}', '\u{2029}',
                ])
                .next()
                .unwrap_or_default();
            return clause.contains("生まれ") && ctx.after().is_none_or(|i| ctx.word(i) != "か");
        }
        true
    }
}
