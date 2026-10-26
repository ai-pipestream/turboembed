//! Chinese quantities with source-preserving CJK number segmentation.
mod lexicon;
mod rules;
use super::cjk;
use crate::engine::{Ctx, Dim, Grammar, Lexicon, Node, NumeralStyle, Output, Rule, WordForms};
use crate::unit::{CurrencyUnit, Information, Minor, Temperature, Unit};
use crate::value::Value;
use std::sync::OnceLock;
const RAW: Dim = Dim(cjk::DIMS);
const UNIT: Dim = Dim(cjk::DIMS + 1);
const QTY: Dim = Dim(cjk::DIMS + 2);
/// Chinese measures and currencies; values retain the original byte spans.
pub struct ChineseUnits {
    lexicon: OnceLock<Lexicon>,
}
pub static CHINESE_UNITS: ChineseUnits = ChineseUnits { lexicon: OnceLock::new() };
static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
impl WordForms for ChineseUnits {
    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        cjk::split(word, cuts, self.lexicon());
    }
}
impl Grammar for ChineseUnits {
    fn word_boundary(&self) -> Option<fn(&str, &str) -> bool> {
        Some(|left, right| {
            left.chars()
                .next_back()
                .into_iter()
                .chain(right.chars().next())
                .any(|c| turbo_unicode::script::script(c) == turbo_unicode::script::Script::Han)
        })
    }
    fn name(&self) -> &'static str {
        "zh-units"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| {
            let mut entries = cjk::entries();
            entries.retain(|entry| {
                !entry.word.chars().any(|c| {
                    matches!(
                        turbo_unicode::script::script(c),
                        turbo_unicode::script::Script::Hiragana | turbo_unicode::script::Script::Katakana
                    )
                })
            });
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
    fn output(&self, dim: Dim) -> Option<Output> {
        matches!(dim, UNIT | QTY).then_some(Output::Quantity)
    }
    fn keep(&self, _: Output, node: &Node, ctx: &Ctx<'_>) -> bool {
        let (unit, amount) = match node.value {
            Value::Unit(u) => (u, false),
            Value::Quantity(q) => (q.unit(), true),
            _ => return false,
        };
        if !amount
            && !lexicon::WORDS.iter().any(|w| w.bare && w.unit == unit && w.word == ctx.word(ctx.end - 1))
            && !matches!(unit, Unit::Temperature(Temperature::Celsius | Temperature::Fahrenheit | Temperature::Kelvin))
        {
            return false;
        }
        let source = &ctx.text[ctx.lexeme(ctx.start).start as usize..ctx.lexeme(ctx.end - 1).end as usize];
        // The Chinese fullwidth comma separates clauses, not digit groups.
        if source.contains('，') {
            return false;
        }
        let tail = &ctx.text[ctx.lexeme(ctx.end - 1).end as usize..];
        if matches!(unit, Unit::Information(Information::Bit)) && ctx.word(ctx.end - 1) == "位" {
            return [
                "是最小的存储单位",
                "是最小的儲存單位",
                "表示一个比特",
                "表示一個位元",
                "等于一个比特",
                "等於一個位元",
            ]
            .iter()
            .any(|s| tail.starts_with(s));
        }
        if unit == Unit::Currency(CurrencyUnit::Minor(Minor::Jiao))
            && (tail.starts_with("大楼") || tail.starts_with("大樓"))
        {
            return false;
        }
        true
    }
}
