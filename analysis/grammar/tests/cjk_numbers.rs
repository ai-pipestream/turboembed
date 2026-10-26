//! Number values taken from the pinned Recognizers-Text Japanese/Chinese unit
//! fixtures, plus malformed-input checks. Full corpus coverage is tested by
//! each unit grammar, not claimed by this shared arithmetic target.
use std::sync::OnceLock;
use turbo_grammar::engine::{Dim, Grammar, Lexicon, NumeralStyle, Output, Rule, WordForms};
use turbo_grammar::lang::cjk;
use turbo_grammar::{Dec, Parser};

#[test]
fn exact_upstream_numeric_values() {
    for (text, integer, scale) in [
        ("35億", 3500000000, 0),
        ("36.57億", 3657000000, 0),
        ("1万億", 1000000000000, 0),
        ("四十九兆九千五百九十二億", 49959200000000, 0),
        ("3 万 4,500", 34500, 0),
        ("29万1,890", 291890, 0),
        ("１．０７５３", 10753, 4),
        ("202107211321363588510367", 202107211321363588510367, 0),
        ("にせんにひゃくごじゅうろく", 2256, 0),
        ("ろく", 6, 0),
        ("负两点五", -25, 1),
        ("負兩點五", -25, 1),
        ("マイナス三十", -30, 0),
        ("1/4", 25, 2),
        ("一〇", 10, 0),
        ("0.9兆", 900000000000, 0),
        ("一千万", 10000000, 0),
        ("1万6千", 16000, 0),
    ] {
        assert_eq!(cjk::parse_number(text), Dec::new(integer, scale), "{text}");
    }
}

#[test]
fn malformed_values_do_not_turn_into_numbers() {
    for text in [
        "",
        " ",
        "-",
        "负負一",
        "一..二",
        "1,23",
        "1,234,",
        "１．",
        "二百百",
        "一万万",
        "1/0",
        "1/2/3",
        "1 2",
        "1.2.3",
        "に3",
        "3に",
        "999999999999999999999999999999999999999999999",
    ] {
        assert_eq!(cjk::parse_number(text), None, "{text}");
    }
}

struct Numbers(OnceLock<Lexicon>);
static NUMBERS: Numbers = Numbers(OnceLock::new());
impl WordForms for Numbers {
    fn split(&self, word: &str, cuts: &mut Vec<u32>) {
        cjk::split(word, cuts, self.lexicon());
    }
}
impl Grammar for Numbers {
    fn name(&self) -> &'static str {
        "cjk-test-numbers"
    }
    fn lexicon(&self) -> &Lexicon {
        self.0.get_or_init(|| Lexicon::new(&cjk::entries()))
    }
    fn rules(&self) -> &'static [Rule] {
        cjk::RULES
    }
    fn dims(&self) -> u16 {
        cjk::DIMS
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
        (dim == cjk::NUM).then_some(Output::Number)
    }
}

#[test]
fn real_chart_cjk_and_fullwidth_source_spans() {
    let mut parser = Parser::new(&NUMBERS);
    for text in
        ["四十九兆九千五百九十二億", "１．０７５３", "にせんにひゃくごじゅうろく", "36.57億", "负两点五", "3 万 4,500"]
    {
        parser.parse(text);
        let mut found = Vec::new();
        parser.select(text, Output::Number, &mut found);
        assert_eq!(found.len(), 1, "{text}: {found:?}");
        assert_eq!(found[0].value.num().unwrap().value, cjk::parse_number(text).unwrap(), "{text}");
        assert_eq!(found[0].start, 0);
        assert_eq!(found[0].end as usize, text.len());
    }
}

#[test]
fn longest_units_do_not_split_into_numeric_scales() {
    use turbo_grammar::engine::Entry;
    let mut entries = cjk::entries();
    for word in ["千米", "千克", "千比特", "兆字节"] {
        entries.push(Entry { word, dim: Dim(cjk::DIMS), value: 0, flags: 0 });
    }
    let lexicon = Lexicon::new(&entries);
    for unit in ["千米", "千克", "千比特", "兆字节"] {
        let text = format!("田二{unit}長");
        let mut cuts = Vec::new();
        cjk::split(&text, &mut cuts, &lexicon);
        assert_eq!(cuts, [3, 6, (6 + unit.len()) as u32]);
        for cut in cuts {
            assert!(text.is_char_boundary(cut as usize));
        }
    }
}

#[test]
fn cjk_script_boundaries_preserve_latin_identifiers() {
    assert!(cjk::word_boundary("私", "29"));
    assert!(cjk::word_boundary("歳", "で"));
    assert!(!cjk::word_boundary("x", "29"));
    assert!(!cjk::word_boundary("mm", "suffix"));
    let mut parser = Parser::new(&NUMBERS);
    let text = "田三百長";
    parser.parse(text);
    let mut found = Vec::new();
    parser.select(text, Output::Number, &mut found);
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].start, found[0].end), (3, 9));
    assert_eq!(found[0].value.num().unwrap().value, Dec::from_i64(300));
}

#[test]
fn unicode_minus_and_leading_decimal_values() {
    for sign in ["−", "‐", "－", "-"] {
        assert_eq!(cjk::parse_number(&format!("{sign}5")), Some(Dec::from_i64(-5)));
        assert_eq!(cjk::parse_number(&format!("{sign}．５")), Dec::new(-5, 1));
    }
    for text in ["．5", "．５", "０．５", ".5"] {
        assert_eq!(cjk::parse_number(text), Dec::new(5, 1), "{text}");
    }
    for text in ["．", "−", "‐", "−‐5", "．５．６", "一万．五"] {
        assert_eq!(cjk::parse_number(text), None, "{text}");
    }
}
