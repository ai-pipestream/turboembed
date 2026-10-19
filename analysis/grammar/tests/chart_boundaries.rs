//! Adversarial custom grammars at the public parser boundary.
use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};
use turbo_grammar::engine::numeral::NumeralStyle;
use turbo_grammar::engine::{Ctx, Dim, Entry, Lexicon, Rule, node, opt_word};
use turbo_grammar::{Grammar, Output, Parser, Value};

const A: Dim = Dim(0);
const B: Dim = Dim(1);
static CALLS: AtomicUsize = AtomicUsize::new(0);

fn guarded_copy(v: &[Value], _: &Ctx<'_>) -> Option<Value> {
    // Bound the failing implementation without letting a cyclic grammar exhaust memory.
    assert!(CALLS.fetch_add(1, Ordering::Relaxed) < 100, "same-span derivation did not terminate");
    Some(v[0])
}

struct Custom {
    rules: &'static [Rule],
    numeral: Dim,
    seed: Dim,
    lexicon: OnceLock<Lexicon>,
}

impl Grammar for Custom {
    fn name(&self) -> &'static str {
        "boundary"
    }
    fn lexicon(&self) -> &Lexicon {
        self.lexicon.get_or_init(|| Lexicon::new(&[Entry { word: "a", dim: self.seed, value: 1, flags: 0 }]))
    }
    fn rules(&self) -> &'static [Rule] {
        self.rules
    }
    fn dims(&self) -> u16 {
        2
    }
    fn numeral(&self) -> Dim {
        self.numeral
    }
    fn style(&self) -> NumeralStyle {
        NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false }
    }
    fn output(&self, _: Dim) -> Option<Output> {
        Some(Output::Number)
    }
}

const fn custom(rules: &'static [Rule], numeral: Dim, seed: Dim) -> Custom {
    Custom { rules, numeral, seed, lexicon: OnceLock::new() }
}

#[test]
fn positive_unary_and_optional_cycles_terminate() {
    static UNARY: Custom =
        custom(&[Rule { name: "loop", dim: A, items: &[node(A)], weight: 1, build: guarded_copy }], A, A);
    static OPTIONAL: Custom = custom(
        &[Rule { name: "optional loop", dim: A, items: &[node(A), opt_word("x")], weight: 1, build: guarded_copy }],
        A,
        A,
    );
    static INDIRECT: Custom = custom(
        &[
            Rule { name: "a to b", dim: B, items: &[node(A)], weight: 1, build: guarded_copy },
            Rule { name: "b to a", dim: A, items: &[node(B)], weight: 1, build: guarded_copy },
        ],
        A,
        A,
    );
    static CHANGING: Custom = custom(
        &[Rule {
            name: "changing loop",
            dim: A,
            items: &[node(A)],
            weight: 1,
            build: |v, ctx| {
                let Value::Term { value, flags } = guarded_copy(v, ctx)? else { return None };
                Some(Value::Term { value: value + 1, flags })
            },
        }],
        A,
        A,
    );
    for grammar in [&UNARY, &OPTIONAL, &INDIRECT, &CHANGING] {
        CALLS.store(0, Ordering::Relaxed);
        let mut parser = Parser::new(grammar);
        parser.parse("a");
        assert!(parser.chart().nodes.len() <= 3);
        let first = parser.chart().nodes.clone();
        parser.parse("a");
        assert_eq!(parser.chart().nodes, first);
    }
}

#[test]
fn selection_rejects_offsets_inside_a_different_texts_character() {
    static SIMPLE: Custom = custom(&[], A, A);
    let mut parser = Parser::new(&SIMPLE);
    parser.parse("a");
    let mut mentions = Vec::new();
    parser.select("é", Output::Number, &mut mentions);
    assert!(mentions.is_empty(), "selection returned a split UTF-8 character");
    parser.select("", Output::Number, &mut mentions);
    assert!(mentions.is_empty(), "selection returned an out-of-bounds span");
}

#[test]
#[should_panic(expected = "unknown category")]
fn constructor_checks_noninitial_rule_categories() {
    static BAD: Custom = custom(
        &[Rule {
            name: "bad second category",
            dim: A,
            items: &[node(A), node(Dim(2))],
            weight: 0,
            build: guarded_copy,
        }],
        A,
        A,
    );
    Parser::new(&BAD);
}

#[test]
#[should_panic(expected = "unknown category")]
fn constructor_checks_numeral_category() {
    static BAD: Custom = custom(&[], Dim(2), A);
    Parser::new(&BAD);
}

#[test]
#[should_panic(expected = "unknown category")]
fn constructor_checks_lexicon_categories() {
    static BAD: Custom = custom(&[], A, Dim(2));
    Parser::new(&BAD);
}
