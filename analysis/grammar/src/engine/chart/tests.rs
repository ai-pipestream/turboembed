use std::sync::OnceLock;

use super::*;
use crate::engine::numeral::NumeralStyle;
use crate::engine::rule::{Entry, Lexicon, node, word};

const A: Dim = Dim(0);
const B: Dim = Dim(1);
const S: Dim = Dim(2);
const N: Dim = Dim(3);

/// A toy grammar: `S` counts the lexemes of an `a` followed by `b`s.
struct Toy(OnceLock<Lexicon>);

static TOY: Toy = Toy(OnceLock::new());

fn count(_: &[Value], ctx: &Ctx<'_>) -> Option<Value> {
    Some(Value::Term { value: (ctx.end - ctx.start) as i64, flags: 0 })
}

fn ranked(n: i64) -> Option<Value> {
    Some(Value::Term { value: n, flags: 0 })
}

static RULES: &[Rule] = &[
    Rule { name: "s_a", dim: S, items: &[node(A)], weight: 0, build: count },
    Rule { name: "s_more", dim: S, items: &[node(S), node(B)], weight: 0, build: count },
    Rule { name: "s_ab", dim: S, items: &[node(A), node(B)], weight: 1, build: count },
    // Six readings of `c` with different values, for the cell cap.
    Rule { name: "c1", dim: S, items: &[word("c")], weight: 1, build: |_, _| ranked(1) },
    Rule { name: "c2", dim: S, items: &[word("c")], weight: 2, build: |_, _| ranked(2) },
    Rule { name: "c3", dim: S, items: &[word("c")], weight: 3, build: |_, _| ranked(3) },
    Rule { name: "c4", dim: S, items: &[word("c")], weight: 4, build: |_, _| ranked(4) },
    Rule { name: "c5", dim: S, items: &[word("c")], weight: 5, build: |_, _| ranked(5) },
    Rule { name: "c0", dim: S, items: &[word("c")], weight: 0, build: |_, _| ranked(0) },
    Rule { name: "n", dim: S, items: &[node(N)], weight: 0, build: count },
];

impl Grammar for Toy {
    fn name(&self) -> &'static str {
        "toy"
    }
    fn lexicon(&self) -> &Lexicon {
        self.0.get_or_init(|| {
            Lexicon::new(&[
                Entry { word: "a", dim: A, value: 0, flags: 0 },
                Entry { word: "b", dim: B, value: 0, flags: 0 },
            ])
        })
    }
    fn rules(&self) -> &'static [Rule] {
        RULES
    }
    fn dims(&self) -> u16 {
        4
    }
    fn numeral(&self) -> Dim {
        N
    }
    fn style(&self) -> NumeralStyle {
        NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false, decimal_comma: false }
    }
    fn output(&self, dim: Dim) -> Option<Output> {
        (dim == S).then_some(Output::Number)
    }
}

fn live(p: &Parser, start: u32, end: u32) -> Vec<&Node> {
    p.chart().nodes.iter().filter(|n| n.dim == S && n.start == start && n.end == end).collect()
}

#[test]
fn equal_readings_keep_the_better_one() {
    let mut p = Parser::new(&TOY);
    p.parse("a b b");
    let two = live(&p, 0, 2);
    assert_eq!(two.len(), 1, "{two:?}");
    assert_eq!(RULES[usize::from(two[0].rule)].name, "s_ab");
    assert_eq!(two[0].score, 3);
    // The longer node is built on the better one.
    let three = live(&p, 0, 3);
    assert_eq!(three.len(), 1);
    assert_eq!(p.chart().children(p.chart().nodes.iter().position(|n| n == three[0]).unwrap() as u32).len(), 2);
}

#[test]
fn a_cell_keeps_the_best_four() {
    let mut p = Parser::new(&TOY);
    p.parse("c");
    let mut kept: Vec<i64> = live(&p, 0, 1).iter().map(|n| n.value.term().unwrap().0).collect();
    kept.sort();
    assert_eq!(kept, [2, 3, 4, 5]);
    let mut out = Vec::new();
    p.select("c", Output::Number, &mut out);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].value.term(), Some((5, 0)));
}

#[test]
fn selection_takes_the_longest_and_does_not_overlap() {
    let mut p = Parser::new(&TOY);
    let text = "a b a";
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, Output::Number, &mut out);
    let spans: Vec<(u32, u32)> = out.iter().map(|m| (m.start, m.end)).collect();
    assert_eq!(spans, [(0, 3), (4, 5)]);
}

#[test]
fn selection_does_not_cut_a_written_token() {
    let mut p = Parser::new(&TOY);
    let text = "a1 b";
    p.parse(text);
    let mut out = Vec::new();
    p.select(text, Output::Number, &mut out);
    assert!(out.is_empty(), "{out:?}");
    let text = "1 a";
    p.parse(text);
    out.clear();
    p.select(text, Output::Number, &mut out);
    assert_eq!(out.len(), 2);
}

#[test]
fn starting_at_lists_every_node_of_a_position() {
    let mut p = Parser::new(&TOY);
    p.parse("a b");
    let at0 = p.chart().starting_at(0).count();
    let at1 = p.chart().starting_at(1).count();
    assert!(at0 >= 3, "{at0}");
    assert_eq!(at1, 1);
    assert_eq!(p.chart().starting_at(9).count(), 0);
    assert_eq!(p.lexed().len(), 2);
    assert_eq!(p.grammar().name(), "toy");
}

#[test]
#[should_panic(expected = "optional word")]
fn a_rule_may_not_open_with_an_optional_word() {
    struct Bad;
    static BAD_RULES: &[Rule] =
        &[Rule { name: "bad", dim: S, items: &[crate::engine::rule::opt_word("x")], weight: 0, build: count }];
    impl Grammar for Bad {
        fn name(&self) -> &'static str {
            "bad"
        }
        fn lexicon(&self) -> &Lexicon {
            TOY.lexicon()
        }
        fn rules(&self) -> &'static [Rule] {
            BAD_RULES
        }
        fn dims(&self) -> u16 {
            4
        }
        fn numeral(&self) -> Dim {
            N
        }
        fn style(&self) -> NumeralStyle {
            TOY.style()
        }
        fn output(&self, _: Dim) -> Option<Output> {
            None
        }
    }
    static BAD: Bad = Bad;
    let _ = Parser::new(&BAD);
}
