//! Boundary regressions for `Parse` mutation and span arithmetic (#649):
//! child-mutating methods must return typed errors where Java throws
//! `IndexOutOfBoundsException`, never panic, and offset arithmetic must not
//! wrap past the u32 span limit.

use std::collections::HashSet;
use std::sync::Arc;

use turbo_analysis::Span;
use turbo_parser::head_rules::HeadRules;
use turbo_parser::parse::{Head, Parse, ParseError, TOK_NODE, TOP_NODE};

struct NoHeads(HashSet<String>);

impl HeadRules for NoHeads {
    fn get_head(&self, _: &[Parse], _: &str) -> Option<Head> {
        None
    }
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.0
    }
}

fn rules() -> NoHeads {
    NoHeads(HashSet::new())
}

fn tree() -> Parse {
    let text: Arc<str> = "a b ".into();
    let mut root = Parse::new(Arc::clone(&text), Span::new(0, 4), TOP_NODE, 1.0, 0);
    root.insert(Parse::new(Arc::clone(&text), Span::new(0, 1), "A", 1.0, 0)).unwrap();
    root.insert(Parse::new(Arc::clone(&text), Span::new(2, 3), "B", 1.0, 1)).unwrap();
    root
}

#[test]
fn adjoin_on_a_childless_node_fails_instead_of_panicking() {
    let text: Arc<str> = "a ".into();
    let mut leaf = Parse::new(Arc::clone(&text), Span::new(0, 1), TOK_NODE, 1.0, 0);
    let sister = Parse::new(Arc::clone(&text), Span::new(0, 1), "X", 1.0, 0);
    assert_eq!(leaf.adjoin(sister, &rules()).unwrap_err(), ParseError::ChildIndex { index: 0, len: 0 });
}

#[test]
fn adjoin_root_with_an_index_past_the_children_fails() {
    let mut root = tree();
    let sister = Parse::new(root.text_arc().clone(), Span::new(0, 1), "X", 1.0, 0);
    assert_eq!(root.adjoin_root(sister, &rules(), 2).unwrap_err(), ParseError::ChildIndex { index: 2, len: 2 });
}

#[test]
fn expand_top_node_with_an_index_past_the_children_fails() {
    let mut root = tree();
    assert_eq!(root.expand_top_node(2).unwrap_err(), ParseError::ChildIndex { index: 2, len: 2 });
    assert!(root.expand_top_node(0).is_ok(), "a valid index still works");
}

#[test]
fn set_child_and_remove_with_an_index_past_the_children_fail() {
    let mut root = tree();
    assert_eq!(root.set_child(2, "X").unwrap_err(), ParseError::ChildIndex { index: 2, len: 2 });
    assert_eq!(root.remove(2).unwrap_err(), ParseError::ChildIndex { index: 2, len: 2 });
    root.set_child(0, "X").unwrap();
    root.remove(1).unwrap();
    assert_eq!(root.child_count(), 1);
}

struct FirstHeads(HashSet<String>);

impl HeadRules for FirstHeads {
    fn get_head(&self, constituents: &[Parse], _: &str) -> Option<Head> {
        constituents.first().map(Parse::as_head)
    }
    fn punctuation_tags(&self) -> &HashSet<String> {
        &self.0
    }
}

#[test]
fn adjoin_root_and_adjoin_still_work_on_valid_input() {
    let mut root = tree();
    let text = root.text_arc().clone();
    let sister = Parse::new(Arc::clone(&text), Span::new(0, 1), "X", 1.0, 0);
    root.adjoin_root(sister, &rules(), 0).unwrap();
    let sister = Parse::new(text, Span::new(3, 4), "Y", 1.0, 1);
    root.adjoin(sister, &FirstHeads(HashSet::new())).unwrap();
    assert_eq!(root.child_count(), 2);
}
