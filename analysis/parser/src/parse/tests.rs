/// `Parse.getType(rest)` for the text after a `(`: an escaped bracket type or `-NONE-` when the
/// text starts with one, otherwise the leading run of characters other than space, `=` and `-`
/// (`^([^ =-]+)`), and with function tags the first tag after it (`^[^ =-]+-([^ =-]+)`). The
/// type is always a prefix of `rest`, so it is returned as a slice of it. [`TypeScan`] gives the
/// same for each `(` of an input in turn.
#[cfg(test)]
fn java_type(rest: &str, use_function_tags: bool) -> Option<&str> {
    if let Some(escaped) = ESCAPED_TYPES.iter().find(|t| rest.starts_with(**t)) {
        return Some(escaped);
    }
    let bytes = rest.as_bytes();
    let run = |from: usize| bytes[from..].iter().take_while(|&&b| !matches!(b, b' ' | b'=' | b'-')).count();
    let n = run(0);
    if n == 0 {
        return None;
    }
    if use_function_tags && bytes.get(n) == Some(&b'-') {
        let m = run(n + 1);
        if m > 0 {
            return Some(&rest[..n + 1 + m]);
        }
    }
    Some(&rest[..n])
}

use super::io::{ESCAPED_TYPES, TypeScan};
use super::*;

#[test]
fn sentence_text_length_is_checked_against_the_u32_span_limit() {
    assert_eq!(checked_text_len(u32::MAX as usize).unwrap(), u32::MAX);
    assert_eq!(checked_text_len(0).unwrap(), 0);
    assert_eq!(checked_text_len(u32::MAX as usize + 1), Err(ParseError::TooLong { len: u32::MAX as usize + 1 }));
}

#[test]
fn the_type_scan_matches_java_type_at_every_bracket() {
    let inputs = [
        "(TOP (S (NP-SBJ-1 (NNP A)) (VP (VB b))))",
        "((((((x y",
        "(A-((((B-C= (-LRB- -LRB-) (-NONE-1 *) (=X) (-Y) (A--B) (A-",
        "(NP-SBJ=2 (WHNP-1 x)) ( (S (A b)))",
    ];
    for input in inputs {
        for tags in [false, true] {
            let mut scan = TypeScan::default();
            for (ci, _) in input.match_indices('(') {
                assert_eq!(scan.kind(input, ci + 1, tags), java_type(&input[ci + 1..], tags), "{input} at {ci}");
            }
        }
    }
}

#[test]
fn deep_trees_are_walked_and_dropped_without_recursion() {
    let depth = 100_000;
    let input = format!("{}(NN x){}", "(A ".repeat(depth), ")".repeat(depth));
    let (parse, sources) = Parse::parse_with_sources(&input, false).unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(parse.tag_nodes().len(), 1);
    assert_eq!(parse.token_nodes().len(), 1);
    drop(parse);
}

#[test]
fn the_tree_builder_matches_insert() {
    let text: Arc<str> = "a b c d e ".into();
    let spans = [(0, 1), (2, 3), (0, 3), (0, 3), (4, 5), (6, 7), (4, 7), (8, 9), (0, 9), (3, 5), (8, 9), (8, 9)];
    let root = || Parse::new(Arc::clone(&text), Span::new(0, 10), TOP_NODE, 1.0, 0);
    let mut slow = root();
    let mut fast = TreeBuilder::new(root());
    for (i, (s, e)) in spans.into_iter().enumerate() {
        let node = || Parse::new(Arc::clone(&text), Span::new(s, e), format!("N{i}"), 1.0, i as i32);
        slow.insert(node()).unwrap();
        fast.insert(node()).unwrap();
        let (mut a, mut b) = (String::new(), String::new());
        slow.show(&mut a);
        fast.root.show(&mut b);
        assert_eq!(a, b, "after {i}");
    }
}
