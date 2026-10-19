//! Feature-string and frontier regressions for the tree-insertion context generators.

use super::*;

#[test]
fn compact_frontier_matches_path_walk_for_repeated_and_missing_heads() {
    let punct_set = HashSet::new();
    for depth in 1..=6 {
        for mut pattern in 0..3usize.pow(depth) {
            let mut root = None;
            for level in 0..depth {
                let mut node = Parse::new("x", Span::new(0, 1), format!("N{level}"), 1.0, (pattern % 3) as i32 - 1);
                pattern /= 3;
                if let Some(child) = root {
                    node.insert(child).unwrap();
                }
                root = Some(node);
            }
            let root = root.unwrap();
            let paths = right_frontier_paths(&root, &punct_set);
            let nodes: Vec<&Parse> = paths.iter().map(|path| root.node_at_path(path)).collect();
            for trim_at in 0..=nodes.len() {
                let trim = nodes.get(trim_at).copied();
                let retained = if trim.is_some() { &nodes[trim_at + 1..] } else { &nodes[..] };
                let mut previous = -1;
                let expected: Vec<&str> = retained
                    .iter()
                    .filter_map(|node| {
                        if node.head_index() == previous {
                            None
                        } else {
                            previous = node.head_index();
                            Some(node.kind())
                        }
                    })
                    .take(2)
                    .collect();
                let (actual, any) = frontier_nodes(&root, &punct_set, trim).unwrap();
                assert!(any);
                assert_eq!(actual.into_iter().flatten().map(Parse::kind).collect::<Vec<_>>(), expected);
            }
        }
    }
}

#[test]
fn production_keeps_existing_punctuation_on_equal_probability() {
    let tree = Parse::parse("(TOP (S (NP (NN cat)) (, ,) (: :) (VP (VB runs))))").unwrap();
    let mut parent = tree.children()[0].clone();
    let existing = Parse::new(parent.text_arc().clone(), Span::new(0, 0), "EXISTING", 1.0, 0);
    parent.children_mut()[0].add_next_punctuation(existing);
    let punctuation = [",".to_owned(), ":".to_owned()].into();
    assert_eq!(production(&parent, true, &punctuation).to_string(), "S->NP,EXISTING,VP");
    assert_eq!(production(&parent, false, &punctuation).to_string(), "S->NP,VP");
}
