use super::{Item, Rule};

/// Only rules on a possible same-span category cycle need a runtime walk.
/// A same-span derivation has one node and otherwise absent optional words.
/// The graph deliberately ignores predicates and builders: rejecting fewer
/// impossible edges keeps this test conservative for custom grammars.
pub(super) fn guards(rules: &[Rule], dims: usize) -> Vec<bool> {
    let inputs: Vec<_> = rules
        .iter()
        .map(|r| {
            let mut input = None;
            for item in r.items {
                match item {
                    Item::Node { dim, .. } if input.is_none() => input = Some(usize::from(dim.0)),
                    Item::Word { optional: true, .. } => {}
                    _ => return None,
                }
            }
            input
        })
        .collect();
    let mut next = vec![Vec::new(); dims];
    for (r, input) in rules.iter().zip(&inputs) {
        if let Some(input) = input {
            next[*input].push(usize::from(r.dim.0));
        }
    }
    let mut seen = vec![false; dims];
    let mut stack = Vec::new();
    rules
        .iter()
        .zip(inputs)
        .map(|(r, input)| {
            let Some(input) = input else { return false };
            seen.fill(false);
            stack.clear();
            stack.push(usize::from(r.dim.0));
            while let Some(dim) = stack.pop() {
                if dim == input {
                    return true;
                }
                if !seen[dim] {
                    seen[dim] = true;
                    stack.extend_from_slice(&next[dim]);
                }
            }
            false
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Dim, node, opt_word, word};

    #[test]
    fn same_span_graph_ignores_consuming_edges() {
        static RULES: &[Rule] = &[
            Rule { name: "a to b", dim: Dim(1), items: &[node(Dim(0))], weight: 0, build: |_, _| None },
            Rule { name: "b to a", dim: Dim(0), items: &[node(Dim(1)), opt_word("x")], weight: 0, build: |_, _| None },
            Rule { name: "acyclic", dim: Dim(2), items: &[node(Dim(1))], weight: 0, build: |_, _| None },
            Rule { name: "word", dim: Dim(1), items: &[node(Dim(2)), word("x")], weight: 0, build: |_, _| None },
            Rule { name: "two kids", dim: Dim(1), items: &[node(Dim(2)), node(Dim(0))], weight: 0, build: |_, _| None },
        ];
        assert_eq!(guards(RULES, 3), [true, true, false, false, false]);
    }
}
