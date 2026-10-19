use super::*;

/// Adds `node` to its cell, unless an equal or better node with the same
/// value is there, or the cell is full of better ones. A node it beats is
/// marked dead. Returns whether it was added.
#[inline]
pub(super) fn insert(chart: &mut Chart, mut node: Node) -> bool {
    // Replaced nodes remain stored, so input length cannot bound this index.
    if chart.nodes.len() >= NONE as usize {
        return false;
    }
    let start = node.start as usize;
    let mut count = 0;
    let mut worst: Option<u32> = None;
    let mut id = chart.head[start];
    while id != NONE {
        let n = &chart.nodes[id as usize];
        if n.dim == node.dim && n.end == node.end {
            if n.value == node.value {
                if !node.better_than(n) {
                    return false;
                }
                chart.nodes[id as usize].dim = Dim::DEAD;
                break;
            }
            count += 1;
            if worst.is_none_or(|w| chart.nodes[w as usize].better_than(n)) {
                worst = Some(id);
            }
        }
        id = n.next;
    }
    if count >= CELL {
        let w = worst.expect("a full cell has a worst node");
        if !node.better_than(&chart.nodes[w as usize]) {
            return false;
        }
        chart.nodes[w as usize].dim = Dim::DEAD;
    }
    node.next = chart.head[start];
    chart.head[start] = chart.nodes.len() as u32;
    chart.nodes.push(node);
    true
}

/// Matches `items` from lexeme `pos` on, recording each complete match.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(super) fn walk(
    chart: &Chart,
    lexed: &Lexed,
    text: &str,
    items: &[Item],
    pos: u32,
    stack: &mut Vec<u32>,
    words: u32,
    done: &mut Vec<Done>,
    done_kids: &mut Vec<u32>,
) {
    let Some(item) = items.first() else {
        // Each child consumes a lexeme, bounded by parse's byte-length check.
        // Matches consume the scratch children consecutively; no stored offset
        // or 32-bit bound on the whole scratch buffer is needed.
        done.push(Done { end: pos, kids_len: stack.len() as u32, words });
        done_kids.extend_from_slice(stack);
        return;
    };
    let p = pos as usize;
    match *item {
        Item::Word { text: w, glued, optional, exact_case } => {
            if optional {
                walk(chart, lexed, text, &items[1..], pos, stack, words, done, done_kids);
            }
            let Some(l) = lexed.lexemes.get(p) else { return };
            if glued && !l.glued {
                return;
            }
            let matches = if exact_case { lexed.raw(text, p) == w } else { lexed.word(p) == w };
            if matches {
                walk(chart, lexed, text, &items[1..], pos + 1, stack, words + 1, done, done_kids);
            }
        }
        Item::Node { dim, test, glued } => {
            let Some(l) = lexed.lexemes.get(p) else { return };
            if glued && !l.glued {
                return;
            }
            for (id, n) in chart.starting_at(p) {
                if n.dim != dim || test.is_some_and(|t| !t(&n.value)) {
                    continue;
                }
                stack.push(id);
                walk(chart, lexed, text, &items[1..], n.end, stack, words, done, done_kids);
                stack.pop();
            }
        }
    }
}

/// A rule cannot repeat along a chain that consumes no additional lexemes.
/// Nonempty children have disjoint spans, so at most one can cover its parent.
#[inline]
pub(super) fn repeats(chart: &Chart, mut id: u32, rule: u16) -> bool {
    loop {
        let node = &chart.nodes[id as usize];
        if node.rule == rule {
            return true;
        }
        // Multiple nonempty children cannot share their parent's span.
        // Terminals have no children, so this also ends seeded chains.
        if node.kids.1 - node.kids.0 != 1 {
            return false;
        }
        let next = chart.kids[node.kids.0 as usize];
        let child = &chart.nodes[next as usize];
        if child.start != node.start || child.end != node.end {
            return false;
        }
        id = next;
    }
}
