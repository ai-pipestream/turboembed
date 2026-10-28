//! Decoding a biaffine parser's arc scores to a tree (parser.rs): the
//! maximum spanning tree with one word on the root (Chu-Liu-Edmonds), or
//! the best projective tree with one word on the root (Eisner).
//!
//! Scores are [n, n + 1]: row `d - 1` holds dependent `d`'s score for
//! each head `h` in `0..=n`, node 0 being the root. A tree is each
//! word's head, `heads[d - 1]`. Sums are taken in f64 in a fixed order,
//! and every choice among equal scores goes to the lower index, so a
//! decode is the same on every machine. The one-root search follows
//! Stanza's `chuliu_edmonds_one_root` (Apache-2.0); Eisner is the
//! textbook first-order algorithm with the root's split fixed at 0.

use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result};

const NEG: f64 = f64::NEG_INFINITY;

/// Square scores [node as dependent][node as head] over `0..=n`, the
/// root's row and every self loop at -inf.
fn square(scores: &[f32], n: usize) -> Result<Vec<Vec<f64>>> {
    if Some(scores.len()) != n.checked_mul(n + 1) {
        return Err(Error::new(INVALID_SHAPE, format!("{} arc scores for {n} words", scores.len())));
    }
    if !scores.iter().all(|v| v.is_finite()) {
        return Err(Error::new(INVALID_ARGUMENT, "an arc score is not finite"));
    }
    let mut s = vec![vec![NEG; n + 1]; n + 1];
    for d in 1..=n {
        for h in 0..=n {
            if h != d {
                s[d][h] = scores[(d - 1) * (n + 1) + h] as f64;
            }
        }
    }
    Ok(s)
}

/// The first index of the greatest value; -inf rows give 0.
fn argmax(row: &[f64]) -> usize {
    let mut best = 0;
    for (j, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = j;
        }
    }
    best
}

/// The nodes of the first cycle the head map has, by lowest node, as a
/// mask; none for a tree. Node 0 is its own head and never in a cycle.
fn cycle(tree: &[usize]) -> Option<Vec<bool>> {
    let n = tree.len();
    // 0 unvisited, 1 on the current walk, 2 done.
    let mut state = vec![0u8; n];
    state[0] = 2;
    for start in 1..n {
        let mut v = start;
        while state[v] == 0 {
            state[v] = 1;
            v = tree[v];
        }
        if state[v] == 1 {
            let mut mask = vec![false; n];
            let mut u = v;
            loop {
                mask[u] = true;
                u = tree[u];
                if u == v {
                    break;
                }
            }
            return Some(mask);
        }
        let mut u = start;
        while state[u] == 1 {
            state[u] = 2;
            u = tree[u];
        }
    }
    None
}

/// Chu-Liu-Edmonds over `s` [dep][head]; row 0 is the root's.
fn chu_liu_edmonds(s: &[Vec<f64>]) -> Vec<usize> {
    let n = s.len();
    let mut tree: Vec<usize> = s.iter().map(|row| argmax(row)).collect();
    tree[0] = 0;
    let Some(in_cycle) = cycle(&tree) else { return tree };
    let cyc: Vec<usize> = (0..n).filter(|&v| in_cycle[v]).collect();
    let rest: Vec<usize> = (0..n).filter(|&v| !in_cycle[v]).collect();
    let cycle_scores: Vec<f64> = cyc.iter().map(|&v| s[v][tree[v]]).collect();
    let total: f64 = cycle_scores.iter().sum();
    // For each outside node as the cycle's head: the best cycle node to
    // hang from it, and the score of the cycle broken there.
    let gain = |c: usize, o: usize| {
        let v = s[cyc[c]][rest[o]] - cycle_scores[c] + total;
        if v.is_nan() { NEG } else { v }
    };
    let m = rest.len();
    let mut meta_head = vec![0usize; m];
    let mut meta_head_score = vec![NEG; m];
    for o in 0..m {
        meta_head_score[o] = gain(0, o);
        for c in 1..cyc.len() {
            let g = gain(c, o);
            if g > meta_head_score[o] {
                meta_head_score[o] = g;
                meta_head[o] = c;
            }
        }
    }
    // For each outside node as a dependent of the cycle: its best head in it.
    let mut meta_dep = vec![0usize; m];
    let mut meta_dep_score = vec![NEG; m];
    for o in 0..m {
        let row: Vec<f64> = cyc.iter().map(|&c| s[rest[o]][c]).collect();
        meta_dep[o] = argmax(&row);
        meta_dep_score[o] = row[meta_dep[o]];
    }
    let mut sub = vec![vec![NEG; m + 1]; m + 1];
    for a in 0..m {
        for b in 0..m {
            sub[a][b] = s[rest[a]][rest[b]];
        }
        sub[m][a] = meta_head_score[a];
        sub[a][m] = meta_dep_score[a];
    }
    let contracted = chu_liu_edmonds(&sub);
    let mut out = tree.clone();
    for a in 0..m {
        let h = contracted[a];
        out[rest[a]] = if h < m { rest[h] } else { cyc[meta_dep[a]] };
    }
    let cycle_head = contracted[m];
    out[cyc[meta_head[cycle_head]]] = rest[cycle_head];
    out[0] = 0;
    out
}

fn total(s: &[Vec<f64>], tree: &[usize]) -> f64 {
    (1..tree.len()).map(|d| s[d][tree[d]]).sum()
}

/// The maximum spanning tree of `scores` [n, n + 1] with exactly one word
/// on the root, into `heads`.
pub fn mst(scores: &[f32], n: usize, heads: &mut Vec<u32>) -> Result<()> {
    let s = square(scores, n)?;
    heads.clear();
    if n == 0 {
        return Ok(());
    }
    let tree = chu_liu_edmonds(&s);
    let roots: Vec<usize> = (1..=n).filter(|&d| tree[d] == 0).collect();
    let best = if roots.len() == 1 {
        tree
    } else {
        let mut best: Option<(f64, Vec<usize>)> = None;
        for r in roots {
            let mut t = s.clone();
            for row in t.iter_mut().skip(1) {
                row[0] = NEG;
            }
            t[r] = vec![NEG; n + 1];
            t[r][0] = s[r][0];
            let tree = chu_liu_edmonds(&t);
            let score = total(&t, &tree);
            if score > NEG && best.as_ref().is_none_or(|(b, _)| score > *b) {
                best = Some((score, tree));
            }
        }
        best.map(|(_, t)| t).ok_or_else(|| Error::new(INVALID_ARGUMENT, "no tree has one word on the root"))?
    };
    heads.extend(best[1..].iter().map(|&h| h as u32));
    Ok(())
}

/// The best projective tree of `scores` [n, n + 1] with exactly one word
/// on the root, into `heads`.
pub fn eisner(scores: &[f32], n: usize, heads: &mut Vec<u32>) -> Result<()> {
    let s = square(scores, n)?;
    heads.clear();
    if n == 0 {
        return Ok(());
    }
    let k = n + 1;
    let at = |a: usize, b: usize| a * k + b;
    // Complete and incomplete spans [a, b], left (head b) and right
    // (head a), with the split each was built from. An incomplete right
    // span is the arc a -> b, so it adds s[b][a] (dependent b, head a).
    let (mut cl, mut cr, mut il, mut ir) = (vec![NEG; k * k], vec![NEG; k * k], vec![NEG; k * k], vec![NEG; k * k]);
    let (mut bcl, mut bcr, mut bil, mut bir) = (vec![0; k * k], vec![0; k * k], vec![0; k * k], vec![0; k * k]);
    for i in 0..k {
        cl[at(i, i)] = 0.0;
        cr[at(i, i)] = 0.0;
    }
    for w in 1..k {
        for a in 0..k - w {
            let b = a + w;
            // Incomplete: an arc between a and b over a split a..r | r+1..b.
            // The root heads one word only, so its split is always at 0.
            let last = if a == 0 { a } else { b - 1 };
            let (mut best, mut arg) = (NEG, a);
            for r in a..=last {
                let v = cr[at(a, r)] + cl[at(r + 1, b)];
                if v > best {
                    best = v;
                    arg = r;
                }
            }
            if a > 0 {
                il[at(a, b)] = best + s[a][b];
                bil[at(a, b)] = arg;
            }
            ir[at(a, b)] = best + s[b][a];
            bir[at(a, b)] = arg;
            let (mut best, mut arg) = (NEG, a);
            for r in a..b {
                let v = cl[at(a, r)] + il[at(r, b)];
                if v > best {
                    best = v;
                    arg = r;
                }
            }
            cl[at(a, b)] = best;
            bcl[at(a, b)] = arg;
            let (mut best, mut arg) = (NEG, b);
            for r in a + 1..=b {
                let v = ir[at(a, r)] + cr[at(r, b)];
                if v > best {
                    best = v;
                    arg = r;
                }
            }
            cr[at(a, b)] = best;
            bcr[at(a, b)] = arg;
        }
    }
    if cr[at(0, n)] == NEG {
        return Err(Error::new(INVALID_ARGUMENT, "no projective tree has one word on the root"));
    }
    let mut tree = vec![0usize; k];
    // (a, b, complete, right)
    let mut stack = vec![(0usize, n, true, true)];
    while let Some((a, b, complete, right)) = stack.pop() {
        if a == b {
            continue;
        }
        match (complete, right) {
            (true, true) => {
                let r = bcr[at(a, b)];
                stack.push((a, r, false, true));
                stack.push((r, b, true, true));
            }
            (true, false) => {
                let r = bcl[at(a, b)];
                stack.push((a, r, true, false));
                stack.push((r, b, false, false));
            }
            (false, right) => {
                if right {
                    tree[b] = a;
                } else {
                    tree[a] = b;
                }
                let r = if right { bir[at(a, b)] } else { bil[at(a, b)] };
                stack.push((a, r, true, true));
                stack.push((r + 1, b, true, false));
            }
        }
    }
    heads.extend(tree[1..].iter().map(|&h| h as u32));
    Ok(())
}
