//! Pseudo-projective encoding and decoding (Nivre & Nilsson 2005).
//!
//! Non-projective arcs are lifted to their lowest projective ancestor and
//! annotated with the directional relation path `base^<step1^>step2...` from
//! the lifted head down to the original head. Literal `^`, `<`, `>`, and `\`
//! characters inside custom relation labels are escaped (`\^`, `\<`, `\>`,
//! `\\`) when non-projective lifting occurs, and only well-formed directional
//! lift suffixes (`^<rel` or `^>rel`) are decoded by [`DependencyGraph::deprojectivize`].
//! Ordinary custom labels such as `root^literal` on projective trees are
//! preserved unchanged.

use crate::{DependencyGraph, Result};
use turbo_analysis::Ix;
use turbo_analysis_layers::WordOrTokenAnchor;

fn escape_rel(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        if matches!(ch, '\\' | '^' | '<' | '>') {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn unescape_rel(s: &str) -> String {
    if !s.contains('\\') {
        return s.to_owned();
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            } else {
                out.push('\\');
            }
        } else {
            out.push(ch);
        }
    }
    out
}

fn has_unescaped_caret(s: &str) -> bool {
    let mut escaped = false;
    for ch in s.chars() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
        } else if ch == '^' {
            return true;
        }
    }
    false
}

fn split_unescaped_carets(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    for (idx, ch) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
        } else if ch == '^' {
            parts.push(&s[start..idx]);
            start = idx + 1;
        }
    }
    parts.push(&s[start..]);
    parts
}

fn parse_pseudo_projective_label(relation: &str) -> Option<(String, Vec<(bool, String)>)> {
    if !relation.contains('^') {
        return None;
    }
    let parts = split_unescaped_carets(relation);
    if parts.len() == 1 {
        if relation.contains("\\^") {
            let base_unescaped = unescape_rel(parts[0]);
            let base = if base_unescaped.chars().all(turbo_unicode::props::white_space) {
                "dep".to_owned()
            } else {
                base_unescaped
            };
            return Some((base, Vec::new()));
        }
        return None;
    }
    let base_raw = parts[0];
    if base_raw.is_empty() {
        return None;
    }
    let mut steps = Vec::with_capacity(parts.len() - 1);
    for &step in &parts[1..] {
        let (want_left, rel_raw) =
            if let Some(rest) = step.strip_prefix('<') { (true, rest) } else { (false, step.strip_prefix('>')?) };
        let rel = unescape_rel(rel_raw);
        if rel.is_empty() || rel.chars().all(turbo_unicode::props::white_space) {
            return None;
        }
        steps.push((want_left, rel));
    }
    let base_unescaped = unescape_rel(base_raw);
    let base =
        if base_unescaped.chars().all(turbo_unicode::props::white_space) { "dep".to_owned() } else { base_unescaped };
    Some((base, steps))
}

fn is_descendant<A: WordOrTokenAnchor>(heads: &[Option<Ix<A>>], ancestor: usize, mut node: usize) -> bool {
    loop {
        if node == ancestor {
            return true;
        }
        let Some(parent) = heads[node] else {
            return false;
        };
        node = parent.get();
    }
}

fn arc_is_nonprojective<A: WordOrTokenAnchor>(heads: &[Option<Ix<A>>], dependent: usize) -> bool {
    let Some(head) = heads[dependent] else {
        return false;
    };
    let h = head.get();
    let (lo, hi) = (h.min(dependent), h.max(dependent));
    ((lo + 1)..hi).any(|k| !is_descendant(heads, h, k))
}

fn depth<A: WordOrTokenAnchor>(heads: &[Option<Ix<A>>], mut node: usize) -> usize {
    let mut d = 0;
    while let Some(parent) = heads[node] {
        d += 1;
        node = parent.get();
    }
    d
}

impl<A: WordOrTokenAnchor> DependencyGraph<A> {
    /// Lifts non-projective arcs to their lowest projective ancestor, encoding
    /// the lifted head path in each relation label so [`Self::deprojectivize`]
    /// can restore the original non-projective arcs.
    pub fn projectivize(&self) -> Result<Self> {
        self.check_tree()?;
        if self.is_projective() {
            return Ok(self.clone());
        }
        let n = self.len();
        let mut heads = Vec::with_capacity(n);
        let mut base_rels = Vec::with_capacity(n);
        for i in 0..n {
            let arc = self.arc(Ix::new(i as u32))?;
            heads.push(arc.head);
            base_rels.push(arc.relation.to_owned());
        }
        let mut steps: Vec<Vec<String>> = vec![Vec::new(); n];
        loop {
            let candidate = (0..n).filter(|&i| arc_is_nonprojective(&heads, i)).min_by(|&a, &b| {
                let ha = heads[a].expect("nonprojective arc has head").get();
                let hb = heads[b].expect("nonprojective arc has head").get();
                depth(&heads, ha)
                    .cmp(&depth(&heads, hb))
                    .then_with(|| ha.abs_diff(a).cmp(&hb.abs_diff(b)))
                    .then_with(|| a.cmp(&b))
            });
            let Some(d) = candidate else {
                break;
            };
            let h = heads[d].expect("nonprojective arc has head").get();
            let p = heads[h].expect("root arc cannot be nonprojective in a single-root tree");
            let dir = if h < p.get() { '<' } else { '>' };
            steps[d].push(format!("{dir}{}", escape_rel(&base_rels[h])));
            heads[d] = Some(p);
        }
        let relations = base_rels
            .into_iter()
            .zip(steps)
            .map(|(base, mut st)| {
                if st.is_empty() {
                    if has_unescaped_caret(&base) { escape_rel(&base) } else { base }
                } else {
                    st.reverse();
                    format!("{}^{}", escape_rel(&base), st.join("^"))
                }
            })
            .collect();
        Self::new(heads, relations)
    }

    /// Decodes pseudo-projective relation labels (`base^<step1^>step2...`) back
    /// into lower (potentially non-projective) arcs while preserving tree
    /// invariants. Ordinary relation labels without directional `^<` / `^>`
    /// lift steps (such as `root^literal`) are preserved unchanged.
    pub fn deprojectivize(&self) -> Result<Self> {
        self.check_tree()?;
        let n = self.len();
        let mut heads = Vec::with_capacity(n);
        let mut base_rels = Vec::with_capacity(n);
        let mut steps: Vec<Vec<(bool, String)>> = Vec::with_capacity(n);
        let mut any_lifted = false;
        for i in 0..n {
            let arc = self.arc(Ix::new(i as u32))?;
            heads.push(arc.head);
            if let Some((base, parsed_steps)) = parse_pseudo_projective_label(arc.relation) {
                any_lifted = true;
                base_rels.push(base);
                steps.push(parsed_steps);
            } else {
                base_rels.push(arc.relation.to_owned());
                steps.push(Vec::new());
            }
        }
        if !any_lifted {
            return Ok(self.clone());
        }
        let mut order = Vec::with_capacity(n);
        let mut queue: Vec<usize> = self.roots().iter().map(|r| r.get()).collect();
        let mut idx = 0;
        while idx < queue.len() {
            let u = queue[idx];
            idx += 1;
            order.push(u);
            for &c in self.children(Ix::new(u as u32))? {
                queue.push(c.get());
            }
        }
        let mut step_idx = vec![0usize; n];
        loop {
            let mut progressed = false;
            for d in &order {
                let d = *d;
                while step_idx[d] < steps[d].len() {
                    let Some(cur) = heads[d].map(|h| h.get()) else {
                        break;
                    };
                    let (want_left, ref target_rel) = steps[d][step_idx[d]];
                    let mut cands: Vec<usize> = (0..n)
                        .filter(|&i| {
                            i != d
                                && heads[i].map(|h| h.get()) == Some(cur)
                                && !is_descendant(&heads, d, i)
                                && base_rels[i] == target_rel.as_str()
                        })
                        .collect();
                    let dir_cands: Vec<usize> = cands.iter().copied().filter(|&i| (i < cur) == want_left).collect();
                    if !dir_cands.is_empty() {
                        cands = dir_cands;
                    }
                    let next = if let Some(best) = cands.into_iter().min_by_key(|&i| (i.abs_diff(d), i)) {
                        Some(best)
                    } else {
                        let mut bfs: Vec<usize> = (0..n)
                            .filter(|&i| {
                                i != d && heads[i].map(|h| h.get()) == Some(cur) && !is_descendant(&heads, d, i)
                            })
                            .collect();
                        let mut b_idx = 0;
                        let mut found = None;
                        while b_idx < bfs.len() {
                            let u = bfs[b_idx];
                            b_idx += 1;
                            if base_rels[u] == target_rel.as_str() {
                                found = Some(u);
                                break;
                            }
                            for i in 0..n {
                                if i != d && heads[i].map(|h| h.get()) == Some(u) && !is_descendant(&heads, d, i) {
                                    bfs.push(i);
                                }
                            }
                        }
                        found
                    };
                    if let Some(u) = next {
                        heads[d] = Some(Ix::new(u as u32));
                        step_idx[d] += 1;
                        progressed = true;
                    } else {
                        break;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
        Self::new(heads, base_rels)
    }
}
