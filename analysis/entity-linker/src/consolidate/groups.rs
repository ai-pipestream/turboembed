//! Which names and mentions are one entity: a union-find over the
//! `Entities` rows (`0..n`) and the coreference mentions no name lifts
//! (`n..n + m`).

use std::collections::HashMap;

use turbo_analysis::{Ix, Layer};
use turbo_analysis_layers::Entities;

use super::{ConsolidatorConfig, Inputs};

/// Disjoint sets, each represented by its lowest node, so the grouping is
/// the same whatever order the unions come in.
pub(super) struct Sets {
    parent: Vec<u32>,
}

impl Sets {
    fn new(len: usize) -> Sets {
        Sets { parent: (0..len as u32).collect() }
    }

    pub(super) fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            let up = self.parent[self.parent[x] as usize];
            self.parent[x] = up;
            x = up as usize;
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            let (lo, hi) = (a.min(b), a.max(b));
            self.parent[hi] = lo as u32;
        }
    }
}

/// The node of mention `i`: the name it was lifted from, else its own.
pub(super) fn mention_node(inp: &Inputs<'_>, i: usize) -> usize {
    let n = inp.entities.spans.len();
    match inp.mentions.and_then(|m| m.entity_of(Ix::new(i as u32))) {
        Some(e) if e.get() < n => e.get(),
        _ => n + i,
    }
}

/// A name's surface folded for comparison: lower case, whitespace runs
/// as one space.
pub(super) fn fold(text: &str) -> String {
    text.split_whitespace().map(str::to_lowercase).collect::<Vec<_>>().join(" ")
}

/// The text of `Entities` row `row`.
pub(super) fn surface<'a>(inp: &Inputs<'a>, row: usize) -> &'a str {
    let s = inp.entities.spans.get(Ix::new(row as u32));
    inp.view.text().get(s.start as usize..s.end as usize).unwrap_or_default()
}

pub(super) fn group(inp: &Inputs<'_>, config: ConsolidatorConfig) -> Sets {
    let e = inp.entities;
    let n = e.spans.len();
    let m = inp.mentions.map_or(0, Layer::len);
    let mut sets = Sets::new(n + m);
    if let (Some(mentions), Some(chains)) = (inp.mentions, inp.mention_groups) {
        for g in 0..chains.len() {
            let members = chains.get(Ix::new(g as u32));
            for w in members.windows(2) {
                if w[0].get() < mentions.len() && w[1].get() < mentions.len() {
                    sets.union(mention_node(inp, w[0].get()), mention_node(inp, w[1].get()));
                }
            }
        }
    }
    if let Some(chains) = inp.entity_groups {
        for g in 0..chains.len() {
            let members = chains.get(Ix::new(g as u32));
            for w in members.windows(2) {
                if w[0].get() < n && w[1].get() < n {
                    sets.union(w[0].get(), w[1].get());
                }
            }
        }
    }
    if let Some(linked) = inp.linked {
        let mut first: HashMap<&str, usize> = HashMap::new();
        for (row, id) in linked.kb_id.iter().enumerate() {
            if let Some(id) = id {
                let at = *first.entry(id.as_str()).or_insert(row);
                sets.union(at, row);
            }
        }
    }
    let folded: Vec<String> = (0..n).map(|r| fold(surface(inp, r))).collect();
    if config.surface_merge {
        let mut first: HashMap<(u16, &str), usize> = HashMap::new();
        for (row, f) in folded.iter().enumerate() {
            let at = *first.entry((e.kind[row].0, f.as_str())).or_insert(row);
            sets.union(at, row);
        }
        overlaps(e, &mut sets);
    }
    if config.alias_merge {
        for row in 0..n {
            if folded[row].is_empty() || folded[row].contains(' ') {
                continue;
            }
            let mut found = None;
            let mut alone = true;
            for (other, f) in folded.iter().enumerate() {
                if e.kind[other] == e.kind[row] && f.contains(' ') && f.rsplit(' ').next() == Some(&folded[row]) {
                    let root = sets.find(other);
                    alone &= found.is_none_or(|r| r == root);
                    found = Some(root);
                }
            }
            if let (Some(root), true) = (found, alone) {
                sets.union(root, row);
            }
        }
    }
    sets
}

/// Names of one kind whose spans overlap are one entity.
fn overlaps(e: &Entities, sets: &mut Sets) {
    let mut rows: Vec<usize> = (0..e.spans.len()).collect();
    rows.sort_by_key(|&r| {
        let s = e.spans.get(Ix::new(r as u32));
        (e.kind[r].0, s.start, s.end, r)
    });
    let mut open: Option<(u16, u32, usize)> = None;
    for r in rows {
        let s = e.spans.get(Ix::new(r as u32));
        match open {
            Some((kind, end, at)) if kind == e.kind[r].0 && s.start < end => {
                sets.union(at, r);
                open = Some((kind, end.max(s.end), at));
            }
            _ => open = Some((e.kind[r].0, s.end, r)),
        }
    }
}
