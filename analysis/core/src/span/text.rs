use std::ops::Range;

use crate::text::ViewId;

use super::{Overlap, Span};

/// A borrowed view of a [`Spans<L>`]'s envelope columns and optional
/// fragment columns, returned by [`Spans::text_spans`] and
/// [`crate::Layer::text_spans`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TextSpans<'a> {
    pub view: ViewId,
    pub start: &'a [u32],
    pub end: &'a [u32],
    pub frag_start: &'a [u32],
    pub frag_end: &'a [u32],
    pub frag_item_end: &'a [u32],
}

impl<'a> TextSpans<'a> {
    /// Whether the `start` column is in non-decreasing text order.
    pub fn is_start_ordered(&self) -> bool {
        self.start.windows(2).all(|w| w[0] <= w[1])
    }

    /// Whether the envelope spans are in text order and do not overlap.
    pub fn is_ordered(&self) -> bool {
        self.is_start_ordered() && (1..self.start.len()).all(|i| self.end[i - 1] <= self.start[i])
    }

    /// The flat fragment index range `start_f..end_f` for item `idx` when
    /// `!self.frag_item_end.is_empty()`.
    pub fn item_frag_range(&self, idx: usize) -> Range<usize> {
        let start = if idx == 0 { 0 } else { self.frag_item_end[idx - 1] as usize };
        let end = self.frag_item_end[idx] as usize;
        start..end
    }

    /// Checks the span columns, the fragment table (when present), and the
    /// layer's [`Overlap`] rule.
    pub fn check(&self, overlap: Overlap) -> Result<(), String> {
        if self.start.len() != self.end.len() {
            return Err(format!("{} starts and {} ends", self.start.len(), self.end.len()));
        }
        for i in 0..self.start.len() {
            if self.start[i] > self.end[i] {
                return Err(format!("span {i} ends before it starts: {}..{}", self.start[i], self.end[i]));
            }
        }
        if !self.frag_item_end.is_empty() {
            if self.frag_item_end.len() != self.start.len() {
                return Err(format!("{} fragment item ends for {} spans", self.frag_item_end.len(), self.start.len()));
            }
            if self.frag_start.len() != self.frag_end.len() {
                return Err(format!(
                    "{} fragment starts and {} fragment ends",
                    self.frag_start.len(),
                    self.frag_end.len()
                ));
            }
            let total_frags = self.frag_item_end.last().copied().unwrap_or(0) as usize;
            if total_frags != self.frag_start.len() {
                return Err(format!(
                    "fragment item ends reach {total_frags}, fragment columns have {}",
                    self.frag_start.len()
                ));
            }
            for i in 0..self.start.len() {
                let r = self.item_frag_range(i);
                let (start_f, end_f) = (r.start, r.end);
                if start_f >= end_f || end_f > self.frag_start.len() {
                    return Err(format!("item {i} has an empty or out-of-bounds fragment range"));
                }
                let multi = end_f - start_f > 1;
                for f in start_f..end_f {
                    if self.frag_start[f] > self.frag_end[f] {
                        return Err(format!(
                            "item {i} fragment ends before it starts: {}..{}",
                            self.frag_start[f], self.frag_end[f]
                        ));
                    }
                    if multi && self.frag_start[f] == self.frag_end[f] {
                        return Err(format!(
                            "item {i} has an empty fragment in a discontinuous span: {}..{}",
                            self.frag_start[f], self.frag_end[f]
                        ));
                    }
                    if f > start_f && self.frag_end[f - 1] >= self.frag_start[f] {
                        return Err(format!(
                            "item {i} fragments {}..{} and {}..{} are not separated in text order",
                            self.frag_start[f - 1],
                            self.frag_end[f - 1],
                            self.frag_start[f],
                            self.frag_end[f]
                        ));
                    }
                }
                if self.start[i] != self.frag_start[start_f] || self.end[i] != self.frag_end[end_f - 1] {
                    return Err(format!(
                        "item {i} envelope {}..{} does not match its fragments {}..{}",
                        self.start[i],
                        self.end[i],
                        self.frag_start[start_f],
                        self.frag_end[end_f - 1]
                    ));
                }
            }
        }
        match overlap {
            Overlap::None => {
                if !self.frag_item_end.is_empty() && (0..self.start.len()).any(|i| self.item_frag_range(i).len() > 1) {
                    return Err("discontinuous spans are not allowed when overlap is None".into());
                }
                if !self.is_ordered() {
                    return Err("spans in text order, not overlapping".into());
                }
                Ok(())
            }
            Overlap::Nesting => {
                if self.frag_item_end.is_empty() {
                    check_no_crossing(self.start, self.end, "span")?;
                } else {
                    check_no_crossing(self.frag_start, self.frag_end, "fragment")?;
                    self.check_fragment_nesting()?;
                }
                Ok(())
            }
            Overlap::Any => Ok(()),
        }
    }

    /// When any item is discontinuous under [`Overlap::Nesting`], verifies
    /// that whenever two items share covered bytes (a fragment of one
    /// overlaps a fragment of the other), one item contains the other
    /// (every fragment of the inner item lies inside a fragment of the
    /// outer item).
    fn check_fragment_nesting(&self) -> Result<(), String> {
        let n = self.start.len();
        let frags_of = |i: usize| -> (&[u32], &[u32]) {
            let r = self.item_frag_range(i);
            (&self.frag_start[r.clone()], &self.frag_end[r])
        };
        let contains_item = |outer: usize, inner: usize| -> bool {
            let (os, oe) = frags_of(outer);
            let (is, ie) = frags_of(inner);
            is.iter().zip(ie).all(|(&cs, &ce)| {
                let child = Span { start: cs, end: ce };
                os.iter().zip(oe).any(|(&ps, &pe)| Span { start: ps, end: pe }.contains(child))
            })
        };
        let shares_bytes = |a: usize, b: usize| -> bool {
            let (as_, ae) = frags_of(a);
            let (bs, be) = frags_of(b);
            as_.iter().zip(ae).any(|(&s1, &e1)| {
                let f1 = Span { start: s1, end: e1 };
                bs.iter().zip(be).any(|(&s2, &e2)| f1.overlaps(Span { start: s2, end: e2 }))
            })
        };
        let check_pair = |a: usize, b: usize| -> Result<(), String> {
            let (first, second) = if a < b { (a, b) } else { (b, a) };
            let env_first = Span { start: self.start[first], end: self.end[first] };
            let env_second = Span { start: self.start[second], end: self.end[second] };
            if env_first.overlaps(env_second)
                && shares_bytes(first, second)
                && !contains_item(first, second)
                && !contains_item(second, first)
            {
                return Err(format!(
                    "item {second} ({}..{}) partially crosses item {first} ({}..{}) across fragments",
                    env_second.start, env_second.end, env_first.start, env_first.end
                ));
            }
            Ok(())
        };
        if self.is_start_ordered() {
            for a in 0..n {
                let end_a = self.end[a];
                for b in (a + 1)..n {
                    if self.start[b] >= end_a {
                        break;
                    }
                    check_pair(a, b)?;
                }
            }
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by_key(|&i| self.start[i]);
            for i in 0..n {
                let a = order[i];
                let end_a = self.end[a];
                for &b in &order[(i + 1)..] {
                    if self.start[b] >= end_a {
                        break;
                    }
                    check_pair(a, b)?;
                }
            }
        }
        Ok(())
    }
}

/// Checks that no two non-empty intervals `[start[i], end[i])` partially
/// cross (`a.start < b.start && b.start < a.end && a.end < b.end`).
pub(super) fn check_no_crossing(starts: &[u32], ends: &[u32], kind: &str) -> Result<(), String> {
    let n = starts.len();
    if n <= 1 {
        return Ok(());
    }
    let ordered = (1..n).all(|i| starts[i - 1] < starts[i] || (starts[i - 1] == starts[i] && ends[i - 1] >= ends[i]));
    let mut stack: Vec<(usize, u32, u32)> = Vec::new();
    let mut step = |idx: usize| -> Result<(), String> {
        let s = starts[idx];
        let e = ends[idx];
        if s == e {
            return Ok(());
        }
        while let Some(&(_, _, top_end)) = stack.last() {
            if top_end <= s {
                stack.pop();
            } else {
                break;
            }
        }
        if let Some(&(anc_idx, anc_start, anc_end)) = stack.last()
            && e > anc_end
        {
            return Err(format!("{kind} {idx} ({s}..{e}) partially crosses {kind} {anc_idx} ({anc_start}..{anc_end})"));
        }
        stack.push((idx, s, e));
        Ok(())
    };
    if ordered {
        for i in 0..n {
            step(i)?;
        }
    } else {
        let mut order: Vec<usize> = (0..n).filter(|&i| starts[i] < ends[i]).collect();
        order.sort_by(|&a, &b| starts[a].cmp(&starts[b]).then(ends[b].cmp(&ends[a])));
        for i in order {
            step(i)?;
        }
    }
    Ok(())
}
