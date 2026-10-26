use super::*;

impl Parser {
    /// Chooses the nodes for `output`: the candidates the grammar keeps
    /// that do not cut a written token, longest first (then by score,
    /// fewer applications, earlier start and lower rule index), each taken
    /// when it overlaps none taken before. Appends them to `out` in text
    /// order. `text` must be the source passed to `parse`. An incompatible
    /// byte layout produces no mentions.
    ///
    /// A number that lies inside a chosen ordinal is part of that ordinal,
    /// not a value the text states (`twenty` in `the twenty first`), so it
    /// is left out unless the grammar's [`Grammar::shadows`] says the
    /// ordinal leaves its numbers standing.
    pub fn select(&mut self, text: &str, output: Output, out: &mut Vec<Mention>) {
        if text.len() != self.chart.text_len {
            return;
        }
        self.shadowed.clear();
        if output == Output::Number {
            let mut shadows = std::mem::take(&mut self.shadows);
            shadows.clear();
            self.pick(text, Output::Ordinal, &mut shadows, &|_, _, _, end| end);
            self.shadowed.resize(self.lexed.len(), NONE);
            for (i, m) in shadows.iter().enumerate() {
                let node = &self.chart.nodes[m.node as usize];
                if self.grammar.shadows(node) {
                    self.shadowed[node.start as usize..node.end as usize].fill(i as u32);
                }
            }
            self.shadows = shadows;
        }
        self.pick(text, output, out, &|_, _, _, end| end);
    }

    /// Select time mentions while reserving a recognized suffix during
    /// ranking and overlap checks. Returned mentions retain the original
    /// value span so the caller can resolve the suffix separately.
    pub(crate) fn select_time_suffix(
        &mut self,
        text: &str,
        out: &mut Vec<Mention>,
        suffix: impl Fn(&Interned, &Value, u32, u32) -> u32,
    ) {
        if text.len() != self.chart.text_len {
            return;
        }
        self.pick(text, Output::Time, out, &suffix);
    }

    #[inline]
    fn pick(
        &mut self,
        text: &str,
        output: Output,
        out: &mut Vec<Mention>,
        suffix: &impl Fn(&Interned, &Value, u32, u32) -> u32,
    ) {
        let n = self.lexed.len();
        let lex = &self.lexed.lexemes;
        self.cands.clear();
        for (id, node) in self.chart.nodes.iter().enumerate() {
            let needs_zone = self.grammar.output(node.dim) != Some(output);
            if node.dim == Dim::DEAD
                || needs_zone && !(output == Output::Time && self.grammar.time_zone_context(node.dim))
            {
                continue;
            }
            let (s, e) = (node.start as usize, node.end as usize);
            // A node may not start or end inside a word glued to a numeral
            // (`29km`, `x10`).
            if lex[s].glued && s > 0 && lex[s - 1].is_alnum() && lex[s].is_alnum() {
                continue;
            }
            if e < n && lex[e].glued && lex[e].is_alnum() && lex[e - 1].is_alnum() {
                continue;
            }
            // Chosen ordinals do not overlap, so one index at both ends
            // means one ordinal covers the whole span.
            if output == Output::Number && !self.shadowed.is_empty() {
                let owner = self.shadowed[s];
                if owner != NONE && owner == self.shadowed[e - 1] {
                    continue;
                }
            }
            if !text.is_char_boundary(lex[s].start as usize) || !text.is_char_boundary(lex[e - 1].end as usize) {
                continue;
            }
            let ctx = Ctx { text, lexed: &self.lexed, start: s, end: e, times: &self.times };
            if needs_zone
                && ctx.before().and_then(|i| ctx.punct(i)).is_some_and(|c| {
                    turbo_unicode::category::general_category(c) == turbo_unicode::category::GeneralCategory::Sc
                })
            {
                continue;
            }
            if !self.grammar.keep(output, node, &ctx) {
                continue;
            }
            let end = suffix(&self.times, &node.value, lex[s].start, lex[e - 1].end);
            if end < lex[e - 1].end
                || needs_zone && end == lex[e - 1].end
                || end as usize > text.len()
                || !text.is_char_boundary(end as usize)
            {
                continue;
            }
            self.cands.push((id as u32, lex[s].start, end));
        }
        let nodes = &self.chart.nodes;
        self.cands.sort_by(|a, b| {
            let (x, y) = (&nodes[a.0 as usize], &nodes[b.0 as usize]);
            (b.2 - b.1)
                .cmp(&(a.2 - a.1))
                .then(y.score.cmp(&x.score))
                .then(x.apps.cmp(&y.apps))
                .then(a.1.cmp(&b.1))
                .then(x.rule.cmp(&y.rule))
                .then(a.0.cmp(&b.0))
        });
        self.taken.clear();
        self.taken.resize(n, false);
        let from = out.len();
        for &(id, start, end) in &self.cands {
            let node = &nodes[id as usize];
            let extended = lex.partition_point(|l| l.start < end);
            let span = node.start as usize..extended;
            if self.taken[span.clone()].iter().any(|&t| t) {
                continue;
            }
            self.taken[span].fill(true);
            out.push(Mention { start, end: lex[node.end as usize - 1].end, value: node.value, node: id });
        }
        out[from..].sort_by_key(|p| p.start);
    }
}
