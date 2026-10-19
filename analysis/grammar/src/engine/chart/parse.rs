use super::*;

impl Parser {
    /// Lexes and parses `text`, replacing the chart.
    ///
    /// Each rule can occur only once along a derivation of the same span.
    /// This stops unary cycles, including rules with absent optional words.
    /// A recursive rule can still extend a span by consuming more text.
    ///
    /// # Panics
    ///
    /// If the input exceeds the 32-bit byte-offset limit.
    pub fn parse(&mut self, text: &str) {
        assert!(u32::try_from(text.len()).is_ok(), "text exceeds the byte-offset limit");
        self.lexed.lex_words(text, &self.grammar.style(), self.grammar.words());
        let n = self.lexed.len();
        self.chart.text_len = text.len();
        self.chart.nodes.clear();
        self.chart.kids.clear();
        self.chart.head.clear();
        self.chart.head.resize(n, NONE);
        self.times.clear();
        let rules = self.grammar.rules();
        for s in (0..n).rev() {
            let first = self.chart.nodes.len();
            self.seed(s);
            let w = self.lexed.word(s);
            let lo = self.by_word.partition_point(|(k, _)| k.as_str() < w);
            let hi = lo + self.by_word[lo..].partition_point(|(k, _)| k.as_str() == w);
            for k in lo..hi {
                self.complete(text, rules, self.by_word[k].1, s, None);
            }
            let mut cursor = first;
            while cursor < self.chart.nodes.len() {
                let id = cursor as u32;
                cursor += 1;
                let dim = self.chart.nodes[id as usize].dim;
                if dim == Dim::DEAD {
                    continue;
                }
                for k in 0..self.by_dim[usize::from(dim.0)].len() {
                    let r = self.by_dim[usize::from(dim.0)][k];
                    self.complete(text, rules, r, s, Some(id));
                }
            }
        }
    }

    /// The lexicon's nodes, or the numeral's, for lexeme `s`.
    #[inline]
    fn seed(&mut self, s: usize) {
        let lexeme = self.lexed.lexemes[s];
        let add = |chart: &mut Chart, dim: Dim, value: Value| {
            insert(
                chart,
                Node {
                    start: s as u32,
                    end: s as u32 + 1,
                    dim,
                    rule: TERMINAL,
                    score: 1,
                    apps: 0,
                    value,
                    kids: (0, 0),
                    next: NONE,
                },
            );
        };
        match lexeme.kind {
            Kind::Numeral => {
                let v = self.lexed.numeral(s).expect("a numeral lexeme has a value");
                add(&mut self.chart, self.grammar.numeral(), Value::Num(Num::new(v, NumKind::of(v), flag::DIGITS)));
            }
            Kind::Word | Kind::Punct => {
                for e in self.grammar.lexicon().lookup(self.lexed.word(s)) {
                    add(&mut self.chart, e.dim, Value::Term { value: e.value, flags: e.flags });
                }
            }
        }
    }

    /// Matches rule `r` at lexeme `s`, with `first` as its first child when
    /// the rule opens with a node, and adds the nodes it builds.
    #[inline]
    fn complete(&mut self, text: &str, rules: &'static [Rule], r: u16, s: usize, first: Option<u32>) {
        let rule = &rules[usize::from(r)];
        self.done.clear();
        self.done_kids.clear();
        self.stack.clear();
        match first {
            Some(id) => {
                let n = &self.chart.nodes[id as usize];
                let Item::Node { test, .. } = rule.items[0] else { unreachable!("indexed by its first node") };
                if test.is_some_and(|t| !t(&n.value)) {
                    return;
                }
                self.stack.push(id);
                walk(
                    &self.chart,
                    &self.lexed,
                    text,
                    &rule.items[1..],
                    n.end,
                    &mut self.stack,
                    0,
                    &mut self.done,
                    &mut self.done_kids,
                );
            }
            None => walk(
                &self.chart,
                &self.lexed,
                text,
                rule.items,
                s as u32,
                &mut self.stack,
                0,
                &mut self.done,
                &mut self.done_kids,
            ),
        }
        let Some(first_match) = self.done.first() else { return };
        // Optional words change the span, never the number of child nodes.
        self.values.resize(first_match.kids_len as usize, Value::Term { value: 0, flags: 0 });
        let mut remaining_kids = self.done_kids.as_slice();
        for d in 0..self.done.len() {
            let Done { end, kids_len, words } = self.done[d];
            let (kids, rest) = remaining_kids.split_at(kids_len as usize);
            remaining_kids = rest;
            if self.cycle_guard[usize::from(r)] && words == 0 && kids_len == 1 && repeats(&self.chart, kids[0], r) {
                continue;
            }
            debug_assert_eq!(self.values.len(), kids.len());
            // At most u32::MAX nonempty children fit in the input, so these
            // wider totals cannot overflow even when every child is maximal.
            let mut score = i64::from(rule.weight) + i64::from(words);
            let mut apps = 1u64;
            for (value, &k) in self.values.iter_mut().zip(kids) {
                let child = &self.chart.nodes[k as usize];
                *value = child.value;
                score += i64::from(child.score);
                apps += u64::from(child.apps);
            }
            let (Ok(score), Ok(apps)) = (i32::try_from(score), u32::try_from(apps)) else { continue };
            let ctx = Ctx { text, lexed: &self.lexed, start: s, end: end as usize, times: &self.times };
            let Some(value) = (rule.build)(&self.values, &ctx) else { continue };
            let Ok(k0) = u32::try_from(self.chart.kids.len()) else { continue };
            let Some(k1) = k0.checked_add(kids_len) else { continue };
            let node =
                Node { start: s as u32, end, dim: rule.dim, rule: r, score, apps, value, kids: (k0, k1), next: NONE };
            if insert(&mut self.chart, node) {
                self.chart.kids.extend_from_slice(kids);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ENGLISH;
    use crate::engine::node;

    fn rejects_overflow(score: i32, apps: u32) {
        static RULES: &[Rule] = &[Rule {
            name: "weighted child",
            dim: Dim(0),
            items: &[node(Dim(0))],
            weight: 1,
            build: |v, _| Some(v[0]),
        }];
        let mut parser = Parser::new(&ENGLISH);
        parser.lexed.lex("a", &ENGLISH.style());
        parser.chart.head.push(NONE);
        insert(
            &mut parser.chart,
            Node {
                start: 0,
                end: 1,
                dim: Dim(0),
                rule: TERMINAL,
                score,
                apps,
                value: Value::Term { value: 0, flags: 0 },
                kids: (0, 0),
                next: NONE,
            },
        );
        parser.complete("a", RULES, 0, 0, Some(0));
        assert_eq!(parser.chart.nodes.len(), 1, "overflowed candidate must not enter the chart");
        assert!(parser.chart.kids.is_empty());
    }

    #[test]
    fn overflowing_score_is_not_ranked() {
        rejects_overflow(i32::MAX, 0);
    }

    #[test]
    fn overflowing_application_count_is_not_ranked() {
        rejects_overflow(1, u32::MAX);
    }
}
