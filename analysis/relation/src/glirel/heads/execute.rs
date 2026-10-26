use super::*;

impl Heads {
    /// Project `n` encoder rows into caller-owned output storage.
    pub fn project_into(&self, rows: &[f32], n: usize, out: &mut Vec<f32>) {
        self.projection.rows_into(rows, n, out);
    }

    /// Project `n` encoder rows, allocating the returned storage.
    pub fn project(&self, rows: &[f32], n: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.project_into(rows, n, &mut out);
        out
    }

    /// Bidirectional LSTM with reusable input products, states and outputs.
    pub fn lstm_into(&self, words: &[f32], n: usize, scratch: &mut Scratch, out: &mut Vec<f32>) {
        let l = self.width / 2;
        let (directions, work) = scratch.buffers.split_at_mut(2);
        self.lstm[0].states_into(words, n, false, &mut work[..4], &mut directions[0]);
        self.lstm[1].states_into(words, n, true, &mut work[..4], &mut directions[1]);
        out.clear();
        out.reserve(n * 2 * l);
        for t in 0..n {
            out.extend_from_slice(&directions[0][t * l..(t + 1) * l]);
            out.extend_from_slice(&directions[1][t * l..(t + 1) * l]);
        }
    }

    /// The LSTM's `[n][H]` word states, forward then backward per word.
    pub fn lstm(&self, words: &[f32], n: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.lstm_into(words, n, &mut Scratch::default(), &mut out);
        out
    }

    /// Entity representations with reusable projection and gather buffers.
    /// Panics for an invalid inclusive word range or incomplete input rows.
    pub fn entities_into(&self, lstm: &[f32], spans: &[(usize, usize)], scratch: &mut Scratch, out: &mut Vec<f32>) {
        let h = self.width;
        assert!(lstm.len().is_multiple_of(h), "LSTM rows of the head width");
        let n = lstm.len() / h;
        assert!(spans.iter().all(|&(s, e)| s <= e && e < n), "entity words inside the text");
        let k = spans.len();
        let [rows, tmp, projected, a, b, hidden, _, _] = &mut scratch.buffers;
        rows.clear();
        rows.extend(spans.iter().flat_map(|&(s, _)| lstm[s * h..(s + 1) * h].iter().copied()));
        mlp_into(&self.start, rows, k, tmp, projected);
        relu(projected);
        self.entity_first[0].rows_into(projected, k, a);
        rows.clear();
        rows.extend(spans.iter().flat_map(|&(_, e)| lstm[e * h..(e + 1) * h].iter().copied()));
        mlp_into(&self.end, rows, k, tmp, projected);
        relu(projected);
        self.entity_first[1].rows_into(projected, k, b);
        hidden.clear();
        hidden.extend(
            a.chunks_exact(self.entity_bias.len())
                .zip(b.chunks_exact(self.entity_bias.len()))
                .flat_map(|(a, b)| a.iter().zip(b).zip(&self.entity_bias).map(|((a, b), c)| (a + b + c).max(0.0))),
        );
        self.entity_last.rows_into(hidden, k, out);
    }

    /// Each inclusive entity word range's representation, `[k][H]`.
    pub fn entities(&self, lstm: &[f32], spans: &[(usize, usize)]) -> Vec<f32> {
        let mut out = Vec::new();
        self.entities_into(lstm, spans, &mut Scratch::default(), &mut out);
        out
    }

    /// Label representations using caller-owned intermediate and output storage.
    pub fn types_into(&self, means: &[f32], m: usize, scratch: &mut Scratch, out: &mut Vec<f32>) {
        mlp_into(&self.prompt, means, m, &mut scratch.buffers[0], out);
    }

    /// Each label's representation, `[m][H]`, from its projected mean.
    pub fn types(&self, means: &[f32], m: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.types_into(means, m, &mut Scratch::default(), &mut out);
        out
    }

    /// Pair logits, `[pairs][labels]`, with caller-reused storage.
    /// Panics for incomplete rows or a pair outside the entities.
    pub fn logits_into(
        &self,
        entities: &[f32],
        pairs: &[(usize, usize)],
        types: &[f32],
        scratch: &mut Scratch,
        out: &mut Vec<f32>,
    ) {
        let (h, f) = (self.width, self.pair_bias.len());
        assert!(entities.len().is_multiple_of(h) && types.len().is_multiple_of(h), "rows of the head width");
        let (k, m) = (entities.len() / h, types.len() / h);
        assert!(pairs.iter().all(|&(a, b)| a < k && b < k), "pairs of the entities given");
        let [head, tail, folded, bias, hidden, _, _, _] = &mut scratch.buffers;
        self.pair_first[0].rows_into(entities, k, head);
        self.pair_first[1].rows_into(entities, k, tail);
        folded.resize(m * f, 0.0);
        folded.fill(0.0);
        for (t, fold) in types.chunks_exact(h).zip(folded.chunks_exact_mut(f)) {
            for (o, &q) in t.iter().enumerate() {
                for (x, &w) in fold.iter_mut().zip(&self.pair_last.weight()[o * f..(o + 1) * f]) {
                    *x += q * w;
                }
            }
        }
        bias.clear();
        bias.extend(types.chunks_exact(h).map(|t| dot(t, self.pair_last.bias())));
        hidden.resize(f, 0.0);
        out.clear();
        out.reserve(pairs.len() * m);
        for &(a, b) in pairs {
            let (x, y) = (&head[a * f..(a + 1) * f], &tail[b * f..(b + 1) * f]);
            for (((z, &x), &y), &c) in hidden.iter_mut().zip(x).zip(y).zip(&self.pair_bias) {
                *z = (x + y + c).max(0.0);
            }
            out.extend(folded.chunks_exact(f).zip(bias.iter()).map(|(w, &c)| dot(hidden, w) + c));
        }
    }

    /// Each pair's logit for every label, allocating returned storage.
    pub fn logits(&self, entities: &[f32], pairs: &[(usize, usize)], types: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.logits_into(entities, pairs, types, &mut Scratch::default(), &mut out);
        out
    }
}
