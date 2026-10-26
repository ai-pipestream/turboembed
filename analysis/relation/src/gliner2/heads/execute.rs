use super::*;

impl Heads {
    /// The count head's logits on a relation's `[P]` vector.
    ///
    /// # Panics
    /// When `p` is not one row of the model width.
    pub fn count_logits(&self, p: &[f32]) -> Vec<f32> {
        assert_eq!(p.len(), self.width, "a [P] vector of the model width");
        self.count_pred.apply(p)
    }

    /// How many instances a relation has: the count head's argmax, the
    /// first of equal values, on its `[P]` vector.
    pub fn count(&self, p: &[f32]) -> usize {
        let logits = self.count_logits(p);
        let mut best = 0;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best.min(self.positions.len() / self.width)
    }

    /// The `k` instances' head and tail queries, `[k][2][D]` flattened,
    /// from the two `[R]` vectors.
    ///
    /// # Panics
    /// As [`Heads::queries_of`].
    pub fn queries(&self, fields: [&[f32]; 2], k: usize) -> Vec<f32> {
        self.queries_of(&fields, k)
    }

    /// The `k` instances' queries for `M` fields, `[k][M][D]` flattened,
    /// from each field's marker vector; the transformer attends across the
    /// `M` fields of an instance.
    ///
    /// # Panics
    /// When a field is not one row of the model width, or `k` is past the
    /// position table (the count head never predicts that many).
    pub fn queries_of(&self, fields: &[&[f32]], k: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.queries_of_into(fields, k, &mut Scratch::default(), &mut out);
        out
    }

    /// Project all field queries using caller-reused projection buffers.
    /// Recurrent and attention temporaries remain local to this call.
    pub fn queries_of_into(&self, fields: &[&[f32]], k: usize, scratch: &mut Scratch, out: &mut Vec<f32>) {
        let (d, n) = (self.width, fields.len());
        assert!(fields.iter().all(|f| f.len() == d), "field vectors of the model width");
        assert!(k <= self.positions.len() / d, "{k} instances past the position table");
        let mut states: Vec<Vec<f32>> = fields.iter().map(|f| f.to_vec()).collect();
        let (mut gi, mut gh) = (vec![0.0; 3 * d], vec![0.0; 3 * d]);
        let mut steps = vec![vec![0.0; n * d]; k];
        for (t, step) in steps.iter_mut().enumerate() {
            self.gru_ih.apply(&self.positions[t * d..(t + 1) * d], &mut gi);
            for (m, h) in states.iter_mut().enumerate() {
                self.gru_hh.apply(h, &mut gh);
                for j in 0..d {
                    let r = sigmoid(gi[j] + gh[j]);
                    let z = sigmoid(gi[d + j] + gh[d + j]);
                    let g = (gi[2 * d + j] + r * gh[2 * d + j]).tanh();
                    h[j] = (1.0 - z) * g + z * h[j];
                }
                for (s, (&a, &b)) in step[m * d..(m + 1) * d].iter_mut().zip(h.iter().zip(fields[m])) {
                    *s = a + b;
                }
            }
        }
        let m = self.in_projector.out_dim;
        let mut cats = Vec::with_capacity(n * k * (m + d));
        for y in &steps {
            let mut x = vec![0.0; n * m];
            for f in 0..n {
                self.in_projector.apply(&y[f * d..(f + 1) * d], &mut x[f * m..(f + 1) * m]);
            }
            for l in &self.layers {
                l.apply(&mut x, n);
            }
            for f in 0..n {
                cats.extend_from_slice(&x[f * m..(f + 1) * m]);
                cats.extend_from_slice(&y[f * d..(f + 1) * d]);
            }
        }
        // Every instance's fields through the out-projector at once, so its
        // weights are read once per task.
        let [a, b, ..] = &mut scratch.buffers;
        self.out_projector[0].rows_into(&cats, n * k, a);
        relu(a);
        self.out_projector[1].rows_into(a, n * k, b);
        relu(b);
        self.out_projector[2].rows_into(b, n * k, out);
    }

    /// The span layer's hidden rows for every span of the `n` word vectors
    /// `words` (`[n][D]`): `[n][max_width][4D]`, empty rows for spans past
    /// the last word.
    ///
    /// # Panics
    /// When `words` is not `n` rows of the model width.
    pub fn spans(&self, words: &[f32], n: usize) -> Spans {
        let mut out = Spans::default();
        self.spans_into(words, n, &mut Scratch::default(), &mut out);
        out
    }

    /// Span rows with reusable word projection buffers and output storage.
    pub fn spans_into(&self, words: &[f32], n: usize, scratch: &mut Scratch, out: &mut Spans) {
        assert_eq!(Some(words.len()), n.checked_mul(self.width), "word rows of the model width");
        let w = self.max_width;
        let hidden = self.first_bias.len();
        let [tmp, projected, start, end, ..] = &mut scratch.buffers;
        mlp_into(&self.start, words, n, tmp, projected);
        relu(projected);
        self.first_start.rows_into(projected, n, start);
        mlp_into(&self.end, words, n, tmp, projected);
        relu(projected);
        self.first_end.rows_into(projected, n, end);
        out.rows.resize(n * w * hidden, 0.0);
        out.rows.fill(0.0);
        for i in 0..n {
            for k in 0..w.min(n - i) {
                let row = &mut out.rows[(i * w + k) * hidden..][..hidden];
                let (s, e) = (&start[i * hidden..][..hidden], &end[(i + k) * hidden..][..hidden]);
                for (((r, &a), &b), &c) in row.iter_mut().zip(s).zip(e).zip(&self.first_bias) {
                    *r = (a + b + c).max(0.0);
                }
            }
        }
        out.n = n;
        out.width = w;
        out.hidden = hidden;
    }

    /// One query's logits over the spans, `[n][max_width]`, with spans
    /// past the last word at negative infinity.
    pub fn logits(&self, spans: &Spans, query: &[f32]) -> Vec<f32> {
        self.logits_all(spans, query)
    }

    /// [`Heads::logits`] for each query of `queries` (`[q][D]`), one after
    /// another: the last span layer folds into all queries in one pass
    /// over its weights, and each span row is read once for all of them.
    ///
    /// # Panics
    /// When `queries` is not whole rows of the model width, or `spans`
    /// came from heads of another span width.
    pub fn logits_all(&self, spans: &Spans, queries: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        self.logits_all_into(spans, queries, &mut Scratch::default(), &mut out);
        out
    }

    /// All query logits with reusable folded-query and output buffers.
    pub fn logits_all_into(&self, spans: &Spans, queries: &[f32], scratch: &mut Scratch, out: &mut Vec<f32>) {
        let (h, d) = (spans.hidden, self.width);
        assert!(queries.len().is_multiple_of(d), "queries of the model width");
        assert!(h == self.last.in_dim && spans.width == self.max_width, "spans from these heads");
        let q = queries.len() / d;
        let [folded, bias, ..] = &mut scratch.buffers;
        self.fold.rows_into(queries, q, folded);
        bias.clear();
        bias.extend(queries.chunks_exact(d).map(|x| dot(x, self.last.bias())));
        let size = spans.n * spans.width;
        out.resize(q * size, f32::NEG_INFINITY);
        out.fill(f32::NEG_INFINITY);
        for i in 0..spans.n {
            for k in 0..spans.width.min(spans.n - i) {
                let row = &spans.rows[(i * spans.width + k) * h..][..h];
                for (j, (f, &b)) in folded.chunks_exact(h).zip(bias.iter()).enumerate() {
                    out[j * size + i * spans.width + k] = dot(row, f) + b;
                }
            }
        }
    }
}
