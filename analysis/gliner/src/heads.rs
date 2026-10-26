//! The markerV0 heads on the encoder's hidden states, batched over rows:
//! every linear layer is one packed product (turbo::cpu::Matrix) over all
//! the rows it applies to. The BiLSTM is turbo::cpu::lstm (#859): the input
//! projections of every word are one product, then one recurrent product
//! per step. Spans go through
//! out_project and label scoring in blocks of SPANS rows, all buffers come
//! from one workspace retained across windows and documents.
use crate::{Candidate, Config};
use turbo::{
    cpu::{
        Matrix, Mode,
        lstm::{self, Lstm, Weights},
    },
    safetensors::{Dtype, File},
};

/// Spans per out_project and scoring block.
const SPANS: usize = 128;

pub(crate) use turbo::head::sigmoid;
fn finite(y: &[f32], what: &str) -> Result<(), String> {
    if y.iter().all(|v| v.is_finite()) { Ok(()) } else { Err(format!("gliner non-finite {what}")) }
}
fn memory(bytes: usize) -> String {
    format!("gliner head allocation failed ({bytes} bytes)")
}
/// rows * n zeros in y, reusing its memory.
fn size(y: &mut Vec<f32>, rows: usize, n: usize) -> &mut [f32] {
    y.clear();
    y.resize(rows * n, 0.0);
    y
}

/// One linear layer, W [n, k] packed, y = x W^T + b over any rows.
struct Dense {
    w: Matrix,
    b: Vec<f32>,
    n: usize,
    k: usize,
}
impl Dense {
    fn new(w: &[f32], b: Vec<f32>, n: usize, k: usize) -> Result<Self, String> {
        if w.len() != n * k || b.len() != n {
            return Err("gliner head shape mismatch".into());
        }
        Ok(Self { w: Matrix::from_rows(&[w], k).map_err(memory)?, b, n, k })
    }
    fn run(&self, x: &[f32], rows: usize, y: &mut Vec<f32>) -> Result<(), String> {
        let y = size(y, rows, self.n);
        self.w.apply(x, self.k, rows, Some(&self.b), y, self.n, Mode::Plain);
        finite(y, "head activation")
    }
}
struct Mlp {
    first: Dense,
    last: Dense,
}
impl Mlp {
    fn run(&self, x: &[f32], rows: usize, mid: &mut Vec<f32>, y: &mut Vec<f32>) -> Result<(), String> {
        self.first.run(x, rows, mid)?;
        mid.iter_mut().for_each(|v| *v = v.max(0.0));
        self.last.run(mid, rows, y)
    }
}
/// The buffers of one call.
#[derive(Default)]
pub(crate) struct Work {
    x: Vec<f32>,
    projected: Vec<f32>,
    prompts: Vec<f32>,
    mid: Vec<f32>,
    lstm: lstm::Work,
    words: Vec<f32>,
    starts: Vec<f32>,
    ends: Vec<f32>,
    spans: Vec<f32>,
    logits: Vec<f32>,
}
pub(crate) struct Heads {
    projection: Dense,
    prompt: Mlp,
    start: Mlp,
    end: Mlp,
    span: Mlp,
    /// The forward and reverse directions over projected words, h =
    /// hidden / 2 each, bias_ih and bias_hh summed.
    lstm: Lstm,
    hidden: usize,
}
impl Heads {
    pub fn load(c: &Config, ec: &turbo::deberta::Config, bytes: &[u8]) -> Result<Self, String> {
        let f = File::parse("gliner heads", bytes).map_err(|e| e.message)?;
        for (name, shape) in c.tensor_shapes(ec)? {
            let t = f.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
            if t.shape != shape || t.data.as_chunks::<4>().0.iter().any(|v| !f32::from_le_bytes(*v).is_finite()) {
                return Err(format!("gliner {name}: shape or finite-weight validation failed"));
            }
        }
        // Weights are rank 2, biases rank 1; shapes are checked above.
        let tensor = |name: &str| -> Result<(Vec<f32>, Vec<usize>), String> {
            let t = f.get(name, Dtype::F32, if name.contains("bias") { 1 } else { 2 }).map_err(|e| e.message)?;
            Ok((t.f32s(), t.shape.iter().map(|&d| d as usize).collect()))
        };
        let linear = |s: &str| -> Result<Dense, String> {
            let ((w, shape), (b, _)) = (tensor(&format!("{s}.weight"))?, tensor(&format!("{s}.bias"))?);
            Dense::new(&w, b, shape[0], shape[1])
        };
        let mlp = |s: &str| -> Result<Mlp, String> {
            Ok(Mlp { first: linear(&format!("{s}.0"))?, last: linear(&format!("{s}.3"))? })
        };
        let lstm = |name: &str, suffix: &str| tensor(&format!("rnn.lstm.{name}_l0{suffix}"));
        let direction = |suffix: &str| -> Result<_, String> {
            let ((w, shape), (u, _)) = (lstm("weight_ih", suffix)?, lstm("weight_hh", suffix)?);
            let ((bi, _), (bh, _)) = (lstm("bias_ih", suffix)?, lstm("bias_hh", suffix)?);
            if bi.len() != bh.len() {
                return Err("gliner recurrent shape mismatch".into());
            }
            Ok((w, u, bi.iter().zip(&bh).map(|(a, b)| a + b).collect::<Vec<f32>>(), shape[1]))
        };
        let ((wf, uf, bf, k), (wr, ur, br, _)) = (direction("")?, direction("_reverse")?);
        let dirs =
            [Weights { w: &wf, u: &uf, b: &bf, reverse: false }, Weights { w: &wr, u: &ur, b: &br, reverse: true }];
        let lstm = Lstm::new(k, c.hidden / 2, &dirs).map_err(|e| format!("gliner recurrent: {}", e.message))?;
        Ok(Self {
            projection: linear("token_rep_layer.projection")?,
            prompt: mlp("prompt_rep_layer")?,
            start: mlp("span_rep_layer.span_rep_layer.project_start")?,
            end: mlp("span_rep_layer.span_rep_layer.project_end")?,
            span: mlp("span_rep_layer.span_rep_layer.out_project")?,
            lstm,
            hidden: c.hidden,
        })
    }
    /// Every span of at most `max_width` words with every label, in start,
    /// width, label order. `states` holds rows of `width` floats; `prompts`
    /// and `words` index them.
    pub fn score(
        &self,
        states: &[f32],
        width: usize,
        prompts: &[usize],
        words: &[usize],
        max_width: usize,
    ) -> Result<Vec<Candidate>, String> {
        self.score_with_work(states, width, prompts, words, max_width, &mut Work::default())
    }

    pub(crate) fn score_with_work(
        &self,
        states: &[f32],
        width: usize,
        prompts: &[usize],
        words: &[usize],
        max_width: usize,
        w: &mut Work,
    ) -> Result<Vec<Candidate>, String> {
        if width != self.projection.k || prompts.iter().chain(words).any(|&i| (i + 1) * width > states.len()) {
            return Err("gliner hidden state layout mismatch".into());
        }
        let (d, labels, n) = (self.hidden, prompts.len(), words.len());
        w.x.clear();
        for &i in prompts.iter().chain(words) {
            w.x.extend_from_slice(&states[i * width..(i + 1) * width]);
        }
        self.projection.run(&w.x, labels + n, &mut w.projected)?;
        self.prompt.run(&w.projected[..labels * d], labels, &mut w.mid, &mut w.prompts)?;
        let scorer = Matrix::from_rows(&[&w.prompts], d).map_err(memory)?;
        let projected = std::mem::take(&mut w.projected);
        let result = self.words(&projected[labels * d..], n, w);
        w.projected = projected;
        result?;
        self.start.run(&w.words, n, &mut w.mid, &mut w.starts)?;
        self.end.run(&w.words, n, &mut w.mid, &mut w.ends)?;
        // out_project reads relu([start, end]).
        w.starts.iter_mut().chain(w.ends.iter_mut()).for_each(|v| *v = v.max(0.0));
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n.min(i + max_width)).map(move |j| (i, j))).collect();
        let mut out = Vec::new();
        out.try_reserve_exact(pairs.len() * labels)
            .map_err(|_| memory(pairs.len() * labels * size_of::<Candidate>()))?;
        for block in pairs.chunks(SPANS) {
            w.x.clear();
            for &(i, j) in block {
                w.x.extend_from_slice(&w.starts[i * d..(i + 1) * d]);
                w.x.extend_from_slice(&w.ends[j * d..(j + 1) * d]);
            }
            self.span.run(&w.x, block.len(), &mut w.mid, &mut w.spans)?;
            let logits = size(&mut w.logits, block.len(), labels);
            scorer.apply(&w.spans, d, block.len(), None, logits, labels, Mode::Plain);
            finite(logits, "span score")?;
            for (&(i, j), row) in block.iter().zip(logits.chunks_exact(labels)) {
                for (label, &logit) in row.iter().enumerate() {
                    out.push(Candidate { start: i, end: j + 1, label, logit, score: sigmoid(logit) });
                }
            }
        }
        Ok(out)
    }
    /// The BiLSTM over `n` projected words into w.words [n, hidden]: forward
    /// states in the first half of each row, reverse in the second.
    fn words(&self, x: &[f32], n: usize, w: &mut Work) -> Result<(), String> {
        size(&mut w.words, n, self.hidden);
        if n == 0 {
            return Ok(());
        }
        self.lstm.run(x, &[n], &mut w.words, &mut w.lstm).map_err(|e| format!("gliner recurrent: {}", e.message))
    }
}

#[cfg(test)]
#[path = "heads_tests.rs"]
mod lstm_tests;

#[cfg(test)]
impl Work {
    pub(crate) fn allocations(&self) -> [(usize, usize); 9] {
        [
            &self.x,
            &self.projected,
            &self.prompts,
            &self.mid,
            &self.words,
            &self.starts,
            &self.ends,
            &self.spans,
            &self.logits,
        ]
        .map(|v| (v.as_ptr() as usize, v.capacity()))
    }
}
