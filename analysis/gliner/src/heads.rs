//! The markerV0 heads on the encoder's hidden states, batched over rows:
//! every linear layer is one packed product (turbo::cpu::Matrix) over all
//! the rows it applies to. The BiLSTM input projections of every word are
//! one product; only the recurrent step runs word by word. Spans go through
//! out_project and label scoring in blocks of SPANS rows, all buffers come
//! from one workspace retained across windows and documents.
use crate::{Candidate, Config};
use turbo::{
    cpu::{Matrix, Mode},
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
    gates: Vec<f32>,
    step: Vec<f32>,
    state: Vec<f32>,
    cell: Vec<f32>,
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
    /// weight_ih of the forward then the reverse direction, [2 * 4h, hidden].
    input: Dense,
    /// weight_hh [4h, h] and bias_hh of each direction, h = hidden / 2.
    recurrent: [(Matrix, Vec<f32>); 2],
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
        let ((wf, shape), (wr, _)) = (lstm("weight_ih", "")?, lstm("weight_ih", "_reverse")?);
        let ((bf, _), (br, _)) = (lstm("bias_ih", "")?, lstm("bias_ih", "_reverse")?);
        let (n, k) = (shape[0], shape[1]);
        if wf.len() != n * k || wr.len() != n * k {
            return Err("gliner recurrent shape mismatch".into());
        }
        let input = Dense { w: Matrix::from_rows(&[&wf, &wr], k).map_err(memory)?, b: [bf, br].concat(), n: 2 * n, k };
        let recurrent = |suffix: &str| -> Result<(Matrix, Vec<f32>), String> {
            let ((w, shape), (b, _)) = (lstm("weight_hh", suffix)?, lstm("bias_hh", suffix)?);
            Ok((Matrix::from_rows(&[&w], shape[1]).map_err(memory)?, b))
        };
        Ok(Self {
            projection: linear("token_rep_layer.projection")?,
            prompt: mlp("prompt_rep_layer")?,
            start: mlp("span_rep_layer.span_rep_layer.project_start")?,
            end: mlp("span_rep_layer.span_rep_layer.project_end")?,
            span: mlp("span_rep_layer.span_rep_layer.out_project")?,
            input,
            recurrent: [recurrent("")?, recurrent("_reverse")?],
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
    /// states in the first half of each row, reverse in the second. Gates are
    /// i, f, g, o, each h = hidden / 2 wide.
    fn words(&self, x: &[f32], n: usize, w: &mut Work) -> Result<(), String> {
        let h = self.hidden / 2;
        self.input.run(x, n, &mut w.gates)?;
        size(&mut w.words, n, self.hidden);
        for (back, (hh, bias)) in self.recurrent.iter().enumerate() {
            size(&mut w.state, 1, h);
            size(&mut w.cell, 1, h);
            size(&mut w.step, 1, 4 * h);
            for t in 0..n {
                let i = if back == 1 { n - t - 1 } else { t };
                hh.apply(&w.state, h, 1, Some(bias), &mut w.step, 4 * h, Mode::Plain);
                let a = &w.gates[(i * 2 + back) * 4 * h..][..4 * h];
                let row = &mut w.words[i * self.hidden + back * h..][..h];
                for j in 0..h {
                    let gate = |k: usize| a[k * h + j] + w.step[k * h + j];
                    let (ig, fg, gg, og) = (gate(0), gate(1), gate(2), gate(3));
                    if !(ig.is_finite() && fg.is_finite() && gg.is_finite() && og.is_finite()) {
                        return Err("gliner non-finite recurrent activation".into());
                    }
                    w.cell[j] = sigmoid(fg) * w.cell[j] + sigmoid(ig) * gg.tanh();
                    w.state[j] = sigmoid(og) * w.cell[j].tanh();
                    if !w.state[j].is_finite() {
                        return Err("gliner non-finite recurrent state".into());
                    }
                    row[j] = w.state[j];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl Work {
    pub(crate) fn allocations(&self) -> [(usize, usize); 13] {
        [
            &self.x,
            &self.projected,
            &self.prompts,
            &self.mid,
            &self.gates,
            &self.step,
            &self.state,
            &self.cell,
            &self.words,
            &self.starts,
            &self.ends,
            &self.spans,
            &self.logits,
        ]
        .map(|v| (v.as_ptr() as usize, v.capacity()))
    }
}
