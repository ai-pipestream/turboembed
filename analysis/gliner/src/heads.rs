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
enum OutputHead {
    Mlp(Mlp),
    Linear(Dense),
}
impl OutputHead {
    fn run(&self, x: &[f32], rows: usize, mid: &mut Vec<f32>, y: &mut Vec<f32>) -> Result<(), String> {
        match self {
            Self::Mlp(head) => head.run(x, rows, mid, y),
            Self::Linear(head) => head.run(x, rows, y),
        }
    }
}

enum PromptHead {
    Mlp(Box<Mlp>),
    Identity,
}

impl PromptHead {
    fn run(&self, x: &[f32], rows: usize, mid: &mut Vec<f32>, y: &mut Vec<f32>) -> Result<(), String> {
        match self {
            Self::Mlp(head) => head.run(x, rows, mid, y),
            Self::Identity => {
                y.clear();
                y.extend_from_slice(x);
                Ok(())
            }
        }
    }
}

enum SpanHead {
    Marker { start: Box<Mlp>, end: Box<Mlp>, span: OutputHead },
    Token { proj_token: Dense, proj_label: Dense, out: Box<Mlp> },
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
    prompt: PromptHead,
    head: SpanHead,
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
        let head = if c.token_level {
            SpanHead::Token {
                proj_token: linear("scorer.proj_token")?,
                proj_label: linear("scorer.proj_label")?,
                out: Box::new(mlp("scorer.out_mlp")?),
            }
        } else {
            SpanHead::Marker {
                start: Box::new(mlp("span_rep_layer.span_rep_layer.project_start")?),
                end: Box::new(mlp("span_rep_layer.span_rep_layer.project_end")?),
                span: if c.out_project_mlp {
                    OutputHead::Mlp(mlp("span_rep_layer.span_rep_layer.out_project")?)
                } else {
                    OutputHead::Linear(linear("span_rep_layer.span_rep_layer.out_project")?)
                },
            }
        };
        Ok(Self {
            projection: linear("token_rep_layer.projection")?,
            prompt: if c.prompt_projection {
                PromptHead::Mlp(Box::new(mlp("prompt_rep_layer")?))
            } else {
                PromptHead::Identity
            },
            head,
            lstm,
            hidden: c.hidden,
        })
    }
    /// Every span of at most `max_width` words with every label, in start,
    /// width, label order. `states` holds rows of `width` floats; `prompts`
    /// and `words` index them. The model scores through score_with_work.
    #[cfg(test)]
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
        if matches!(self.head, SpanHead::Token { .. }) {
            let words = std::mem::take(&mut w.words);
            let prompts = std::mem::take(&mut w.prompts);
            return self.score_token(&words, &prompts, max_width, w);
        }
        let SpanHead::Marker { start, end, span } = &self.head else { unreachable!() };
        start.run(&w.words, n, &mut w.mid, &mut w.starts)?;
        end.run(&w.words, n, &mut w.mid, &mut w.ends)?;
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
            span.run(&w.x, block.len(), &mut w.mid, &mut w.spans)?;
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
    fn score_token(
        &self,
        words: &[f32],
        prompts: &[f32],
        max_width: usize,
        w: &mut Work,
    ) -> Result<Vec<Candidate>, String> {
        let SpanHead::Token { proj_token, proj_label, out: output } = &self.head else { unreachable!() };
        let (d, labels, n) = (self.hidden, prompts.len() / self.hidden, words.len() / self.hidden);
        proj_token.run(words, n, &mut w.starts)?;
        proj_label.run(prompts, labels, &mut w.ends)?;
        w.x.clear();
        for i in 0..n {
            let token = &w.starts[i * 2 * d..(i + 1) * 2 * d];
            for label in 0..labels {
                let prompt = &w.ends[label * 2 * d..(label + 1) * 2 * d];
                w.x.extend_from_slice(&token[..d]);
                w.x.extend_from_slice(&prompt[..d]);
                w.x.extend(token[d..].iter().zip(&prompt[d..]).map(|(a, b)| a * b));
            }
        }
        output.run(&w.x, n * labels, &mut w.mid, &mut w.logits)?;
        finite(&w.logits, "token score")?;
        let pair_count = (0..n).map(|start| n.min(start + max_width) - start).sum::<usize>();
        let mut out = Vec::new();
        out.try_reserve_exact(pair_count * labels).map_err(|_| memory(pair_count * labels * size_of::<Candidate>()))?;
        w.spans.clear();
        w.spans.resize(labels, f32::INFINITY);
        for start in 0..n {
            w.spans.fill(f32::INFINITY);
            for end in start..n.min(start + max_width) {
                for label in 0..labels {
                    let start_score = w.logits[(start * labels + label) * 3];
                    let end_score = w.logits[(end * labels + label) * 3 + 1];
                    let inside = &mut w.spans[label];
                    *inside = inside.min(w.logits[(end * labels + label) * 3 + 2]);
                    let logit = start_score.min(end_score).min(*inside);
                    out.push(Candidate { start, end: end + 1, label, logit, score: sigmoid(logit) });
                }
            }
        }
        Ok(out)
    }

    /// Frozen word and prompt features for output and boundary training,
    /// plus the ordinary inference candidates used to mine hard negatives.
    pub(crate) fn training(
        &self,
        states: &[f32],
        width: usize,
        prompts: &[usize],
        words: &[usize],
        max_width: usize,
    ) -> Result<(crate::train::TrainingInput, Vec<Candidate>), String> {
        if width != self.projection.k || prompts.iter().chain(words).any(|&i| (i + 1) * width > states.len()) {
            return Err("gliner hidden state layout mismatch".into());
        }
        let (d, labels, n) = (self.hidden, prompts.len(), words.len());
        let mut w = Work::default();
        for &i in prompts.iter().chain(words) {
            w.x.extend_from_slice(&states[i * width..(i + 1) * width]);
        }
        self.projection.run(&w.x, labels + n, &mut w.projected)?;
        self.prompt.run(&w.projected[..labels * d], labels, &mut w.mid, &mut w.prompts)?;
        let projected = std::mem::take(&mut w.projected);
        self.words(&projected[labels * d..], n, &mut w)?;
        let SpanHead::Marker { start, end, span } = &self.head else {
            return Err("gliner token-level output training is not supported".into());
        };
        start.run(&w.words, n, &mut w.mid, &mut w.starts)?;
        end.run(&w.words, n, &mut w.mid, &mut w.ends)?;
        w.starts.iter_mut().chain(w.ends.iter_mut()).for_each(|v| *v = v.max(0.0));
        let pairs: Vec<(usize, usize)> = (0..n).flat_map(|i| (i..n.min(i + max_width)).map(move |j| (i, j))).collect();
        let mut input = crate::train::TrainingInput {
            words: w.words.clone(),
            spans: Vec::new(),
            prompts: w.prompts.clone(),
            prompt_inputs: projected[..labels * d].to_vec(),
            pairs: pairs.iter().map(|&(start, end)| (start, end + 1)).collect(),
        };
        let scorer = Matrix::from_rows(&[&w.prompts], d).map_err(memory)?;
        let mut candidates = Vec::new();
        for block in pairs.chunks(SPANS) {
            w.x.clear();
            for &(i, j) in block {
                w.x.extend_from_slice(&w.starts[i * d..(i + 1) * d]);
                w.x.extend_from_slice(&w.ends[j * d..(j + 1) * d]);
            }
            input.spans.extend_from_slice(&w.x);
            span.run(&w.x, block.len(), &mut w.mid, &mut w.spans)?;
            let logits = size(&mut w.logits, block.len(), labels);
            scorer.apply(&w.spans, d, block.len(), None, logits, labels, Mode::Plain);
            finite(logits, "span score")?;
            for (&(i, j), row) in block.iter().zip(logits.chunks_exact(labels)) {
                for (label, &logit) in row.iter().enumerate() {
                    candidates.push(Candidate { start: i, end: j + 1, label, logit, score: sigmoid(logit) });
                }
            }
        }
        Ok((input, candidates))
    }

    pub(crate) fn set_output(&mut self, params: &[Vec<f32>]) -> Result<(), String> {
        let (h, f) = match &self.head {
            SpanHead::Marker { start, .. } => (self.hidden, start.first.n),
            SpanHead::Token { .. } => return Err("gliner token-level output training is not supported".into()),
        };
        if !matches!(params.len(), 2 | 4 | 8 | 10 | 12 | 14 | 16) {
            return Err("gliner output training requires output tensors with optional boundary and prompt MLPs".into());
        }
        let linear = matches!(params.len(), 2 | 10 | 14);
        let output_len = if linear { 2 } else { 4 };
        let span = match &params[..output_len] {
            [w0, b0, w1, b1] => OutputHead::Mlp(Mlp {
                first: Dense::new(w0, b0.clone(), f, 2 * h)?,
                last: Dense::new(w1, b1.clone(), h, f)?,
            }),
            [weight, bias] => OutputHead::Linear(Dense::new(weight, bias.clone(), h, 2 * h)?),
            _ => unreachable!(),
        };
        let mlp = |values: &[Vec<f32>]| -> Result<Box<Mlp>, String> {
            Ok(Box::new(Mlp {
                first: Dense::new(&values[0], values[1].clone(), f, h)?,
                last: Dense::new(&values[2], values[3].clone(), h, f)?,
            }))
        };
        let boundaries = if matches!(params.len(), 10 | 12 | 14 | 16) {
            Some((mlp(&params[output_len..output_len + 4])?, mlp(&params[output_len + 4..output_len + 8])?))
        } else {
            None
        };
        let prompt = if matches!(params.len(), 8 | 14 | 16) {
            Some(PromptHead::Mlp(mlp(&params[params.len() - 4..])?))
        } else {
            None
        };
        let SpanHead::Marker { start, end, span: current } = &mut self.head else { unreachable!() };
        *current = span;
        if let Some((new_start, new_end)) = boundaries {
            *start = new_start;
            *end = new_end;
        }
        if let Some(prompt) = prompt {
            self.prompt = prompt;
        }
        Ok(())
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
