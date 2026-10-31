//! An open-label span typing head (`heads[]` of kind HEAD_OPEN_SPAN):
//! zero-shot named-entity and span recognition where caller-supplied label
//! names are embedded in the encoder's hidden space and matched against
//! every candidate word span (Zaratiana et al. 2024, GLiNER).
//!
//! Each candidate span `(i, j)` of 1 to `span.max_words` words is projected
//! into `span.width_dim` dimensions from its first word, its last word, and
//! a learned embedding of its width `j - i + 1`:
//!
//!   `h_{i,j} = leaky_relu_0.1(W_start v_i + W_end v_j + W_width w_{j-i} + b_span)`
//!
//! where `span.weight` `[dim, 2 * width + dim]` stores `[W_start | W_end | W_width]`.
//! Each caller label vector `q_l` `[hidden]` is projected into the same space:
//!
//!   `p_l = W_label q_l + b_label`
//!
//! and combined with the direct encoder-space inner products `<x_i, q_l>`
//! and `<x_j, q_l>` of the span's first and last word vectors `x_i, x_j`
//! (`[hidden]`):
//!
//!   `z_{i,j,l} = out.bias[0] + out.bias[1] <x_i, q_l> + out.bias[2] <x_j, q_l> + <h_{i,j}, p_l>`.

use crate::bundle::Bundle;
use crate::manifest::{Decode, HeadKind, HeadSpec};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::Linear;
use super::span::{Span, columns, flat};
use super::tagger::load_tensors;
use super::words::{self, Take, WordReader, Work};

/// Negative slope of the span projection's leaky ReLU activation.
pub const LEAKY_SLOPE: f32 = 0.1;

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let (mut out, width) = words::tensors(spec, hidden);
    let (max, dim) = spec.span.map_or((0, 0), |s| (s.max_words as u64, s.width_dim as u64));
    out.push(("span.width".into(), vec![max, dim]));
    out.push(("span.weight".into(), vec![dim, 2 * width + dim]));
    out.push(("span.bias".into(), vec![dim]));
    out.push(("label.weight".into(), vec![dim, hidden as u64]));
    out.push(("label.bias".into(), vec![dim]));
    out.push(("out.bias".into(), vec![3]));
    out.push(("center.weight".into(), vec![2, hidden as u64]));
    out
}

/// Splits a label name (such as `"creative-work"` or `"person_name"`) into
/// the words a tokenizer and encoder embed to represent the label.
pub fn label_words(label: &str) -> Vec<&str> {
    let parts: Vec<&str> = label
        .split(|c: char| c.is_whitespace() || c == '-' || c == '_' || c == '/')
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() { vec![label] } else { parts }
}

/// Mean-pools the `[n_words, hidden]` word features of one label name and
/// L2-normalizes the result to unit length (or zeros if the norm is zero).
pub fn pool_label_words(word_features: &[f32], hidden: usize) -> Result<Vec<f32>> {
    if hidden == 0 || word_features.is_empty() || !word_features.len().is_multiple_of(hidden) {
        return Err(Error::new(
            INVALID_SHAPE,
            format!("{} label word features for hidden {hidden}", word_features.len()),
        ));
    }
    if word_features.iter().any(|v| !v.is_finite()) {
        return Err(Error::new(INVALID_ARGUMENT, "a label word feature is not finite"));
    }
    let n = word_features.len() / hidden;
    let mut mean = vec![0.0f64; hidden];
    for row in word_features.chunks_exact(hidden) {
        for (m, &v) in mean.iter_mut().zip(row) {
            *m += v as f64;
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut mean {
        *m *= inv_n;
    }
    let norm = mean.iter().map(|v| v * v).sum::<f64>().sqrt();
    let scale = if norm > 1e-12 { 1.0 / norm } else { 0.0 };
    Ok(mean.into_iter().map(|v| (v * scale) as f32).collect())
}

/// Scratch buffers for [`OpenSpanMatcher::scores`], kept across calls.
#[derive(Default)]
pub struct OpenSpanWork {
    raw_words: Vec<f32>,
    raw_count: Vec<u32>,
    centered_labels: Vec<f32>,
    word_label_sim: Vec<f32>,
    starts: Vec<f32>,
    ends: Vec<f32>,
    spans: Vec<f32>,
    proj_labels: Vec<f32>,
}

/// A loaded open-label span matcher. Read-only after loading, so one head
/// serves many threads, each with its own [`Work`] and [`OpenSpanWork`].
#[derive(Debug)]
pub struct OpenSpanMatcher {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    start: Linear,
    end: Linear,
    widths: Vec<f32>,
    label: Linear,
    out_bias: [f32; 3],
    center_x: Vec<f32>,
    center_q: Vec<f32>,
    max_words: usize,
    dim: usize,
}

impl OpenSpanMatcher {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<OpenSpanMatcher> {
        let l = load_tensors(bundle, name, HeadKind::OpenSpan, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::OpenSpan {
            return Err(invalid(format!("head {:?} is not an open-span head", spec.name)));
        }
        spec.validate("head")?;
        let shape = spec.span.ok_or_else(|| invalid(format!("head {:?}: no span", spec.name)))?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let width = reader.width();
        let (max_words, dim) = (shape.max_words as usize, shape.width_dim as usize);
        let span_width_raw = t.take("span.width")?;
        let span_weight = t.take("span.weight")?;
        let span_bias = t.take("span.bias")?;
        let label_weight = t.take("label.weight")?;
        let label_bias = t.take("label.bias")?;
        let out_bias_vec = t.take("out.bias")?;
        let center_weight = t.take("center.weight")?;
        let all_finite = span_width_raw
            .iter()
            .chain(&span_weight)
            .chain(&span_bias)
            .chain(&label_weight)
            .chain(&label_bias)
            .chain(&out_bias_vec)
            .chain(&center_weight)
            .all(|v| v.is_finite());
        if !all_finite {
            return Err(Error::new(INVALID_ARGUMENT, format!("head {:?}: a weight is not finite", spec.name)));
        }
        let cols = 2 * width + dim;
        let zeros = vec![0.0; dim];
        let start = Linear::new(width, dim, columns(&span_weight, cols, 0, width), zeros.clone())?;
        let end = Linear::new(width, dim, columns(&span_weight, cols, width, width), zeros)?;
        let by_width = Linear::new(dim, dim, columns(&span_weight, cols, 2 * width, dim), span_bias)?;
        let mut widths = vec![0.0; max_words * dim];
        by_width.apply_rows(&span_width_raw, max_words, &mut widths);
        let label = Linear::new(hidden, dim, label_weight, label_bias)?;
        let center_x = center_weight[..hidden].to_vec();
        let center_q = center_weight[hidden..2 * hidden].to_vec();
        Ok(OpenSpanMatcher {
            spec: spec.clone(),
            hidden,
            reader,
            start,
            end,
            widths,
            label,
            out_bias: [out_bias_vec[0], out_bias_vec[1], out_bias_vec[2]],
            center_x,
            center_q,
            max_words,
            dim,
        })
    }

    pub fn spec(&self) -> &HeadSpec {
        &self.spec
    }

    pub fn labels(&self) -> &[String] {
        &self.spec.labels
    }

    /// The encoder width the head reads.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Whether the head reads each word's text (a `hash_embed`).
    pub fn reads_forms(&self) -> bool {
        self.reader.reads_forms()
    }

    /// The number of candidate spans a sequence of `n` words has.
    pub fn span_count(&self, n: usize) -> usize {
        (1..=self.max_words.min(n)).fold(0usize, |t, width| t.saturating_add(n - width + 1))
    }

    /// Scores every candidate span of one sequence against the `L` caller
    /// label vectors `label_states` (`[L, hidden]`, `L >= 1`), writing
    /// `[spans, L]` logits into `out` in order of first word, then last
    /// word, then label index. Returns the number of words.
    #[allow(clippy::too_many_arguments)]
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        label_states: &[f32],
        out: &mut Vec<f32>,
        w: &mut Work,
        sw: &mut OpenSpanWork,
    ) -> Result<usize> {
        out.clear();
        if label_states.is_empty() || !label_states.len().is_multiple_of(self.hidden) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("{} label state floats for hidden {}", label_states.len(), self.hidden),
            ));
        }
        if label_states.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a label state is not finite"));
        }
        let num_labels = label_states.len() / self.hidden;
        let n = self.reader.read(states, words, forms, w)?;
        let spans = self.span_count(n);
        if spans == 0 {
            return Ok(n);
        }
        self.reader.pool_into(states, words, n, &mut sw.raw_words, &mut sw.raw_count);
        for row in sw.raw_words.chunks_exact_mut(self.hidden) {
            center_and_normalize_in_place(row, &self.center_x);
        }
        sw.centered_labels.clear();
        sw.centered_labels.extend_from_slice(label_states);
        for row in sw.centered_labels.chunks_exact_mut(self.hidden) {
            center_and_normalize_in_place(row, &self.center_q);
        }
        let raw_sim = Linear::new(self.hidden, num_labels, sw.centered_labels.clone(), vec![0.0; num_labels])?;
        sw.word_label_sim.clear();
        sw.word_label_sim.resize(n * num_labels, 0.0);
        raw_sim.apply_rows(&sw.raw_words, n, &mut sw.word_label_sim);

        let dim = self.dim;
        for (lin, dst) in [(&self.start, &mut sw.starts), (&self.end, &mut sw.ends)] {
            dst.clear();
            dst.resize(n * dim, 0.0);
            lin.apply_rows(&w.pooled, n, dst);
        }
        sw.proj_labels.clear();
        sw.proj_labels.resize(num_labels * dim, 0.0);
        self.label.apply_rows(&sw.centered_labels, num_labels, &mut sw.proj_labels);

        sw.spans.clear();
        sw.spans.reserve(spans * dim);
        for i in 0..n {
            let s = &sw.starts[i * dim..][..dim];
            for j in i..n.min(i + self.max_words) {
                let (e, wd) = (&sw.ends[j * dim..][..dim], &self.widths[(j - i) * dim..][..dim]);
                sw.spans.extend((0..dim).map(|d| leaky(s[d] + e[d] + wd[d])));
            }
        }
        if sw.spans.iter().any(|v| !v.is_finite())
            || sw.proj_labels.iter().any(|v| !v.is_finite())
            || sw.word_label_sim.iter().any(|v| !v.is_finite())
        {
            return Err(Error::new(INVALID_ARGUMENT, "an open-span score is not finite"));
        }
        let matcher = Linear::new(dim, num_labels, sw.proj_labels.clone(), vec![self.out_bias[0]; num_labels])?;
        out.resize(spans * num_labels, 0.0);
        matcher.apply_rows(&sw.spans, spans, out);
        let (a_start, a_end) = (self.out_bias[1], self.out_bias[2]);
        let mut idx = 0;
        for i in 0..n {
            let sim_i = &sw.word_label_sim[i * num_labels..][..num_labels];
            for j in i..n.min(i + self.max_words) {
                let sim_j = &sw.word_label_sim[j * num_labels..][..num_labels];
                let row = &mut out[idx..idx + num_labels];
                for l in 0..num_labels {
                    row[l] += a_start * sim_i[l] + a_end * sim_j[l];
                }
                idx += num_labels;
            }
        }
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "an open-span score is not finite"));
        }
        Ok(n)
    }

    /// Decodes `[spans, num_labels]` logits at the default logit threshold `0.0`
    /// (`sigmoid(logit) > 0.5`, i.e. scoring above `O`'s `0.0` logit).
    pub fn decode(&self, scores: &[f32], n: usize, num_labels: usize, out: &mut Vec<Span>) -> Result<()> {
        self.decode_min_logit(scores, n, num_labels, 0.0, out)
    }

    /// Decodes `[spans, num_labels]` logits keeping every span whose best
    /// label logit is strictly greater than `min_logit`, applying flat
    /// non-maximum suppression when `spec.decode == Decode::SpansFlat`.
    pub fn decode_min_logit(
        &self,
        scores: &[f32],
        n: usize,
        num_labels: usize,
        min_logit: f32,
        out: &mut Vec<Span>,
    ) -> Result<()> {
        if num_labels == 0 {
            return Err(Error::new(INVALID_SHAPE, "0 open-span labels"));
        }
        if n > u32::MAX as usize {
            return Err(Error::new(INVALID_SHAPE, format!("{n} words is past the word index limit")));
        }
        if Some(scores.len()) != self.span_count(n).checked_mul(num_labels) {
            return Err(Error::new(
                INVALID_SHAPE,
                format!("{} open-span scores for {n} words and {num_labels} labels", scores.len()),
            ));
        }
        if !min_logit.is_finite() || scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "an open-span score or threshold is not finite"));
        }
        out.clear();
        let mut rows = scores.chunks_exact(num_labels);
        for i in 0..n {
            for j in i..n.min(i + self.max_words) {
                let Some(row) = rows.next() else { break };
                let best = (0..num_labels).fold(0, |b, l| if row[l] > row[b] { l } else { b });
                let margin = row[best];
                if margin > min_logit {
                    out.push(Span { first: i as u32, last: j as u32, label: best as u32, margin });
                }
            }
        }
        if self.spec.decode == Decode::SpansFlat {
            flat(out);
        }
        Ok(())
    }
}

#[inline]
fn leaky(x: f32) -> f32 {
    if x >= 0.0 { x } else { LEAKY_SLOPE * x }
}

fn center_and_normalize_in_place(row: &mut [f32], center: &[f32]) {
    let norm = row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt().max(1e-6);
    let inv = (1.0 / norm) as f32;
    for (v, &c) in row.iter_mut().zip(center) {
        *v = *v * inv - c;
    }
    let r = (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() + 1e-6).sqrt();
    let inv_r = (1.0 / r) as f32;
    for v in row {
        *v *= inv_r;
    }
}
