//! A span classifier head (`heads[]` of kind HEAD_SPAN_CLASSIFIER): a
//! label for every span of 1 to `span.max_words` words of a sequence, for
//! typed and nested spans (nested entities, coreference mentions) with
//! exact offsets, since a span is whole words.
//!
//! A span from word i to word j is read as its first and last word
//! vectors and a learned embedding of its width j - i + 1 (Lee et al.
//! 2017 without the attention over its words), joined and scored by one
//! linear layer: `out.weight` [labels, 2 * width + width_dim]. The score
//! splits by the parts of that row, so each word's start and end scores
//! are computed once and a span's score is their sum plus its width's.
//!
//! Spans are in order of their first word, then their last. A span whose
//! best label is `O` is no span. Under DECODE_SPANS_NESTED every other
//! span is kept; under DECODE_SPANS_FLAT the spans are taken from the
//! highest margin over `O` down, each one kept unless it overlaps one
//! already kept.

use crate::bundle::Bundle;
use crate::manifest::{Decode, HeadKind, HeadSpec};
use crate::status::{Error, INVALID_ARGUMENT, INVALID_SHAPE, Result, invalid};

use super::Linear;
use super::tagger::load_tensors;
use super::words::{self, Take, WordReader, Work};

/// Each tensor of `spec` as a suffix of its `tensor_prefix` and its shape,
/// for an encoder `hidden` wide, in the order `load` reads them.
pub fn tensors(spec: &HeadSpec, hidden: usize) -> Vec<(String, Vec<u64>)> {
    let labels = spec.labels.len() as u64;
    let (mut out, width) = words::tensors(spec, hidden);
    let (max, dim) = spec.span.map_or((0, 0), |s| (s.max_words as u64, s.width_dim as u64));
    out.push(("span.width".into(), vec![max, dim]));
    out.push(("out.weight".into(), vec![labels, 2 * width + dim]));
    out.push(("out.bias".into(), vec![labels]));
    out
}

/// One kept span: words `first..=last` of its sequence, its label, and
/// its label's score less the score of `O`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub first: u32,
    pub last: u32,
    pub label: u32,
    pub margin: f32,
}

/// A loaded span classifier. Read-only after loading, so one head serves
/// many threads, each with its own [`Work`] and [`SpanWork`].
#[derive(Debug)]
pub struct SpanClassifier {
    spec: HeadSpec,
    hidden: usize,
    reader: WordReader,
    /// The start and end parts of `out.weight`, without bias.
    start: Linear,
    end: Linear,
    /// Each width's part of the score and the bias, [max_words, labels].
    widths: Vec<f32>,
    max_words: usize,
    none: usize,
}

/// Each word's start and end scores, kept across calls.
#[derive(Default)]
pub struct SpanWork {
    starts: Vec<f32>,
    ends: Vec<f32>,
}

/// Columns `from..from + n` of the rows of `w`, each `cols` wide.
pub(super) fn columns(w: &[f32], cols: usize, from: usize, n: usize) -> Vec<f32> {
    w.chunks_exact(cols).flat_map(|row| row[from..from + n].iter().copied()).collect()
}

impl SpanClassifier {
    /// The head named `name` in `bundle`, its tensors checked for shape
    /// and dtype and read from its artifact.
    pub fn load(bundle: &Bundle, name: &str) -> Result<SpanClassifier> {
        let l = load_tensors(bundle, name, HeadKind::SpanClassifier, tensors)?;
        Self::from_tensors(&l.spec, l.hidden, &|s| l.get(s))
    }

    /// A head for `spec` over an encoder `hidden` wide, from `get`, which
    /// gives each tensor [`tensors`] lists by its suffix, row-major.
    pub fn from_tensors(spec: &HeadSpec, hidden: usize, get: &dyn Fn(&str) -> Option<Vec<f32>>) -> Result<Self> {
        if spec.kind != HeadKind::SpanClassifier {
            return Err(invalid(format!("head {:?} is not a span classifier", spec.name)));
        }
        spec.validate("head")?;
        let shape = spec.span.ok_or_else(|| invalid(format!("head {:?}: no span", spec.name)))?;
        if hidden == 0 {
            return Err(invalid(format!("head {:?}: hidden 0", spec.name)));
        }
        let t = Take { name: &spec.name, sizes: words::sizes(spec, &tensors(spec, hidden))?, get };
        let reader = WordReader::new(spec, hidden, &t)?;
        let (labels, width) = (spec.labels.len(), reader.width());
        let (max_words, dim) = (shape.max_words as usize, shape.width_dim as usize);
        let emb = t.take("span.width")?;
        let w = t.take("out.weight")?;
        let bias = t.take("out.bias")?;
        if !emb.iter().chain(&w).chain(&bias).all(|v| v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, format!("head {:?}: a weight is not finite", spec.name)));
        }
        let cols = 2 * width + dim;
        let zeros = vec![0.0; labels];
        let start = Linear::new(width, labels, columns(&w, cols, 0, width), zeros.clone())?;
        let end = Linear::new(width, labels, columns(&w, cols, width, width), zeros)?;
        let by_width = Linear::new(dim, labels, columns(&w, cols, 2 * width, dim), bias)?;
        let mut widths = vec![0.0; max_words * labels];
        by_width.apply_rows(&emb, max_words, &mut widths);
        let none = spec.labels.iter().position(|l| l == "O").ok_or_else(|| invalid("no label \"O\""))?;
        Ok(SpanClassifier { spec: spec.clone(), hidden, reader, start, end, widths, max_words, none })
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

    /// The number of spans a sequence of `n` words has.
    pub fn span_count(&self, n: usize) -> usize {
        (1..=self.max_words.min(n)).fold(0usize, |t, width| t.saturating_add(n - width + 1))
    }

    /// The label scores [spans, labels] of one sequence into `out`, spans
    /// in order of first word then last, from its subword states and word
    /// map as a token tagger reads them (`TokenTagger::scores_forms`).
    /// Returns the number of words.
    pub fn scores(
        &self,
        states: &[f32],
        words: &[u32],
        forms: Option<&[&str]>,
        out: &mut Vec<f32>,
        w: &mut Work,
        sw: &mut SpanWork,
    ) -> Result<usize> {
        out.clear();
        let n = self.reader.read(states, words, forms, w)?;
        let labels = self.spec.labels.len();
        for (lin, dst) in [(&self.start, &mut sw.starts), (&self.end, &mut sw.ends)] {
            dst.clear();
            dst.resize(n * labels, 0.0);
            if n > 0 {
                lin.apply_rows(&w.pooled, n, dst);
            }
        }
        out.reserve(self.span_count(n) * labels);
        for i in 0..n {
            let s = &sw.starts[i * labels..][..labels];
            for j in i..n.min(i + self.max_words) {
                let (e, wd) = (&sw.ends[j * labels..][..labels], &self.widths[(j - i) * labels..][..labels]);
                out.extend((0..labels).map(|l| s[l] + e[l] + wd[l]));
            }
        }
        // A start or end projection past f32 makes a sum infinite or NaN.
        if out.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a span score is not finite"));
        }
        Ok(n)
    }

    /// The spans of a sequence of `n` words kept from their `scores`
    /// [spans, labels] under the head's decode, in order of first word
    /// then last.
    pub fn decode(&self, scores: &[f32], n: usize, out: &mut Vec<Span>) -> Result<()> {
        let labels = self.spec.labels.len();
        if n > u32::MAX as usize {
            return Err(Error::new(INVALID_SHAPE, format!("{n} words is past the word index limit")));
        }
        if Some(scores.len()) != self.span_count(n).checked_mul(labels) {
            return Err(Error::new(INVALID_SHAPE, format!("{} span scores for {n} words", scores.len())));
        }
        if scores.iter().any(|v| !v.is_finite()) {
            return Err(Error::new(INVALID_ARGUMENT, "a span score is not finite"));
        }
        out.clear();
        let mut rows = scores.chunks_exact(labels);
        for i in 0..n {
            for j in i..n.min(i + self.max_words) {
                let Some(row) = rows.next() else { break };
                let best = (0..labels).fold(0, |b, l| if row[l] > row[b] { l } else { b });
                if best != self.none {
                    // Two margins past f32 would tie at infinity.
                    let margin = row[best] - row[self.none];
                    if !margin.is_finite() {
                        return Err(Error::new(INVALID_ARGUMENT, "a span margin is past f32"));
                    }
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

/// Keeps the spans that do not overlap a span of a higher margin (on a
/// tie, the one earlier in order wins), in order of first word.
pub(super) fn flat(spans: &mut Vec<Span>) {
    let mut order: Vec<usize> = (0..spans.len()).collect();
    order.sort_by(|&a, &b| spans[b].margin.partial_cmp(&spans[a].margin).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<Span> = Vec::new();
    for i in order {
        let s = spans[i];
        if kept.iter().all(|k| s.last < k.first || s.first > k.last) {
            kept.push(s);
        }
    }
    kept.sort_by_key(|s| (s.first, s.last));
    *spans = kept;
}
