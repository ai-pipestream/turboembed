//! The BiLSTM forward pass on the CPU backend's packed kernels
//! (`turbo::cpu::Matrix`) in f32, the tagger's default numerics.
//!
//! Every linear map is one packed product over all the rows it applies
//! to: the character LSTMs' input projections are a table made once per
//! model (one row per character, both directions), the sentence LSTMs'
//! input projections are one product over every token of every sentence
//! of the call (both directions), the pretrained adapter is one product
//! over the words that have a vector, and the output layer is one product
//! over every token. Each BiLSTM is one `turbo::device::lstm::Lstm` over
//! both directions, on the CPU or a GPU, which runs every sequence of the
//! call (sorted longest first) in one recurrent product per step; the
//! character one reads its input projections gathered from the table,
//! which lives where the LSTM runs. Word representations are
//! cached per token string, as the reference path caches its own.
//!
//! The JVM order of operations is not kept (that is the reference path,
//! [`BilstmPosModel::score`]); a row's result does not depend on the
//! other rows of the call, so a sentence gets the same scores alone or in
//! a document.

mod lstm;

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, RwLock};

use turbo::cpu::{Matrix, Mode};
use turbo::device::Device;
use turbo::device::lstm::{self as device_lstm, Lstm, Table};

use self::lstm::{allocation, bidirectional, pack, run_error, to_f32};
use super::BilstmPosModel;
use crate::neural::NeuralError;

const CACHE_LIMIT: usize = 100_000;

/// A [`BilstmPosModel`]'s weights packed for the CPU backend.
pub(crate) struct Packed {
    /// `[chars, 8 hc]`: `E W^T + b` of the forward then the backward
    /// character LSTM for every character row.
    char_table: Table,
    /// The character BiLSTM, run on projections gathered from the table.
    chars: Lstm,
    word_dim: usize,
    pretrained_dim: usize,
    adapter: Option<(Matrix, Vec<f32>)>,
    /// The sentence BiLSTMs, input projections included.
    layers: Vec<Lstm>,
    out: Matrix,
    out_bias: Vec<f32>,
    tags: usize,
    cache: RwLock<HashMap<String, Box<[f32]>>>,
    /// The LSTM workspaces of finished calls, kept so that a call on a GPU
    /// does not make its stream and buffers again.
    spare: Mutex<Vec<device_lstm::Work>>,
}

impl fmt::Debug for Packed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Packed").field("layers", &self.layers.len()).field("tags", &self.tags).finish_non_exhaustive()
    }
}

/// The buffers of one call.
#[derive(Default)]
struct Work {
    x: Vec<f32>,
    states: Vec<f32>,
    steps: Steps,
}

/// The buffers of the recurrences: the character ids in the order the
/// LSTM reads them, the character states, and the LSTM's own.
#[derive(Default)]
struct Steps {
    index: Vec<usize>,
    out: Vec<f32>,
    lstm: device_lstm::Work,
}

/// Indices of `lens` sorted longest first, and the sorted lengths.
fn longest_first(lens: impl Iterator<Item = usize>) -> (Vec<usize>, Vec<usize>) {
    let lens: Vec<usize> = lens.collect();
    let mut order: Vec<usize> = (0..lens.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(lens[i]));
    let sorted = order.iter().map(|&i| lens[i]).collect();
    (order, sorted)
}

fn size(v: &mut Vec<f32>, len: usize) -> &mut [f32] {
    v.clear();
    v.resize(len, 0.0);
    v
}

fn finite(v: &[f32], what: &str) -> Result<(), NeuralError> {
    match v.iter().find(|x| !x.is_finite()) {
        Some(x) => Err(NeuralError::NonFiniteScore(format!("Non-finite {what}: {x}"))),
        None => Ok(()),
    }
}

impl Packed {
    /// `m`'s weights with its BiLSTMs on `device`.
    pub(crate) fn new(m: &BilstmPosModel, device: Device) -> Result<Self, NeuralError> {
        let (cf, cb) = (&m.char_forward, &m.char_backward);
        let hc = cf.hidden_size();
        let char_dim = cf.input_size();
        let char_input = pack(&[cf.w(), cb.w()], char_dim)?;
        let char_bias: Vec<f32> = cf.b().iter().chain(cb.b()).map(|&v| v as f32).collect();
        let embeddings: Vec<f32> = m.char_embeddings.iter().flatten().map(|&v| v as f32).collect();
        let rows = m.char_embeddings.len();
        let mut char_table = vec![0.0f32; rows * 8 * hc];
        char_input.apply(&embeddings, char_dim, rows, Some(&char_bias), &mut char_table, 8 * hc, Mode::Plain);
        finite(&char_table, "character projection")?;
        let chars = bidirectional(device, cf, cb)?;
        let char_table = chars.table(&char_table).map_err(|e| NeuralError::InvalidArgument(e.message))?;

        let pretrained_dim = m.pretrained_vectors.as_ref().and_then(|v| v.first()).map_or(0, Vec::len);
        let adapter = match (&m.adapter_weights, &m.adapter_bias) {
            (Some(aw), Some(ab)) => {
                // rep[i] = b[i] + sum over r of aw[r][i] p[r]: W[i][r] = aw[r][i].
                let flat: Vec<f32> = aw.iter().flatten().map(|&v| v as f32).collect();
                let mut w = Matrix::empty();
                w.pack_cols(&flat, pretrained_dim, pretrained_dim, pretrained_dim).map_err(allocation)?;
                Some((w, to_f32(ab)))
            }
            _ => None,
        };

        let mut layers = vec![bidirectional(device, &m.word_forward, &m.word_backward)?];
        if let (Some(f), Some(b)) = (&m.word_forward2, &m.word_backward2) {
            layers.push(bidirectional(device, f, b)?);
        }
        let last = layers.last().map_or(0, Lstm::width);
        Ok(Self {
            char_table,
            chars,
            word_dim: m.word_embeddings[0].len(),
            pretrained_dim,
            adapter,
            layers,
            out: pack(&[&m.output_weights], last)?,
            out_bias: to_f32(&m.output_bias),
            tags: m.tags.len(),
            cache: RwLock::new(HashMap::new()),
            spare: Mutex::new(Vec::new()),
        })
    }

    /// Where the BiLSTMs run.
    pub(crate) fn device(&self) -> Device {
        self.chars.device()
    }

    fn rep_dim(&self) -> usize {
        self.word_dim + self.chars.width() + self.pretrained_dim
    }

    /// The word representations of `tokens` into `out` `[tokens, rep_dim]`.
    fn represent(
        &self,
        m: &BilstmPosModel,
        tokens: &[&str],
        out: &mut [f32],
        steps: &mut Steps,
    ) -> Result<(), NeuralError> {
        let (rd, wd, hc) = (self.rep_dim(), self.word_dim, self.chars.hidden());
        for (row, tok) in out.chunks_exact_mut(rd).zip(tokens) {
            row.fill(0.0);
            for (o, &v) in row[..wd].iter_mut().zip(&m.word_embeddings[m.word_id(tok)]) {
                *o = v as f32;
            }
        }
        // As in Java, a token without characters keeps zero character
        // states; sorted longest first, those come last and are left out.
        let ids: Vec<Vec<usize>> = tokens.iter().map(|t| m.char_ids(t)).collect();
        let (order, mut lens) = longest_first(ids.iter().map(Vec::len));
        lens.truncate(lens.partition_point(|&l| l > 0));
        steps.index.clear();
        steps.index.extend(order.iter().flat_map(|&i| &ids[i]));
        let states = size(&mut steps.out, steps.index.len() * 2 * hc);
        self.chars.run_gathered(&self.char_table, &steps.index, &lens, states, &mut steps.lstm).map_err(run_error)?;
        let mut at = 0;
        for (&i, &len) in order.iter().zip(&lens) {
            // The forward state after the last character; the backward one
            // after reading back to the first.
            let row = &mut out[i * rd + wd..][..2 * hc];
            row[..hc].copy_from_slice(&states[(at + len - 1) * 2 * hc..][..hc]);
            row[hc..].copy_from_slice(&states[at * 2 * hc + hc..][..hc]);
            at += len;
        }
        if self.pretrained_dim == 0 {
            return Ok(());
        }
        let (pd, offset) = (self.pretrained_dim, wd + 2 * hc);
        let vecs = m.pretrained_vectors.as_deref().unwrap_or_default();
        let with: Vec<(usize, &Vec<f32>)> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, t)| m.pretrained_ids.get(&BilstmPosModel::normalize(t)).map(|&r| (i, &vecs[r])))
            .collect();
        match &self.adapter {
            Some((w, b)) => {
                let x: Vec<f32> = with.iter().flat_map(|(_, v)| v.iter().copied()).collect();
                let mut y = vec![0.0f32; with.len() * pd];
                w.apply(&x, pd, with.len(), Some(b), &mut y, pd, Mode::Plain);
                for ((i, _), r) in with.iter().zip(y.chunks_exact(pd)) {
                    out[i * rd + offset..][..pd].copy_from_slice(r);
                }
            }
            None => {
                for (i, v) in with {
                    out[i * rd + offset..][..pd].copy_from_slice(v);
                }
            }
        }
        Ok(())
    }

    /// The representations of `tokens` into `x` `[tokens, rep_dim]`, from
    /// the cache or computed in one batch for the distinct tokens missing.
    fn representations(
        &self,
        m: &BilstmPosModel,
        tokens: &[&str],
        x: &mut Vec<f32>,
        steps: &mut Steps,
    ) -> Result<(), NeuralError> {
        let rd = self.rep_dim();
        let x = size(x, tokens.len() * rd);
        let mut missing: Vec<&str> = Vec::new();
        let mut slot: HashMap<&str, usize> = HashMap::new();
        {
            let cache = self.cache.read().unwrap_or_else(|e| e.into_inner());
            for (row, &tok) in x.chunks_exact_mut(rd).zip(tokens) {
                if let Some(rep) = cache.get(tok) {
                    row.copy_from_slice(rep);
                } else if !slot.contains_key(tok) {
                    slot.insert(tok, missing.len());
                    missing.push(tok);
                }
            }
        }
        if missing.is_empty() {
            return Ok(());
        }
        let mut reps = vec![0.0f32; missing.len() * rd];
        self.represent(m, &missing, &mut reps, steps)?;
        finite(&reps, "word representation")?;
        for (row, tok) in x.chunks_exact_mut(rd).zip(tokens) {
            if let Some(&i) = slot.get(tok) {
                row.copy_from_slice(&reps[i * rd..][..rd]);
            }
        }
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        for (tok, rep) in missing.iter().zip(reps.chunks_exact(rd)) {
            if cache.len() >= CACHE_LIMIT {
                break;
            }
            cache.entry((*tok).to_owned()).or_insert_with(|| rep.into());
        }
        Ok(())
    }

    /// Raw tag scores `[tokens, tags]` of sentences `lens` long whose
    /// tokens are `tokens`, one after another. Every length is positive.
    pub(crate) fn score(&self, m: &BilstmPosModel, tokens: &[&str], lens: &[usize]) -> Result<Vec<f32>, NeuralError> {
        debug_assert_eq!(lens.iter().sum::<usize>(), tokens.len());
        let mut w = Work::default();
        w.steps.lstm = self.spare.lock().unwrap_or_else(|e| e.into_inner()).pop().unwrap_or_default();
        let scores = self.score_in(m, tokens, lens, &mut w);
        self.spare.lock().unwrap_or_else(|e| e.into_inner()).push(std::mem::take(&mut w.steps.lstm));
        scores
    }

    fn score_in(
        &self,
        m: &BilstmPosModel,
        tokens: &[&str],
        lens: &[usize],
        w: &mut Work,
    ) -> Result<Vec<f32>, NeuralError> {
        let n = tokens.len();
        let mut offsets = Vec::with_capacity(lens.len());
        let mut at = 0;
        for &len in lens {
            offsets.push(at);
            at += len;
        }
        // Every row in sentence order longest first, as the LSTM reads them.
        let (order, sorted) = longest_first(lens.iter().copied());
        let in_order: Vec<&str> =
            order.iter().flat_map(|&i| &tokens[offsets[i]..offsets[i] + lens[i]]).copied().collect();
        self.representations(m, &in_order, &mut w.x, &mut w.steps)?;
        for layer in &self.layers {
            let states = size(&mut w.states, n * layer.width());
            layer.run(&w.x, &sorted, states, &mut w.steps.lstm).map_err(run_error)?;
            std::mem::swap(&mut w.x, &mut w.states);
        }
        let (k, tags) = (self.layers.last().map_or(0, Lstm::width), self.tags);
        let in_order = size(&mut w.states, n * tags);
        self.out.apply(&w.x, k, n, Some(&self.out_bias), in_order, tags, Mode::Plain);
        finite(in_order, "tag score")?;
        let mut scores = vec![0.0f32; n * tags];
        let mut at = 0;
        for (&i, &len) in order.iter().zip(&sorted) {
            scores[offsets[i] * tags..][..len * tags].copy_from_slice(&in_order[at * tags..][..len * tags]);
            at += len;
        }
        Ok(scores)
    }
}
