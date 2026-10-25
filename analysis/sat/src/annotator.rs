//! The SaT step: the encoder's final state for every token of every
//! window, the classifier's score on each, and the sentences the scores
//! cut.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, Result as CoreResult};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::turbo_token_range;
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Execution, ModelRef, Refs, Span, Spans, View};
use turbo_analysis_embed::{Device, Encoder, EncoderOptions, Precision, Rows};
use turbo_analysis_layers::Sentences;

use crate::cuts::{self, Window};
use crate::head::{BOUNDARY_LABEL, Head};

/// A text's tokens' spans, as byte offsets, and each token's stitched
/// score, in token order.
pub type Scored = (Vec<[u32; 2]>, Vec<f32>);

/// wtpsplit's stride in `split`.
const DEFAULT_STRIDE: u32 = 64;

/// How a `SatAnnotator` runs its model and reads its scores.
#[derive(Clone, Debug)]
pub struct SatOptions {
    pub encoder: EncoderOptions,
    /// A token ends a sentence when the logistic of its score is over
    /// this: wtpsplit's `threshold`. The default, 0.25, is wtpsplit's for
    /// the supervised mixture (`-sm`) models; it picks 0.025 or 0.01 for
    /// the others, which need limited lookahead (docs/neural-annotators.md).
    pub threshold: f64,
    /// Tokens from the start of one window to the start of the next:
    /// wtpsplit's `stride`. At most the block size; 0 is wtpsplit's 64
    /// in `split`, or the block size when that is smaller.
    pub stride: u32,
    /// Tokens in a window besides the template's special tokens:
    /// wtpsplit's `block_size`. 0 is all a row has room for (510 for a
    /// model of 512 tokens, as wtpsplit cuts its 512).
    pub block_size: u32,
    /// Also end a sentence at every line feed, which the model reads as a
    /// space: wtpsplit's `split_on_input_newlines`, on by default.
    pub split_on_newlines: bool,
    /// The classifier's label whose score is the boundary, 0 as in
    /// wtpsplit.
    pub label: u32,
}

impl Default for SatOptions {
    fn default() -> Self {
        SatOptions {
            encoder: EncoderOptions::default(),
            threshold: 0.25,
            stride: 0,
            block_size: 0,
            split_on_newlines: true,
            label: BOUNDARY_LABEL,
        }
    }
}

/// Writes `opennlp:sentences` from a SaT model (Segment any Text,
/// wtpsplit): a sentence boundary score for every token from one encoder
/// pass per window, and the sentences wtpsplit's `split` cuts from them.
///
/// - The view is tokenized whole with the bundle's tokenizer and read in
///   windows of `block_size` tokens every `stride` tokens, the last one
///   ending at the last token; each window is one row with the template's
///   special tokens around it. A token in several windows gets the mean
///   of its scores.
/// - A token's score is the classifier's label `label` on its final
///   state. Its probability, over `threshold`, puts a boundary at the end
///   of its span; the boundary moves past the whitespace after it.
/// - Each boundary then moves forward to the end of the grapheme cluster
///   it falls in, so no sentence ends inside a character a reader sees
///   as one; the sentences are trimmed of whitespace clusters at both
///   ends and empty ones are left out. With `split_on_newlines`, every
///   line feed also ends a sentence.
///
/// The windows of every document of a batch go through the same runs, up
/// to the session's rows per run. A document that cannot be tokenized
/// fails alone; a run that fails fails every document in it.
pub struct SatAnnotator {
    tokenizer: Tokenizer,
    head: Head,
    run: Mutex<Run>,
    model: ModelRef,
    execution: Execution,
    threshold: f64,
    stride: u32,
    block: u32,
    split_on_newlines: bool,
    max_batch: u32,
    /// The bare word start piece, U+2581 alone, when the vocabulary has one.
    word_start: Option<i32>,
    /// The template's special tokens before and after a window.
    row_head: Vec<i32>,
    row_tail: Vec<i32>,
    config: u64,
}

/// One run being built: its rows, a range of one token for every token of
/// every row, and whose token each range is.
struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    owners: Vec<(u32, u32)>,
    states: Vec<f32>,
}

/// One document's tokens and the scores their windows gave.
#[derive(Default)]
struct Doc {
    ids: Vec<i32>,
    spans: Vec<[u32; 2]>,
    sums: Vec<f64>,
    counts: Vec<u32>,
}

impl SatAnnotator {
    /// The bundle at `dir` on the device `o` names. The bundle must hold
    /// an encoder (a static model has no state per token) and declare a
    /// `CLASSIFIER_TOKEN` head in its manifest.
    pub fn open(dir: &Path, o: &SatOptions) -> CoreResult<SatAnnotator> {
        let bundle = Bundle::open(dir)?;
        let m = &bundle.manifest;
        let id = &m.model.id;
        if m.architecture.is_none() {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("{id}: a static model has no state per token")));
        }
        let embed =
            m.embed.as_ref().ok_or_else(|| CoreError::new(INVALID_ARGUMENT, format!("{id}: no embed block")))?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let max_batch = if o.encoder.max_batch == 0 { embed.max_batch } else { o.encoder.max_batch };
        let max_seq = if o.encoder.max_seq == 0 { embed.max_seq } else { o.encoder.max_seq };
        let (row_head, row_tail) = tokenizer.template_ends();
        let overhead = (row_head.len() + row_tail.len()) as u32;
        let room = max_seq.saturating_sub(overhead);
        let block = if o.block_size == 0 { room } else { o.block_size };
        if block == 0 || block > room {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("block_size {block}: a row of {max_seq} tokens has room for 1 to {room} beside {overhead}"),
            ));
        }
        let stride = if o.stride == 0 { DEFAULT_STRIDE.min(block) } else { o.stride };
        if stride > block {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("stride {stride}: 1 to the block size, {block}, so every token is read"),
            ));
        }
        if !(0.0..=1.0).contains(&o.threshold) {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("threshold {}: a probability, 0 to 1", o.threshold)));
        }
        let mut eo = o.encoder.clone();
        eo.max_batch = max_batch;
        eo.max_seq = max_seq;
        eo.max_ranges = max_batch.saturating_mul(max_seq);
        let encoder = Encoder::open(dir, &eo)?;
        let info = encoder.info().clone();
        let head = Head::load(&bundle, o.label, info.dim)?;
        let config = config_hash(o, block, stride);
        let word_start = (0..tokenizer.vocab_size() as i32).find(|&i| tokenizer.piece(i) == Some("\u{2581}"));
        Ok(SatAnnotator {
            run: Mutex::new(Run {
                rows: Rows::new(info.max_seq, tokenizer.fill_id()),
                ranges: Vec::new(),
                owners: Vec::new(),
                states: Vec::new(),
                encoder,
            }),
            tokenizer,
            head,
            model: crate::model_ref(&bundle),
            execution: info.execution,
            threshold: o.threshold + 0.0,
            stride,
            block,
            split_on_newlines: o.split_on_newlines,
            max_batch: info.max_batch,
            word_start,
            row_head,
            row_tail,
            config,
        })
    }

    /// The tokens a window holds besides the special tokens.
    pub fn block_size(&self) -> u32 {
        self.block
    }

    /// The tokens from the start of one window to the start of the next.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The stitched score of every token of each text, and each token's
    /// span, the prefix piece's the text's first character
    /// (`cuts::place_prefix`): what the sentences are cut from. A text
    /// of only whitespace has no tokens read, as wtpsplit reads none. One
    /// result per text.
    pub fn scores(&self, texts: &[&str]) -> Vec<Result<Scored, Error>> {
        let mut docs: Vec<Result<Doc, Error>> = texts.iter().map(|t| self.tokens(t)).collect();
        if let Err(e) = self.encode(&mut docs) {
            let e = Error::Annotator { annotator: Self::ID, detail: e.to_string() };
            for d in docs.iter_mut().filter(|d| d.is_ok()) {
                *d = Err(e.clone());
            }
        }
        docs.into_iter()
            .map(|d| {
                d.map(|d| {
                    let scores = d.sums.iter().zip(&d.counts).map(|(&s, &c)| (s / f64::from(c)) as f32).collect();
                    (d.spans, scores)
                })
            })
            .collect()
    }

    /// The sentences of `text` from its tokens' spans and scores.
    pub fn sentences(&self, text: &str, spans: &[[u32; 2]], scores: &[f32], out: &mut Vec<Span>) {
        let mut b = Vec::new();
        cuts::boundaries(spans, scores, self.threshold, &mut b);
        let mut c = Vec::with_capacity(b.len());
        cuts::cut_points(text, &b, &mut c);
        let graphemes = cuts::grapheme_index(text);
        cuts::snap(&graphemes, &mut c);
        let mut pieces = Vec::with_capacity(c.len() + 1);
        cuts::pieces(text, &c, self.split_on_newlines, &mut pieces);
        out.extend(pieces.into_iter().filter_map(|p| cuts::trim(text, &graphemes, p)));
    }

    /// The text's tokens, none for a text of whitespace alone.
    fn tokens(&self, text: &str) -> Result<Doc, Error> {
        if u32::try_from(text.len()).is_err() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("a text of {} bytes is past the u32 offset range", text.len()),
            });
        }
        let mut d = Doc::default();
        if text.chars().all(cuts::is_python_space) {
            return Ok(d);
        }
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        self.tokenizer
            .encode_spans(text, opts, &mut d.ids, &mut d.spans)
            .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        if let Some(w) = self.word_start {
            cuts::place_prefix(text, w, &d.ids, &mut d.spans);
        }
        d.sums = vec![0.0; d.ids.len()];
        d.counts = vec![0; d.ids.len()];
        Ok(d)
    }

    /// Every window of every document through the encoder, its tokens'
    /// scores added to their documents.
    fn encode(&self, docs: &mut [Result<Doc, Error>]) -> CoreResult<()> {
        let mut guard = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        let run = &mut *guard;
        run.rows.clear();
        run.ranges.clear();
        run.owners.clear();
        let lead = self.row_head.len() as u32;
        let mut windows: Vec<Window> = Vec::new();
        for d in 0..docs.len() {
            let Ok(doc) = &docs[d] else { continue };
            windows.clear();
            cuts::windows(doc.ids.len() as u32, self.block, self.stride, &mut windows);
            for w in &windows {
                if run.rows.len() == self.max_batch {
                    self.flush(run, docs)?;
                }
                let Ok(doc) = &docs[d] else { unreachable!("checked above") };
                let r = run.rows.push(&[&self.row_head, &doc.ids[w.start as usize..w.end as usize], &self.row_tail]);
                for t in w.start..w.end {
                    let at = lead + t - w.start;
                    run.ranges.push(turbo_token_range { row: r, start: at, end: at + 1 });
                    run.owners.push((d as u32, t));
                }
            }
        }
        if !run.rows.is_empty() {
            self.flush(run, docs)?;
        }
        Ok(())
    }

    /// Runs the rows built so far and adds each token's score to its
    /// document. Leaves the run empty.
    fn flush(&self, run: &mut Run, docs: &mut [Result<Doc, Error>]) -> CoreResult<()> {
        run.encoder.run_ranges(&run.rows, &run.ranges, &mut run.states)?;
        let dim = self.head.dim();
        for (state, &(d, t)) in run.states.chunks_exact(dim).zip(&run.owners) {
            if let Ok(doc) = &mut docs[d as usize] {
                doc.sums[t as usize] += f64::from(self.head.score(state));
                doc.counts[t as usize] += 1;
            }
        }
        run.rows.clear();
        run.ranges.clear();
        run.owners.clear();
        Ok(())
    }
}

impl Annotator for SatAnnotator {
    const ID: &'static str = "turbo:sat";
    type Reads = (View,);
    type Writes = (Sentences,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<(Sentences,), Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<(Sentences,), Error>> {
        let texts: Vec<&str> = inputs.iter().map(|(v,)| v.text()).collect();
        let scored = self.scores(&texts);
        let mut out = Vec::with_capacity(inputs.len());
        let mut found = Vec::new();
        for (((view,), text), s) in inputs.iter().zip(&texts).zip(scored) {
            out.push(s.map(|(spans, scores)| {
                found.clear();
                self.sentences(text, &spans, &scores, &mut found);
                let mut col = Spans::with_capacity(view.id(), found.len());
                for &f in &found {
                    col.push(f);
                }
                (Sentences { spans: col },)
            }));
        }
        out
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.config
    }
}

/// FNV-1a over everything besides the model and where it runs that
/// changes the sentences: the threshold's bits, the stride, the block
/// size, the line feed switch and the classifier's label.
fn config_hash(o: &SatOptions, block: u32, stride: u32) -> u64 {
    turbo_analysis::Fnv1a::new()
        .bytes(&(o.threshold + 0.0).to_bits().to_le_bytes())
        .u32(stride)
        .u32(block)
        .bool(o.split_on_newlines)
        .u32(o.label)
        .finish()
}

/// Offers `SatAnnotator` to pipelines built from configuration, as "sat".
/// The spec's location is the bundle directory; its options:
///
/// - `threshold`: a probability, 0.25 by default;
/// - `stride` (0, which is 64 or the block size when that is smaller)
///   and `block_size` (0, all a row has room for), in tokens;
/// - `newlines`: `split` (the default) to end a sentence at every line
///   feed, or `model` to leave them to the model;
/// - `device`: `cpu` (the default), `select`, a device index, or a backend
///   name such as `cuda`;
/// - `precision`: `model` (the default), `fastest` or `exact`.
pub struct SatProvider;

fn parse_options(spec: &Spec) -> Result<SatOptions, String> {
    fn number<T: std::str::FromStr>(spec: &Spec, key: &str, default: &str) -> Result<T, String> {
        let v = spec.option(key, default);
        v.parse().map_err(|_| format!("{key} {v:?} is not a number"))
    }
    let device = match spec.option("device", "cpu") {
        "cpu" => Device::Cpu,
        "select" => Device::Select,
        d => match d.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(d.to_owned()),
        },
    };
    let precision = match spec.option("precision", "model") {
        "model" => Precision::Model,
        "fastest" => Precision::Fastest,
        "exact" => Precision::Exact,
        p => return Err(format!("precision {p:?}: model, fastest or exact")),
    };
    let split_on_newlines = match spec.option("newlines", "split") {
        "split" => true,
        "model" => false,
        n => return Err(format!("newlines {n:?}: split or model")),
    };
    Ok(SatOptions {
        encoder: EncoderOptions { device, precision, ..EncoderOptions::default() },
        threshold: number(spec, "threshold", "0.25")?,
        stride: number(spec, "stride", "0")?,
        block_size: number(spec, "block_size", "0")?,
        split_on_newlines,
        label: BOUNDARY_LABEL,
    })
}

impl Provider<dyn Step> for SatProvider {
    fn name(&self) -> &'static str {
        "sat"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("sat: no bundle location".into()))?;
        let o = parse_options(spec).map_err(|e| Error::Pipeline(format!("sat: {e}")))?;
        let a = SatAnnotator::open(dir, &o).map_err(|e| Error::Pipeline(format!("sat: {}: {e}", dir.display())))?;
        Ok(Box::new(Bound::new(a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_hash_normalizes_negative_zero_threshold() {
        let pos = SatOptions { threshold: 0.0, ..SatOptions::default() };
        let neg = SatOptions { threshold: -0.0, ..SatOptions::default() };
        assert_eq!(config_hash(&pos, 510, 64), config_hash(&neg, 510, 64));
    }
}

#[cfg(test)]
mod hash_compat;
