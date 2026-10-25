//! A vector per sentence and per chunk, from one encoder pass over each
//! window of the text's subword pieces (late pooling).

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::ProjectionHead;
use turbo::manifest::{PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, INVALID_SHAPE, UNSUPPORTED_OPTION};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{TURBO_POOLING_MEAN, turbo_token_range};
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Execution, Ix, Layer, ModelRef, Refs, Span, Spans, ViewId};
use turbo_analysis_layers::{Chunks, Levels, Sentences, Subwords, VectorKind, Vectors};

use crate::encoder::{Device, Encoder, EncoderInfo, EncoderOptions, Precision, Rows, normalize};
use crate::window::{Placed, Plan, Target, Which, default_token_stride, pieces_of};

/// How a `VectorAnnotator` runs its model.
#[derive(Clone, Debug)]
pub struct VectorOptions {
    pub encoder: EncoderOptions,
    /// The prompt whose prefix every row carries, after the template's
    /// leading special tokens: the document role by default, as chunks
    /// and sentences are what a search finds. The chunker's budget is
    /// counted with the same role (`SubwordAnnotator::overhead`).
    pub prompt: PromptRole,
    /// Sliding-window stride (in subword pieces) when splitting long
    /// inputs for per-token contextual vectors (`#709`). `0` selects the
    /// default half-window stride `(width / 2).max(1)`.
    pub stride: u32,
}

impl Default for VectorOptions {
    fn default() -> Self {
        VectorOptions { encoder: EncoderOptions::default(), prompt: PromptRole::Document, stride: 0 }
    }
}

/// One document's layers for `VectorAnnotator::embed`: the subwords, and
/// the sentences or the chunks or both, in the subwords' view; or the
/// levels alone (see `crate::levels`).
pub struct Input<'d> {
    pub subwords: &'d Subwords,
    pub sentences: Option<&'d Spans<Sentences>>,
    pub chunks: Option<&'d Spans<Chunks>>,
    pub levels: Option<&'d Levels>,
}

/// One document's vectors: a layer for each kind of item it was given.
pub struct Output {
    pub sentences: Option<Vectors<Sentences>>,
    pub chunks: Option<Vectors<Chunks>>,
    /// For levels: each unit's local vector, then its contextual one.
    pub levels: Option<[Vectors<Levels>; 2]>,
}

/// Embeds windows of a text's subword pieces once each and pools a vector
/// per sentence and per chunk over the encoder's final states, through
/// the core's late pooling (`turbo_embed_write_ranges`). docs/analysis.md
/// and the `window` module give the window policy; in short:
///
/// - A row is the template's leading special tokens, the prompt's
///   prefix, a window of pieces and the trailing special tokens, so a
///   window holds `max_seq` less that overhead: the overhead the
///   chunker's budget counts.
/// - An item is pooled over the pieces its span overlaps; the special
///   tokens and the prefix are never pooled, so a vector is not the one
///   the text interface gives for the same span.
/// - The sentences lay out the windows when the step has them, else the
///   chunks. Every other item is pooled from a window that holds it
///   whole, or in parts over several, its vector their mean weighted by
///   pieces; a window is added only for pieces no window holds.
/// - An item with no pieces has no token to pool: its row is the zero
///   vector and the layer lists it in `Vectors::empty`.
///
/// Vectors are normalized as the bundle says, on the host, with the
/// arithmetic the core's CPU backend uses. The windows of every document
/// in a batch go through the same runs, up to the session's rows and
/// ranges per run.
pub struct VectorAnnotator {
    state: Mutex<State>,
    tokenizer: Tokenizer,
    projection: Option<ProjectionHead>,
    model: ModelRef,
    execution: Execution,
    dim: u32,
    l2: bool,
    /// The template's leading special tokens, then the prompt's prefix.
    head: Vec<i32>,
    tail: Vec<i32>,
    max_batch: u32,
    max_ranges: u32,
    stride: u32,
    /// The settings every vector depends on besides the model and where
    /// it runs: the row length and the prompt, as `config_hash` takes
    /// them, before the layer that lays out the windows.
    settings: [u32; 2],
}

/// What one batch works in, kept between batches.
struct State {
    plan: Plan,
    run: Run,
}

/// The run being built: its rows and ranges, and the session it goes to.
struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    /// For each range, its placement in the plan.
    targets: Vec<u32>,
    /// The run's vectors, `[ranges, dim]`.
    flat: Vec<f32>,
}

impl Run {
    /// Runs the rows and ranges built so far and writes each range's vector
    /// to its item's row: copied for a whole item, added times its pieces
    /// for a part. Leaves the run empty.
    fn flush(&mut self, dim: usize, placed: &[Placed], out: &mut [Result<Output, Error>]) -> Result<(), CoreError> {
        self.encoder.run_ranges(&self.rows, &self.ranges, &mut self.flat)?;
        for (v, &t) in self.flat.chunks_exact(dim).zip(&self.targets) {
            let x = placed[t as usize];
            let Ok(o) = &mut out[x.target.doc as usize] else { continue };
            let dst = row_of(o, x.target, dim);
            if x.whole {
                dst.copy_from_slice(v);
            } else {
                let n = (x.end - x.start) as f32;
                for (d, &s) in dst.iter_mut().zip(v) {
                    *d += s * n;
                }
            }
        }
        self.rows.clear();
        self.ranges.clear();
        self.targets.clear();
        Ok(())
    }
}

impl VectorAnnotator {
    /// The bundle at `dir`, loaded on the device `o` names. A bundle
    /// pooled another way than by the mean is refused: late pooling is the
    /// mean, and a CLS model's mean vectors are not what it was trained to
    /// give.
    pub fn open(dir: &Path, o: &VectorOptions) -> Result<VectorAnnotator, CoreError> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let projection = ProjectionHead::load_optional(&bundle)?;
        let encoder = Encoder::open(dir, &o.encoder)?;
        let info = encoder.info().clone();
        if info.pooling != TURBO_POOLING_MEAN {
            return Err(CoreError::new(
                UNSUPPORTED_OPTION,
                format!(
                    "{}: the bundle pools by {}, and sentence and chunk vectors are pooled by the mean",
                    bundle.manifest.model.id,
                    pooling_name(info.pooling)
                ),
            ));
        }
        let (mut head, tail) = tokenizer.template_ends();
        let bare =
            Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: o.prompt };
        tokenizer.encode_into("", bare, &mut head)?;
        let overhead = (head.len() + tail.len()) as u32;
        if info.max_seq <= overhead {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("max_seq {} leaves no room for text beside {overhead} tokens of overhead", info.max_seq),
            ));
        }
        let width = info.max_seq - overhead;
        if o.stride > width {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("stride {} exceeds window width {width}", o.stride)));
        }
        let stride = if o.stride == 0 { default_token_stride(width) } else { o.stride };
        let fill_id = tokenizer.fill_id();
        let state = State {
            plan: Plan::new(width),
            run: Run {
                rows: Rows::new(info.max_seq, fill_id),
                ranges: Vec::new(),
                targets: Vec::new(),
                flat: Vec::new(),
                encoder,
            },
        };
        let prompt = match o.prompt {
            PromptRole::None => 0,
            PromptRole::Query => 1,
            PromptRole::Document => 2,
        };
        Ok(VectorAnnotator {
            state: Mutex::new(state),
            tokenizer,
            projection,
            model: crate::model_ref(&bundle),
            execution: info.execution,
            dim: info.dim,
            l2: info.l2,
            head,
            tail,
            max_batch: info.max_batch,
            max_ranges: info.max_ranges,
            stride,
            settings: [info.max_seq, prompt],
        })
    }

    /// Attach or override the linear projection head applied to per-token
    /// contextual vectors (`#709`).
    pub fn with_projection(mut self, projection: ProjectionHead) -> Result<Self, CoreError> {
        projection.validate()?;
        if projection.hidden != self.dim as usize {
            return Err(CoreError::new(
                INVALID_SHAPE,
                format!("projection input width {} does not match encoder width {}", projection.hidden, self.dim),
            ));
        }
        self.projection = Some(projection);
        Ok(self)
    }

    /// Override the sliding-window stride (in subword pieces) used when
    /// splitting long inputs for per-token contextual vectors (`0` uses the
    /// default half-window stride).
    pub fn with_stride(mut self, stride: u32) -> Result<Self, CoreError> {
        let width = self.width();
        if stride > width {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("stride {stride} exceeds window width {width}")));
        }
        self.stride = if stride == 0 { default_token_stride(width) } else { stride };
        Ok(self)
    }

    /// The configuration hash of the vectors a step writes when `layout`
    /// lays out its windows: the sentences for a step that reads them,
    /// else the chunks. A chunk's vector depends on which.
    pub fn config(&self, layout: Which) -> u64 {
        let layout = match layout {
            Which::Sentences => 0,
            Which::Chunks => 1,
            Which::Levels(_) => 2,
        };
        config_hash(&[self.settings[0], self.settings[1], layout])
    }

    /// The configuration hash of per-token contextual vectors (`SubwordVectors`).
    pub fn token_config(&self) -> u64 {
        let seed = config_hash(&[self.settings[0], self.settings[1], self.stride, self.token_dim()]);
        let mut h = turbo_analysis::Fnv1a::with_seed(seed);
        if let Some(proj) = &self.projection {
            h.bool(proj.has_bias);
            proj.linear.weight().iter().chain(proj.linear.bias()).for_each(|&v| {
                h.f32(v);
            });
        }
        h.finish()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn projection(&self) -> Option<&ProjectionHead> {
        self.projection.as_ref()
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    pub fn dim(&self) -> u32 {
        self.dim
    }

    /// Output width of per-token contextual vectors (`projection.dim` when
    /// a projection head is present, else `self.dim()`).
    pub fn token_dim(&self) -> u32 {
        self.projection.as_ref().map_or(self.dim, |p| p.dim as u32)
    }

    /// Sliding-window stride (in subword pieces) for per-token contextual
    /// vectors.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The session's settings as it resolved them.
    pub fn info(&self) -> EncoderInfo {
        self.lock().run.encoder.info().clone()
    }

    /// The most pieces one window holds.
    pub fn width(&self) -> u32 {
        self.lock().plan.width()
    }

    /// What every row holds besides its text: the template's
    /// special tokens and this annotator's prompt prefix, in
    /// tokens. A `Chunks` layer whose `prompt_overhead` records a
    /// different nonzero count was budgeted for another prompt.
    pub fn overhead(&self) -> u32 {
        (self.head.len() + self.tail.len()) as u32
    }

    /// What every row holds besides its window's pieces: the template's
    /// special tokens and the prompt's prefix, before and after.
    pub fn row_ends(&self) -> (&[i32], &[i32]) {
        (&self.head, &self.tail)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic in a run leaves nothing half changed that the next
        // batch reads: every buffer is cleared before it is used.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The vector of item `t` in a document's output.
fn row_of(o: &mut Output, t: Target, dim: usize) -> &mut [f32] {
    let data = match t.which {
        Which::Sentences => &mut o.sentences.as_mut().expect("sentences were given").data,
        Which::Chunks => &mut o.chunks.as_mut().expect("chunks were given").data,
        Which::Levels(kind) => &mut crate::levels::of_kind(o.levels.as_mut().expect("levels were given"), kind).data,
    };
    &mut data[t.item as usize * dim..(t.item as usize + 1) * dim]
}

fn pooling_name(pooling: u32) -> &'static str {
    match pooling {
        turbo::TURBO_POOLING_CLS => "its first token",
        turbo::TURBO_POOLING_LAST => "its last token",
        _ => "another pooling",
    }
}

/// FNV-1a over the settings that change a vector besides the model and
/// where it runs: the row length, the prompt and the layer that lays out
/// the windows.
fn config_hash(values: &[u32]) -> u64 {
    let mut h = turbo_analysis::Fnv1a::new();
    values.iter().for_each(|&v| {
        h.u32(v);
    });
    h.finish()
}

mod execute;
mod steps;
mod tokens;
pub use steps::{ChunkVectors, SentenceChunkVectors, SentenceVectors, SubwordVectors, VectorsProvider};
pub use tokens::ContextualTokens;
mod cached;
pub(crate) use cached::CachedWindows;
#[cfg(test)]
mod cache_tests;
