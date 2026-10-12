//! A vector per sentence and per chunk, from one encoder pass over each
//! window of the text's subword pieces (late pooling).

use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, UNSUPPORTED_OPTION};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{TURBO_POOLING_MEAN, turbo_token_range};
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Execution, ModelRef, Refs, Spans};
use turbo_analysis_layers::{Chunks, Sentences, Subwords, Vectors};

use crate::encoder::{Device, Encoder, EncoderInfo, EncoderOptions, Precision, Rows, normalize};
use crate::window::{Placed, Plan, Target, Which, pieces_of};

/// How a `VectorAnnotator` runs its model.
#[derive(Clone, Debug)]
pub struct VectorOptions {
    pub encoder: EncoderOptions,
    /// The prompt whose prefix every row carries, after the template's
    /// leading special tokens: the document role by default, as chunks
    /// and sentences are what a search finds. The chunker's budget is
    /// counted with the same role (`SubwordAnnotator::overhead`).
    pub prompt: PromptRole,
}

impl Default for VectorOptions {
    fn default() -> Self {
        VectorOptions { encoder: EncoderOptions::default(), prompt: PromptRole::Document }
    }
}

/// One document's layers for `VectorAnnotator::embed`: the subwords, and
/// the sentences or the chunks or both, in the subwords' view.
pub struct Input<'d> {
    pub subwords: &'d Subwords,
    pub sentences: Option<&'d Spans<Sentences>>,
    pub chunks: Option<&'d Spans<Chunks>>,
}

/// One document's vectors: a layer for each kind of item it was given.
pub struct Output {
    pub sentences: Option<Vectors<Sentences>>,
    pub chunks: Option<Vectors<Chunks>>,
}

/// Embeds windows of a text's subword pieces once each and pools a vector
/// per sentence and per chunk over the encoder's final states, through
/// the core's late pooling (`turbo_embed_write_ranges`). docs/analysis.md
/// gives the window policy; in short:
///
/// - A row is the template's leading special tokens, the prompt's
///   prefix, a window of pieces and the trailing special tokens, so a
///   window holds `max_seq` less that overhead: the overhead the
///   chunker's budget counts.
/// - An item is pooled over the pieces its span overlaps, in a window
///   that holds them all; the special tokens and the prefix are never
///   pooled.
/// - An item longer than a window is pooled in parts, each from a window
///   that holds it,
///   and its vector is their mean weighted by pieces: the mean over all
///   its pieces, each encoded in the context of its own part's window.
/// - An item with no pieces has no token to pool and gets the zero
///   vector, which no pooled unit vector is.
///
/// Vectors are normalized as the bundle says, on the host, with the
/// arithmetic the core's CPU backend uses. The windows of every document
/// in a batch go through the same runs, up to the session's rows and
/// ranges per run.
pub struct VectorAnnotator {
    state: Mutex<State>,
    model: ModelRef,
    execution: Execution,
    dim: u32,
    l2: bool,
    /// The template's leading special tokens, then the prompt's prefix.
    head: Vec<i32>,
    tail: Vec<i32>,
    max_batch: u32,
    max_ranges: u32,
    config: u64,
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
        let state = State {
            plan: Plan::new(width),
            run: Run {
                rows: Rows::new(info.max_seq, tokenizer.fill_id()),
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
            model: crate::model_ref(&bundle),
            execution: info.execution,
            dim: info.dim,
            l2: info.l2,
            head,
            tail,
            max_batch: info.max_batch,
            max_ranges: info.max_ranges,
            config: config_hash(&[info.max_seq, prompt]),
        })
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

    /// The session's settings as it resolved them.
    pub fn info(&self) -> EncoderInfo {
        self.lock().run.encoder.info().clone()
    }

    /// The most pieces one window holds.
    pub fn width(&self) -> u32 {
        self.lock().plan.width()
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

    /// The vectors of a batch of documents, a result for each. A document
    /// whose layers do not fit together fails alone; a run that fails
    /// fails every document of the batch. `annotator` names the step in
    /// errors.
    pub fn embed(&self, annotator: &'static str, inputs: &[Input<'_>]) -> Vec<Result<Output, Error>> {
        let mut out: Vec<Result<Output, Error>> = inputs.iter().map(|i| self.output_for(annotator, i)).collect();
        let mut guard = self.lock();
        let st = &mut *guard;
        st.plan.clear();
        for (d, (input, o)) in inputs.iter().zip(&out).enumerate() {
            if o.is_err() {
                continue;
            }
            let (starts, ends) = input.subwords.spans.columns();
            let d = d as u32;
            if let Some(s) = input.sentences {
                for (i, span) in s.iter() {
                    let t = Target { doc: d, which: Which::Sentences, item: i.get() as u32 };
                    st.plan.item(t, pieces_of(starts, ends, span.start, span.end));
                }
            }
            if let Some(c) = input.chunks {
                for (i, span) in c.iter() {
                    let t = Target { doc: d, which: Which::Chunks, item: i.get() as u32 };
                    st.plan.item(t, pieces_of(starts, ends, span.start, span.end));
                }
            }
            st.plan.place(d, starts.len() as u32);
        }
        if let Err(e) = self.run(st, inputs, &mut out) {
            let e = Error::Annotator { annotator, detail: e.to_string() };
            for o in out.iter_mut().filter(|o| o.is_ok()) {
                *o = Err(e.clone());
            }
            return out;
        }
        let dim = self.dim as usize;
        for s in &st.plan.split {
            if let Ok(o) = &mut out[s.target.doc as usize] {
                let inv = 1.0 / s.pieces as f32;
                for x in row_of(o, s.target, dim) {
                    *x *= inv;
                }
            }
        }
        drop(guard);
        if self.l2 {
            for o in out.iter_mut().flatten() {
                let rows = [o.sentences.as_mut().map(|v| &mut v.data), o.chunks.as_mut().map(|v| &mut v.data)];
                for data in rows.into_iter().flatten() {
                    for v in data.chunks_exact_mut(dim) {
                        normalize(v);
                    }
                }
            }
        }
        out
    }

    /// Zero vectors for every item of `i`, or why its layers do not fit
    /// together.
    fn output_for(&self, annotator: &'static str, i: &Input<'_>) -> Result<Output, Error> {
        let fail = |detail: String| Err(Error::Annotator { annotator, detail });
        if i.subwords.model != self.model {
            return fail(format!(
                "the subwords are {}'s pieces (revision {}), the vectors {}'s (revision {})",
                i.subwords.model.id, i.subwords.model.revision, self.model.id, self.model.revision
            ));
        }
        let view = i.subwords.spans.view();
        let views = [i.sentences.map(|s| ("sentences", s.view())), i.chunks.map(|c| ("chunks", c.view()))];
        if let Some((what, v)) = views.into_iter().flatten().find(|(_, v)| *v != view) {
            return fail(format!("the {what} are in view {}, the subwords in view {}", v.0, view.0));
        }
        let zeros = |n: usize| vec![0.0f32; n * self.dim as usize];
        Ok(Output {
            sentences: i.sentences.map(|s| Vectors::new(self.dim, zeros(s.len()), self.model.clone())),
            chunks: i.chunks.map(|c| Vectors::new(self.dim, zeros(c.len()), self.model.clone())),
        })
    }

    /// Every window of the plan through the encoder, in as few runs as the
    /// session's rows and ranges allow. A window whose ranges do not all
    /// fit in the run being built goes again in the next.
    fn run(&self, st: &mut State, inputs: &[Input<'_>], out: &mut [Result<Output, Error>]) -> Result<(), CoreError> {
        let State { plan, run } = st;
        run.rows.clear();
        run.ranges.clear();
        run.targets.clear();
        let dim = self.dim as usize;
        let head = self.head.len() as u32;
        let mut p = 0usize;
        for (w, win) in plan.windows.iter().enumerate() {
            let end = p + plan.placed[p..].iter().take_while(|x| x.window as usize == w).count();
            let ids = &inputs[win.doc as usize].subwords.ids[win.start as usize..win.end as usize];
            while p < end {
                if run.rows.len() == self.max_batch || run.ranges.len() as u32 == self.max_ranges {
                    run.flush(dim, &plan.placed, out)?;
                }
                let row = run.rows.push(&[&self.head, ids, &self.tail]);
                let take = (end - p).min(self.max_ranges as usize - run.ranges.len());
                for (k, x) in plan.placed[p..p + take].iter().enumerate() {
                    run.ranges.push(turbo_token_range { row, start: head + x.start, end: head + x.end });
                    run.targets.push((p + k) as u32);
                }
                p += take;
            }
        }
        if !run.rows.is_empty() {
            run.flush(dim, &plan.placed, out)?;
        }
        Ok(())
    }
}

/// The vector of item `t` in a document's output.
fn row_of(o: &mut Output, t: Target, dim: usize) -> &mut [f32] {
    let data = match t.which {
        Which::Sentences => &mut o.sentences.as_mut().expect("sentences were given").data,
        Which::Chunks => &mut o.chunks.as_mut().expect("chunks were given").data,
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
/// where it runs: the row length and the prompt.
fn config_hash(values: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in values.iter().flat_map(|v| v.to_le_bytes()) {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

/// The provenance every vector step gives: the model, where it ran and
/// the settings.
macro_rules! provenance {
    () => {
        fn model(&self) -> Option<ModelRef> {
            Some(self.0.model.clone())
        }

        fn execution(&self) -> Option<Execution> {
            Some(self.0.execution.clone())
        }

        fn config_hash(&self) -> u64 {
            self.0.config
        }
    };
}

/// Writes `opennlp:embeddings<opennlp:sentences>`: a vector per sentence.
#[derive(Clone)]
pub struct SentenceVectors(pub Arc<VectorAnnotator>);

impl Annotator for SentenceVectors {
    const ID: &'static str = "turbo:sentence-vectors";
    type Reads = (Subwords, Sentences);
    type Writes = (Vectors<Sentences>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, s)| Input { subwords: sub, sentences: Some(&s.spans), chunks: None })
            .collect();
        self.0.embed(Self::ID, &inputs).into_iter().map(|r| r.map(|o| (o.sentences.expect("asked for"),))).collect()
    }

    provenance!();
}

/// Writes `opennlp:embeddings<turbo:chunks>`: a vector per chunk, at
/// every level of the chunk hierarchy.
#[derive(Clone)]
pub struct ChunkVectors(pub Arc<VectorAnnotator>);

impl Annotator for ChunkVectors {
    const ID: &'static str = "turbo:chunk-vectors";
    type Reads = (Subwords, Chunks);
    type Writes = (Vectors<Chunks>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, c)| Input { subwords: sub, sentences: None, chunks: Some(&c.spans) })
            .collect();
        self.0.embed(Self::ID, &inputs).into_iter().map(|r| r.map(|o| (o.chunks.expect("asked for"),))).collect()
    }

    provenance!();
}

/// Writes both, from the same windows: a sentence and the chunk it is in
/// are pooled from one encoder pass when one window holds them.
#[derive(Clone)]
pub struct SentenceChunkVectors(pub Arc<VectorAnnotator>);

impl Annotator for SentenceChunkVectors {
    const ID: &'static str = "turbo:sentence-chunk-vectors";
    type Reads = (Subwords, Sentences, Chunks);
    type Writes = (Vectors<Sentences>, Vectors<Chunks>);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_batch(vec![input]).remove(0)
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        let inputs: Vec<Input<'_>> = inputs
            .into_iter()
            .map(|(sub, s, c)| Input { subwords: sub, sentences: Some(&s.spans), chunks: Some(&c.spans) })
            .collect();
        self.0
            .embed(Self::ID, &inputs)
            .into_iter()
            .map(|r| r.map(|o| (o.sentences.expect("asked for"), o.chunks.expect("asked for"))))
            .collect()
    }

    provenance!();
}

/// Offers the vector steps to pipelines built from configuration, as
/// "vectors". The spec's location is the bundle directory; its options:
///
/// - `layers`: `sentences`, `chunks` or `both` (the default);
/// - `device`: `cpu` (the default), `select`, a device index, or a backend
///   name such as `cuda`;
/// - `precision`: `model` (the default), `fastest` or `exact`;
/// - `prompt`: `document` (the default), `query` or `none`.
pub struct VectorsProvider;

fn parse_options(spec: &Spec) -> Result<VectorOptions, String> {
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
    let prompt = match spec.option("prompt", "document") {
        "document" => PromptRole::Document,
        "query" => PromptRole::Query,
        "none" => PromptRole::None,
        p => return Err(format!("prompt {p:?}: document, query or none")),
    };
    Ok(VectorOptions { encoder: EncoderOptions { device, precision, ..EncoderOptions::default() }, prompt })
}

impl Provider<dyn Step> for VectorsProvider {
    fn name(&self) -> &'static str {
        "vectors"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("vectors: no bundle location".into()))?;
        let o = parse_options(spec).map_err(|e| Error::Pipeline(format!("vectors: {e}")))?;
        let layers = spec.option("layers", "both");
        if !matches!(layers, "sentences" | "chunks" | "both") {
            return Err(Error::Pipeline(format!("vectors: layers {layers:?}: sentences, chunks or both")));
        }
        let a =
            VectorAnnotator::open(dir, &o).map_err(|e| Error::Pipeline(format!("vectors: {}: {e}", dir.display())))?;
        let a = Arc::new(a);
        Ok(match layers {
            "sentences" => Box::new(Bound::new(SentenceVectors(a))),
            "chunks" => Box::new(Bound::new(ChunkVectors(a))),
            _ => Box::new(Bound::new(SentenceChunkVectors(a))),
        })
    }
}
