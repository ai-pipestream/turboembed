//! A document's learned sparse terms (the embed task's sparse output,
//! `embed.sparse`): the vocabulary terms an MLM head gives a text, pooled
//! as SPLADE pools (SPLADE-v3, arXiv 2403.06789) and cut by top-k or a
//! weight floor, with the source span of every term the input carries.
//!
//! The encoder runs each window of the text's subword pieces once with a
//! range of one token per piece, exactly as the SaT step reads per-token
//! states (analysis/sat), and the MLM head pools
//! `log(1 + relu(logit))` over them on the host (turbo::mlm,
//! turbo::sparse). Max pooling composes over windows, so a long text is
//! read in sequential windows of what a row holds and pools as one span;
//! a text of no pieces has no logits and no terms.
//!
//! A term whose id is among the input's pieces is an input term: its
//! span runs from its first occurrence's start to its last occurrence's
//! end, and its occurrences, one span per piece, cover the pieces byte
//! for byte. A term the model adds is expanded: no span, no occurrences.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::manifest::PromptRole;
use turbo::mlm::MlmHead;
use turbo::sparse::{MaxLogPool, SparseOptions as Cut, SparseTerm};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT};
use turbo::tokenizer::Tokenizer;
use turbo::turbo_token_range;
use turbo_analysis::annotate::Step;
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{Annotator, Bound, Error, Execution, Layer, ModelRef, Refs, Span, View};
use turbo_analysis_layers::{SparseTerms, Subwords};

use crate::encoder::{Encoder, EncoderOptions, Rows};
use crate::model_ref;

/// How a `SparseAnnotator` runs its model and cuts the terms.
#[derive(Clone, Debug)]
pub struct SparseOptions {
    pub encoder: EncoderOptions,
    /// The prompt whose prefix every row carries: the document role by
    /// default, as for the vectors.
    pub prompt: PromptRole,
    /// The most terms a document keeps; 0 keeps every term over `floor`.
    pub top_k: u32,
    /// The least weight a term keeps; 0 keeps every positive weight.
    pub floor: f32,
}

impl Default for SparseOptions {
    fn default() -> SparseOptions {
        SparseOptions { encoder: EncoderOptions::default(), prompt: PromptRole::Document, top_k: 0, floor: 0.0 }
    }
}

/// Writes `turbo:sparse-terms` over the view: the document's learned
/// sparse vector, one unit holding every term. See the module comment.
pub struct SparseAnnotator {
    state: Mutex<State>,
    mlm: MlmHead,
    model: ModelRef,
    execution: Execution,
    /// The template's leading special tokens, then the prompt's prefix.
    row_head: Vec<i32>,
    row_tail: Vec<i32>,
    /// Pieces per window: the session's row length less the overhead.
    width: u32,
    vocab: usize,
    cut: Cut,
    max_batch: u32,
    max_ranges: usize,
    settings: [u32; 4],
}

/// What one batch works in, kept between batches.
struct State {
    run: Run,
}

/// The run being built: its rows, a one-token range per piece, whose
/// piece each range is, and the session it goes to.
struct Run {
    encoder: Encoder,
    rows: Rows,
    ranges: Vec<turbo_token_range>,
    /// `(document, piece)` per range, `document` the index into the
    /// batch's `pools`.
    owners: Vec<(u32, u32)>,
    states: Vec<f32>,
}

impl SparseAnnotator {
    /// The bundle at `dir` on the device `o` names, its MLM head loaded
    /// as the core loads it on seal.
    pub fn open(dir: &Path, o: &SparseOptions) -> Result<SparseAnnotator, CoreError> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let mlm = MlmHead::load(&bundle)?;
        let mut eo = o.encoder.clone();
        let max_batch = if eo.max_batch == 0 { bundle.manifest.embed().max_batch } else { eo.max_batch };
        let max_seq = if eo.max_seq == 0 { bundle.manifest.embed().max_seq } else { eo.max_seq };
        eo.max_batch = max_batch;
        eo.max_seq = max_seq;
        eo.max_ranges = max_batch.saturating_mul(max_seq);
        let encoder = Encoder::open(dir, &eo)?;
        let info = encoder.info().clone();
        if mlm.hidden != info.dim as usize {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("the MLM head reads width {} hidden states, the encoder gives {}", mlm.hidden, info.dim),
            ));
        }
        let vocab = bundle.manifest.architecture.as_ref().map_or(0, |a| a.vocab_size) as usize;
        if mlm.vocab != vocab {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("the MLM head decodes {} terms, the model's vocabulary has {vocab}", mlm.vocab),
            ));
        }
        let (mut head_tokens, tail) = tokenizer.template_ends();
        let bare = turbo::tokenizer::Encode {
            add_special_tokens: false,
            truncation: turbo::manifest::Truncation::None,
            max_tokens: u32::MAX,
            prompt: o.prompt,
        };
        tokenizer.encode_into("", bare, &mut head_tokens)?;
        let overhead = (head_tokens.len() + tail.len()) as u32;
        if info.max_seq <= overhead {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("max_seq {} leaves no room for text beside {overhead} tokens of overhead", info.max_seq),
            ));
        }
        if !o.floor.is_finite() || o.floor < 0.0 {
            return Err(CoreError::new(INVALID_ARGUMENT, format!("sparse floor {}: a number over 0", o.floor)));
        }
        let prompt = match o.prompt {
            PromptRole::None => 0,
            PromptRole::Query => 1,
            PromptRole::Document => 2,
        };
        let cut = Cut { top_k: o.top_k, floor: o.floor };
        Ok(SparseAnnotator {
            state: Mutex::new(State {
                run: Run {
                    rows: Rows::new(info.max_seq, tokenizer.fill_id()),
                    ranges: Vec::new(),
                    owners: Vec::new(),
                    states: Vec::new(),
                    encoder,
                },
            }),
            mlm,
            model: model_ref(&bundle),
            execution: info.execution,
            row_head: head_tokens,
            row_tail: tail,
            width: info.max_seq - overhead,
            vocab,
            cut,
            max_batch,
            max_ranges: (max_batch as usize).saturating_mul(max_seq as usize),
            settings: [info.max_seq, prompt, o.top_k, o.floor.to_bits()],
        })
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    /// The configuration hash of the terms a step writes: the row length,
    /// the prompt and the top-k cut, before the layer that lays out the
    /// windows.
    pub fn config(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        self.settings.iter().for_each(|&v| {
            h.u32(v);
        });
        h.finish()
    }

    /// The terms of one document per input, a result for each. A document
    /// whose layers do not fit fails alone; a run that fails fails the
    /// batch. `annotator` names the step in errors.
    pub fn encode(&self, annotator: &'static str, inputs: &[Refs<'_, Self>]) -> Vec<Result<SparseTerms<View>, Error>> {
        let mut out: Vec<Result<SparseTerms<View>, Error>> =
            inputs.iter().map(|(view, sub)| self.check(annotator, view, sub)).collect();
        let mut pools: Vec<Option<MaxLogPool>> = vec![None; inputs.len()];
        {
            let mut guard = self.lock();
            let st = &mut *guard;
            st.run.rows.clear();
            st.run.ranges.clear();
            st.run.owners.clear();
            for (d, slot) in out.iter_mut().enumerate() {
                if slot.is_err() {
                    continue;
                }
                let (_, sub) = inputs[d];
                for w in 0..sub.ids.len().div_ceil(self.width as usize) {
                    let full = st.run.rows.len() == self.max_batch || st.run.ranges.len() == self.max_ranges;
                    if full && let Err(e) = self.flush(&mut st.run, &mut pools) {
                        return Self::fail_all(annotator, e, out);
                    }
                    let lo = w * self.width as usize;
                    let hi = (lo + self.width as usize).min(sub.ids.len());
                    let row = st.run.rows.push(&[&self.row_head, &sub.ids[lo..hi], &self.row_tail]);
                    for (k, p) in (lo..hi).enumerate() {
                        let at = self.row_head.len() as u32 + k as u32;
                        st.run.ranges.push(turbo_token_range { row, start: at, end: at + 1 });
                        st.run.owners.push((d as u32, p as u32));
                    }
                    if pools[d].is_none() {
                        pools[d] = Some(MaxLogPool::new(self.vocab));
                    }
                }
            }
            if let Err(e) = self.flush(&mut st.run, &mut pools) {
                return Self::fail_all(annotator, e, out);
            }
        }
        for ((d, slot), pool) in (0..out.len()).zip(out.iter_mut()).zip(pools) {
            let Ok(terms) = slot else { continue };
            let pool = pool.unwrap_or_else(|| MaxLogPool::new(self.vocab));
            match pool.terms(&self.cut) {
                Ok(selected) => Self::fill(terms, inputs[d].1, &selected),
                Err(e) => *slot = Err(Error::Annotator { annotator, detail: e.to_string() }),
            }
        }
        out
    }

    /// An empty result for `input`, or why its layers do not fit.
    fn check(&self, annotator: &'static str, view: &View, sub: &Subwords) -> Result<SparseTerms<View>, Error> {
        let fail = |detail: String| Err(Error::Annotator { annotator, detail });
        if sub.spans.view() != view.id() {
            return fail(format!("the subwords are in view {}, the text in view {}", sub.spans.view().0, view.id().0));
        }
        if let Err(e) = sub.check() {
            return fail(format!("the subwords: {e}"));
        }
        // The terms publish these spans as their source: every piece span
        // must lie on the text, whole UTF-8 characters, or the spans a
        // term carries would name bytes the text has not.
        let text = view.text();
        for k in 0..sub.ids.len() {
            let s = sub.spans.get(turbo_analysis::Ix::new(k as u32));
            if text.get(s.range()).is_none() {
                return fail(format!(
                    "piece {k} span {}..{} is out of bounds or not on a UTF-8 char boundary (text len {})",
                    s.start,
                    s.end,
                    text.len()
                ));
            }
        }
        if sub.model != self.model {
            return fail(format!(
                "the subwords are {}'s pieces (revision {}), the terms {}'s (revision {})",
                sub.model.id, sub.model.revision, self.model.id, self.model.revision
            ));
        }
        Ok(SparseTerms::new(self.model.clone()))
    }

    /// Run what is written and pool each piece's logits into its
    /// document's pool. Leaves the run empty.
    fn flush(&self, run: &mut Run, pools: &mut [Option<MaxLogPool>]) -> Result<(), CoreError> {
        if run.rows.is_empty() {
            return Ok(());
        }
        run.encoder.run_ranges(&run.rows, &run.ranges, &mut run.states)?;
        let mut at = 0usize;
        while at < run.owners.len() {
            // Pool the states of one document together: a run is built a
            // document at a time, so each document's states are a run of
            // its owners.
            let d = run.owners[at].0;
            let end = run.owners[at..].iter().position(|&(x, _)| x != d).map_or(run.owners.len(), |k| at + k);
            let pool = pools[d as usize].as_mut().expect("a document with rows has a pool");
            self.mlm.relu_peak_into(&run.states[at * self.mlm.hidden..end * self.mlm.hidden], pool)?;
            at = end;
        }
        run.rows.clear();
        run.ranges.clear();
        run.owners.clear();
        Ok(())
    }

    /// Fill one document's output from its pool's selection: an input
    /// term keeps its pieces' spans, one occurrence per piece, and a
    /// bounding span from the first occurrence's start to the last's
    /// end; an expanded term keeps none.
    fn fill(out: &mut SparseTerms<View>, sub: &Subwords, selected: &[SparseTerm]) {
        // The pieces are in text order, not id order: collect each input
        // term's occurrences first, and bound them once they are all in.
        let mut spans: std::collections::HashMap<i32, (Span, Vec<Span>)> = std::collections::HashMap::new();
        for k in 0..sub.ids.len() {
            let s = sub.spans.get(turbo_analysis::Ix::new(k as u32));
            spans
                .entry(sub.ids[k])
                .and_modify(|(hull, occ): &mut (Span, Vec<Span>)| {
                    hull.start = hull.start.min(s.start);
                    hull.end = hull.end.max(s.end);
                    occ.push(s);
                })
                .or_insert((s, vec![s]));
        }
        for t in selected {
            match spans.get(&(t.term as i32)) {
                Some((hull, occ)) => {
                    out.push(t.term, t.weight, Some(*hull));
                    out.push_occurrences(occ);
                }
                None => {
                    out.push(t.term, t.weight, None);
                    out.push_occurrences(&[]);
                }
            }
        }
        out.unit();
    }

    /// Run a failure into every still-ok result, and give the batch back.
    fn fail_all(
        annotator: &'static str,
        e: CoreError,
        mut out: Vec<Result<SparseTerms<View>, Error>>,
    ) -> Vec<Result<SparseTerms<View>, Error>> {
        let e = Error::Annotator { annotator, detail: e.to_string() };
        for o in out.iter_mut().filter(|o| o.is_ok()) {
            *o = Err(e.clone());
        }
        out
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        // A panic in a run leaves nothing half changed that the next
        // batch reads: every buffer is cleared before it is used.
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Annotator for SparseAnnotator {
    const ID: &'static str = "turbo:sparse";
    type Reads = (View, Subwords);
    type Writes = (SparseTerms<View>,);

    fn annotate(&self, input: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.encode(Self::ID, &[input]).remove(0).map(|t| (t,))
    }

    fn annotate_batch<'d>(&self, inputs: Vec<Refs<'d, Self>>) -> Vec<Result<Self::Writes, Error>> {
        self.encode(Self::ID, &inputs).into_iter().map(|r| r.map(|t| (t,))).collect()
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn config_hash(&self) -> u64 {
        self.config()
    }
}

/// Offers `SparseAnnotator` to pipelines built from configuration, as
/// "sparse": the spec's location is the bundle directory.
pub struct SparseProvider;

impl Provider<dyn Step> for SparseProvider {
    fn name(&self) -> &'static str {
        "sparse"
    }

    fn supports(&self, spec: &Spec) -> bool {
        spec.location.is_some()
    }

    fn create(&self, spec: &Spec) -> Result<Box<dyn Step>, Error> {
        let dir = spec.location.as_deref().ok_or_else(|| Error::Pipeline("sparse: no bundle location".into()))?;
        let o = SparseOptions::default();
        let a =
            SparseAnnotator::open(dir, &o).map_err(|e| Error::Pipeline(format!("sparse: {}: {e}", dir.display())))?;
        Ok(Box::new(Bound::new(a)))
    }
}
