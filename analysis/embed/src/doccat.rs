//! Sequence-classification transformer scorer (`BundleSequenceScorer`) for
//! [`turbo_doccat::DocumentCategorizerDL`] (`#208`), running on turboembed's
//! backends (`CPU`, `CUDA`, `Metal`, `Level Zero`, and `NPU`).
//!
//! Opens a bundle carrying a `CLASSIFIER_SEQUENCE` head (`manifest.classifier`
//! with `classifier.weight`, `classifier.bias`, and optional `pooler.dense.weight`
//! and `pooler.dense.bias`), encodes each text chunk or `opennlp:subwords`
//! window with `TURBO_POOLING_CLS` and `TURBO_NORMALIZE_NONE` through
//! [`Encoder::run_cls`], and evaluates [`turbo::head::Head::logits_into`] on
//! the resulting CLS hidden states.

use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::Head;
use turbo::manifest::{ClassifierKind, PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, INVALID_SHAPE};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Execution, ModelRef};
use turbo_doccat::{
    AverageClassificationScoringStrategy, DlDoccatError, DlInferenceOptions, DlSequenceScorer, DocumentCategorizerDL,
    dl_preprocess_text,
};

use crate::encoder::{Encoder, EncoderInfo, EncoderOptions, Rows};

struct SeqRun {
    encoder: Encoder,
    rows: Rows,
    cls_states: Vec<f32>,
}

/// Sequence-classification transformer session over a turboembed bundle,
/// implementing [`DlSequenceScorer`] for [`DocumentCategorizerDL`].
pub struct BundleSequenceScorer {
    run: Mutex<SeqRun>,
    tokenizer: Tokenizer,
    head: Head,
    model: ModelRef,
    execution: Execution,
    info: EncoderInfo,
    row_head: Vec<i32>,
    row_tail: Vec<i32>,
    max_content_tokens: usize,
    max_batch: u32,
}

impl BundleSequenceScorer {
    /// Opens the sequence-classification bundle at `dir` on the device and
    /// precision named by `options`.
    pub fn open(dir: &Path, options: &EncoderOptions) -> Result<Self, CoreError> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let head = Head::load(&bundle)?;
        if head.kind != ClassifierKind::Sequence {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "{}: BundleSequenceScorer requires CLASSIFIER_SEQUENCE, found {:?}",
                    bundle.manifest.model.id, head.kind
                ),
            ));
        }
        let embed_cfg = bundle.manifest.embed.as_ref().ok_or_else(|| {
            CoreError::new(INVALID_ARGUMENT, format!("{}: bundle has no embed block", bundle.manifest.model.id))
        })?;
        let max_batch = if options.max_batch == 0 { embed_cfg.max_batch } else { options.max_batch };
        let max_seq = if options.max_seq == 0 { embed_cfg.max_seq } else { options.max_seq };
        let (row_head, row_tail) = tokenizer.template_ends();
        let overhead = row_head.len().saturating_add(row_tail.len());
        if (max_seq as usize) <= overhead {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!("max_seq {max_seq} leaves no room for content tokens beside {overhead} template tokens"),
            ));
        }
        let max_content_tokens = (max_seq as usize) - overhead;
        let mut eo = options.clone();
        eo.max_batch = max_batch;
        eo.max_seq = max_seq;
        if eo.max_ranges == 0 {
            // CLS sequence classification uses row writes (run_cls), so
            // max_ranges == max_batch lets sessions open on backends that do
            // not offer embed_write_ranges.
            eo.max_ranges = max_batch;
        }
        let encoder = Encoder::open(dir, &eo)?;
        let info = encoder.info().clone();
        if head.hidden != info.dim as usize {
            return Err(CoreError::new(
                INVALID_SHAPE,
                format!("sequence head hidden width {} does not match encoder dim {}", head.hidden, info.dim),
            ));
        }
        let fill_id = tokenizer.fill_id();
        let execution = info.execution.clone();
        let max_batch = info.max_batch;
        Ok(Self {
            run: Mutex::new(SeqRun { rows: Rows::new(info.max_seq, fill_id), cls_states: Vec::new(), encoder }),
            tokenizer,
            head,
            model: crate::model_ref(&bundle),
            execution,
            info,
            row_head,
            row_tail,
            max_content_tokens,
            max_batch,
        })
    }

    /// Opens a [`DocumentCategorizerDL`] backed by [`BundleSequenceScorer`] on
    /// the device and precision named by `encoder_options`, with
    /// `dl_options`.
    pub fn open_categorizer(
        dir: &Path,
        encoder_options: &EncoderOptions,
        dl_options: DlInferenceOptions,
    ) -> Result<DocumentCategorizerDL<Self, AverageClassificationScoringStrategy>, CoreError> {
        let scorer = Self::open(dir, encoder_options)?;
        DocumentCategorizerDL::with_options(scorer, dl_options)
            .map_err(|e| CoreError::new(INVALID_ARGUMENT, e.to_string()))
    }

    /// Returns the bundle's [`Tokenizer`].
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Returns the loaded sequence [`Head`].
    pub fn head(&self) -> &Head {
        &self.head
    }

    /// Returns the resolved [`EncoderInfo`].
    pub fn info(&self) -> &EncoderInfo {
        &self.info
    }

    fn score_piece_batches(&self, windows: &[&[i32]]) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        let num_labels = self.head.num_labels();
        let mut guard = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        let run = &mut *guard;
        let mut all_logits = Vec::with_capacity(windows.len());
        let mut batch_logits = Vec::new();
        for batch in windows.chunks(self.max_batch as usize) {
            run.rows.clear();
            for &win in batch {
                let capped = &win[..win.len().min(self.max_content_tokens)];
                run.rows.push(&[&self.row_head, capped, &self.row_tail]);
            }
            run.encoder.run_cls(&run.rows, &mut run.cls_states).map_err(|e| DlDoccatError::Backend(e.to_string()))?;
            batch_logits.resize(batch.len() * num_labels, 0.0f32);
            self.head
                .logits_into(&run.cls_states, &mut batch_logits)
                .map_err(|e| DlDoccatError::InvalidLogits(e.to_string()))?;
            for row in batch_logits.chunks_exact(num_labels) {
                all_logits.push(row.to_vec());
            }
        }
        run.rows.clear();
        Ok(all_logits)
    }
}

impl DlSequenceScorer for BundleSequenceScorer {
    fn categories(&self) -> &[String] {
        &self.head.labels
    }

    fn model_ref(&self) -> ModelRef {
        self.model.clone()
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn max_content_tokens(&self) -> usize {
        self.max_content_tokens
    }

    fn score_chunks(&self, chunks: &[String], options: &DlInferenceOptions) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        let encode_opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let mut encoded_chunks: Vec<Vec<i32>> = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let processed = dl_preprocess_text(chunk, options);
            let mut ids = Vec::new();
            self.tokenizer
                .encode_into(&processed, encode_opts, &mut ids)
                .map_err(|e| DlDoccatError::Backend(e.to_string()))?;
            if ids.len() > self.max_content_tokens {
                ids.truncate(self.max_content_tokens);
            }
            encoded_chunks.push(ids);
        }
        let refs: Vec<&[i32]> = encoded_chunks.iter().map(Vec::as_slice).collect();
        self.score_piece_batches(&refs)
    }

    fn score_subword_windows(&self, windows: &[&[i32]]) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        self.score_piece_batches(windows)
    }
}
