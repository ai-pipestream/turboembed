//! Sequence-pair NLI scorer (`BundleNliScorer`) implementing
//! [`turbo_doccat::NliScorer`] over a turboembed session (`#909`).
//!
//! Opens a sequence-classification bundle carrying a `CLASSIFIER_SEQUENCE`
//! head with 3-way (`entailment`, `neutral`, `contradiction`) or 2-way
//! (`entailment`, `not_entailment`) NLI outcome labels, encodes each
//! `(premise, hypothesis)` pair via [`Tokenizer::encode_pair`],
//! batches the pairs of a document into `max_batch` chunks through
//! [`Encoder::run_cls`], and evaluates [`turbo::head::Head::logits_into`] on
//! the resulting CLS hidden states.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::Head;
use turbo::manifest::{ClassifierKind, PromptRole, Truncation};
use turbo::status::{Error as CoreError, INVALID_ARGUMENT, INVALID_SHAPE};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Execution, ModelRef};
use turbo_doccat::{NliScorer, ZeroShotAnnotator, ZeroShotCategorizer, ZeroShotConfig, ZeroShotError};

use crate::encoder::{Encoder, EncoderInfo, EncoderOptions, Rows};

struct NliRun {
    encoder: Encoder,
    rows: Rows,
    cls_states: Vec<f32>,
    runs: u64,
}

fn fold(s: &str) -> String {
    s.trim().chars().flat_map(char::to_lowercase).collect()
}

fn valid_nli_outcomes(labels: &[String]) -> bool {
    let names: Vec<String> = labels.iter().map(|n| fold(n)).collect();
    let has = |want: &str| names.iter().any(|n| n == want);
    let three = names.len() == 3 && has("entailment") && has("neutral") && has("contradiction");
    let two = names.len() == 2 && has("entailment") && has("not_entailment");
    three || two
}

/// Sequence-pair NLI scorer over a turboembed bundle, implementing
/// [`NliScorer`] for [`ZeroShotCategorizer`] (`#909`).
pub struct BundleNliScorer {
    run: Mutex<NliRun>,
    tokenizer: Tokenizer,
    head: Head,
    model: ModelRef,
    execution: Execution,
    info: EncoderInfo,
    pair_specials: usize,
    max_pair_tokens: u32,
    max_batch: u32,
}

impl BundleNliScorer {
    /// Opens the NLI sequence-classification bundle at `dir` on the device and
    /// precision named by `options`.
    pub fn open(dir: &Path, options: &EncoderOptions) -> Result<Self, CoreError> {
        let bundle = Bundle::open(dir)?;
        let tokenizer = Tokenizer::load(&bundle)?;
        let head = Head::load(&bundle)?;
        if head.kind != ClassifierKind::Sequence {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "{}: BundleNliScorer requires CLASSIFIER_SEQUENCE, found {:?}",
                    bundle.manifest.model.id, head.kind
                ),
            ));
        }
        if !valid_nli_outcomes(&head.labels) {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "{}: NLI classifier labels {:?} must be entailment/neutral/contradiction or entailment/not_entailment",
                    bundle.manifest.model.id, head.labels
                ),
            ));
        }
        let pair_template = tokenizer.pair_template().ok_or_else(|| {
            CoreError::new(
                INVALID_ARGUMENT,
                format!("{}: bundle tokenizer does not support sequence-pair encoding", bundle.manifest.model.id),
            )
        })?;
        let pair_specials = pair_template.specials() as usize;
        let embed_cfg = bundle.manifest.embed.as_ref().ok_or_else(|| {
            CoreError::new(INVALID_ARGUMENT, format!("{}: bundle has no embed block", bundle.manifest.model.id))
        })?;
        let max_batch = if options.max_batch == 0 { embed_cfg.max_batch } else { options.max_batch };
        let max_seq = if options.max_seq == 0 { embed_cfg.max_seq } else { options.max_seq };
        if (max_seq as usize) <= pair_specials + 1 {
            return Err(CoreError::new(
                INVALID_ARGUMENT,
                format!(
                    "max_seq {max_seq} leaves no room for premise and hypothesis tokens beside {pair_specials} pair specials"
                ),
            ));
        }
        let mut eo = options.clone();
        eo.max_batch = max_batch;
        eo.max_seq = max_seq;
        if eo.max_ranges == 0 {
            eo.max_ranges = max_batch;
        }
        let encoder = Encoder::open(dir, &eo)?;
        let info = encoder.info().clone();
        if head.hidden != info.dim as usize {
            return Err(CoreError::new(
                INVALID_SHAPE,
                format!("NLI head hidden width {} does not match encoder dim {}", head.hidden, info.dim),
            ));
        }
        let fill_id = tokenizer.fill_id();
        let execution = info.execution.clone();
        let max_pair_tokens = info.max_seq;
        let max_batch = info.max_batch;
        Ok(Self {
            run: Mutex::new(NliRun {
                rows: Rows::new(max_pair_tokens, fill_id),
                cls_states: Vec::new(),
                runs: 0,
                encoder,
            }),
            tokenizer,
            head,
            model: crate::model_ref(&bundle),
            execution,
            info,
            pair_specials,
            max_pair_tokens,
            max_batch,
        })
    }

    /// Opens a [`ZeroShotCategorizer`] backed by [`BundleNliScorer`] on the
    /// device and precision named by `encoder_options`.
    pub fn open_categorizer(
        dir: &Path,
        encoder_options: &EncoderOptions,
        labels: Vec<String>,
        config: ZeroShotConfig,
    ) -> Result<ZeroShotCategorizer<Self>, CoreError> {
        let scorer = Self::open(dir, encoder_options)?;
        ZeroShotCategorizer::new(scorer, labels, config).map_err(|e| CoreError::new(INVALID_ARGUMENT, e.to_string()))
    }

    /// Opens a [`ZeroShotAnnotator`] backed by [`BundleNliScorer`] on the
    /// device and precision named by `encoder_options`.
    pub fn open_annotator(
        dir: &Path,
        encoder_options: &EncoderOptions,
        labels: Vec<String>,
        config: ZeroShotConfig,
    ) -> Result<ZeroShotAnnotator<Self>, CoreError> {
        let categorizer = Self::open_categorizer(dir, encoder_options, labels, config)?;
        Ok(ZeroShotAnnotator::new(categorizer))
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

    /// Returns how many encoder batch runs have been executed by this scorer.
    pub fn run_count(&self) -> u64 {
        self.run.lock().unwrap_or_else(PoisonError::into_inner).runs
    }
}

impl NliScorer for BundleNliScorer {
    fn outcome_labels(&self) -> &[String] {
        &self.head.labels
    }

    fn model_ref(&self) -> ModelRef {
        self.model.clone()
    }

    fn execution(&self) -> Option<Execution> {
        Some(self.execution.clone())
    }

    fn max_pair_tokens(&self) -> u32 {
        self.max_pair_tokens
    }

    fn pair_special_tokens(&self) -> usize {
        self.pair_specials
    }

    fn count_tokens(&self, text: &str) -> usize {
        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        self.tokenizer.encode(text, opts).map_or(0, |ids| ids.len())
    }

    fn score_pairs(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, ZeroShotError> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let num_labels = self.head.num_labels();
        let max_pair = self.max_pair_tokens as usize;
        let mut hyp_lengths: HashMap<&str, usize> = HashMap::new();
        let mut guard = self.run.lock().unwrap_or_else(PoisonError::into_inner);
        let run = &mut *guard;
        let mut all_logits = Vec::with_capacity(pairs.len());
        let mut batch_logits = Vec::new();
        for batch in pairs.chunks(self.max_batch as usize) {
            run.rows.clear();
            for &(premise, hypothesis) in batch {
                let hyp_tokens = *hyp_lengths.entry(hypothesis).or_insert_with(|| self.count_tokens(hypothesis));
                if hyp_tokens + self.pair_specials >= max_pair {
                    return Err(ZeroShotError::Scorer(format!(
                        "hypothesis of {hyp_tokens} tokens plus {} pair specials fills max_pair_tokens {}",
                        self.pair_specials, self.max_pair_tokens
                    )));
                }
                let query_tokens = (max_pair - self.pair_specials - hyp_tokens.max(1)).max(1) as u32;
                let windows = self
                    .tokenizer
                    .encode_pair(premise, hypothesis, query_tokens, self.max_pair_tokens, 0)
                    .map_err(|e| ZeroShotError::Scorer(e.to_string()))?;
                let w = &windows[0];
                run.rows.push_with_types(&w.ids, &w.types);
            }
            run.encoder.run_cls(&run.rows, &mut run.cls_states).map_err(|e| ZeroShotError::Scorer(e.to_string()))?;
            run.runs += 1;
            batch_logits.resize(batch.len() * num_labels, 0.0f32);
            self.head
                .logits_into(&run.cls_states, &mut batch_logits)
                .map_err(|e| ZeroShotError::BadScorerOutput(e.to_string()))?;
            for row in batch_logits.chunks_exact(num_labels) {
                all_logits.push(row.to_vec());
            }
        }
        run.rows.clear();
        Ok(all_logits)
    }
}
