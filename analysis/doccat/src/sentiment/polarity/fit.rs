//! End-to-end training of [`PolarityVectorHead`] with `turbo-trainer`
//! (`AdamW`, `linear::forward`, `linear::backward`, `xent::loss`, and `Rng`).

use turbo_analysis::ModelRef;
use turbo_analysis_layers::Polarity;
use turbo_trainer::adamw::{AdamW, Settings};
use turbo_trainer::linear;
use turbo_trainer::rng::Rng;
use turbo_trainer::xent;

use super::head::{POLARITY_COUNT, PolarityVectorHead, polarity_index};
use super::metrics::evaluate_polarity;

/// One labeled embedding vector for training or evaluating a [`PolarityVectorHead`].
#[derive(Clone, Debug, PartialEq)]
pub struct PolarityExample {
    pub vector: Vec<f32>,
    pub polarity: Polarity,
}

impl PolarityExample {
    pub fn new(vector: Vec<f32>, polarity: Polarity) -> Self {
        Self { vector, polarity }
    }
}

/// Hyperparameters for fitting a [`PolarityVectorHead`] with `turbo-trainer`'s
/// AdamW optimizer and softmax cross-entropy loss.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct PolarityFitConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub warmup_steps: u64,
    pub clip_norm: f64,
    /// Early-stopping patience on validation macro-F1 when a validation split
    /// is provided (`0` disables early stopping).
    pub patience: usize,
    pub seed: u64,
}

impl Default for PolarityFitConfig {
    fn default() -> Self {
        Self {
            epochs: 40,
            batch_size: 16,
            learning_rate: 0.05,
            weight_decay: 1e-3,
            warmup_steps: 0,
            clip_norm: 5.0,
            patience: 10,
            seed: 0x504F_4C41_5249_5459,
        }
    }
}

/// Summary of a [`PolarityVectorHead`] training run.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct PolarityFitReport {
    pub epochs_run: usize,
    pub steps_run: u64,
    pub final_train_loss: f64,
    pub best_dev_accuracy: Option<f64>,
    pub best_dev_macro_f1: Option<f64>,
}

/// Parses a polarity label string (`"positive"` / `"pos"` / `"1"`,
/// `"negative"` / `"neg"` / `"-1"`, `"neutral"` / `"neu"` / `"0"`) into
/// [`Polarity`].
pub fn parse_polarity_label(raw: &str) -> Result<Polarity, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "positive" | "pos" | "+1" | "1" => Ok(Polarity::Positive),
        "negative" | "neg" | "-1" => Ok(Polarity::Negative),
        "neutral" | "neu" | "0" => Ok(Polarity::Neutral),
        other => Err(format!("unknown polarity label {other:?}")),
    }
}

/// Parses a two-column TSV (`<label>\t<text>`) for polarity training or
/// evaluation. Blank lines and `#` comment lines are skipped.
pub fn parse_polarity_tsv(tsv: &str) -> Result<Vec<(Polarity, String)>, String> {
    let mut rows = Vec::new();
    for (line_idx, raw_line) in tsv.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((label_str, text)) = raw_line.split_once('\t') else {
            return Err(format!("line {}: expected `<label>\\t<text>`", line_idx + 1));
        };
        let polarity = parse_polarity_label(label_str).map_err(|e| format!("line {}: {e}", line_idx + 1))?;
        let cleaned = text.trim();
        if cleaned.is_empty() {
            return Err(format!("line {}: empty text column", line_idx + 1));
        }
        rows.push((polarity, cleaned.to_owned()));
    }
    if rows.is_empty() {
        return Err("polarity TSV contains no examples".into());
    }
    Ok(rows)
}

impl PolarityVectorHead {
    /// Fits a [`PolarityVectorHead`] on `examples` using `turbo-trainer`.
    pub fn fit(
        examples: &[PolarityExample],
        config: &PolarityFitConfig,
        encoder: Option<ModelRef>,
    ) -> Result<Self, String> {
        let (head, _) = Self::fit_with_dev(examples, &[], config, encoder)?;
        Ok(head)
    }

    /// Fits a [`PolarityVectorHead`] on `train` with optional validation early
    /// stopping on `dev`, returning the trained head and [`PolarityFitReport`].
    pub fn fit_with_dev(
        train: &[PolarityExample],
        dev: &[PolarityExample],
        config: &PolarityFitConfig,
        encoder: Option<ModelRef>,
    ) -> Result<(Self, PolarityFitReport), String> {
        if train.is_empty() {
            return Err("polarity training requires at least one example".into());
        }
        if config.epochs == 0 || config.batch_size == 0 {
            return Err("epochs and batch_size must be > 0".into());
        }
        let dim = train[0].vector.len();
        if dim == 0 {
            return Err("polarity training vector dimension must be > 0".into());
        }
        validate_examples(train, dim)?;
        if !dev.is_empty() {
            validate_examples(dev, dim)?;
        }

        let batches_per_epoch = train.len().div_ceil(config.batch_size) as u64;
        let total_steps = batches_per_epoch.saturating_mul(config.epochs as u64);
        let warmup = config.warmup_steps.min(total_steps.saturating_sub(1));
        let settings = Settings {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            clip: config.clip_norm,
            warmup,
            total: total_steps,
            ..Settings::default()
        };

        let weight_len = POLARITY_COUNT.checked_mul(dim).ok_or("polarity weight dimension overflow")?;
        let mut opt = AdamW::<f32>::new(settings, &[weight_len, POLARITY_COUNT]).map_err(|e| e.to_string())?;
        let mut rng = Rng::new(config.seed);

        let init_bound = (1.0 / (dim as f64).sqrt()) * 0.1;
        let weights: Vec<f32> = (0..weight_len).map(|_| rng.uniform(init_bound) as f32).collect();
        let biases = vec![0.0f32; POLARITY_COUNT];
        let mut params = vec![weights, biases];
        let mut grads = vec![vec![0.0f32; weight_len], vec![0.0f32; POLARITY_COUNT]];
        let decay = [true, false];

        let mut order: Vec<usize> = (0..train.len()).collect();
        let cap = config.batch_size.min(train.len());
        let x_cap = cap.checked_mul(dim).ok_or("polarity batch buffer capacity overflow")?;
        let logit_cap = cap.checked_mul(POLARITY_COUNT).ok_or("polarity batch logit capacity overflow")?;
        let mut x_batch = Vec::with_capacity(x_cap);
        let mut targets = Vec::with_capacity(cap);
        let mut logits = Vec::with_capacity(logit_cap);
        let mut d_logits = Vec::with_capacity(logit_cap);

        let mut best_weights = params[0].clone();
        let mut best_biases: [f32; POLARITY_COUNT] = params[1].as_slice().try_into().unwrap();
        let mut best_dev_f1: Option<f64> = None;
        let mut best_dev_acc: Option<f64> = None;
        let mut epochs_without_improvement = 0usize;
        let mut epochs_run = 0usize;
        let mut steps_run = 0u64;
        let mut final_train_loss = 0.0f64;

        for _epoch in 0..config.epochs {
            epochs_run += 1;
            rng.shuffle(&mut order);
            let mut epoch_loss_sum = 0.0f64;
            let mut epoch_items = 0usize;

            for chunk in order.chunks(config.batch_size) {
                let batch_n = chunk.len();
                x_batch.clear();
                targets.clear();
                for &idx in chunk {
                    x_batch.extend_from_slice(&train[idx].vector);
                    targets.push(polarity_index(train[idx].polarity) as u32);
                }

                logits.resize(batch_n * POLARITY_COUNT, 0.0);
                linear::forward(&params[0], &params[1], &x_batch, dim, POLARITY_COUNT, &mut logits);

                d_logits.clear();
                d_logits.resize(batch_n * POLARITY_COUNT, 0.0);
                let batch_loss =
                    xent::loss(&logits, &targets, POLARITY_COUNT, &mut d_logits).map_err(|e| e.to_string())?;
                epoch_loss_sum += batch_loss;
                epoch_items += batch_n;
                let inv_n = 1.0 / (batch_n as f32);
                for g in &mut d_logits {
                    *g *= inv_n;
                }

                let [dw, db] = &mut grads[..] else { unreachable!() };
                dw.fill(0.0);
                db.fill(0.0);
                linear::backward(&params[0], &x_batch, &d_logits, dim, POLARITY_COUNT, dw, db, None);

                opt.step(&mut params, &mut grads, &decay).map_err(|e| e.to_string())?;
                steps_run += 1;
            }

            final_train_loss = if epoch_items > 0 { epoch_loss_sum / (epoch_items as f64) } else { 0.0 };

            if !dev.is_empty() {
                let b_arr: [f32; POLARITY_COUNT] = params[1].as_slice().try_into().unwrap();
                let candidate = Self::new(dim, params[0].clone(), b_arr, encoder.clone())?;
                let dev_metrics = evaluate_polarity(&candidate, dev)?;
                let improved = match (best_dev_f1, best_dev_acc) {
                    (None, _) => true,
                    (Some(best_f1), Some(best_acc)) => {
                        dev_metrics.macro_f1 > best_f1 + 1e-12
                            || ((dev_metrics.macro_f1 - best_f1).abs() <= 1e-12
                                && dev_metrics.accuracy > best_acc + 1e-12)
                    }
                    (Some(best_f1), None) => dev_metrics.macro_f1 > best_f1 + 1e-12,
                };
                if improved {
                    best_dev_f1 = Some(dev_metrics.macro_f1);
                    best_dev_acc = Some(dev_metrics.accuracy);
                    best_weights.clone_from(&params[0]);
                    best_biases = b_arr;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                    if config.patience > 0 && epochs_without_improvement >= config.patience {
                        break;
                    }
                }
            } else {
                best_weights.clone_from(&params[0]);
                best_biases = params[1].as_slice().try_into().unwrap();
            }
        }

        let head = Self::new(dim, best_weights, best_biases, encoder)?;
        let report = PolarityFitReport {
            epochs_run,
            steps_run,
            final_train_loss,
            best_dev_accuracy: best_dev_acc,
            best_dev_macro_f1: best_dev_f1,
        };
        Ok((head, report))
    }
}

fn validate_examples(examples: &[PolarityExample], dim: usize) -> Result<(), String> {
    for (idx, ex) in examples.iter().enumerate() {
        if ex.vector.len() != dim {
            return Err(format!("example {idx} has dimension {}, expected {dim}", ex.vector.len()));
        }
        if ex.vector.iter().any(|v| !v.is_finite()) {
            return Err(format!("example {idx} contains non-finite values"));
        }
    }
    Ok(())
}
