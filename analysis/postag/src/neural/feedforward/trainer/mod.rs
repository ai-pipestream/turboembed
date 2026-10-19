//! Mini-batch Adagrad trainer and hyperparameters for [`FeedforwardPosModel`]
//! (`FeedforwardPOSTrainer` and `FeedforwardPOSTrainer.Settings`).

use std::collections::{HashMap, HashSet};

use super::context::{ABSENT, FeedforwardPosContext, PRETRAINED_SLOTS, SLOTS, SentenceFeatures, UNKNOWN};
use super::model::FeedforwardPosModel;
use crate::neural::{JavaRng, NeuralError, PosSample};

mod adagrad;
mod settings;

use adagrad::{adagrad_matrix, adagrad_vector, uniform_f32_matrix, update_weight};

pub use settings::FeedforwardSettings;

struct Example {
    features: [usize; SLOTS],
    pretrained_rows: Option<[isize; PRETRAINED_SLOTS]>,
    gold: usize,
}

/// Mini-batch Adagrad trainer for [`FeedforwardPosModel`], ported from `FeedforwardPOSTrainer`.
pub struct FeedforwardPosTrainer;

impl FeedforwardPosTrainer {
    /// Trains a [`FeedforwardPosModel`] without pretrained word vectors (`ONLP-FFPT-1`).
    pub fn train(samples: &[PosSample], settings: FeedforwardSettings) -> Result<FeedforwardPosModel, NeuralError> {
        Self::train_internal(samples, settings, None::<fn(&str) -> Option<Vec<f32>>>, None)
    }

    /// Trains a [`FeedforwardPosModel`] with pretrained word vectors (`ONLP-FFPT-2`).
    pub fn train_with_vectors<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: F,
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        Self::train_internal(samples, settings, Some(vectors), None)
    }

    /// Trains a [`FeedforwardPosModel`] with pretrained word vectors and an additional lexicon (`ONLP-FFPT-2`).
    pub fn train_with_lexicon<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: F,
        additional_words: &[String],
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        Self::train_internal(samples, settings, Some(vectors), Some(additional_words))
    }

    fn train_internal<F>(
        samples: &[PosSample],
        settings: FeedforwardSettings,
        vectors: Option<F>,
        additional_words: Option<&[String]>,
    ) -> Result<FeedforwardPosModel, NeuralError>
    where
        F: Fn(&str) -> Option<Vec<f32>>,
    {
        let mut corpus: Vec<&PosSample> = Vec::new();
        let mut word_order: Vec<String> = Vec::new();
        let mut word_counts: HashMap<String, usize> = HashMap::new();
        let mut suffix_order: Vec<String> = Vec::new();
        let mut suffix_counts: HashMap<String, usize> = HashMap::new();
        let mut tag_order: Vec<String> = Vec::new();
        let mut tag_set: HashSet<String> = HashSet::new();

        for sample in samples {
            if sample.sentence.is_empty() {
                continue;
            }
            corpus.push(sample);
            for i in 0..sample.sentence.len() {
                let lower = FeedforwardPosContext::lowercase(&sample.sentence[i]);
                if let Some(c) = word_counts.get_mut(&lower) {
                    *c += 1;
                } else {
                    word_order.push(lower.clone());
                    word_counts.insert(lower, 1);
                }
                let lowered = FeedforwardPosContext::lowercase_all(&sample.sentence[i]);
                let s2 = FeedforwardPosContext::suffix_str(&lowered, 2);
                if let Some(c) = suffix_counts.get_mut(s2) {
                    *c += 1;
                } else {
                    let owned = s2.to_owned();
                    suffix_order.push(owned.clone());
                    suffix_counts.insert(owned, 1);
                }
                let s3 = FeedforwardPosContext::suffix_str(&lowered, 3);
                if let Some(c) = suffix_counts.get_mut(s3) {
                    *c += 1;
                } else {
                    let owned = s3.to_owned();
                    suffix_order.push(owned.clone());
                    suffix_counts.insert(owned, 1);
                }
                let tg = &sample.tags[i];
                if tag_set.insert(tg.clone()) {
                    tag_order.push(tg.clone());
                }
            }
        }
        if corpus.is_empty() {
            return Err(NeuralError::InvalidArgument("Training stream produced no non-empty POS samples.".into()));
        }

        let mut next_row = 0usize;
        let mut word_entries = Vec::new();
        let mut suffix_entries = Vec::new();
        let mut shape_entries = Vec::new();
        let mut tag_feature_entries = Vec::new();

        next_row = register_vocab(&mut word_entries, &word_order, &word_counts, settings.word_cutoff, next_row);
        next_row = register_vocab(&mut suffix_entries, &suffix_order, &suffix_counts, settings.suffix_cutoff, next_row);
        next_row = register_all(&mut shape_entries, FeedforwardPosContext::all_shapes().iter().copied(), next_row);
        next_row = register_all(&mut tag_feature_entries, tag_order.iter().map(String::as_str), next_row);
        let total_rows = next_row;

        let mut pretrained_entries = Vec::new();
        let mut pretrained_rows_vec: Vec<Vec<f32>> = Vec::new();
        let mut pretrained_size = 0usize;
        if let Some(ref vec_fn) = vectors {
            let mut seen_p = HashSet::new();
            for word in &word_order {
                if let Some(found) = vec_fn(word) {
                    let row = check_vector(word, &found, &mut pretrained_size)?;
                    pretrained_entries.push((word.clone(), pretrained_rows_vec.len()));
                    seen_p.insert(word.clone());
                    pretrained_rows_vec.push(row);
                }
            }
            if let Some(extra) = additional_words {
                for raw in extra {
                    let word = FeedforwardPosContext::lowercase(raw);
                    if seen_p.contains(&word) || word_counts.contains_key(&word) {
                        continue;
                    }
                    if let Some(found) = vec_fn(&word) {
                        let row = check_vector(&word, &found, &mut pretrained_size)?;
                        pretrained_entries.push((word.clone(), pretrained_rows_vec.len()));
                        seen_p.insert(word);
                        pretrained_rows_vec.push(row);
                    }
                }
            }
            if pretrained_rows_vec.is_empty() {
                return Err(NeuralError::InvalidArgument("vectors returned null for every vocabulary word".into()));
            }
        }

        let tags = tag_order;
        let mut tag_index = HashMap::with_capacity(tags.len());
        for (i, t) in tags.iter().enumerate() {
            tag_index.insert(t.clone(), i);
        }

        let emb_size = settings.embedding_size;
        let hid_size = settings.hidden_size;
        let input_size = SLOTS * emb_size + PRETRAINED_SLOTS * pretrained_size;
        let num_tags = tags.len();

        let shell = FeedforwardPosModel::with_pretrained(
            word_entries.clone(),
            suffix_entries.clone(),
            shape_entries.clone(),
            tag_feature_entries.clone(),
            tags.clone(),
            emb_size,
            vec![vec![0.0; emb_size]; total_rows],
            vec![vec![0.0; input_size]; hid_size],
            vec![0.0; hid_size],
            vec![vec![0.0; hid_size]; num_tags],
            vec![0.0; num_tags],
            pretrained_size,
            pretrained_entries.clone(),
            pretrained_rows_vec.clone(),
        );

        let mut examples: Vec<Example> = Vec::new();
        for sample in &corpus {
            let sent = &sample.sentence;
            let gold_tags = &sample.tags;
            let sent_features = SentenceFeatures::new(sent);
            for i in 0..sent.len() {
                let prev = if i >= 1 { Some(gold_tags[i - 1].as_str()) } else { None };
                let prev_prev = if i >= 2 { Some(gold_tags[i - 2].as_str()) } else { None };
                examples.push(Example {
                    features: shell.feature_ids_at(&sent_features, i, prev, prev_prev),
                    pretrained_rows: shell.pretrained_rows_at(&sent_features, i),
                    gold: tag_index[&gold_tags[i]],
                });
            }
        }

        let mut init_rng = JavaRng::new(settings.seed);
        let mut embeddings = uniform_f32_matrix(&mut init_rng, total_rows, emb_size, 0.01);
        let mut hidden_weights =
            uniform_f32_matrix(&mut init_rng, hid_size, input_size, (6.0 / (input_size + hid_size) as f64).sqrt());
        let mut hidden_bias = vec![0.0f32; hid_size];
        let mut output_weights =
            uniform_f32_matrix(&mut init_rng, num_tags, hid_size, (6.0 / (hid_size + num_tags) as f64).sqrt());
        let mut output_bias = vec![0.0f32; num_tags];

        let mut g2_emb = vec![vec![0.0f64; emb_size]; total_rows];
        let mut g2_hw = vec![vec![0.0f64; input_size]; hid_size];
        let mut g2_hb = vec![0.0f64; hid_size];
        let mut g2_ow = vec![vec![0.0f64; hid_size]; num_tags];
        let mut g2_ob = vec![0.0f64; num_tags];

        let mut d_emb = vec![vec![0.0f64; emb_size]; total_rows];
        let mut d_hw = vec![vec![0.0f64; input_size]; hid_size];
        let mut d_hb = vec![0.0f64; hid_size];
        let mut d_ow = vec![vec![0.0f64; hid_size]; num_tags];
        let mut d_ob = vec![0.0f64; num_tags];
        let mut touched_rows: Vec<usize> = Vec::new();
        let mut row_touched = vec![false; total_rows];

        let mut x = vec![0.0f64; input_size];
        let mut dx = vec![0.0f64; input_size];
        let mut hidden_delta = vec![0.0f64; hid_size];
        let mut pre = vec![0.0f64; hid_size];
        let mut h = vec![0.0f64; hid_size];
        let mut mask = vec![false; hid_size];
        let mut probs = vec![0.0f64; num_tags];

        let mut rng = JavaRng::new(settings.seed);
        let mut order: Vec<usize> = (0..examples.len()).collect();
        let keep_prob = 1.0 - settings.dropout;
        let pretrained_offset = SLOTS * emb_size;

        for _epoch in 0..settings.epochs {
            rng.shuffle(&mut order);
            let mut batch_start = 0;
            while batch_start < order.len() {
                let batch_end = (batch_start + settings.batch_size).min(order.len());
                let batch = (batch_end - batch_start) as f64;

                for j in 0..hid_size {
                    d_hw[j].fill(0.0);
                }
                d_hb.fill(0.0);
                for o in 0..num_tags {
                    d_ow[o].fill(0.0);
                }
                d_ob.fill(0.0);
                for &r in &touched_rows {
                    d_emb[r].fill(0.0);
                    row_touched[r] = false;
                }
                touched_rows.clear();

                for idx in &order[batch_start..batch_end] {
                    let ex = &examples[*idx];
                    for s in 0..SLOTS {
                        let e_row = &embeddings[ex.features[s]];
                        let offset = s * emb_size;
                        for d in 0..emb_size {
                            x[offset + d] = f64::from(e_row[d]);
                        }
                    }
                    if pretrained_size > 0
                        && let Some(ref p_rows) = ex.pretrained_rows
                    {
                        for w in 0..PRETRAINED_SLOTS {
                            let offset = pretrained_offset + w * pretrained_size;
                            let row = p_rows[w];
                            if row < 0 {
                                x[offset..offset + pretrained_size].fill(0.0);
                            } else {
                                let vec = &pretrained_rows_vec[row as usize];
                                for i in 0..pretrained_size {
                                    x[offset + i] = f64::from(vec[i]);
                                }
                            }
                        }
                    }

                    for j in 0..hid_size {
                        mask[j] = rng.next_double() < keep_prob;
                        if !mask[j] {
                            pre[j] = 0.0;
                            h[j] = 0.0;
                            continue;
                        }
                        let w = &hidden_weights[j];
                        let mut sum = f64::from(hidden_bias[j]);
                        for i in 0..input_size {
                            sum += f64::from(w[i]) * x[i];
                        }
                        pre[j] = sum;
                        h[j] = sum * sum * sum / keep_prob;
                    }

                    let mut max_score = f64::NEG_INFINITY;
                    for o in 0..num_tags {
                        let w = &output_weights[o];
                        let mut sum = f64::from(output_bias[o]);
                        for j in 0..hid_size {
                            sum += f64::from(w[j]) * h[j];
                        }
                        probs[o] = sum;
                        if sum > max_score {
                            max_score = sum;
                        }
                    }
                    let mut z = 0.0f64;
                    for o in 0..num_tags {
                        let p = (probs[o] - max_score).exp();
                        probs[o] = p;
                        z += p;
                    }
                    for o in 0..num_tags {
                        probs[o] /= z;
                    }

                    hidden_delta.fill(0.0);
                    dx.fill(0.0);
                    for o in 0..num_tags {
                        let delta = probs[o] - if o == ex.gold { 1.0 } else { 0.0 };
                        d_ob[o] += delta;
                        let dow = &mut d_ow[o];
                        let w_row = &output_weights[o];
                        for j in 0..hid_size {
                            dow[j] += delta * h[j];
                            hidden_delta[j] += delta * f64::from(w_row[j]);
                        }
                    }
                    for j in 0..hid_size {
                        if !mask[j] {
                            continue;
                        }
                        let d_pre = hidden_delta[j] * 3.0 * pre[j] * pre[j] / keep_prob;
                        d_hb[j] += d_pre;
                        let w = &hidden_weights[j];
                        let dw = &mut d_hw[j];
                        for i in 0..input_size {
                            dw[i] += d_pre * x[i];
                            dx[i] += d_pre * f64::from(w[i]);
                        }
                    }
                    for s in 0..SLOTS {
                        let row = ex.features[s];
                        if !row_touched[row] {
                            row_touched[row] = true;
                            touched_rows.push(row);
                        }
                        let de = &mut d_emb[row];
                        let offset = s * emb_size;
                        for i in 0..emb_size {
                            de[i] += dx[offset + i];
                        }
                    }
                }

                let lr = settings.learning_rate;
                let l2 = settings.l2;
                adagrad_matrix(&mut hidden_weights, &d_hw, &mut g2_hw, batch, lr, l2)?;
                adagrad_vector(&mut hidden_bias, &d_hb, &mut g2_hb, batch, lr)?;
                adagrad_matrix(&mut output_weights, &d_ow, &mut g2_ow, batch, lr, l2)?;
                adagrad_vector(&mut output_bias, &d_ob, &mut g2_ob, batch, lr)?;
                for &row in &touched_rows {
                    let e_row = &mut embeddings[row];
                    let g_row = &d_emb[row];
                    let a_row = &mut g2_emb[row];
                    for d in 0..emb_size {
                        let g = g_row[d] / batch;
                        a_row[d] += g * g;
                        e_row[d] = update_weight(e_row[d], g, a_row[d], lr)?;
                    }
                }

                batch_start = batch_end;
            }
        }

        Ok(FeedforwardPosModel::with_pretrained(
            word_entries,
            suffix_entries,
            shape_entries,
            tag_feature_entries,
            tags,
            emb_size,
            embeddings,
            hidden_weights,
            hidden_bias,
            output_weights,
            output_bias,
            pretrained_size,
            pretrained_entries,
            pretrained_rows_vec,
        ))
    }
}

fn check_vector(word: &str, found: &[f32], pretrained_size: &mut usize) -> Result<Vec<f32>, NeuralError> {
    if found.is_empty() {
        return Err(NeuralError::InvalidArgument(format!("empty vector returned for '{word}'")));
    }
    if *pretrained_size == 0 {
        *pretrained_size = found.len();
    } else if found.len() != *pretrained_size {
        return Err(NeuralError::InvalidArgument(format!(
            "vector length mismatch for '{word}': expected {}, got {}",
            *pretrained_size,
            found.len()
        )));
    }
    for (i, &v) in found.iter().enumerate() {
        if !v.is_finite() {
            return Err(NeuralError::InvalidArgument(format!(
                "non-finite vector component for '{word}' at index {i}: {v}"
            )));
        }
    }
    Ok(found.to_vec())
}

fn register_vocab(
    entries: &mut Vec<(String, usize)>,
    order: &[String],
    counts: &HashMap<String, usize>,
    cutoff: usize,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in order {
        if counts[s] >= cutoff && seen.insert(s.clone()) {
            entries.push((s.clone(), next_row));
            next_row += 1;
        }
    }
    next_row
}

fn register_all<'a>(
    entries: &mut Vec<(String, usize)>,
    symbols: impl IntoIterator<Item = &'a str>,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in symbols {
        if seen.insert(s.to_owned()) {
            entries.push((s.to_owned(), next_row));
            next_row += 1;
        }
    }
    next_row
}
