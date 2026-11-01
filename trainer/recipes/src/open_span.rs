//! Open-label span typing (`HEAD_OPEN_SPAN`) recipes (#1735): building
//! `HeadSpec`s, embedding caller label names, preparing training and
//! zero-shot held-out examples from BIO corpora (`wnut17`, `multiconer`),
//! and scoring against both the majority baseline and a case-insensitive
//! dictionary matcher over the label names.

use std::collections::BTreeSet;

use turbo::head::OpenSpanMatcher;
use turbo::head::open_span::{OpenSpanWork, label_words, pool_label_words};
use turbo::head::tagger::Work;
use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, SpanShape, WordPooling};
use turbo_tagger::WordEncoder;
use turbo_trainer::OpenSpanExample;
use turbo_trainer::rng::Rng;
use turbo_trainer::span::GoldSpan;

use crate::corpus::{Corpus, CorpusRecord, Sentence, Task};
use crate::eval;
use crate::run::PREDICT_CHUNK;
use crate::{Error, Result, fixed_entity_types};

/// Default maximum span length in words for `HEAD_OPEN_SPAN`.
pub const DEFAULT_MAX_WORDS: u32 = 8;
/// Default span projection width for `HEAD_OPEN_SPAN`.
pub const DEFAULT_WIDTH_DIM: u32 = 64;
/// Entity type extracted from contiguous `PROPN` runs in a UD treebank.
pub const UD_PROPN_TYPE: &str = "proper noun";

/// Converts a sequence of UD `UPOS` tags into BIO labels over contiguous
/// `PROPN` runs (`"B-proper noun"`, `"I-proper noun"`, `"O"`).
pub fn propn_to_bio(upos: &[String]) -> Vec<String> {
    let mut out = Vec::with_capacity(upos.len());
    let mut in_propn = false;
    for tag in upos {
        if tag == "PROPN" {
            out.push(if in_propn { format!("I-{UD_PROPN_TYPE}") } else { format!("B-{UD_PROPN_TYPE}") });
            in_propn = true;
        } else {
            out.push("O".to_owned());
            in_propn = false;
        }
    }
    out
}

/// Reads a BIO NER corpus or a UD treebank (`ud-<lang>-<bank>` or
/// `ud-<lang>-<bank>-upos`, converting contiguous `PROPN` runs into
/// `"proper noun"` BIO spans) for `train-open-span`.
pub fn read_open_span_corpus(id: &str, dir: &std::path::Path, unrecorded: bool) -> Result<Corpus> {
    if let Some(bank) = crate::ud::TREEBANKS.iter().find(|b| b.id == id) {
        let mut c = crate::ud::read(bank, crate::ud::Column::Upos, dir, unrecorded)?;
        for s in c.splits.train.iter_mut().chain(&mut c.splits.dev).chain(&mut c.splits.test) {
            s.labels = propn_to_bio(&s.labels);
        }
        c.record.id = bank.id.into();
        c.record.name = format!("{}, PROPN spans", bank.name);
        c.record.task = Task::Bio;
        return Ok(c);
    }
    let mut c = crate::read_corpus(id, dir, unrecorded)?;
    if c.record.task == Task::Tags && c.record.id.starts_with("ud-") {
        for s in c.splits.train.iter_mut().chain(&mut c.splits.dev).chain(&mut c.splits.test) {
            s.labels = propn_to_bio(&s.labels);
        }
        c.record.task = Task::Bio;
    }
    Ok(c)
}

fn sentence_spans(s: &Sentence) -> Vec<(usize, usize, String)> {
    if s.labels.iter().all(|l| l == "O" || l.starts_with("B-") || l.starts_with("I-")) {
        eval::spans(&s.labels)
    } else {
        eval::spans(&propn_to_bio(&s.labels))
    }
}

/// Collects all entity type names from `corpus` in deterministic order.
pub fn entity_types(corpus: &Corpus) -> Result<Vec<String>> {
    if corpus.record.task != Task::Bio && !(corpus.record.task == Task::Tags && corpus.record.id.starts_with("ud-")) {
        return Err(Error(format!("corpus {:?} is {:?}, expected BIO or UD", corpus.record.id, corpus.record.task)));
    }
    if let Some(fixed) = fixed_entity_types(&corpus.record.id) {
        return Ok(fixed.iter().map(|s| (*s).to_owned()).collect());
    }
    let mut set = BTreeSet::new();
    for s in corpus.splits.train.iter().chain(&corpus.splits.dev).chain(&corpus.splits.test) {
        for (_, _, kind) in sentence_spans(s) {
            set.insert(kind);
        }
    }
    if set.is_empty() {
        return Err(Error(format!("corpus {:?} has no entity spans", corpus.record.id)));
    }
    Ok(set.into_iter().collect())
}

/// Builds and validates a `HEAD_OPEN_SPAN` [`HeadSpec`].
pub fn spec(
    labels: Vec<String>,
    record: &CorpusRecord,
    pooling: WordPooling,
    bilstm: Option<(u32, u32)>,
    max_words: u32,
    width_dim: u32,
    nested: bool,
) -> Result<HeadSpec> {
    let s = HeadSpec {
        name: "open_ner".into(),
        layer: "turboembed:entities".into(),
        kind: HeadKind::OpenSpan,
        labels,
        scheme: Scheme::Tags,
        word_pooling: pooling,
        adapter: bilstm.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: if nested { Decode::SpansNested } else { Decode::SpansFlat },
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        layer_mix: None,
        span: Some(SpanShape { max_words, width_dim }),
        relation: None,
        license_class: Some(record.class()?),
        artifact: "weights-f32".into(),
        tensor_prefix: "heads.open_ner".into(),
    };
    s.validate("head").map_err(|e| Error(e.message))?;
    Ok(s)
}

/// Encodes and L2-normalizes each label name in `labels` into `[labels.len() * hidden]`.
pub fn embed_label_names(enc: &mut WordEncoder, labels: &[String], pooling: WordPooling) -> Result<Vec<f32>> {
    let hidden = enc.hidden();
    let seqs: Vec<Vec<&str>> = labels.iter().map(|l| label_words(l)).collect();
    let feats = enc.features(&seqs, pooling).map_err(|e| Error(e.message))?;
    let mut out = Vec::with_capacity(labels.len() * hidden);
    for wf in &feats {
        let pooled = pool_label_words(wf, hidden).map_err(|e| Error(e.message))?;
        out.extend_from_slice(&pooled);
    }
    Ok(out)
}

/// Encodes `sentences` into per-word feature vectors in chunks of [`PREDICT_CHUNK`].
pub fn encode_sentences(enc: &mut WordEncoder, sentences: &[Sentence], pooling: WordPooling) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(sentences.len());
    for chunk in sentences.chunks(PREDICT_CHUNK) {
        let refs: Vec<Vec<&str>> = chunk.iter().map(|s| s.words.iter().map(String::as_str).collect()).collect();
        let feats = enc.features(&refs, pooling).map_err(|e| Error(e.message))?;
        out.extend(feats);
    }
    Ok(out)
}

/// Encodes `sentences` into [`OpenSpanExample`]s over `active_labels`.
pub fn examples(
    enc: &mut WordEncoder,
    sentences: &[Sentence],
    active_labels: &[String],
    hold_out: &[String],
    pooling: WordPooling,
    max_words: u32,
    train_seed: Option<u64>,
) -> Result<(Vec<OpenSpanExample>, usize)> {
    let hidden = enc.hidden();
    let label_features = embed_label_names(enc, active_labels, pooling)?;
    let word_feats = encode_sentences(enc, sentences, pooling)?;
    examples_from_features(
        sentences,
        &word_feats,
        &label_features,
        active_labels,
        hold_out,
        max_words,
        hidden,
        train_seed,
    )
}

/// Builds [`OpenSpanExample`]s from pre-extracted sentence `word_feats` and
/// `label_features`.
///
/// When `train_seed` is `Some(seed)`, sentences containing any span in
/// `hold_out` are excluded so held-out entity mentions are never trained as
/// outside (`O`) negatives, and a second pass of prototype-conditioned label
/// vectors (with negative non-entity distractor labels) is appended so the
/// label projection learns across the encoder's full hidden space.
#[allow(clippy::too_many_arguments)]
pub fn examples_from_features(
    sentences: &[Sentence],
    word_feats: &[Vec<f32>],
    label_features: &[f32],
    active_labels: &[String],
    hold_out: &[String],
    max_words: u32,
    hidden: usize,
    train_seed: Option<u64>,
) -> Result<(Vec<OpenSpanExample>, usize)> {
    let max_w = max_words as usize;
    let num_labels = active_labels.len();
    let mut out = Vec::with_capacity(sentences.len() * if train_seed.is_some() { 2 } else { 1 });
    let mut mention_pool: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_labels];
    let mut outside_pool: Vec<Vec<f32>> = Vec::new();
    let mut skipped = 0usize;

    for (s, wf) in sentences.iter().zip(word_feats) {
        let sp = sentence_spans(s);
        let has_held_out = train_seed.is_some() && sp.iter().any(|(_, _, k)| hold_out.iter().any(|h| h == k));
        if has_held_out {
            skipped += 1;
            continue;
        }
        let mut gold = Vec::new();
        let mut occupied = vec![false; s.words.len()];
        for (start, end, kind) in sp {
            if end <= start || end - start > max_w {
                continue;
            }
            let Some(l) = active_labels.iter().position(|a| a == &kind) else { continue };
            for flag in &mut occupied[start..end] {
                *flag = true;
            }
            let span = GoldSpan { first: start as u32, last: (end - 1) as u32, label: l as u32 };
            if gold.iter().all(|g: &GoldSpan| g.first != span.first || g.last != span.last) {
                gold.push(span);
                if train_seed.is_some()
                    && mention_pool[l].len() < 256
                    && let Ok(p) = pool_label_words(&wf[start * hidden..end * hidden], hidden)
                {
                    mention_pool[l].push(p);
                }
            }
        }
        if train_seed.is_some() && outside_pool.len() < 512 {
            for (i, &is_ent) in occupied.iter().enumerate() {
                if !is_ent
                    && outside_pool.len() < 512
                    && let Ok(p) = pool_label_words(&wf[i * hidden..(i + 1) * hidden], hidden)
                {
                    outside_pool.push(p);
                }
            }
        }
        out.push(OpenSpanExample { features: wf.clone(), label_features: label_features.to_vec(), spans: gold });
    }

    if let Some(seed) = train_seed {
        let mut rng = Rng::new(seed ^ 0x517c_c1b7_2722_0a95);
        let mut cq = vec![0.0f32; hidden];
        for l in 0..num_labels {
            for d in 0..hidden {
                cq[d] += label_features[l * hidden + d] / num_labels as f32;
            }
        }
        let mut cx = vec![0.0f32; hidden];
        let mut cx_count = 0usize;
        for pool in &mention_pool {
            for m in pool {
                for d in 0..hidden {
                    cx[d] += m[d];
                }
                cx_count += 1;
            }
        }
        if cx_count > 0 {
            for val in &mut cx {
                *val /= cx_count as f32;
            }
        }
        let shift_to_q = |proto: &[f32]| -> Vec<f32> {
            let centered: Vec<f32> = proto.iter().zip(&cx).map(|(&p, &c)| p - c).collect();
            let unit_c = pool_label_words(&centered, hidden).unwrap_or_else(|_| proto.to_vec());
            let shifted: Vec<f32> = cq.iter().zip(&unit_c).map(|(&q0, &u)| q0 + 0.35 * u).collect();
            pool_label_words(&shifted, hidden).unwrap_or_else(|_| proto.to_vec())
        };
        let base_len = out.len();
        for idx in 0..base_len {
            let mut lf = Vec::with_capacity((num_labels + 2) * hidden);
            for l in 0..num_labels {
                let canon = &label_features[l * hidden..(l + 1) * hidden];
                let pool = &mention_pool[l];
                if !pool.is_empty() {
                    let mut avg = vec![0.0f32; hidden];
                    for _ in 0..4 {
                        let m = &pool[rng.below(pool.len())];
                        for d in 0..hidden {
                            avg[d] += 0.25 * m[d];
                        }
                    }
                    lf.extend_from_slice(&shift_to_q(&avg));
                } else {
                    lf.extend_from_slice(canon);
                }
            }
            for _ in 0..2 {
                if !outside_pool.is_empty() {
                    let mut avg = vec![0.0f32; hidden];
                    for _ in 0..2 {
                        let o = &outside_pool[rng.below(outside_pool.len())];
                        for d in 0..hidden {
                            avg[d] += 0.5 * o[d];
                        }
                    }
                    lf.extend_from_slice(&shift_to_q(&avg));
                }
            }
            out.push(OpenSpanExample {
                features: out[idx].features.clone(),
                label_features: lf,
                spans: out[idx].spans.clone(),
            });
        }
    }

    Ok((out, skipped))
}

/// Predicts BIO label sequences on `sentences` using `head` and `eval_labels`.
pub fn predict(
    enc: &mut WordEncoder,
    head: &OpenSpanMatcher,
    sentences: &[Sentence],
    eval_labels: &[String],
) -> Result<Vec<Vec<String>>> {
    let pooling = head.spec().word_pooling;
    let label_features = embed_label_names(enc, eval_labels, pooling)?;
    let word_feats = encode_sentences(enc, sentences, pooling)?;
    predict_from_features(head, sentences, &word_feats, &label_features, eval_labels)
}

/// Predicts BIO label sequences from pre-extracted `word_feats` and `label_features`.
pub fn predict_from_features(
    head: &OpenSpanMatcher,
    sentences: &[Sentence],
    word_feats: &[Vec<f32>],
    label_features: &[f32],
    eval_labels: &[String],
) -> Result<Vec<Vec<String>>> {
    let (mut scores, mut kept, mut w, mut sw) = (Vec::new(), Vec::new(), Work::default(), OpenSpanWork::default());
    let num_labels = eval_labels.len();
    let mut out = Vec::with_capacity(sentences.len());
    for (s, wf) in sentences.iter().zip(word_feats) {
        let n = s.words.len();
        let words: Vec<u32> = (0..n as u32).collect();
        let forms_vec: Vec<&str> = s.words.iter().map(String::as_str).collect();
        let forms = head.reads_forms().then_some(forms_vec.as_slice());
        head.scores(wf, &words, forms, label_features, &mut scores, &mut w, &mut sw).map_err(|e| Error(e.message))?;
        head.decode(&scores, n, num_labels, &mut kept).map_err(|e| Error(e.message))?;
        let mut bio = vec!["O".to_owned(); n];
        for sp in &kept {
            let (first, last) = (sp.first as usize, sp.last as usize);
            let Some(kind) = eval_labels.get(sp.label as usize) else { continue };
            if first <= last && last < n && bio[first..=last].iter().all(|b| b == "O") {
                bio[first] = format!("B-{kind}");
                for b in &mut bio[first + 1..=last] {
                    *b = format!("I-{kind}");
                }
            }
        }
        out.push(bio);
    }
    Ok(out)
}

/// Filters a sentence split's BIO labels to keep only spans whose type is in `keep`.
pub fn filter_sentences(sentences: &[Sentence], keep: &[String]) -> Vec<Sentence> {
    sentences
        .iter()
        .map(|s| {
            let mut labels = vec!["O".to_owned(); s.words.len()];
            for (start, end, kind) in sentence_spans(s) {
                if keep.iter().any(|k| k == &kind) {
                    labels[start] = format!("B-{kind}");
                    for l in &mut labels[start + 1..end] {
                        *l = format!("I-{kind}");
                    }
                }
            }
            Sentence { words: s.words.clone(), labels }
        })
        .collect()
}

/// Case-insensitive dictionary matcher over the words of each label name in `labels`.
pub fn dictionary_baseline(sentences: &[Sentence], labels: &[String]) -> Vec<Vec<String>> {
    let patterns: Vec<(Vec<String>, &str)> = labels
        .iter()
        .map(|l| (label_words(l).into_iter().map(|w| w.to_ascii_lowercase()).collect(), l.as_str()))
        .collect();
    sentences
        .iter()
        .map(|s| {
            let n = s.words.len();
            let lower: Vec<String> = s.words.iter().map(|w| w.to_ascii_lowercase()).collect();
            let mut bio = vec!["O".to_owned(); n];
            let mut i = 0usize;
            while i < n {
                let mut matched: Option<(usize, &str)> = None;
                for (pat, kind) in &patterns {
                    let len = pat.len();
                    if len > 0
                        && i + len <= n
                        && lower[i..i + len] == pat[..]
                        && matched.is_none_or(|(best_len, _)| len > best_len)
                    {
                        matched = Some((len, kind));
                    }
                }
                if let Some((len, kind)) = matched {
                    bio[i] = format!("B-{kind}");
                    for b in &mut bio[i + 1..i + len] {
                        *b = format!("I-{kind}");
                    }
                    i += len;
                } else {
                    i += 1;
                }
            }
            bio
        })
        .collect()
}
