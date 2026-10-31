//! Loading and running a `HEAD_OPEN_SPAN` bundle (`turbo::head::OpenSpanMatcher`)
//! inside `turbo-gliner` (#1735).
//!
//! A `HEAD_OPEN_SPAN` bundle runs any default-tier encoder (such as
//! `ModernBERT-base`) once over the document's words and once over the
//! caller's `ExtractionSchema` label names, then matches every candidate word
//! span against the normalized label embeddings on the host.

use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};

use turbo::bundle::Bundle;
use turbo::head::OpenSpanMatcher;
use turbo::head::open_span::{OpenSpanWork, label_words, pool_label_words};
use turbo::head::sigmoid;
use turbo::head::tagger::Work as CoreWork;
use turbo::manifest::{HeadKind, HeadSpec, PromptRole, Truncation, WordPooling};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::turbo_token_range;
use turbo_analysis::{Execution, IxRange, Layer as _, ModelRef, Span, Spans, Tag, TagSet, View};
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_zoo::{Acceptances, Format, Kind, UserId, Zoo};

use crate::{Candidate, Decisions, ExtractionSchema, decode};

const BARE: Encode =
    Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: PromptRole::None };

struct Runner {
    encoder: Encoder,
    tokenizer: Tokenizer,
    cls: Vec<i32>,
    sep: Vec<i32>,
    width: usize,
    max_batch: usize,
    max_ranges: usize,
    hidden: usize,
}

impl Runner {
    fn open(dir: &Path, bundle: &Bundle, o: &EncoderOptions) -> Result<(Runner, Execution), String> {
        let tokenizer = Tokenizer::load(bundle).map_err(|e| e.message)?;
        let embed = bundle.manifest.embed();
        let mut eo = o.clone();
        eo.max_batch = if eo.max_batch == 0 { embed.max_batch } else { eo.max_batch };
        eo.max_seq = if eo.max_seq == 0 { embed.max_seq } else { eo.max_seq };
        eo.max_ranges = eo.max_batch.saturating_mul(eo.max_seq);
        let encoder = Encoder::open(dir, &eo).map_err(|e| e.message)?;
        let (cls, sep) = tokenizer.template_ends();
        let overhead = cls.len() + sep.len();
        let info = encoder.info().clone();
        if info.max_seq as usize <= overhead {
            return Err(format!("max_seq {} leaves no room beside {overhead} special tokens", info.max_seq));
        }
        let exec = info.execution.clone();
        Ok((
            Runner {
                width: info.max_seq as usize - overhead,
                max_batch: info.max_batch as usize,
                max_ranges: info.max_ranges as usize,
                hidden: info.dim as usize,
                encoder,
                tokenizer,
                cls,
                sep,
            },
            exec,
        ))
    }

    fn pieces(&self, word: &str) -> Result<Vec<i32>, String> {
        let (mut ids, mut spans) = (Vec::new(), Vec::new());
        self.tokenizer.encode_spans(word, BARE, &mut ids, &mut spans).map_err(|e| e.message)?;
        if ids.is_empty() {
            ids.push(self.tokenizer.unk_id);
        }
        ids.truncate(self.width);
        Ok(ids)
    }

    fn word_features(&mut self, words: &[&str], pooling: WordPooling) -> Result<Vec<f32>, String> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let pieces: Vec<Vec<i32>> = words.iter().map(|w| self.pieces(w)).collect::<Result<_, _>>()?;
        let mut out = vec![0.0f32; words.len() * self.hidden];
        let mut rows = Rows::new((self.width + self.cls.len() + self.sep.len()) as u32, self.tokenizer.fill_id());
        let mut ranges: Vec<turbo_token_range> = Vec::new();
        let mut owners: Vec<(usize, usize, usize)> = Vec::new();
        let mut states: Vec<f32> = Vec::new();
        let mut row_ids: Vec<i32> = Vec::new();
        let mut a = 0usize;
        while a < pieces.len() {
            let (mut b, mut used) = (a, 0usize);
            while b < pieces.len() && used + pieces[b].len() <= self.width {
                used += pieces[b].len();
                b += 1;
            }
            let count = b - a;
            if rows.len() as usize == self.max_batch || ranges.len() + count > self.max_ranges {
                self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, pooling)?;
            }
            row_ids.clear();
            let mut offsets = Vec::with_capacity(count);
            for p in &pieces[a..b] {
                offsets.push(row_ids.len());
                row_ids.extend_from_slice(p);
            }
            let row = rows.push(&[&self.cls, &row_ids, &self.sep]);
            for (k, &at) in offsets.iter().enumerate() {
                let w = a + k;
                let start = (self.cls.len() + at) as u32;
                let n = pieces[w].len();
                if count > self.max_ranges {
                    for piece in 0..n {
                        let p_at = start + piece as u32;
                        ranges.push(turbo_token_range { row, start: p_at, end: p_at + 1 });
                        owners.push((w, piece, n));
                    }
                } else {
                    let len = match pooling {
                        WordPooling::First => 1,
                        WordPooling::Mean => n as u32,
                    };
                    ranges.push(turbo_token_range { row, start, end: start + len });
                    owners.push((w, 0, 0));
                }
            }
            a = b;
        }
        self.flush(&mut rows, &mut ranges, &mut owners, &mut states, &mut out, pooling)?;
        Ok(out)
    }

    fn flush(
        &mut self,
        rows: &mut Rows,
        ranges: &mut Vec<turbo_token_range>,
        owners: &mut Vec<(usize, usize, usize)>,
        states: &mut Vec<f32>,
        out: &mut [f32],
        pooling: WordPooling,
    ) -> Result<(), String> {
        if !ranges.is_empty() {
            self.encoder.run_ranges(rows, ranges, states).map_err(|e| e.message)?;
            let h = self.hidden;
            for (r, &(w, piece, pieces)) in owners.iter().enumerate() {
                let state = &states[r * h..(r + 1) * h];
                let dst = &mut out[w * h..(w + 1) * h];
                if pieces == 0 {
                    dst.copy_from_slice(state);
                } else {
                    match pooling {
                        WordPooling::First if piece == 0 => dst.copy_from_slice(state),
                        WordPooling::First => {}
                        WordPooling::Mean => {
                            if piece == 0 {
                                dst.fill(0.0);
                            }
                            for (d, &s) in dst.iter_mut().zip(state) {
                                *d += s;
                            }
                            if piece + 1 == pieces {
                                let inv = 1.0 / pieces as f32;
                                for d in dst.iter_mut() {
                                    *d *= inv;
                                }
                            }
                        }
                    }
                }
            }
        }
        rows.clear();
        ranges.clear();
        owners.clear();
        Ok(())
    }
}

/// A loaded `HEAD_OPEN_SPAN` bundle and encoder.
pub struct OpenSpanModel {
    head: OpenSpanMatcher,
    runner: Mutex<Runner>,
    identity: ModelRef,
    execution: Execution,
    config_bytes: Vec<u8>,
}

impl OpenSpanModel {
    /// Opens the `HEAD_OPEN_SPAN` head named `name` (or the sole `HEAD_OPEN_SPAN`
    /// head when `name` is `None`) in the bundle at `dir`.
    pub fn open(dir: &Path, name: Option<&str>, options: &EncoderOptions) -> Result<Self, String> {
        let bundle = Bundle::open(dir).map_err(|e| e.message)?;
        let spec = find_open_span_head(&bundle, name)?;
        let head = OpenSpanMatcher::load(&bundle, &spec.name).map_err(|e| e.message)?;
        let (runner, execution) = Runner::open(dir, &bundle, options)?;
        if runner.hidden != head.hidden() {
            return Err(format!("open-span head expects hidden {}, encoder gives {}", head.hidden(), runner.hidden));
        }
        let identity = ModelRef {
            id: bundle.manifest.model.id.clone(),
            revision: bundle.manifest.model.revision.clone(),
            sha256: bundle.manifest_sha256.clone(),
        };
        let config_bytes = serde_json::to_vec(head.spec()).map_err(|e| e.to_string())?;
        Ok(OpenSpanModel { head, runner: Mutex::new(runner), identity, execution, config_bytes })
    }

    /// Fetches `id` from `zoo` (verifying its manifest, sizes, SHA-256/SHA-512,
    /// and licence class) and opens its `HEAD_OPEN_SPAN` head.
    pub fn from_zoo(
        zoo: &Zoo<'_>,
        id: &str,
        name: Option<&str>,
        options: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<Self, String> {
        let a = zoo.index().get(id).map_err(|e| e.to_string())?;
        if !is_bundle_artifact(a) {
            return Err(format!(
                "{id} is {} {}, not a FORMAT_SAFETENSORS model or head bundle with manifest.json",
                a.kind, a.format
            ));
        }
        let fetched = zoo.ensure(id, user, acceptances).map_err(|e| e.to_string())?;
        Self::open(&fetched.dir, name, options)
    }

    /// Finds the first redistributable [`Component::EntityExtractor`](turbo_zoo::Component::EntityExtractor)
    /// entry for `language` in `zoo`, fetches it, and opens its `HEAD_OPEN_SPAN` head.
    pub fn from_zoo_language(
        zoo: &Zoo<'_>,
        language: &str,
        name: Option<&str>,
        options: &EncoderOptions,
        user: &UserId,
        acceptances: &Acceptances,
    ) -> Result<Self, String> {
        let a = zoo
            .index()
            .find(language, turbo_zoo::Component::EntityExtractor)
            .find(|a| is_bundle_artifact(a))
            .ok_or_else(|| format!("no redistributable open-span entity extractor in zoo for language {language:?}"))?;
        Self::from_zoo(zoo, a.id.as_str(), name, options, user, acceptances)
    }

    pub fn head(&self) -> &OpenSpanMatcher {
        &self.head
    }

    pub fn identity(&self) -> &ModelRef {
        &self.identity
    }

    pub fn execution(&self) -> &Execution {
        &self.execution
    }

    pub(crate) fn config_bytes(&self) -> &[u8] {
        &self.config_bytes
    }

    /// Encodes `words` into `[words.len() * hidden]` pooled word features.
    pub fn encode_words(&self, words: &[&str]) -> Result<Vec<f32>, String> {
        let pooling = self.head.spec().word_pooling;
        let mut runner = self.runner.lock().unwrap_or_else(PoisonError::into_inner);
        runner.word_features(words, pooling)
    }

    /// Encodes and L2-normalizes each label name in `labels` into `[labels.len(), hidden]`.
    pub fn embed_labels(&self, labels: &[String]) -> Result<Vec<f32>, String> {
        let pooling = self.head.spec().word_pooling;
        let hidden = self.head.hidden();
        let mut runner = self.runner.lock().unwrap_or_else(PoisonError::into_inner);
        let mut out = Vec::with_capacity(labels.len() * hidden);
        for label in labels {
            let ws = label_words(label);
            let wf = runner.word_features(&ws, pooling)?;
            let pooled = pool_label_words(&wf, hidden).map_err(|e| e.message)?;
            out.extend_from_slice(&pooled);
        }
        Ok(out)
    }

    /// Scores `words` against `schema` and returns decoded [`Candidate`] spans.
    pub fn predict(&self, words: &[&str], schema: &ExtractionSchema) -> Result<Vec<Candidate>, String> {
        let label_states = self.embed_labels(schema.labels())?;
        let candidates = self.score_words(words, &label_states)?;
        Ok(decode(candidates, schema))
    }

    pub(crate) fn score_words(&self, words: &[&str], label_states: &[f32]) -> Result<Vec<Candidate>, String> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let features = self.encode_words(words)?;
        let n = words.len();
        let word_map: Vec<u32> = (0..n as u32).collect();
        let forms = self.head.reads_forms().then_some(words);
        let (mut raw_scores, mut w, mut sw) = (Vec::new(), CoreWork::default(), OpenSpanWork::default());
        self.head
            .scores(&features, &word_map, forms, label_states, &mut raw_scores, &mut w, &mut sw)
            .map_err(|e| e.message)?;
        let num_labels = label_states.len() / self.head.hidden();
        let max_words = self.head.spec().span.map_or(0, |s| s.max_words as usize);
        let mut out = Vec::with_capacity(raw_scores.len());
        let mut rows = raw_scores.chunks_exact(num_labels);
        for i in 0..n {
            for j in i..n.min(i + max_words) {
                let Some(row) = rows.next() else { break };
                for (l, &logit) in row.iter().enumerate() {
                    out.push(Candidate { start: i, end: j + 1, label: l, logit, score: sigmoid(logit) });
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn annotate_tokens(
        &self,
        schema: &ExtractionSchema,
        label_states: &[f32],
        view: &View,
        tokens: &Tokens,
        sentences: Option<&Sentences>,
        kinds: &Arc<TagSet>,
    ) -> Result<(Entities, Decisions), String> {
        tokens.check()?;
        if tokens.spans.view() != view.id() || tokens.spans.fragment_columns().is_some() {
            return Err("gliner needs contiguous tokens in its input view".into());
        }
        let (starts, ends) = tokens.spans.columns();
        if !starts.windows(2).all(|x| x[0] <= x[1]) || !ends.windows(2).all(|x| x[0] <= x[1]) {
            return Err("gliner tokens must be in source order".into());
        }
        let units = sentence_units(view, tokens, sentences)?;
        let mut entities = Entities {
            spans: Spans::new(view.id()),
            tokens: Vec::new(),
            kind: Vec::new(),
            kinds: kinds.clone(),
            score: Vec::new(),
        };
        let mut decisions = Decisions { logits: Vec::new(), threshold: schema.threshold() };
        for unit in units {
            if unit.is_empty() {
                continue;
            }
            let mut words = Vec::with_capacity(unit.len());
            for i in unit.clone() {
                let word = view
                    .text()
                    .get(starts[i] as usize..ends[i] as usize)
                    .ok_or_else(|| format!("token {i}: UTF-8 bounds"))?;
                words.push(word);
            }
            let candidates = decode(self.score_words(&words, label_states)?, schema);
            for c in candidates {
                let start = unit.start + c.start;
                let end = unit.start + c.end;
                entities.spans.push(Span::new(starts[start], ends[end - 1]));
                entities.tokens.push(IxRange::new(start as u32, end as u32));
                entities.kind.push(Tag(c.label as u16));
                entities.score.push(c.score as f64);
                decisions.logits.push(c.logit);
            }
        }
        Ok((entities, decisions))
    }
}

fn is_bundle_artifact(a: &turbo_zoo::Artifact) -> bool {
    a.format == Format::Safetensors
        && matches!(a.kind, Kind::Model | Kind::Head)
        && a.files.iter().any(|f| f.path.as_str() == "manifest.json")
}

fn find_open_span_head<'a>(bundle: &'a Bundle, name: Option<&str>) -> Result<&'a HeadSpec, String> {
    if let Some(name) = name {
        let h = bundle
            .manifest
            .heads
            .iter()
            .find(|h| h.name == name)
            .ok_or_else(|| format!("bundle has no head named {name:?}"))?;
        if h.kind != HeadKind::OpenSpan {
            return Err(format!("head {:?} is {:?}, not HEAD_OPEN_SPAN", h.name, h.kind));
        }
        return Ok(h);
    }
    let mut matching = bundle.manifest.heads.iter().filter(|h| h.kind == HeadKind::OpenSpan);
    let first = matching.next().ok_or_else(|| "bundle has no HEAD_OPEN_SPAN head".to_string())?;
    if matching.next().is_some() {
        return Err("bundle has multiple HEAD_OPEN_SPAN heads; pass a head name".into());
    }
    Ok(first)
}

fn sentence_units(view: &View, tokens: &Tokens, sentences: Option<&Sentences>) -> Result<Vec<Range<usize>>, String> {
    let (starts, ends) = tokens.spans.columns();
    let Some(sentences) = sentences else {
        return Ok((!starts.is_empty()).then_some(0..starts.len()).into_iter().collect());
    };
    sentences.check()?;
    if sentences.spans.view() != view.id() || sentences.spans.fragment_columns().is_some() {
        return Err("gliner needs contiguous sentences in its input view".into());
    }
    let (a, b) = sentences.spans.columns();
    let mut at = 0;
    let mut units = Vec::with_capacity(a.len());
    for (sentence, (&start, &end)) in a.iter().zip(b).enumerate() {
        if view.text().get(start as usize..end as usize).is_none() {
            return Err(format!("sentence {sentence}: UTF-8 bounds"));
        }
        let first = at;
        while at < starts.len() && starts[at] < end {
            if starts[at] < start || ends[at] > end {
                return Err(format!("token {at}: outside sentence {sentence}"));
            }
            if tokens.sentence.as_ref().is_some_and(|s| s[at].get() != sentence) {
                return Err(format!("token {at}: sentence index mismatch at sentence {sentence}"));
            }
            at += 1;
        }
        units.push(first..at);
    }
    if at != starts.len() {
        return Err(format!("token {at}: outside supplied sentences"));
    }
    Ok(units)
}
