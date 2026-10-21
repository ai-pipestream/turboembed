//! GLiNER2's relation extraction on the native DeBERTa-v3 encoder, as a
//! [`RelationScorer`](crate::RelationScorer) and on its own.
//!
//! [`Gliner2Relations`] reads a GLiNER2 checkpoint (`fastino/gliner2-base-v1`
//! is Apache-2.0) as two parts: the encoder as `turbo-bundle
//! deberta-convert` packages it, and the checkpoint's own directory for the
//! relation heads, the SentencePiece model and the special tokens. One
//! encoder pass reads every relation label and the text together:
//!
//! 1. the input (`prompt`): a schema per label, then the text's words;
//! 2. the encoder's output at each word's first subword, and at each
//!    label's `[P]` and two `[R]` markers;
//! 3. per label, the count head's number of instances K and K head and
//!    tail queries (`heads`);
//! 4. per query, a logit for each span of up to `max_width` words.
//!
//! [`Gliner2Relations::extract`] decodes as the reference does: for each
//! instance, the head and the tail are each the span of highest score
//! (the logistic of its logit) at or above the threshold, ties to the
//! earliest start and then the shortest; an instance with both is a
//! relation. As a scorer, a pair of mentions under a label scores the best,
//! over the label's instances, of the lower of its head span's and its
//! tail span's scores; a mention whose span is not a run of whole words
//! no longer than `max_width` scores 0.

pub mod heads;
mod kernel;
pub mod prompt;
pub mod words;

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbo::deberta::Encoder;
use turbo::deberta::package::{PackageManifest, digest};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::{ModelRef, Span};

use crate::zeroshot::{Mention, RelationError, RelationScorer};
use heads::{Heads, sigmoid};
use prompt::{Prompt, Specials};

/// One end of an extracted relation: its byte range in the text as
/// normalized (the caller's text, with a final `.` added when it lacks
/// sentence punctuation) and its score.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Found {
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

/// One extracted relation: the label's index, its head and its tail.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Extracted {
    pub label: usize,
    pub head: Found,
    pub tail: Found,
}

/// What one pass produced for one label: its instance count and, per
/// instance, the head and tail logits over the spans.
pub struct LabelScores {
    pub count: usize,
    /// `[count][2][words][max_width]` flattened.
    pub logits: Vec<f32>,
}

/// One pass over a text and its labels.
pub struct Pass {
    pub prompt: Prompt,
    pub labels: Vec<LabelScores>,
}

/// A GLiNER2 checkpoint's relation extraction on the native encoder.
pub struct Gliner2Relations {
    encoder: Encoder,
    heads: Heads,
    tokenizer: SentencePiece,
    specials: Specials,
    model: ModelRef,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

impl Gliner2Relations {
    /// The model from a converted encoder package and the checkpoint's
    /// directory (`model.safetensors`, `config.json`, `added_tokens.json`,
    /// `spm.model`). `id` and `revision` name the checkpoint; its SHA-256
    /// is computed here.
    pub fn load(encoder_package: &Path, checkpoint: &Path, id: &str, revision: &str) -> Result<Self, String> {
        let config = read_json(&checkpoint.join("config.json"))?;
        let field = |k: &str| config.get(k).and_then(Value::as_str);
        if field("counting_layer") != Some("count_lstm_v2") {
            return Err(format!("GLiNER2 counting_layer {:?}: only count_lstm_v2 is read", field("counting_layer")));
        }
        if field("token_pooling").is_some_and(|p| p != "first") {
            return Err(format!("GLiNER2 token_pooling {:?}: only first is read", field("token_pooling")));
        }
        let max_width = config.get("max_width").and_then(Value::as_u64).unwrap_or(8) as usize;
        let (_, encoder) = PackageManifest::load(encoder_package).map_err(|e| e.message)?;
        let bytes = fs::read(checkpoint.join("model.safetensors")).map_err(|e| e.to_string())?;
        let file = File::parse("GLiNER2 checkpoint", &bytes).map_err(|e| e.message)?;
        let heads = Heads::load(&file, encoder.config().hidden, max_width)?;
        let model = ModelRef { id: id.to_owned(), revision: revision.to_owned(), sha256: digest(&bytes) };
        let added = read_json(&checkpoint.join("added_tokens.json"))?;
        let id_of = |t: &str| {
            added.get(t).and_then(Value::as_u64).map(|v| v as u32).ok_or_else(|| format!("added_tokens.json: no {t}"))
        };
        let specials = Specials {
            sep_struct: id_of("[SEP_STRUCT]")?,
            sep_text: id_of("[SEP_TEXT]")?,
            p: id_of("[P]")?,
            r: id_of("[R]")?,
        };
        let spm = fs::read(checkpoint.join("spm.model")).map_err(|e| e.to_string())?;
        let tokenizer = SentencePiece::parse(&spm).map_err(|e| e.message)?;
        Self::new(encoder, heads, tokenizer, specials, model)
    }

    /// The model from its parts; the special tokens must be inside the
    /// encoder's vocabulary.
    pub fn new(
        encoder: Encoder,
        heads: Heads,
        tokenizer: SentencePiece,
        specials: Specials,
        model: ModelRef,
    ) -> Result<Self, String> {
        let v = encoder.config().vocabulary as u32;
        if [specials.sep_struct, specials.sep_text, specials.p, specials.r].iter().any(|&t| t >= v) {
            return Err("a GLiNER2 special token outside the encoder vocabulary".into());
        }
        if heads.width != encoder.config().hidden {
            return Err("GLiNER2 heads and encoder widths differ".into());
        }
        Ok(Gliner2Relations { encoder, heads, tokenizer, specials, model })
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    /// The input for `labels` over `text`, without running the model.
    pub fn prompt(&self, text: &str, labels: &[&str]) -> Prompt {
        prompt::build(&self.tokenizer, &self.specials, text, labels)
    }

    /// One encoder pass and the heads over `text` and `labels`.
    pub fn pass(&self, text: &str, labels: &[&str]) -> Result<Pass, RelationError> {
        let prompt = self.prompt(text, labels);
        self.pass_prompt(prompt)
    }

    /// The heads over a built input.
    pub fn pass_prompt(&self, prompt: Prompt) -> Result<Pass, RelationError> {
        run(&self.encoder, &self.heads, prompt)
    }

    /// Relations as the reference extracts them, label by label in the
    /// order given, instance by instance.
    pub fn extract(&self, text: &str, labels: &[&str], threshold: f32) -> Result<Vec<Extracted>, RelationError> {
        let pass = self.pass(text, labels)?;
        Ok(decode(&pass, self.heads.max_width, threshold))
    }
}

/// The encoder and the heads over a built input. The parity tests give
/// inputs whose token ids are not a tokenizer's.
pub fn run(encoder: &Encoder, heads: &Heads, prompt: Prompt) -> Result<Pass, RelationError> {
    let c = encoder.config();
    if prompt.ids.len() > c.max_sequence {
        let (max, got) = (c.max_sequence, prompt.ids.len());
        return Err(RelationError::OverLimit { limit: "encoder max_sequence", max, got });
    }
    let t = prompt.ids.len();
    if prompt.word_first.iter().chain(prompt.markers.iter().flatten()).any(|&p| p >= t) {
        return Err(RelationError::Scorer("a word or marker past the end of the input".into()));
    }
    let mask = vec![true; t];
    let hidden = encoder.encode(&prompt.ids, &mask).map_err(|e| RelationError::Scorer(e.message))?;
    let (data, d) = (hidden.data(), hidden.width());
    let row = |p: usize| &data[p * d..(p + 1) * d];
    let words: Vec<f32> = prompt.word_first.iter().flat_map(|&p| row(p).iter().copied()).collect();
    let spans = heads.spans(&words, prompt.word_first.len());
    let labels = prompt
        .markers
        .iter()
        .map(|&[p, h, tl]| {
            let count = heads.count(row(p));
            let queries = heads.queries([row(h), row(tl)], count);
            let logits = queries.chunks_exact(d).flat_map(|q| heads.logits(&spans, q)).collect();
            LabelScores { count, logits }
        })
        .collect();
    Ok(Pass { prompt, labels })
}

/// The best span of one query's logits at or above `threshold`: highest
/// score, ties to the earliest start, then the shortest.
fn best(logits: &[f32], width: usize, threshold: f32) -> Option<(usize, usize, f32)> {
    let mut out: Option<(usize, usize, f32)> = None;
    for (k, &l) in logits.iter().enumerate() {
        // Spans past the last word are negative infinity, never kept.
        if !l.is_finite() {
            continue;
        }
        let s = sigmoid(l);
        if s >= threshold && out.is_none_or(|b| s > b.2) {
            out = Some((k / width, k % width, s));
        }
    }
    out
}

/// The relations of a pass at `threshold`.
pub fn decode(pass: &Pass, width: usize, threshold: f32) -> Vec<Extracted> {
    let words = &pass.prompt.words;
    let size = words.len() * width;
    let found = |(i, w, score): (usize, usize, f32)| Found { start: words[i].start, end: words[i + w].end, score };
    let mut out = Vec::new();
    for (label, s) in pass.labels.iter().enumerate() {
        for inst in 0..s.count {
            let at = |f: usize| &s.logits[(inst * 2 + f) * size..(inst * 2 + f + 1) * size];
            if let (Some(h), Some(t)) = (best(at(0), width, threshold), best(at(1), width, threshold)) {
                out.push(Extracted { label, head: found(h), tail: found(t) });
            }
        }
    }
    out
}

/// The span index of `span` among the words: its first and last whole
/// word, when they are no more than `width` words.
fn span_of(words: &[words::Word], span: Span, width: usize) -> Option<usize> {
    let first = words.iter().position(|w| w.start >= span.start as usize && w.end <= span.end as usize)?;
    let last = words.iter().rposition(|w| w.start >= span.start as usize && w.end <= span.end as usize)?;
    (last - first < width).then_some(first * width + last - first)
}

impl RelationScorer for Gliner2Relations {
    fn model_ref(&self) -> Option<ModelRef> {
        Some(self.model.clone())
    }

    fn max_labels(&self) -> usize {
        64
    }

    fn max_mentions(&self) -> usize {
        usize::MAX
    }

    fn score(
        &self,
        text: &str,
        mentions: &[Mention<'_>],
        pairs: &[(u32, u32)],
        labels: &[&str],
        out: &mut Vec<f32>,
    ) -> Result<(), RelationError> {
        let pass = self.pass(text, labels)?;
        let width = self.heads.max_width;
        let size = pass.prompt.words.len() * width;
        let at: Vec<Option<usize>> = mentions.iter().map(|m| span_of(&pass.prompt.words, m.span, width)).collect();
        for &(h, t) in pairs {
            for s in &pass.labels {
                let (Some(hs), Some(ts)) = (at[h as usize], at[t as usize]) else {
                    out.push(0.0);
                    continue;
                };
                let best = (0..s.count)
                    .map(|k| {
                        let head = sigmoid(s.logits[(k * 2) * size + hs]);
                        let tail = sigmoid(s.logits[(k * 2 + 1) * size + ts]);
                        head.min(tail)
                    })
                    .fold(0.0f32, f32::max);
                out.push(best);
            }
        }
        Ok(())
    }
}
