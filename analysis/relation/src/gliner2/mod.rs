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
    /// The span width the logits were computed at.
    pub width: usize,
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
        decode(&pass, threshold)
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
    check_prompt(&prompt)?;
    let t = prompt.ids.len();
    let mask = vec![true; t];
    let hidden = encoder.encode(&prompt.ids, &mask).map_err(|e| RelationError::Scorer(e.message))?;
    let (data, d) = (hidden.data(), hidden.width());
    let row = |p: usize| &data[p * d..(p + 1) * d];
    let words: Vec<f32> = prompt.word_first.iter().flat_map(|&p| row(p).iter().copied()).collect();
    let mut scratch = crate::Scratch::default();
    let mut spans = heads::Spans::default();
    heads.spans_into(&words, prompt.word_first.len(), &mut scratch, &mut spans);
    let mut queries = Vec::new();
    let labels = prompt
        .markers
        .iter()
        .map(|&[p, h, tl]| {
            let count = heads.count(row(p));
            heads.queries_of_into(&[row(h), row(tl)], count, &mut scratch, &mut queries);
            let mut logits = Vec::new();
            heads.logits_all_into(&spans, &queries, &mut scratch, &mut logits);
            LabelScores { count, logits }
        })
        .collect();
    Ok(Pass { prompt, width: heads.max_width, labels })
}

fn bad(what: &str) -> RelationError {
    RelationError::Scorer(format!("GLiNER2 input: {what}"))
}

/// A prompt the heads and the decoder can index: a first subword for each
/// word, every position inside the input, and word ranges in order on
/// character boundaries of the text.
fn check_prompt(prompt: &Prompt) -> Result<(), RelationError> {
    let t = prompt.ids.len();
    if prompt.words.len() != prompt.word_first.len() {
        return Err(bad("words and first subwords differ in number"));
    }
    if prompt.word_first.iter().chain(prompt.markers.iter().flatten()).any(|&p| p >= t) {
        return Err(bad("a word or marker past the end of the input"));
    }
    let mut prev = 0;
    for w in &prompt.words {
        let on = |i: usize| prompt.text.is_char_boundary(i);
        if w.start < prev || w.end <= w.start || !on(w.start) || !on(w.end) {
            return Err(bad("word ranges out of order or off the text"));
        }
        prev = w.end;
    }
    Ok(())
}

/// A pass whose logits are the sizes its prompt and width give.
fn check_pass(pass: &Pass) -> Result<usize, RelationError> {
    check_prompt(&pass.prompt)?;
    let size =
        pass.prompt.words.len().checked_mul(pass.width).filter(|_| pass.width > 0).ok_or_else(|| bad("span width"))?;
    if pass.labels.len() != pass.prompt.markers.len() {
        return Err(bad("label scores and markers differ in number"));
    }
    for s in &pass.labels {
        if s.count.checked_mul(2).and_then(|c| c.checked_mul(size)) != Some(s.logits.len()) {
            return Err(bad("logits of another size than the count and spans give"));
        }
    }
    Ok(size)
}

/// The best span of one query's logits over `n` words at or above
/// `threshold`: highest score, ties to the earliest start, then the
/// shortest.
fn best(logits: &[f32], n: usize, width: usize, threshold: f32) -> Option<(usize, usize, f32)> {
    let mut out: Option<(usize, usize, f32)> = None;
    for (k, &l) in logits.iter().enumerate() {
        // A slot past the last word is no span, whatever it holds (the
        // heads put negative infinity there); nor is a non-finite logit.
        if k / width + k % width >= n || !l.is_finite() {
            continue;
        }
        let s = sigmoid(l);
        if s >= threshold && out.is_none_or(|b| s > b.2) {
            out = Some((k / width, k % width, s));
        }
    }
    out
}

/// The relations of a pass at `threshold`; a pass whose parts do not fit
/// together is an error.
pub fn decode(pass: &Pass, threshold: f32) -> Result<Vec<Extracted>, RelationError> {
    let size = check_pass(pass)?;
    let (words, width) = (&pass.prompt.words, pass.width);
    let found = |(i, w, score): (usize, usize, f32)| Found { start: words[i].start, end: words[i + w].end, score };
    let mut out = Vec::new();
    for (label, s) in pass.labels.iter().enumerate() {
        for inst in 0..s.count {
            let at = |f: usize| &s.logits[(inst * 2 + f) * size..(inst * 2 + f + 1) * size];
            if let (Some(h), Some(t)) =
                (best(at(0), words.len(), width, threshold), best(at(1), words.len(), width, threshold))
            {
                out.push(Extracted { label, head: found(h), tail: found(t) });
            }
        }
    }
    Ok(out)
}

/// The span index of `span` among the words: it starts where a word
/// starts and ends where a word ends, no more than `width` words on.
/// Anything else, part of a word included, is no span.
pub fn span_of(words: &[words::Word], span: Span, width: usize) -> Option<usize> {
    let first = words.iter().position(|w| w.start == span.start as usize)?;
    let last = words.iter().position(|w| w.end == span.end as usize)?;
    (first <= last && last - first < width).then_some(first * width + last - first)
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
