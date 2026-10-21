//! GLiREL's zero-shot relation classification on the native DeBERTa-v3
//! encoder, as a [`RelationScorer`](crate::RelationScorer) and on its own.
//!
//! GLiREL (`jackboyla/glirel-large-v0`, CC BY-NC-SA 4.0) takes the
//! entities as given and scores every ordered pair of them against each
//! label. The checkpoint is a `torch.save` file; `turbo-bundle
//! glirel-convert` reads it as data and writes a package: the encoder as a
//! native DeBERTa package, the heads as safetensors and `glirel.json`. This
//! module reads only that package and the tokenizer's `spm.model`.
//!
//! One encoder pass reads every label and the text (`prompt`); the heads
//! (`heads`) project the rows read, run the LSTM over the text's words,
//! build each entity from its first and last word and each pair from its
//! two entities, and score a pair against a label by the dot product with
//! the label's representation. A pair's score is the logistic of that
//! logit. As a scorer, a mention whose span does not start and end on word
//! boundaries scores 0.

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
use heads::Heads;
use prompt::{Prompt, Specials};
use words::Word;

/// One encoder pass and the heads that do not depend on the entities.
pub struct Pass {
    pub prompt: Prompt,
    /// The LSTM's row for each word, `[words][H]`.
    pub lstm: Vec<f32>,
    /// Each label's representation, `[labels][H]`.
    pub types: Vec<f32>,
}

/// One relation at or above a threshold: its head and tail entities (indexes
/// into the entities given), its label's index and its score.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Extracted {
    pub head: usize,
    pub tail: usize,
    pub label: usize,
    pub score: f32,
}

/// A GLiREL package on the native encoder.
pub struct GlirelRelations {
    encoder: Encoder,
    heads: Heads,
    tokenizer: SentencePiece,
    specials: Specials,
    model: ModelRef,
    max_words: usize,
}

fn bad(what: impl std::fmt::Display) -> RelationError {
    RelationError::Scorer(format!("GLiREL input: {what}"))
}

impl GlirelRelations {
    /// The model from a package `turbo-bundle glirel-convert` wrote and the
    /// tokenizer it was written with.
    pub fn load(package: &Path, spm: &Path) -> Result<Self, String> {
        let read = |p: &Path| fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
        let meta: Value = serde_json::from_slice(&read(&package.join("glirel.json"))?).map_err(|e| e.to_string())?;
        let field = |k: &str| meta.get(k).ok_or_else(|| format!("glirel.json: no {k}"));
        let text = |k: &str| field(k)?.as_str().map(str::to_owned).ok_or_else(|| format!("glirel.json: {k}"));
        let id =
            |k: &str| field(k)?.as_u64().and_then(|v| u32::try_from(v).ok()).ok_or_else(|| format!("glirel.json: {k}"));
        if meta.get("version").and_then(Value::as_u64) != Some(1) {
            return Err("glirel.json: version 1 only".into());
        }
        let heads_bytes = read(&package.join("heads.safetensors"))?;
        if digest(&heads_bytes) != text("heads_sha256")? {
            return Err("GLiREL heads do not match their recorded digest".into());
        }
        let (_, encoder) = PackageManifest::load(&package.join("encoder")).map_err(|e| e.message)?;
        let heads =
            Heads::load(&File::parse("GLiREL heads", &heads_bytes).map_err(|e| e.message)?, encoder.config().hidden)?;
        let tokenizer = SentencePiece::parse(&read(spm)?).map_err(|e| e.message)?;
        if digest(&read(spm)?) != text("spm_sha256")? {
            return Err("spm.model is not the one the package was written with".into());
        }
        let specials = Specials { cls: id("cls")?, sep: id("sep")?, rel: id("rel")? };
        let model = ModelRef { id: text("repository")?, revision: text("revision")?, sha256: text("sha256")? };
        let max_words = field("max_words")?.as_u64().ok_or("glirel.json: max_words")? as usize;
        Self::new(encoder, heads, tokenizer, specials, model, max_words)
    }

    /// The model from its parts; the special tokens must be inside the
    /// encoder's vocabulary and the heads must read its width.
    pub fn new(
        encoder: Encoder,
        heads: Heads,
        tokenizer: SentencePiece,
        specials: Specials,
        model: ModelRef,
        max_words: usize,
    ) -> Result<Self, String> {
        let v = encoder.config().vocabulary as u32;
        if [specials.cls, specials.sep, specials.rel].iter().any(|&t| t >= v) {
            return Err("a GLiREL special token outside the encoder vocabulary".into());
        }
        if heads.encoder_width != encoder.config().hidden {
            return Err("GLiREL heads and encoder widths differ".into());
        }
        Ok(GlirelRelations { encoder, heads, tokenizer, specials, model, max_words })
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    /// The input for `labels` over `text`, without running the model.
    pub fn prompt(&self, text: &str, labels: &[&str]) -> Result<Prompt, RelationError> {
        let p = prompt::build(&self.tokenizer, &self.specials, text, labels)
            .ok_or_else(|| bad("a label with no subwords"))?;
        if p.words.len() > self.max_words {
            return Err(RelationError::OverLimit { limit: "GLiREL max_len", max: self.max_words, got: p.words.len() });
        }
        Ok(p)
    }

    /// One encoder pass and the entity-free heads over `text` and `labels`.
    pub fn pass(&self, text: &str, labels: &[&str]) -> Result<Pass, RelationError> {
        run(&self.encoder, &self.heads, self.prompt(text, labels)?)
    }

    /// Each pair's logit for each label, `[pairs][labels]`, for entities
    /// given as inclusive word ranges.
    pub fn logits(
        &self,
        pass: &Pass,
        entities: &[(usize, usize)],
        pairs: &[(usize, usize)],
    ) -> Result<Vec<f32>, RelationError> {
        let n = pass.prompt.words.len();
        if entities.iter().any(|&(s, e)| s > e || e >= n)
            || pairs.iter().any(|&(a, b)| a >= entities.len() || b >= entities.len())
        {
            return Err(bad("an entity outside the words or a pair outside the entities"));
        }
        let reps = self.heads.entities(&pass.lstm, entities);
        Ok(self.heads.logits(&reps, pairs, &pass.types))
    }

    /// Relations as the reference predicts them: every ordered pair of
    /// distinct entities (inclusive word ranges), each label whose score is
    /// above `threshold`, pair by pair in the reference's order.
    pub fn extract(
        &self,
        text: &str,
        entities: &[(usize, usize)],
        labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<Extracted>, RelationError> {
        let pass = self.pass(text, labels)?;
        let pairs: Vec<(usize, usize)> = (0..entities.len())
            .flat_map(|a| (0..entities.len()).filter(move |&b| b != a).map(move |b| (a, b)))
            .collect();
        let logits = self.logits(&pass, entities, &pairs)?;
        let m = labels.len();
        let mut out = Vec::new();
        for (p, &(head, tail)) in pairs.iter().enumerate() {
            for (label, &l) in logits[p * m..(p + 1) * m].iter().enumerate() {
                let score = sigmoid(l);
                if score > threshold {
                    out.push(Extracted { head, tail, label, score });
                }
            }
        }
        Ok(out)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// The encoder and the entity-free heads over a built input. The parity
/// tests give inputs whose token ids are not a tokenizer's.
pub fn run(encoder: &Encoder, heads: &Heads, prompt: Prompt) -> Result<Pass, RelationError> {
    let c = encoder.config();
    let t = prompt.ids.len();
    if t > c.max_sequence {
        return Err(RelationError::OverLimit { limit: "encoder max_sequence", max: c.max_sequence, got: t });
    }
    if prompt.words.len() != prompt.word_first.len()
        || prompt.word_first.iter().flatten().chain(prompt.labels.iter().flatten()).any(|&p| p >= t)
    {
        return Err(bad("a word or label position past the input"));
    }
    let hidden = encoder.encode(&prompt.ids, &vec![true; t]).map_err(|e| RelationError::Scorer(e.message))?;
    let (data, e) = (hidden.data(), hidden.width());
    // A word with no subwords reads a zero row, as the reference fills it.
    let zero = vec![0.0; e];
    let row = |p: Option<usize>| p.map_or(&zero[..], |p| &data[p * e..(p + 1) * e]);
    let rows: Vec<f32> = prompt
        .word_first
        .iter()
        .map(|&p| row(p))
        .chain(prompt.labels.iter().flat_map(|&[r, l]| [row(Some(r)), row(Some(l))]))
        .flat_map(|r| r.iter().copied())
        .collect();
    let (n, m, h) = (prompt.words.len(), prompt.labels.len(), heads.width);
    let projected = heads.project(&rows, n + 2 * m);
    let lstm = heads.lstm(&projected[..n * h], n);
    let means: Vec<f32> = projected[n * h..]
        .chunks_exact(2 * h)
        .flat_map(|pair| pair[..h].iter().zip(&pair[h..]).map(|(a, b)| (a + b) / 2.0))
        .collect();
    let types = heads.types(&means, m);
    Ok(Pass { prompt, lstm, types })
}

/// The inclusive word range of `span`: it starts where a word starts and
/// ends where a word ends. Anything else, part of a word included, is none.
pub fn words_of(words: &[Word], span: Span) -> Option<(usize, usize)> {
    let first = words.iter().position(|w| w.start == span.start as usize)?;
    let last = words.iter().position(|w| w.end == span.end as usize)?;
    (first <= last).then_some((first, last))
}

impl RelationScorer for GlirelRelations {
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
        let at: Vec<Option<(usize, usize)>> = mentions.iter().map(|m| words_of(&pass.prompt.words, m.span)).collect();
        // Only the mentions that are word runs become entities.
        let mut index = vec![usize::MAX; mentions.len()];
        let mut entities = Vec::new();
        for (i, a) in at.iter().enumerate() {
            if let Some(range) = a {
                index[i] = entities.len();
                entities.push(*range);
            }
        }
        let scored: Vec<(usize, usize)> = pairs
            .iter()
            .filter_map(|&(h, t)| {
                let (a, b) = (index[h as usize], index[t as usize]);
                (a != usize::MAX && b != usize::MAX).then_some((a, b))
            })
            .collect();
        let logits = self.logits(&pass, &entities, &scored)?;
        let m = labels.len();
        let mut next = logits.chunks_exact(m.max(1));
        for &(h, t) in pairs {
            if index[h as usize] == usize::MAX || index[t as usize] == usize::MAX {
                out.extend(std::iter::repeat_n(0.0, m));
            } else {
                out.extend(next.next().expect("one row per scored pair").iter().map(|&l| sigmoid(l)));
            }
        }
        Ok(())
    }
}
