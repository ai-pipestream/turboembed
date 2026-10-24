//! A GLiNER2 checkpoint on the native encoder, run over a schema prompt:
//! the encoder once, then for each span task its instance count, its
//! queries over the task's fields and their span logits over every text
//! token, and for each classification the classifier on its label markers.
//!
//! The span heads are the relation scorer's (`turbo_relation::gliner2`),
//! with the count transformer attending across all of a task's fields. The
//! classifier is GLiNER2's `classifier`: Linear(D, 2D), ReLU, Linear(2D, 1)
//! on each `[L]` vector.

use std::fs;
use std::path::Path;

use serde_json::Value;
use turbo::deberta::Encoder;
use turbo::deberta::package::{PackageManifest, digest};
use turbo::head::Linear;
use turbo::safetensors::{Dtype, File};
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::ModelRef;
use turbo_relation::gliner2::heads::Heads;

use crate::prompt::{Kind, Prompt, Specials, build};
use crate::schema::{ExtractionSchema, Limits, SchemaError};

/// GLiNER2's classification head.
pub struct Classifier {
    first: Linear,
    last: Linear,
}

fn linear(f: &File, name: &str) -> Result<Linear, String> {
    let w = f.get(&format!("{name}.weight"), Dtype::F32, 2).map_err(|e| e.message)?;
    let b = f.get(&format!("{name}.bias"), Dtype::F32, 1).map_err(|e| e.message)?;
    Linear::new(w.shape[1] as usize, w.shape[0] as usize, w.f32s(), b.f32s())
        .map_err(|e| format!("{name}: {}", e.message))
}

impl Classifier {
    /// The head from a checkpoint's `classifier.0` and `classifier.2`,
    /// checked against the encoder width.
    pub fn load(f: &File, width: usize) -> Result<Classifier, String> {
        let (first, last) = (linear(f, "classifier.0")?, linear(f, "classifier.2")?);
        if first.in_dim != width || last.in_dim != first.out_dim || last.out_dim != 1 {
            return Err("GLiNER2 classifier widths do not match the encoder width".into());
        }
        Ok(Classifier { first, last })
    }

    /// The logit of one label vector.
    ///
    /// # Panics
    /// When `row` is not one row of the model width.
    pub fn logit(&self, row: &[f32]) -> f32 {
        let mut hidden = vec![0.0; self.first.out_dim];
        self.first.apply(row, &mut hidden);
        hidden.iter_mut().for_each(|x| *x = x.max(0.0));
        let mut out = [0.0];
        self.last.apply(&hidden, &mut out);
        out[0]
    }
}

/// What the heads gave one task.
#[derive(Clone, Debug, PartialEq)]
pub enum Scores {
    /// A record type or the entity types: the predicted instance count and,
    /// per instance kept and per field, the span logits over the text
    /// tokens, `[instances][fields][tokens][width]` flattened, with spans
    /// past the last token at negative infinity. Entities keep instance 0.
    Spans { count: usize, instances: usize, logits: Vec<f32> },
    /// A classification: one logit per label.
    Labels(Vec<f32>),
}

/// One pass over a text and its schema.
pub struct Pass {
    prompt: Prompt,
    width: usize,
    scores: Vec<Scores>,
}

impl Pass {
    /// A pass from its parts, for a runner other than [`run`]. Nothing is
    /// checked here: [`crate::decode`] checks a pass against its schema
    /// before it reads one score.
    pub fn new(prompt: Prompt, width: usize, scores: Vec<Scores>) -> Pass {
        Pass { prompt, width, scores }
    }

    pub fn prompt(&self) -> &Prompt {
        &self.prompt
    }

    /// The span width the logits were computed at.
    pub fn width(&self) -> usize {
        self.width
    }

    /// One per task of the prompt, in its order.
    pub fn scores(&self) -> &[Scores] {
        &self.scores
    }

    /// The prompt, width and scores, for a caller that builds a pass of its
    /// own from them.
    pub fn into_parts(self) -> (Prompt, usize, Vec<Scores>) {
        (self.prompt, self.width, self.scores)
    }
}

/// Why a pass failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ExtractError {
    Schema(SchemaError),
    /// A request over a named limit after tokenization.
    OverLimit {
        limit: &'static str,
        max: usize,
        got: usize,
    },
    /// The model refused its input.
    Model(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Schema(e) => write!(f, "schema: {e}"),
            ExtractError::OverLimit { limit, max, got } => write!(f, "{limit} is {max}, the request has {got}"),
            ExtractError::Model(m) => write!(f, "model: {m}"),
        }
    }
}

impl std::error::Error for ExtractError {}

/// A GLiNER2 checkpoint's schema-driven extraction on the native encoder.
pub struct Gliner2Extractor {
    encoder: Encoder,
    heads: Heads,
    classifier: Classifier,
    tokenizer: SentencePiece,
    specials: Specials,
    model: ModelRef,
    limits: Limits,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

impl Gliner2Extractor {
    /// The model from a converted encoder package and the checkpoint's
    /// directory, as `Gliner2Relations::load` reads them.
    pub fn load(encoder_package: &Path, checkpoint: &Path, id: &str, revision: &str) -> Result<Self, String> {
        let config = read_json(&checkpoint.join("config.json"))?;
        let field = |k: &str| config.get(k).and_then(Value::as_str);
        if field("counting_layer") != Some("count_lstm_v2") || field("token_pooling").is_some_and(|p| p != "first") {
            return Err("GLiNER2: only count_lstm_v2 counting and first-subword pooling are read".into());
        }
        let max_width = config.get("max_width").and_then(Value::as_u64).unwrap_or(8) as usize;
        let (_, encoder) = PackageManifest::load(encoder_package).map_err(|e| e.message)?;
        let bytes = fs::read(checkpoint.join("model.safetensors")).map_err(|e| e.to_string())?;
        let file = File::parse("GLiNER2 checkpoint", &bytes).map_err(|e| e.message)?;
        let hidden = encoder.config().hidden;
        let heads = Heads::load(&file, hidden, max_width)?;
        let classifier = Classifier::load(&file, hidden)?;
        let model = ModelRef { id: id.to_owned(), revision: revision.to_owned(), sha256: digest(&bytes) };
        let added = read_json(&checkpoint.join("added_tokens.json"))?;
        let id_of = |t: &str| {
            added.get(t).and_then(Value::as_u64).map(|v| v as u32).ok_or_else(|| format!("added_tokens.json: no {t}"))
        };
        let specials = Specials {
            sep_struct: id_of("[SEP_STRUCT]")?,
            sep_text: id_of("[SEP_TEXT]")?,
            p: id_of("[P]")?,
            c: id_of("[C]")?,
            e: id_of("[E]")?,
            l: id_of("[L]")?,
            description: id_of("[DESCRIPTION]")?,
        };
        let tokenizer = SentencePiece::parse(&fs::read(checkpoint.join("spm.model")).map_err(|e| e.to_string())?)
            .map_err(|e| e.message)?;
        Self::new(encoder, heads, classifier, tokenizer, specials, model, Limits::default())
    }

    /// The model from its parts; the special tokens must be inside the
    /// encoder's vocabulary.
    pub fn new(
        encoder: Encoder,
        heads: Heads,
        classifier: Classifier,
        tokenizer: SentencePiece,
        specials: Specials,
        model: ModelRef,
        limits: Limits,
    ) -> Result<Self, String> {
        let v = encoder.config().vocabulary as u32;
        let s = &specials;
        if [s.sep_struct, s.sep_text, s.p, s.c, s.e, s.l, s.description].iter().any(|&t| t >= v) {
            return Err("a GLiNER2 special token outside the encoder vocabulary".into());
        }
        if heads.width != encoder.config().hidden || classifier.first.in_dim != heads.width {
            return Err("GLiNER2 heads and encoder widths differ".into());
        }
        Ok(Gliner2Extractor { encoder, heads, classifier, tokenizer, specials, model, limits })
    }

    pub fn model(&self) -> &ModelRef {
        &self.model
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// The widest span, in words, the span heads score.
    pub fn max_width(&self) -> usize {
        self.heads.max_width
    }

    /// The input for `schema` over `text`, checked against every limit,
    /// without running the model.
    pub fn prompt(&self, text: &str, schema: &ExtractionSchema) -> Result<Prompt, ExtractError> {
        schema.validate(&self.limits).map_err(ExtractError::Schema)?;
        let p = build(&self.tokenizer, &self.specials, text, schema);
        let l = &self.limits;
        // The caller's words: not the `.` the model's reading may add.
        let words = p.words.iter().filter(|w| w.start < text.len()).count();
        if words > l.max_words {
            return Err(ExtractError::OverLimit { limit: "max_words", max: l.max_words, got: words });
        }
        let schema_len = p.token_first.first().copied().unwrap_or(p.ids.len());
        if schema_len > l.max_prompt_subwords {
            return Err(ExtractError::OverLimit {
                limit: "max_prompt_subwords",
                max: l.max_prompt_subwords,
                got: schema_len,
            });
        }
        let max = self.encoder.config().max_sequence;
        if p.ids.len() > max {
            return Err(ExtractError::OverLimit { limit: "encoder max_sequence", max, got: p.ids.len() });
        }
        Ok(p)
    }

    /// One encoder pass and every head over `text` and `schema`.
    pub fn pass(&self, text: &str, schema: &ExtractionSchema) -> Result<Pass, ExtractError> {
        run(&self.encoder, &self.heads, &self.classifier, self.prompt(text, schema)?)
    }
}

/// The encoder and every head over a built input. The parity tests give
/// inputs whose token ids are not a tokenizer's.
pub fn run(encoder: &Encoder, heads: &Heads, classifier: &Classifier, prompt: Prompt) -> Result<Pass, ExtractError> {
    let t = prompt.ids.len();
    let positions =
        prompt.token_first.iter().chain(prompt.tasks.iter().flat_map(|k| std::iter::once(&k.p).chain(&k.markers)));
    if positions.into_iter().any(|&p| p >= t) {
        return Err(ExtractError::Model("a text token or marker past the end of the input".into()));
    }
    let hidden = encoder.encode(&prompt.ids, &vec![true; t]).map_err(|e| ExtractError::Model(e.message))?;
    let (data, d) = (hidden.data(), hidden.width());
    let row = |p: usize| &data[p * d..(p + 1) * d];
    let n = prompt.token_first.len();
    let rows: Vec<f32> = prompt.token_first.iter().flat_map(|&p| row(p).iter().copied()).collect();
    let spans = heads.spans(&rows, n);
    let mut scores = Vec::with_capacity(prompt.tasks.len());
    for task in &prompt.tasks {
        let fields: Vec<&[f32]> = task.markers.iter().map(|&m| row(m)).collect();
        scores.push(match task.kind {
            Kind::Classification(_) => Scores::Labels(fields.iter().map(|f| classifier.logit(f)).collect()),
            Kind::Record(_) | Kind::Entities => {
                let count = heads.count(row(task.p));
                // Entities decode instance 0 only, whose query does not
                // depend on the count.
                let instances = if task.kind == Kind::Entities { count.min(1) } else { count };
                let queries = heads.queries_of(&fields, instances);
                let logits = heads.logits_all(&spans, &queries);
                Scores::Spans { count, instances, logits }
            }
        });
    }
    Ok(Pass { prompt, width: heads.max_width, scores })
}
