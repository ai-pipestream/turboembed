//! Production [`TokenVectors`] adapter (`#960`) over [`VectorAnnotator`]'s
//! per-token contextual outputs (`#709`), aligned to coreference word tokens
//! through the [`Subwords`] layer (`subwords::align` and `Subwords::of`).

use std::path::Path;
use std::sync::Arc;

use turbo::manifest::{PromptRole, Truncation};
use turbo::status::Error as CoreError;
use turbo::tokenizer::Encode;
use turbo_analysis::{Error, Execution, Layer, ModelRef, Span, Spans, ViewId};
use turbo_analysis_layers::{Sentences, Subwords, Tokens, Vectors};
use turbo_coref::TokenVectors;

use crate::encoder::normalize;
use crate::subwords::align;
use crate::vectors::{VectorAnnotator, VectorOptions};

const ANNOTATOR_ID: &str = "turbo-coref:token-vectors";
const CONFIG_TAG: &str = "turbo-embed:coref-token-vectors:1";

/// Contextual per-word-token vector provider (`turbo_coref::TokenVectors`)
/// backed by a [`VectorAnnotator`]'s per-subword contextual embeddings (`#709`),
/// aligned to word tokens through the [`Subwords`] layer (`#960`).
///
/// For each sentence (or document with `Tokens` and `Subwords`):
/// 1. Word tokens are aligned to subword pieces via [`crate::subwords::align`]
///    and indexed with [`Subwords::of`].
/// 2. Contextual subword vectors are extracted via
///    [`VectorAnnotator::embed_subword_tokens`] (`#709`), applying the bundle's
///    sliding-window interior selection and optional `ProjectionHead`.
/// 3. Each word token receives the componentwise `f64` mean of its aligned
///    subword vectors (normalized with [`normalize`] when the bundle normalizes
///    its vectors to unit length). When a byte-level BPE piece spans across
///    adjacent word tokens so `Subwords::of(token_ix)` is empty, the token
///    receives the mean of the subword pieces overlapping its byte span (or
///    the nearest piece when no piece overlaps).
#[derive(Clone)]
pub struct BundleTokenVectors {
    annotator: Arc<VectorAnnotator>,
}

impl std::fmt::Debug for BundleTokenVectors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BundleTokenVectors")
            .field("model", self.annotator.model())
            .field("execution", self.annotator.execution())
            .field("dim", &self.annotator.token_dim())
            .field("stride", &self.annotator.stride())
            .finish()
    }
}

impl BundleTokenVectors {
    /// Wraps an opened [`VectorAnnotator`] as a [`TokenVectors`] provider.
    pub fn new(annotator: Arc<VectorAnnotator>) -> Self {
        Self { annotator }
    }

    /// Opens the bundle at `dir` with `options` as a [`BundleTokenVectors`]
    /// provider.
    pub fn open(dir: &Path, options: &VectorOptions) -> Result<Self, CoreError> {
        let annotator = VectorAnnotator::open(dir, options)?;
        Ok(Self::new(Arc::new(annotator)))
    }

    /// Underlying [`VectorAnnotator`].
    pub fn annotator(&self) -> &Arc<VectorAnnotator> {
        &self.annotator
    }

    /// Bundle [`ModelRef`].
    pub fn model(&self) -> &ModelRef {
        self.annotator.model()
    }

    /// Backend [`Execution`] provenance.
    pub fn execution(&self) -> &Execution {
        self.annotator.execution()
    }

    /// Computes one contextual vector per word token of `tokens` by extracting
    /// per-subword vectors from `subwords` (`#709`) and pooling them per word
    /// token through `Subwords` alignment (`subwords::align` / `Subwords::of`).
    pub fn embed_document_tokens(
        &self,
        tokens: &Tokens,
        subwords: &Subwords,
        sentences: Option<&Spans<Sentences>>,
    ) -> Result<Vec<Vec<f32>>, Error> {
        if tokens.spans.view() != subwords.spans.view() {
            return Err(Error::Annotator {
                annotator: ANNOTATOR_ID,
                detail: format!(
                    "the tokens are in view {}, the subwords in view {}",
                    tokens.spans.view().0,
                    subwords.spans.view().0
                ),
            });
        }
        let ctx = self.annotator.embed_subword_tokens(ANNOTATOR_ID, subwords, sentences, self.annotator.stride())?;
        pool_word_tokens(&tokens.spans, subwords, &ctx.data, ctx.dim as usize, self.annotator.info().l2)
    }

    /// Fallible sentence-level contextual word-token encoding used by
    /// [`TokenVectors::vectors`].
    ///
    /// Joins `sentence` tokens with single ASCII spaces into a synthetic view,
    /// builds `Tokens` and `Subwords` with `subwords.token = align(&sub.spans, &tokens)`,
    /// extracts per-subword contextual vectors via [`VectorAnnotator::embed_subword_tokens`],
    /// and pools one vector per word token via [`pool_word_tokens`].
    pub fn try_vectors(&self, sentence: &[&str]) -> Result<Vec<Vec<f32>>, Error> {
        if sentence.is_empty() {
            return Ok(Vec::new());
        }
        let mut text = String::new();
        let mut token_spans = Spans::<Tokens>::with_capacity(ViewId::SOURCE, sentence.len());
        for (i, &word) in sentence.iter().enumerate() {
            if i > 0 {
                text.push(' ');
            }
            let start = text.len() as u32;
            text.push_str(word);
            let end = text.len() as u32;
            token_spans.push(Span::new(start, end));
        }

        let opts = Encode {
            add_special_tokens: false,
            truncation: Truncation::None,
            max_tokens: u32::MAX,
            prompt: PromptRole::None,
        };
        let (mut ids, mut raw_spans) = (Vec::new(), Vec::new());
        self.annotator
            .tokenizer()
            .encode_spans(&text, opts, &mut ids, &mut raw_spans)
            .map_err(|e| Error::Annotator { annotator: ANNOTATOR_ID, detail: e.to_string() })?;
        let mut sub_spans = Spans::<Subwords>::with_capacity(ViewId::SOURCE, raw_spans.len());
        for &[start, end] in &raw_spans {
            sub_spans.push(Span::new(start, end));
        }
        let token_col = align(&sub_spans, &token_spans);
        let subwords = Subwords { spans: sub_spans, ids, model: self.annotator.model().clone(), token: token_col };
        let ctx = self.annotator.embed_subword_tokens(ANNOTATOR_ID, &subwords, None, self.annotator.stride())?;
        pool_word_tokens(&token_spans, &subwords, &ctx.data, ctx.dim as usize, self.annotator.info().l2)
    }
}

impl TokenVectors for BundleTokenVectors {
    fn dimension(&self) -> usize {
        self.annotator.token_dim() as usize
    }

    fn vectors(&self, sentence: &[&str]) -> Vec<Vec<f32>> {
        self.try_vectors(sentence).unwrap_or_default()
    }

    fn config_hash(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for s in
            [CONFIG_TAG, &self.annotator.model().id, &self.annotator.model().revision, &self.annotator.model().sha256]
        {
            for &b in s.as_bytes() {
                h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
            }
            h = (h ^ 0xff).wrapping_mul(0x0100_0000_01b3);
        }
        for b in self.annotator.token_config().to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
        h
    }
}

/// Pools a precomputed `Vectors<Subwords>` layer (`#709`) into one vector per
/// word token of `tokens` using `Subwords` alignment (`subwords::align` /
/// `Subwords::of`).
pub fn pool_subword_vectors(
    tokens: &Tokens,
    subwords: &Subwords,
    vectors: &Vectors<Subwords>,
    l2: bool,
) -> Result<Vec<Vec<f32>>, Error> {
    pool_word_tokens(&tokens.spans, subwords, &vectors.data, vectors.dim as usize, l2)
}

/// Pools per-subword vectors `subword_vectors` (`[subwords.len(), dim]`) into
/// one `dim`-dimensional vector per word token of `tokens`, aligned through
/// `Subwords::of` (computing `subwords::align` when `subwords.token` is `None`).
pub fn pool_word_tokens(
    tokens: &Spans<Tokens>,
    subwords: &Subwords,
    subword_vectors: &[f32],
    dim: usize,
    l2: bool,
) -> Result<Vec<Vec<f32>>, Error> {
    if tokens.view() != subwords.spans.view() {
        return Err(Error::Annotator {
            annotator: ANNOTATOR_ID,
            detail: format!(
                "the tokens are in view {}, the subwords in view {}",
                tokens.view().0,
                subwords.spans.view().0
            ),
        });
    }
    if dim == 0 {
        return Err(Error::Annotator {
            annotator: ANNOTATOR_ID,
            detail: "token vector dimension must be positive".into(),
        });
    }
    let expected = subwords.len().checked_mul(dim).ok_or_else(|| Error::Annotator {
        annotator: ANNOTATOR_ID,
        detail: "subword vector buffer size overflow".into(),
    })?;
    if subword_vectors.len() != expected {
        return Err(Error::Annotator {
            annotator: ANNOTATOR_ID,
            detail: format!(
                "subword vector buffer has {} values, expected {} ({} subwords x {dim})",
                subword_vectors.len(),
                expected,
                subwords.len()
            ),
        });
    }
    if !tokens.is_ordered() {
        return Err(Error::Annotator {
            annotator: ANNOTATOR_ID,
            detail: "the tokens are not in text order, or overlap".into(),
        });
    }
    let (p_starts, p_ends) = subwords.spans.columns();
    if !p_starts.windows(2).all(|w| w[0] <= w[1])
        || !p_ends.windows(2).all(|w| w[0] <= w[1])
        || subwords.spans.has_discontinuous()
    {
        return Err(Error::Annotator { annotator: ANNOTATOR_ID, detail: "the subwords are not in text order".into() });
    }
    if let Some(t) = &subwords.token {
        if t.len() != subwords.spans.len() {
            return Err(Error::Annotator {
                annotator: ANNOTATOR_ID,
                detail: format!("a token for every subword: {} tokens for {} subwords", t.len(), subwords.spans.len()),
            });
        }
        if !t.windows(2).all(|w| w[0] <= w[1]) {
            return Err(Error::Annotator { annotator: ANNOTATOR_ID, detail: "subwords are not in token order".into() });
        }
        if let Some(last) = t.last()
            && last.get() >= tokens.len()
        {
            return Err(Error::Annotator {
                annotator: ANNOTATOR_ID,
                detail: format!("subword token index {} is out of bounds for {} tokens", last.get(), tokens.len()),
            });
        }
    }

    let aligned_view;
    let sub_ref = if subwords.token.is_some() {
        subwords
    } else {
        aligned_view = Subwords {
            spans: subwords.spans.clone(),
            ids: Vec::new(),
            model: subwords.model.clone(),
            token: align(&subwords.spans, tokens),
        };
        &aligned_view
    };

    let (p_starts, p_ends) = sub_ref.spans.columns();
    let mut out = Vec::with_capacity(tokens.len());
    for (t_ix, t_span) in tokens.iter() {
        if sub_ref.spans.is_empty() || sub_ref.token.is_none() {
            out.push(vec![0.0; dim]);
            continue;
        }
        let range = sub_ref.of(t_ix);
        let (mut a, mut b) = (range.start.get(), range.end.get());
        if a == b {
            // A byte-level BPE piece spanning two adjacent word tokens is
            // assigned by `align` to the first token; find overlapping pieces
            // for this token's byte span, or fall back to the nearest piece.
            let mut ov_start = None;
            let mut ov_end = 0usize;
            for p in 0..p_starts.len() {
                let (ps, pe) = (p_starts[p], p_ends[p]);
                if ps < t_span.end && t_span.start < pe {
                    if ov_start.is_none() {
                        ov_start = Some(p);
                    }
                    ov_end = p + 1;
                }
            }
            if let Some(s) = ov_start {
                a = s;
                b = ov_end;
            } else {
                let nearest = a.min(p_starts.len() - 1);
                a = nearest;
                b = nearest + 1;
            }
        }
        let count = (b - a) as f64;
        let mut sum = vec![0.0f64; dim];
        for p in a..b {
            let row = &subword_vectors[p * dim..(p + 1) * dim];
            for (dst, &val) in sum.iter_mut().zip(row) {
                *dst += val as f64;
            }
        }
        let mut vec: Vec<f32> = sum.into_iter().map(|s| (s / count) as f32).collect();
        if l2 {
            normalize(&mut vec);
        }
        out.push(vec);
    }
    Ok(out)
}
