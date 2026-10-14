//! `GoldenLoad` for the shared layers with a golden shape: each builds
//! its Rust columns from the golden's. Spans are source bytes; a span the
//! text cannot hold is refused by the document when the layer is added.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Ix, IxRange, Key, KeyId, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{
    ChunkTags, Confidence, Entities, Language, Lemmas, PhraseChunks, PosTags, Sentences, Tag, TagSet, Tokens,
};

use super::{GoldenLoad, LoadCx, LoadError};
use crate::format::GoldenLayer;

// The shape is checked before `load` runs, so a column the shape names is
// there with its type.
fn u32s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [u32] {
    g.u32s(name).expect("a u32 column the shape checked")
}

fn f64s<'a>(g: &'a GoldenLayer, name: &str) -> &'a [f64] {
    g.f64s(name).expect("an f64 column the shape checked")
}

/// The `start` and `end` columns as spans of the source text. A span
/// that ends before it starts is kept as it is, for the document to
/// refuse with its own message.
fn spans<L>(g: &GoldenLayer) -> Spans<L> {
    let (start, end) = (u32s(g, "start"), u32s(g, "end"));
    let mut spans = Spans::with_capacity(ViewId::SOURCE, start.len());
    for (&start, &end) in start.iter().zip(end) {
        spans.push(Span { start, end });
    }
    spans
}

/// A tag set of the distinct labels of a `str` column, in the order they
/// first appear, and the column as tags of it.
fn tags(g: &GoldenLayer, column: &'static str, key: KeyId) -> Result<(Arc<TagSet>, Vec<Tag>), LoadError> {
    let labels = g.strs(column).expect("a str column the shape checked");
    let mut index: HashMap<&str, u16> = HashMap::new();
    let mut names: Vec<&str> = Vec::new();
    let mut tags = Vec::with_capacity(labels.len());
    for label in labels.iter() {
        let next = names.len();
        let tag = match index.get(label) {
            Some(&t) => t,
            None => {
                let t = u16::try_from(next).map_err(|_| LoadError::Labels { layer: key, column, count: next + 1 })?;
                index.insert(label, t);
                names.push(label);
                t
            }
        };
        tags.push(Tag(tag));
    }
    Ok((TagSet::new(&format!("{key} {column}"), &names), tags))
}

impl GoldenLoad for Sentences {
    fn load(g: &GoldenLayer, _: &LoadCx<'_>) -> Result<Sentences, LoadError> {
        Ok(Sentences { spans: spans(g) })
    }
}

/// Tokens on the source text. When sentences are loaded under the same
/// name, each token's `sentence` is the sentence whose span contains the
/// token's span, and a token inside no sentence is refused: the JVM makes
/// its tokens inside its sentences. `load_document` loads blocks in file
/// order, and the format writes sentences before tokens, so a golden with
/// both gets the column. Without sentences under that name, `sentence` is
/// `None`, since the golden's token block has no sentence column of its
/// own.
impl GoldenLoad for Tokens {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Tokens, LoadError> {
        let spans = spans(g);
        let sentence = match cx.doc.get_key(&Key::<Sentences>::named(cx.name)) {
            Some(sentences) => Some(sentence_of_each(&spans, sentences, cx.key)?),
            None => None,
        };
        Ok(Tokens { spans, sentence })
    }

    /// The sentences, when the tokens were given a sentence column.
    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        let key = Key::<Sentences>::named(cx.name);
        if cx.doc.get_key(&key).is_some() { vec![key.id()] } else { Vec::new() }
    }
}

/// The sentence each token is in, in one forward pass over both: tokens
/// are in text order and sentences do not overlap, so a token's sentence
/// is never before the previous token's.
fn sentence_of_each(
    tokens: &Spans<Tokens>,
    sentences: &Sentences,
    key: KeyId,
) -> Result<Vec<Ix<Sentences>>, LoadError> {
    let (t_start, t_end) = tokens.columns();
    let (s_start, s_end) = sentences.spans.columns();
    let mut out = Vec::with_capacity(t_start.len());
    let mut j = 0;
    for (i, (&start, &end)) in t_start.iter().zip(t_end).enumerate() {
        // Past every sentence that ends before this token does.
        while j < s_end.len() && s_end[j] < end {
            j += 1;
        }
        if j == s_end.len() || s_start[j] > start {
            return Err(LoadError::OutsideSentences { layer: key, token: i, start, end });
        }
        out.push(Ix::new(j as u32));
    }
    Ok(out)
}

/// A confidence per item of `L`, loaded onto the `L` loaded under the same
/// name.
impl<L: GoldenLoad> GoldenLoad for Confidence<L> {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Confidence<L>, LoadError> {
        cx.anchor::<L>()?;
        Ok(Confidence::new(f64s(g, "value").to_vec()))
    }
}

/// Tags with their probabilities, onto the tokens loaded under the same
/// name. The tag set is the distinct tags of the golden, in the order they
/// first appear.
impl GoldenLoad for PosTags {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<PosTags, LoadError> {
        cx.anchor::<Tokens>()?;
        let (tagset, tags) = tags(g, "tag", cx.key)?;
        Ok(PosTags { tagset, tags, prob: Some(f64s(g, "prob").to_vec()) })
    }
}

impl GoldenLoad for Lemmas {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Lemmas, LoadError> {
        cx.anchor::<Tokens>()?;
        Ok(Lemmas { lemma: g.strs("lemma").expect("a str column the shape checked").clone() })
    }
}

/// Entities on the source text. Their token ranges are the golden's
/// `token_start` and `token_end` columns, indexes into the tokens loaded
/// under the same name, and each must be inside them. The golden's `prob`
/// is the Rust layer's `score`.
impl GoldenLoad for Entities {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Entities, LoadError> {
        let n = cx.anchor::<Tokens>()?.len();
        let (token_start, token_end) = (u32s(g, "token_start"), u32s(g, "token_end"));
        let mut tokens = Vec::with_capacity(token_start.len());
        for (row, (&start, &end)) in token_start.iter().zip(token_end).enumerate() {
            if start > end || end as usize > n {
                return Err(LoadError::TokenRange { layer: cx.key, row, start, end, tokens: n });
            }
            tokens.push(IxRange::new(start, end));
        }
        let (kinds, kind) = tags(g, "kind", cx.key)?;
        Ok(Entities { spans: spans(g), tokens, kind, kinds, score: f64s(g, "prob").to_vec() })
    }

    /// The tokens it indexes.
    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        vec![Key::<Tokens>::named(cx.name).id()]
    }
}

/// Chunk tags with their probabilities, onto the tokens loaded under the
/// same name, as `PosTags`.
impl GoldenLoad for ChunkTags {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<ChunkTags, LoadError> {
        cx.anchor::<Tokens>()?;
        let (tagset, tags) = tags(g, "tag", cx.key)?;
        Ok(ChunkTags { tagset, tags, prob: Some(f64s(g, "prob").to_vec()) })
    }
}

/// Phrases on the source text, with their token ranges checked against
/// the tokens loaded under the same name, as `Entities`; a phrase without
/// a token is refused too.
impl GoldenLoad for PhraseChunks {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<PhraseChunks, LoadError> {
        let n = cx.anchor::<Tokens>()?.len();
        let (token_start, token_end) = (u32s(g, "token_start"), u32s(g, "token_end"));
        let mut tokens = Vec::with_capacity(token_start.len());
        for (row, (&start, &end)) in token_start.iter().zip(token_end).enumerate() {
            if start >= end || end as usize > n {
                return Err(LoadError::TokenRange { layer: cx.key, row, start, end, tokens: n });
            }
            tokens.push(IxRange::new(start, end));
        }
        let (kinds, kind) = tags(g, "kind", cx.key)?;
        Ok(PhraseChunks { spans: spans(g), tokens, kind, kinds })
    }

    /// The tokens it indexes.
    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        vec![Key::<Tokens>::named(cx.name).id()]
    }
}

/// Every language the detector ranked, best first, as the golden lists
/// them; the golden's `confidence` is the Rust layer's `score`. A golden
/// with no rows names no language and is refused.
impl GoldenLoad for Language {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Language, LoadError> {
        if g.is_empty() {
            return Err(LoadError::Empty { layer: cx.key });
        }
        let code = g.strs("code").expect("a str column the shape checked").clone();
        Ok(Language { code, score: f64s(g, "confidence").to_vec() })
    }
}
