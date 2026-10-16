//! `GoldenLoad` for the shared layers with a golden shape: each builds
//! its Rust columns from the golden's. Spans are source bytes; a span the
//! text cannot hold is refused by the document when the layer is added.

use std::collections::HashMap;
use std::sync::Arc;

use turbo_analysis::{Ix, IxRange, Key, KeyId, Layer, Span, Spans, ViewId};
use turbo_analysis_layers::{
    Aspects, Categories, ChunkTags, Chunks, Confidence, Entities, Incivility, IncivilityLabel, IncivilityTarget,
    Incongruity, Language, Lemmas, Opinions, PhraseChunks, PosTags, Pragmatics, Sentences, Structure, Tag, TagSet,
    Tokens,
};

use super::{GoldenLoad, LoadCx, LoadError};
use crate::compare::Mismatch;
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

impl GoldenLoad for Opinions {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Opinions, LoadError> {
        let n = cx.anchor::<Tokens>()?.len();
        let (token_start, token_end) = (u32s(g, "token_start"), u32s(g, "token_end"));
        let negated_col = u32s(g, "negated");
        let mut tokens = Vec::with_capacity(token_start.len());
        let mut negated = Vec::with_capacity(negated_col.len());
        for (row, (&start, &end)) in token_start.iter().zip(token_end).enumerate() {
            if start >= end || end as usize > n {
                return Err(LoadError::TokenRange { layer: cx.key, row, start, end, tokens: n });
            }
            tokens.push(IxRange::new(start, end));
            let neg = negated_col[row];
            if neg > 1 {
                return Err(LoadError::Shape(Mismatch::BadGolden {
                    layer: cx.key.to_string(),
                    detail: format!("row {row} has negated {neg}, expected 0 or 1"),
                }));
            }
            negated.push(neg != 0);
        }
        Ok(Opinions {
            tokens,
            polarity: f64s(g, "polarity").to_vec(),
            intensity: f64s(g, "intensity").to_vec(),
            negated,
            tokens_key: Key::<Tokens>::named(cx.name),
        })
    }

    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        vec![Key::<Tokens>::named(cx.name).id()]
    }
}

impl GoldenLoad for Aspects {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Aspects, LoadError> {
        let n_tokens = cx.anchor::<Tokens>()?.len();
        let (token_start, token_end) = (u32s(g, "token_start"), u32s(g, "token_end"));
        let sentence_col = g.strs("sentence").expect("a str column the shape checked");
        let category_col = g.strs("category").expect("a str column the shape checked");
        let opinions_col = g.strs("opinions").expect("a str column the shape checked");

        let mut tokens = Vec::with_capacity(g.len());
        let mut sentence = Vec::with_capacity(g.len());
        let mut category = Vec::with_capacity(g.len());
        let mut cat_index: HashMap<&str, u16> = HashMap::new();
        let mut cat_names: Vec<&str> = Vec::new();
        let mut opinions = Vec::new();
        let mut opinion_end = Vec::with_capacity(g.len());
        let mut uses_sentences = false;
        let mut uses_opinions = false;

        for row in 0..g.len() {
            let (start, end) = (token_start[row], token_end[row]);
            if start == 0 && end == 0 {
                tokens.push(None);
            } else {
                if start >= end || end as usize > n_tokens {
                    return Err(LoadError::TokenRange { layer: cx.key, row, start, end, tokens: n_tokens });
                }
                tokens.push(Some(IxRange::new(start, end)));
            }

            let sent_str = sentence_col.get(row);
            if sent_str.is_empty() {
                sentence.push(None);
            } else {
                uses_sentences = true;
                let s = crate::read::parse_u32(sent_str).ok_or_else(|| {
                    LoadError::Shape(Mismatch::BadGolden {
                        layer: cx.key.to_string(),
                        detail: format!("row {row} has invalid sentence index {sent_str:?}"),
                    })
                })?;
                sentence.push(Some(Ix::new(s)));
            }

            let cat_str = category_col.get(row);
            if cat_str.is_empty() {
                category.push(None);
            } else {
                let next = cat_names.len();
                let tag = match cat_index.get(cat_str) {
                    Some(&t) => t,
                    None => {
                        let t = u16::try_from(next).map_err(|_| LoadError::Labels {
                            layer: cx.key,
                            column: "category",
                            count: next + 1,
                        })?;
                        cat_index.insert(cat_str, t);
                        cat_names.push(cat_str);
                        t
                    }
                };
                category.push(Some(Tag(tag)));
            }

            let ops_str = opinions_col.get(row);
            if !ops_str.is_empty() {
                uses_opinions = true;
                for part in ops_str.split(',') {
                    let op = crate::read::parse_u32(part).ok_or_else(|| {
                        LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has invalid opinion index {part:?}"),
                        })
                    })?;
                    opinions.push(Ix::new(op));
                }
            }
            opinion_end.push(opinions.len() as u32);
        }

        if uses_sentences {
            cx.anchor::<Sentences>()?;
        }
        if uses_opinions {
            cx.anchor::<Opinions>()?;
        }
        let sentences_key = if uses_sentences || cx.doc.get_key(&Key::<Sentences>::named(cx.name)).is_some() {
            Key::<Sentences>::named(cx.name)
        } else {
            Key::DEFAULT
        };
        let opinions_key = if uses_opinions || cx.doc.get_key(&Key::<Opinions>::named(cx.name)).is_some() {
            Key::<Opinions>::named(cx.name)
        } else {
            Key::DEFAULT
        };

        Ok(Aspects {
            tokens,
            sentence,
            category,
            categories: TagSet::new(&format!("{} category", cx.key), &cat_names),
            positive: f64s(g, "positive").to_vec(),
            negative: f64s(g, "negative").to_vec(),
            neutral: f64s(g, "neutral").to_vec(),
            opinions,
            opinion_end,
            tokens_key: Key::<Tokens>::named(cx.name),
            sentences_key,
            opinions_key,
        })
    }

    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        let mut out = vec![Key::<Tokens>::named(cx.name).id()];
        let s = Key::<Sentences>::named(cx.name);
        if cx.doc.get_key(&s).is_some() {
            out.push(s.id());
        }
        let o = Key::<Opinions>::named(cx.name);
        if cx.doc.get_key(&o).is_some() {
            out.push(o.id());
        }
        out
    }
}

impl GoldenLoad for Incivility {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Incivility, LoadError> {
        let n_tokens = cx.anchor::<Tokens>()?.len();
        let target_kind = g.strs("target_kind").expect("a str column the shape checked");
        let target_index = u32s(g, "target_index");
        let evidence_col = g.strs("evidence").expect("a str column the shape checked");

        let mut target = Vec::with_capacity(g.len());
        let mut evidence_label = Vec::new();
        let mut evidence_tokens = Vec::new();
        let mut evidence_end = Vec::with_capacity(g.len());
        let mut uses_sentences = false;
        let mut uses_chunks = false;
        let mut uses_structure = false;

        for (row, &idx) in target_index.iter().enumerate().take(g.len()) {
            let kind = target_kind.get(row);
            let Some(t) = IncivilityTarget::from_kind_and_index(kind, idx) else {
                return Err(LoadError::Shape(Mismatch::BadGolden {
                    layer: cx.key.to_string(),
                    detail: format!("row {row} has unknown target_kind {kind:?}"),
                }));
            };
            match t {
                IncivilityTarget::Sentence(_) => uses_sentences = true,
                IncivilityTarget::Chunk(_) => uses_chunks = true,
                IncivilityTarget::Block(_) => uses_structure = true,
            }
            target.push(t);

            let ev_str = evidence_col.get(row);
            if !ev_str.is_empty() {
                for entry in ev_str.split(',') {
                    let Some((label_str, range_str)) = entry.split_once(':') else {
                        return Err(LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has malformed evidence entry {entry:?}"),
                        }));
                    };
                    let Some(label) = IncivilityLabel::from_name(label_str) else {
                        return Err(LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has unknown incivility label {label_str:?}"),
                        }));
                    };
                    let Some((start_str, end_str)) = range_str.split_once("..") else {
                        return Err(LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has malformed evidence token range {range_str:?}"),
                        }));
                    };
                    let (Some(start), Some(end)) = (crate::read::parse_u32(start_str), crate::read::parse_u32(end_str))
                    else {
                        return Err(LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has invalid evidence token range {range_str:?}"),
                        }));
                    };
                    if start >= end || end as usize > n_tokens {
                        return Err(LoadError::TokenRange { layer: cx.key, row, start, end, tokens: n_tokens });
                    }
                    evidence_label.push(label);
                    evidence_tokens.push(IxRange::new(start, end));
                }
            }
            evidence_end.push(evidence_label.len() as u32);
        }

        if uses_sentences {
            cx.anchor::<Sentences>()?;
        }
        if uses_chunks {
            cx.anchor::<Chunks>()?;
        }
        if uses_structure {
            cx.anchor::<Structure>()?;
        }
        let sentences_key = if uses_sentences || cx.doc.get_key(&Key::<Sentences>::named(cx.name)).is_some() {
            Key::<Sentences>::named(cx.name)
        } else {
            Key::DEFAULT
        };
        let chunks_key = if uses_chunks || cx.doc.get_key(&Key::<Chunks>::named(cx.name)).is_some() {
            Key::<Chunks>::named(cx.name)
        } else {
            Key::DEFAULT
        };
        let structure_key = if uses_structure || cx.doc.get_key(&Key::<Structure>::named(cx.name)).is_some() {
            Key::<Structure>::named(cx.name)
        } else {
            Key::DEFAULT
        };

        Ok(Incivility {
            target,
            insult: f64s(g, "insult").to_vec(),
            profanity: f64s(g, "profanity").to_vec(),
            threat: f64s(g, "threat").to_vec(),
            identity_attack: f64s(g, "identity_attack").to_vec(),
            intolerance: f64s(g, "intolerance").to_vec(),
            severe_toxicity: f64s(g, "severe_toxicity").to_vec(),
            evidence_label,
            evidence_tokens,
            evidence_end,
            tokens_key: Key::<Tokens>::named(cx.name),
            sentences_key,
            chunks_key,
            structure_key,
        })
    }

    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        let mut out = vec![Key::<Tokens>::named(cx.name).id()];
        let s = Key::<Sentences>::named(cx.name);
        if cx.doc.get_key(&s).is_some() {
            out.push(s.id());
        }
        let c = Key::<Chunks>::named(cx.name);
        if cx.doc.get_key(&c).is_some() {
            out.push(c.id());
        }
        let st = Key::<Structure>::named(cx.name);
        if cx.doc.get_key(&st).is_some() {
            out.push(st.id());
        }
        out
    }
}

impl GoldenLoad for Pragmatics {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Pragmatics, LoadError> {
        cx.anchor::<Sentences>()?;
        let n_tokens = cx.anchor::<Tokens>()?.len();
        let (pos_start, pos_end) = (u32s(g, "pos_start"), u32s(g, "pos_end"));
        let (neg_start, neg_end) = (u32s(g, "neg_start"), u32s(g, "neg_end"));
        let mut evidence = Vec::with_capacity(g.len());
        for row in 0..g.len() {
            let (ps, pe, ns, ne) = (pos_start[row], pos_end[row], neg_start[row], neg_end[row]);
            if ps == 0 && pe == 0 && ns == 0 && ne == 0 {
                evidence.push(None);
            } else {
                if ps >= pe || pe as usize > n_tokens {
                    return Err(LoadError::TokenRange { layer: cx.key, row, start: ps, end: pe, tokens: n_tokens });
                }
                if ns >= ne || ne as usize > n_tokens {
                    return Err(LoadError::TokenRange { layer: cx.key, row, start: ns, end: ne, tokens: n_tokens });
                }
                evidence.push(Some(Incongruity { positive: IxRange::new(ps, pe), negative: IxRange::new(ns, ne) }));
            }
        }
        Ok(Pragmatics { score: f64s(g, "score").to_vec(), evidence, tokens_key: Key::<Tokens>::named(cx.name) })
    }

    fn made_on(cx: &LoadCx<'_>) -> Vec<KeyId> {
        vec![Key::<Sentences>::named(cx.name).id(), Key::<Tokens>::named(cx.name).id()]
    }
}

/// The document's categories: the golden lists every category the
/// categorizer scored, best first, so its order is the layer's category
/// order and the first is the best. A golden with no rows names no
/// category and is refused.
impl GoldenLoad for Categories {
    fn load(g: &GoldenLayer, cx: &LoadCx<'_>) -> Result<Categories, LoadError> {
        if g.is_empty() {
            return Err(LoadError::Empty { layer: cx.key });
        }
        let category = g.strs("category").expect("a str column the shape checked");
        let names: Vec<&str> = category.iter().collect();
        Ok(Categories::document(TagSet::new(<Categories>::ID, &names), Tag(0), f64s(g, "score")))
    }
}
