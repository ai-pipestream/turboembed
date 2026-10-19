use std::collections::HashMap;

use turbo_analysis::{Ix, IxRange, Key, KeyId, Layer};
use turbo_analysis_layers::{
    Aspects, Categories, Chunks, Incivility, IncivilityLabel, IncivilityTarget, Incongruity, Opinions, Pragmatics,
    Sentences, Structure, Tag, TagSet, Tokens,
};

use super::super::{GoldenLoad, LoadCx, LoadError};
use super::{f64s, u32s};
use crate::compare::Mismatch;
use crate::format::GoldenLayer;

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
        let mut n_sentences: Option<usize> = None;
        let mut n_opinions: Option<usize> = None;

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
                let n = match n_sentences {
                    Some(n) => n,
                    None => {
                        let n = cx.anchor::<Sentences>()?.len();
                        n_sentences = Some(n);
                        n
                    }
                };
                if s as usize >= n {
                    return Err(LoadError::Shape(Mismatch::BadGolden {
                        layer: cx.key.to_string(),
                        detail: format!("row {row} has sentence {s}, outside {n} sentences"),
                    }));
                }
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
                    let n = match n_opinions {
                        Some(n) => n,
                        None => {
                            let n = cx.anchor::<Opinions>()?.len();
                            n_opinions = Some(n);
                            n
                        }
                    };
                    if op as usize >= n {
                        return Err(LoadError::Shape(Mismatch::BadGolden {
                            layer: cx.key.to_string(),
                            detail: format!("row {row} has opinion {op}, outside {n} opinions"),
                        }));
                    }
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
            let (what, n) = match kind {
                "sentence" => ("sentences", cx.anchor::<Sentences>()?.len()),
                "chunk" => ("chunks", cx.anchor::<Chunks>()?.len()),
                "block" => ("blocks", cx.anchor::<Structure>()?.len()),
                _ => ("", 0),
            };
            if !what.is_empty() && idx as usize >= n {
                return Err(LoadError::Shape(Mismatch::BadGolden {
                    layer: cx.key.to_string(),
                    detail: format!("row {row} has target {idx}, outside {n} {what}"),
                }));
            }
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
