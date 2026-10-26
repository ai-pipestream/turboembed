//! Cross-layer index and span-correspondence checks for [`ProjectableLayer`]
//! implementations in `turbo-analysis-layers`.

use std::borrow::Cow;

use turbo_analysis::{Document, Error, Ix, IxRange, Key, KeyId, Layer, ProjectableLayer, Span, Spans, ViewId};

use crate::{
    ChunkTableHeaders, Chunks, Entities, Events, Levels, Money, Quantities, Sentences, Structure, Tables, Temporals,
    Tokens, Unit,
};

pub(super) fn check_invariants<L: Layer>(layer: &L) -> Result<(), Error> {
    layer.check().map_err(|detail| Error::Invalid { layer: Key::<L>::DEFAULT.id(), detail })
}

fn target_spans<'a, T: Layer>(
    doc: &'a Document,
    from: ViewId,
    to: ViewId,
    layer: KeyId,
    spans: &'a Spans<T>,
) -> Result<Cow<'a, Spans<T>>, Error> {
    if spans.view() == to {
        Ok(Cow::Borrowed(spans))
    } else if spans.view() == from {
        Ok(Cow::Owned(doc.project_spans_monotonic(spans, to)?))
    } else {
        Err(Error::Bounds {
            layer,
            view: to,
            detail: format!("referenced {} is in view {:?}, neither {from:?} nor {to:?}", T::NAME, spans.view()),
        })
    }
}

pub(super) fn verify_token_ranges<L: Layer>(
    doc: &Document,
    from: ViewId,
    to: ViewId,
    spans: &Spans<L>,
    ranges: &[IxRange<Tokens>],
) -> Result<(), Error> {
    let Some(tokens) = doc.get::<Tokens>() else {
        return Ok(());
    };
    let layer = Key::<L>::DEFAULT.id();
    let tok_spans = target_spans(doc, from, to, layer, &tokens.spans)?;
    for (i, span) in spans.iter() {
        let r = ranges[i.get()];
        if r.start > r.end || r.end.get() > tok_spans.len() {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!("token range {:?}..{:?} exceeds tokens length {}", r.start, r.end, tok_spans.len()),
            });
        }
        if !r.is_empty() {
            let first = tok_spans.get(r.start);
            let last = tok_spans.get(Ix::new((r.end.get() - 1) as u32));
            if Span::new(first.start, last.end) != span {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!(
                        "projected span {}..{} does not match token range span {}..{}",
                        span.start, span.end, first.start, last.end
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_token_ixs<L: Layer>(
    doc: &Document,
    from: ViewId,
    to: ViewId,
    spans: &Spans<L>,
    ixs: &[Ix<Tokens>],
    exact: bool,
) -> Result<(), Error> {
    let Some(tokens) = doc.get::<Tokens>() else {
        return Ok(());
    };
    let layer = Key::<L>::DEFAULT.id();
    let tok_spans = target_spans(doc, from, to, layer, &tokens.spans)?;
    for (i, span) in spans.iter() {
        let t = ixs[i.get()];
        if t.get() >= tok_spans.len() {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!("token index {:?} exceeds tokens length {}", t, tok_spans.len()),
            });
        }
        let ts = tok_spans.get(t);
        if (exact && ts != span) || (!exact && !ts.contains(span)) {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!(
                    "projected span {}..{} does not align with token span {}..{}",
                    span.start, span.end, ts.start, ts.end
                ),
            });
        }
    }
    Ok(())
}

pub(super) fn verify_sentence_ixs<L: Layer>(
    doc: &Document,
    from: ViewId,
    to: ViewId,
    spans: &Spans<L>,
    ixs: &[Ix<Sentences>],
) -> Result<(), Error> {
    let Some(sents) = doc.get::<Sentences>() else {
        return Ok(());
    };
    let layer = Key::<L>::DEFAULT.id();
    let sent_spans = target_spans(doc, from, to, layer, &sents.spans)?;
    for (i, span) in spans.iter() {
        let s = ixs[i.get()];
        if s.get() >= sent_spans.len() || !sent_spans.get(s).contains(span) {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!("projected span {}..{} is outside sentence {:?}", span.start, span.end, s),
            });
        }
    }
    Ok(())
}

pub(super) fn verify_sentence_ranges<L: Layer>(
    doc: &Document,
    from: ViewId,
    to: ViewId,
    spans: &Spans<L>,
    ranges: &[IxRange<Sentences>],
) -> Result<(), Error> {
    let Some(sents) = doc.get::<Sentences>() else {
        return Ok(());
    };
    let layer = Key::<L>::DEFAULT.id();
    let sent_spans = target_spans(doc, from, to, layer, &sents.spans)?;
    for (i, span) in spans.iter() {
        let r = ranges[i.get()];
        if r.start > r.end || r.end.get() > sent_spans.len() {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!(
                    "sentence range {:?}..{:?} exceeds sentences length {}",
                    r.start,
                    r.end,
                    sent_spans.len()
                ),
            });
        }
        if !r.is_empty() {
            let first = sent_spans.get(r.start);
            let last = sent_spans.get(Ix::new((r.end.get() - 1) as u32));
            if !Span::new(first.start, last.end).contains(span) {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!(
                        "projected span {}..{} is outside sentence range {}..{}",
                        span.start, span.end, first.start, last.end
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_levels(doc: &Document, from: ViewId, to: ViewId, levels: &Levels) -> Result<(), Error> {
    let layer = Key::<Levels>::DEFAULT.id();
    if let Some(chunks) = doc.get::<Chunks>() {
        let chunk_spans = target_spans(doc, from, to, layer, &chunks.spans)?;
        for (i, span) in levels.spans.iter() {
            if let Unit::Chunk(c) = levels.unit[i.get()]
                && (c.get() >= chunk_spans.len() || chunk_spans.get(c) != span)
            {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!("level chunk unit {:?} does not match target chunk span", c),
                });
            }
        }
    }
    if let Some(st) = doc.get::<Structure>() {
        let st_spans = target_spans(doc, from, to, layer, &st.spans)?;
        for (i, span) in levels.spans.iter() {
            if let Unit::Section(s) = levels.unit[i.get()]
                && (s.get() >= st_spans.len() || st_spans.get(s) != span)
            {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!("level section unit {:?} does not match target structure span", s),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_chunk_table_headers(
    doc: &Document,
    from: ViewId,
    to: ViewId,
    cth: &ChunkTableHeaders,
) -> Result<(), Error> {
    let layer = Key::<ChunkTableHeaders>::DEFAULT.id();
    if let Some(chunks) = doc.get::<Chunks>() {
        let chunk_spans = target_spans(doc, from, to, layer, &chunks.spans)?;
        for (i, span) in cth.spans.iter() {
            let c = Ix::new(cth.contexts[i.get()].chunk_ix);
            if c.get() >= chunk_spans.len() || chunk_spans.get(c) != span {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!("chunk table header chunk {:?} does not match target chunk span", c),
                });
            }
        }
    }
    if let Some(tables) = doc.get::<Tables>() {
        let tb = target_spans(doc, from, to, layer, &tables.table_bounds)?;
        for ctx in &cth.contexts {
            if (ctx.table_id as usize) >= tb.len() {
                return Err(Error::Bounds {
                    layer,
                    view: to,
                    detail: format!("chunk table header table_id {} exceeds tables length {}", ctx.table_id, tb.len()),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn verify_events(doc: &Document, from: ViewId, to: ViewId, events: &Events) -> Result<(), Error> {
    let sent_ixs: Vec<Ix<Sentences>> = events.sentence.iter().copied().map(Ix::new).collect();
    verify_sentence_ixs(doc, from, to, &events.spans, &sent_ixs)?;
    let layer = Key::<Events>::DEFAULT.id();
    for (of, bound) in events.indexes() {
        let len = if of == Key::<Temporals>::DEFAULT.id() {
            doc.get::<Temporals>().map(Layer::len)
        } else if of == Key::<Money>::DEFAULT.id() {
            doc.get::<Money>().map(Layer::len)
        } else if of == Key::<Quantities>::DEFAULT.id() {
            doc.get::<Quantities>().map(Layer::len)
        } else if of == Key::<Entities>::DEFAULT.id() {
            doc.get::<Entities>().map(Layer::len)
        } else {
            None
        };
        if let Some(len) = len
            && bound > len
        {
            return Err(Error::Bounds {
                layer,
                view: to,
                detail: format!("event argument index {bound} exceeds {of} length {len}"),
            });
        }
    }
    Ok(())
}

pub(super) fn retarget_span_key<L: ProjectableLayer>(
    doc: &Document,
    key: Key<L>,
    to: ViewId,
    spans_of: impl Fn(&L) -> &Spans<L>,
) -> Result<(ViewId, Key<L>), Error> {
    let src = doc.require(&key)?;
    let src_spans = spans_of(src);
    let from = src_spans.view();
    if from == to {
        return Ok((from, key));
    }
    let projected = doc.project_spans_monotonic(src_spans, to)?;
    let dst = doc.require(&Key::<L>::DEFAULT).map_err(|_| Error::Bounds {
        layer: Key::<L>::DEFAULT.id(),
        view: to,
        detail: format!("no projected {} at default key matching {}", L::NAME, key.id()),
    })?;
    let dst_spans = spans_of(dst);
    let matches = projected.view() == dst_spans.view()
        && projected.len() == dst_spans.len()
        && (0..projected.len() as u32).all(|i| {
            let ix = Ix::new(i);
            projected.get(ix) == dst_spans.get(ix) && projected.fragments(ix).eq(dst_spans.fragments(ix))
        });
    if !matches {
        return Err(Error::Bounds {
            layer: Key::<L>::DEFAULT.id(),
            view: to,
            detail: format!("default {} does not match projected spans of {}", L::NAME, key.id()),
        });
    }
    Ok((from, Key::DEFAULT))
}
