//! Caller-provided token rows and range validation.

use super::*;

/// # Safety
/// `batch` and the arrays it points to hold what turbo.h says; other
/// pointers are NULL or valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_embed_write_tokens(
    s: *mut turbo_session,
    batch: *const turbo_token_batch,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(s)?.inner;
            let mut state = begin_write(s)?;
            let tb = token_batch(batch)?;
            let o = embed_options(s, opts)?;
            no_text_options(&o)?;
            check_tokens(s, &tb, o.max_tokens)?;
            write_rows(s, &mut state, caller_rows(&o, &tb), &o, false)
        })
    }
}

/// # Safety
/// `batch` and the arrays it points to hold what turbo.h says, `ranges`
/// holds `count` ranges; other pointers are NULL or valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_embed_write_ranges(
    s: *mut turbo_session,
    batch: *const turbo_token_batch,
    ranges: *const turbo_token_range,
    count: u32,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(s)?.inner;
            let mut state = begin_write(s)?;
            let tb = token_batch(batch)?;
            let o = embed_options(s, opts)?;
            if o.pooling_given != TURBO_POOLING_MODEL && o.pooling_given != TURBO_POOLING_MEAN {
                return Err(Error::field(
                    UNSUPPORTED_OPTION,
                    5,
                    format!(
                        "pooling {}: a range is pooled by the mean alone, so pooling is 0 or TURBO_POOLING_MEAN",
                        o.pooling_given
                    ),
                ));
            }
            no_text_options(&o)?;
            check_tokens(s, &tb, o.max_tokens)?;
            if ranges.is_null() && count > 0 {
                return Err(Error::new(INVALID_ARGUMENT, format!("ranges is NULL and count is {count}")));
            }
            if count == 0 {
                return Err(Error::new(INVALID_ARGUMENT, "count is 0: a run needs at least one range"));
            }
            if count > s.max_ranges {
                return Err(Error::new(
                    CAPACITY,
                    format!("{count} ranges is over the session's max_ranges {}", s.max_ranges),
                ));
            }
            let len = array_len::<turbo_token_range>(count as u64, "ranges")?;
            let ranges = std::slice::from_raw_parts(ranges, len);
            check_ranges(&tb, ranges)?;
            // turbo.h: a range is pooled by the mean whatever the bundle says.
            let mut rows = turbo_backend_embed_rows { pooling: TURBO_POOLING_MEAN, ..caller_rows(&o, &tb) };
            if let Some(h) = &s.model.span_head {
                span_head_options(&o, s.model.info.dim)?;
                let output_dim = h.classifier.out_dim as u32;
                rows = turbo_backend_embed_rows { head: 1, output_dim, normalize: TURBO_NORMALIZE_NONE, ..rows };
            }
            let b = s.backend;
            let write = backend::offered!(b, embed_write_ranges)?;
            backend::check(b, "embed_write_ranges", |err| write(s.raw, &rows, ranges.as_ptr(), count, err))?;
            state.written = Some(Written { batch: count, output_dim: rows.output_dim, tokenized: false });
            Ok(())
        })
    }
}

/// turbo.h: the span head reads a range's whole vector, so a range write
/// to a model whose device applies it takes no normalization and no
/// narrower output_dim.
fn span_head_options(o: &EmbedOptions, dim: u32) -> Result<()> {
    if o.normalize != TURBO_NORMALIZE_NONE {
        return Err(Error::field(UNSUPPORTED_OPTION, 4, "normalize: the span head's logits are not normalized"));
    }
    if o.output_dim != dim {
        return Err(Error::field(
            UNSUPPORTED_OPTION,
            6,
            format!("output_dim {}: the span head reads all {dim} values of a range's vector", o.output_dim),
        ));
    }
    Ok(())
}

/// The caller's turbo_token_batch, read once and its struct_size checked.
pub(super) unsafe fn token_batch(batch: *const turbo_token_batch) -> Result<turbo_token_batch> {
    let tb = *unsafe { batch.as_ref() }.ok_or_else(|| Error::new(INVALID_ARGUMENT, "batch is NULL"))?;
    sized(tb.struct_size, size_of::<turbo_token_batch>(), "turbo_token_batch")?;
    Ok(tb)
}

/// Rows written as tokens are already cut and carry their prefix, so the
/// two options that act on text are refused when the caller gives them.
pub(super) fn no_text_options(o: &EmbedOptions) -> Result<()> {
    if o.truncate_given {
        return Err(Error::field(INVALID_ARGUMENT, 1, "truncate: rows written as tokens are already cut"));
    }
    if o.prompt_given {
        return Err(Error::field(INVALID_ARGUMENT, 3, "prompt_role: rows written as tokens carry their prefix"));
    }
    Ok(())
}

/// The caller's checked rows as the backend receives them: the caller's
/// arrays themselves, not a copy.
pub(super) fn caller_rows(o: &EmbedOptions, tb: &turbo_token_batch) -> turbo_backend_embed_rows {
    let stride = if tb.row_stride == 0 { tb.seq } else { tb.row_stride };
    let mut r = backend_rows(o, tb.batch, tb.seq, stride);
    r.ids = tb.ids;
    r.mask = tb.mask;
    r.types = tb.types;
    r
}

/// Every range names a row of the batch, a column span within seq that is
/// not empty, and at least one token of that row whose mask is 1. The
/// mask is read in place through the row stride; check_tokens has already
/// checked the batch's shape and that mask is not NULL.
pub(super) unsafe fn check_ranges(tb: &turbo_token_batch, ranges: &[turbo_token_range]) -> Result<()> {
    let stride = if tb.row_stride == 0 { tb.seq } else { tb.row_stride } as usize;
    for (i, g) in ranges.iter().enumerate() {
        let bad = |rule: String| Err(Error::new(INVALID_ARGUMENT, format!("ranges[{i}]: {rule}")));
        if g.row >= tb.batch {
            return bad(format!("row {} is not under batch {}", g.row, tb.batch));
        }
        if g.start >= g.end {
            return bad(format!("start {} is not under end {}", g.start, g.end));
        }
        if g.end > tb.seq {
            return bad(format!("end {} is over seq {}", g.end, tb.seq));
        }
        let at = g.row as usize * stride + g.start as usize;
        let mask = unsafe { std::slice::from_raw_parts(tb.mask.add(at), (g.end - g.start) as usize) };
        if !mask.contains(&1) {
            return bad(format!(
                "columns {} to {} of row {} hold no token whose mask is 1: a range needs a token",
                g.start, g.end, g.row
            ));
        }
    }
    Ok(())
}

/// Every row the caller wrote, checked as turbo_backend.h promises the
/// backend: the shape within the session, and each id, mask entry and
/// type one the model has. `max_tokens`, when not 0, bounds each row's
/// length through its last live token.
pub(super) unsafe fn check_tokens(s: &SessionInner, b: &turbo_token_batch, max_tokens: u32) -> Result<()> {
    if b.batch == 0 || b.seq == 0 {
        return Err(Error::new(INVALID_SHAPE, format!("batch {} by seq {}: neither may be 0", b.batch, b.seq)));
    }
    if b.batch > s.info.max_batch {
        return Err(Error::new(
            CAPACITY,
            format!("batch {} is over the session's max_batch {}", b.batch, s.info.max_batch),
        ));
    }
    if b.seq > s.info.max_seq {
        return Err(Error::new(CAPACITY, format!("seq {} is over the session's max_seq {}", b.seq, s.info.max_seq)));
    }
    let stride = if b.row_stride == 0 { b.seq } else { b.row_stride } as usize;
    if stride < b.seq as usize {
        return Err(Error::new(INVALID_SHAPE, format!("row_stride {stride} is under seq {}", b.seq)));
    }
    if b.ids.is_null() || b.mask.is_null() {
        return Err(Error::new(INVALID_ARGUMENT, "ids and mask may not be NULL"));
    }
    let (vocab, types) = (s.model.weights.vocab_size(), s.model.weights.token_types());
    let (n, seq) = (b.batch as usize, b.seq as usize);
    let span = array_len::<i32>((b.batch as u64 - 1) * stride as u64 + b.seq as u64, "token batch")?;
    let (ids, mask) = unsafe { (std::slice::from_raw_parts(b.ids, span), std::slice::from_raw_parts(b.mask, span)) };
    let ty = (!b.types.is_null()).then(|| unsafe { std::slice::from_raw_parts(b.types, span) });
    for r in 0..n {
        let at = r * stride;
        let mut live = 0;
        for p in 0..seq {
            let (id, m) = (ids[at + p], mask[at + p]);
            if id < 0 || id as u32 >= vocab {
                if id == -1 && m == 0 && s.model.tokenizer.pad_id < 0 {
                    return Err(Error::new(
                        INVALID_ARGUMENT,
                        format!(
                            "ids[{r}][{p}] is -1: the bundle has no pad token, so pad under mask 0 with \
                             an id in the vocabulary, such as unk id {}",
                            s.model.tokenizer.unk_id
                        ),
                    ));
                }
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("ids[{r}][{p}] is {id}, outside the model's vocabulary of {vocab}"),
                ));
            }
            match m {
                0 => {}
                1 => live = p + 1,
                _ => return Err(Error::new(INVALID_ARGUMENT, format!("mask[{r}][{p}] is {m}, not 0 or 1"))),
            }
            if let Some(ty) = ty
                && (ty[at + p] < 0 || ty[at + p] as u32 >= types)
            {
                return Err(Error::new(
                    INVALID_ARGUMENT,
                    format!("types[{r}][{p}] is {}, and the model has {types} token types", ty[at + p]),
                ));
            }
        }
        if live == 0 {
            return Err(Error::new(INVALID_ARGUMENT, format!("mask row {r} has no 1: a row needs a token")));
        }
        if max_tokens != 0 && live > max_tokens as usize {
            return Err(Error::new(
                CAPACITY,
                format!("row {r}: {live} tokens through its last live one is over max_tokens {max_tokens}"),
            ));
        }
    }
    Ok(())
}

// ---- Run --------------------------------------------------------------------------
