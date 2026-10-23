//! C tokenizer lifecycle, batched encoding, decoding, and counting.

use super::*;

// ---- Tokenizer -----------------------------------------------------------------

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_create(
    rt: *mut turbo_runtime,
    bundle_path: turbo_text,
    out: *mut *mut turbo_tokenizer,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let rt = runtime(rt)?;
            let out = out_ptr(out, "out")?;
            let path = text(bundle_path, "bundle_path")?;
            let bundle = bundle::Bundle::open(Path::new(path))?;
            let tok = Tokenizer::load(&bundle)?;
            let t = turbo_tokenizer { magic: TOKENIZER_MAGIC, _runtime: rt.inner.clone(), tok };
            *out = Box::into_raw(Box::new(t));
            Ok(())
        })
    }
}

/// # Safety
/// `t` is NULL or a handle from turbo_tokenizer_create, released once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_release(t: *mut turbo_tokenizer) {
    if unsafe { tokenizer(t) }.is_ok() {
        let mut b = unsafe { Box::from_raw(t) };
        b.magic = 0;
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_get_info(
    t: *mut turbo_tokenizer,
    out: *mut turbo_tokenizer_info,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let out = out_ptr(out, "out")?;
            sized(out.struct_size, size_of::<turbo_tokenizer_info>(), "turbo_tokenizer_info")?;
            out.vocab_size = tok.vocab_size();
            out.max_seq = tok.max_seq;
            out.specials_per_sequence = tok.specials_per_sequence();
            out.pad_id = tok.pad_id;
            out.bos_id = tok.bos_id;
            out.eos_id = tok.eos_id;
            out.unk_id = tok.unk_id;
            write_str(&mut out.kind, tok.kind_name());
            write_str(&mut out.sha256, &tok.sha256);
            write_str(&mut out.manifest_sha256, &tok.manifest_sha256);
            Ok(())
        })
    }
}

/// The options as the tokenizer takes them; NULL, or 0 in every field, is
/// what the bundle says.
pub(super) unsafe fn encode_options(tok: &Tokenizer, opts: *const turbo_encode_options) -> Result<Encode> {
    let mut e = Encode {
        add_special_tokens: true,
        truncation: tok.truncation(),
        max_tokens: tok.max_seq,
        prompt: PromptRole::None,
    };
    let Some(o) = (unsafe { opts.as_ref() }) else {
        return Ok(e);
    };
    sized(o.struct_size, size_of::<turbo_encode_options>(), "turbo_encode_options")?;
    e.add_special_tokens = match o.omit_special_tokens {
        0 => true,
        1 => false,
        v => {
            return Err(Error::field(INVALID_ARGUMENT, 1, format!("omit_special_tokens is {v}, not 0 or 1")));
        }
    };
    e.truncation = match o.truncate {
        TURBO_TRUNCATE_MODEL => tok.truncation(),
        TURBO_TRUNCATE_NONE => Truncation::None,
        TURBO_TRUNCATE_RIGHT => Truncation::Right,
        TURBO_TRUNCATE_LEFT => Truncation::Left,
        v => {
            return Err(Error::new(INVALID_ENUM, format!("truncate: {v} is not a TURBO_TRUNCATE_* value")));
        }
    };
    if o.max_tokens != 0 {
        e.max_tokens = o.max_tokens;
    }
    e.prompt = prompt_role(o.prompt_role)?;
    Ok(e)
}

pub(super) fn prompt_role(v: u32) -> Result<PromptRole> {
    match v {
        TURBO_PROMPT_NONE => Ok(PromptRole::None),
        TURBO_PROMPT_QUERY => Ok(PromptRole::Query),
        TURBO_PROMPT_DOCUMENT => Ok(PromptRole::Document),
        v => Err(Error::new(INVALID_ENUM, format!("prompt_role: {v} is not a TURBO_PROMPT_* value"))),
    }
}

/// # Safety
/// `texts` holds `count` views; `ids`, `mask` and, when not NULL, `types`
/// hold `count * row_stride` elements and `lengths` holds `count`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_encode(
    t: *mut turbo_tokenizer,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_encode_options,
    ids: *mut i32,
    mask: *mut i32,
    types: *mut i32,
    row_stride: u32,
    lengths: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || encode_rows(t, texts, count, opts, ids, mask, types, std::ptr::null_mut(), row_stride, lengths))
    }
}

/// # Safety
/// As turbo_tokenizer_encode, and `spans` holds `count * row_stride * 2`
/// elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_encode_spans(
    t: *mut turbo_tokenizer,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_encode_options,
    ids: *mut i32,
    mask: *mut i32,
    types: *mut i32,
    spans: *mut u32,
    row_stride: u32,
    lengths: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            if spans.is_null() && count > 0 {
                return Err(Error::new(INVALID_ARGUMENT, "spans may not be NULL"));
            }
            encode_rows(t, texts, count, opts, ids, mask, types, spans, row_stride, lengths)
        })
    }
}

/// turbo_tokenizer_encode, and the spans when `spans` is not NULL.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn encode_rows(
    t: *mut turbo_tokenizer,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_encode_options,
    ids: *mut i32,
    mask: *mut i32,
    types: *mut i32,
    spans: *mut u32,
    row_stride: u32,
    lengths: *mut u32,
) -> Result<()> {
    unsafe {
        let tok = &tokenizer(t)?.tok;
        let e = encode_options(tok, opts)?;
        if count == 0 {
            return Ok(());
        }
        if texts.is_null() || ids.is_null() || mask.is_null() {
            return Err(Error::new(INVALID_ARGUMENT, "texts, ids and mask may not be NULL"));
        }
        if row_stride == 0 {
            return Err(Error::new(INVALID_ARGUMENT, "row_stride is 0"));
        }
        let n = array_len::<i32>(count as u64 * row_stride as u64, "tokenizer rows")?;
        if !spans.is_null() {
            array_len::<[u32; 2]>(n as u64, "tokenizer spans")?;
        }
        let text_count = array_len::<turbo_text>(count as u64, "texts")?;
        let texts = std::slice::from_raw_parts(texts, text_count);
        // Every row is encoded before any is written, so a failed call
        // leaves the caller's arrays as they were.
        let mut rows = Vec::with_capacity(texts.len());
        let mut row_spans = Vec::with_capacity(if spans.is_null() { 0 } else { texts.len() });
        for (i, &tx) in texts.iter().enumerate() {
            let at = |mut err: Error| {
                err.message = format!("texts[{i}]: {}", err.message);
                err
            };
            let tx = text(tx, &format_args!("texts[{i}]"))?;
            let row = if spans.is_null() {
                tok.encode(tx, e).map_err(at)?
            } else {
                let (mut row, mut s) = (Vec::new(), Vec::new());
                tok.encode_spans(tx, e, &mut row, &mut s).map_err(at)?;
                row_spans.push(s);
                row
            };
            if row.len() > row_stride as usize {
                return Err(Error::new(
                    status::CAPACITY,
                    format!("texts[{i}]: {} tokens, row_stride is {row_stride}", row.len()),
                ));
            }
            rows.push(row);
        }
        let ids = std::slice::from_raw_parts_mut(ids, n);
        let mask = std::slice::from_raw_parts_mut(mask, n);
        let mut types = (!types.is_null()).then(|| std::slice::from_raw_parts_mut(types, n));
        let mut spans = (!spans.is_null()).then(|| std::slice::from_raw_parts_mut(spans, 2 * n));
        for (i, row) in rows.iter().enumerate() {
            let at = i * row_stride as usize;
            let end = at + row_stride as usize;
            ids[at..at + row.len()].copy_from_slice(row);
            ids[at + row.len()..end].fill(tok.pad_id);
            mask[at..at + row.len()].fill(1);
            mask[at + row.len()..end].fill(0);
            if let Some(ty) = types.as_deref_mut() {
                ty[at..end].fill(0);
            }
            if let Some(sp) = spans.as_deref_mut() {
                let out = &mut sp[2 * at..2 * end];
                out.fill(0);
                out[..2 * row.len()].copy_from_slice(row_spans[i].as_flattened());
            }
            if !lengths.is_null() {
                *lengths.add(i) = row.len() as u32;
            }
        }
        Ok(())
    }
}

/// # Safety
/// `ids` holds `count` ids and `out` `capacity` bytes; `written` is valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_decode(
    t: *mut turbo_tokenizer,
    ids: *const i32,
    count: u32,
    skip_special_tokens: u32,
    out: *mut c_char,
    capacity: u64,
    written: *mut u64,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let written = out_ptr(written, "written")?;
            if ids.is_null() && count > 0 {
                return Err(Error::new(INVALID_ARGUMENT, "ids may not be NULL"));
            }
            let skip = match skip_special_tokens {
                0 => false,
                1 => true,
                v => return Err(Error::new(INVALID_ARGUMENT, format!("skip_special_tokens is {v}, not 0 or 1"))),
            };
            let len = array_len::<i32>(count as u64, "ids")?;
            let ids = if count == 0 { &[][..] } else { std::slice::from_raw_parts(ids, len) };
            let text = tok.decode(ids, skip)?;
            *written = text.len() as u64;
            if (text.len() as u64) >= capacity {
                return Err(Error::new(
                    status::CAPACITY,
                    format!("{} bytes and a NUL, capacity is {capacity}", text.len()),
                ));
            }
            if out.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "out may not be NULL"));
            }
            std::ptr::copy_nonoverlapping(text.as_ptr(), out.cast::<u8>(), text.len());
            *out.add(text.len()) = 0;
            Ok(())
        })
    }
}

/// # Safety
/// Pointers are NULL or valid for the call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_count(
    t: *mut turbo_tokenizer,
    txt: turbo_text,
    prompt: u32,
    out: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let tok = &tokenizer(t)?.tok;
            let prompt = prompt_role(prompt)?;
            let out = out_ptr(out, "out")?;
            *out = u32::try_from(tok.count(text(txt, "text")?, prompt))
                .map_err(|_| Error::new(status::CAPACITY, "token count exceeds u32"))?;
            Ok(())
        })
    }
}
