use super::*;

pub const TURBO_RERANK_STAGE_TOKENIZE: usize = 0;
pub const TURBO_RERANK_STAGE_ENCODE: usize = 1;
pub const TURBO_RERANK_STAGE_HEAD: usize = 2;
pub const TURBO_RERANK_STAGE_POOL: usize = 3;
pub const TURBO_RERANK_STAGE_COUNT: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct turbo_rerank_options {
    pub struct_size: u32,
    pub query_tokens: u32,
    pub max_tokens: u32,
    pub stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct turbo_text_pair {
    pub query: turbo_text,
    pub passage: turbo_text,
}

pub(super) struct PairBatch {
    ids: Vec<i32>,
    mask: Vec<i32>,
    types: Vec<i32>,
    pair: Vec<usize>,
    seq: usize,
    count: usize,
}

unsafe fn options(opts: *const turbo_rerank_options) -> Result<turbo_rerank_options> {
    match unsafe { opts.as_ref() } {
        None => Ok(turbo_rerank_options::default()),
        Some(o) => {
            sized(o.struct_size, size_of::<turbo_rerank_options>(), "turbo_rerank_options")?;
            Ok(*o)
        }
    }
}

/// # Safety
/// Pointers follow the lifetimes and capacities specified in turbo.h.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_rerank_write_pairs(
    session_ptr: *mut turbo_session,
    pairs: *const turbo_text_pair,
    count: u32,
    opts: *const turbo_rerank_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(session_ptr)?.inner;
            let mut state = s.lock()?;
            state.written = None;
            state.rerank = None;
            if s.model.info.task != TURBO_TASK_RERANK {
                return Err(Error::new(UNSUPPORTED_TASK, "model task is not rerank"));
            }
            let o = options(opts)?;
            if pairs.is_null() || count == 0 {
                return Err(Error::new(INVALID_ARGUMENT, "pairs is NULL or count is zero"));
            }
            if count > s.info.max_batch {
                return Err(Error::new(CAPACITY, "pair count exceeds max_batch"));
            }
            if o.max_tokens > s.info.max_seq {
                return Err(Error::field(CAPACITY, 2, "rerank max_tokens exceeds session max_seq"));
            }
            let max_tokens = if o.max_tokens == 0 { s.info.max_seq } else { o.max_tokens };
            let pairs = std::slice::from_raw_parts(pairs, count as usize);
            let mut windows = Vec::new();
            let mut owners = Vec::new();
            for (i, p) in pairs.iter().enumerate() {
                let at = |e: Error| Error { message: format!("pairs[{i}]: {}", e.message), ..e };
                let query = text(p.query, "pair.query").map_err(at)?;
                let passage = text(p.passage, "pair.passage").map_err(at)?;
                let encoded =
                    s.model.tokenizer.encode_pair(query, passage, o.query_tokens, max_tokens, o.stride).map_err(at)?;
                owners.extend(std::iter::repeat_n(i, encoded.len()));
                windows.extend(encoded);
            }
            let seq = windows.iter().map(|w| w.ids.len()).max().unwrap_or(1);
            let size = windows.len().checked_mul(seq).ok_or_else(|| Error::new(CAPACITY, "pair windows overflow"))?;
            array_len::<i32>(size as u64, "pair windows")?;
            let mut batch = PairBatch {
                ids: vec![s.model.tokenizer.fill_id(); size],
                mask: vec![0; size],
                types: vec![0; size],
                pair: owners,
                seq,
                count: count as usize,
            };
            for (i, w) in windows.iter().enumerate() {
                let at = i * seq..i * seq + w.ids.len();
                batch.ids[at.clone()].copy_from_slice(&w.ids);
                batch.types[at.clone()].copy_from_slice(&w.types);
                batch.mask[at].fill(1);
            }
            state.rerank = Some(batch);
            state.written = Some(Written { batch: count, output_dim: 1, tokenized: true });
            Ok(())
        })
    }
}

pub(super) unsafe fn run_pairs(s: &SessionInner, state: &mut State) -> Result<turbo_backend_run> {
    let batch = state.rerank.take().ok_or_else(|| Error::new(INVALID_STATE, "no rerank pairs written"))?;
    let head = s.model.rerank_head.as_ref().ok_or_else(|| Error::new(INTERNAL, "missing rerank head"))?;
    state.scores[..batch.count].fill(f32::NEG_INFINITY);
    let write = backend::offered!(s.backend, embed_write)?;
    let mut result: turbo_backend_run = unsafe { std::mem::zeroed() };
    let mut allocations = 0u64;
    // A device backend leaves the states in device memory: they are read
    // back here, and the scores the run returns are the session's own.
    let mut staged: Vec<f32> = Vec::new();
    let mut read_back = 0u64;
    // Each encoder run reports its own transfers; the public run is all of them.
    let (mut uploaded, mut downloaded, mut device_allocations) = (0u64, 0u64, 0u64);
    for start in (0..batch.pair.len()).step_by(s.info.max_batch as usize) {
        let n = (batch.pair.len() - start).min(s.info.max_batch as usize);
        let at = start * batch.seq;
        let rows = turbo_backend_embed_rows {
            struct_size: size_of::<turbo_backend_embed_rows>() as u32,
            batch: n as u32,
            seq: batch.seq as u32,
            row_stride: batch.seq as u32,
            ids: batch.ids[at..].as_ptr(),
            mask: batch.mask[at..].as_ptr(),
            types: batch.types[at..].as_ptr(),
            pooling: TURBO_POOLING_CLS,
            normalize: TURBO_NORMALIZE_NONE,
            output_dim: head.hidden as u32,
            head: 0,
        };
        backend::check(s.backend, "rerank encode write", |err| unsafe { write(s.raw, &rows, err) })?;
        result.struct_size = size_of::<turbo_backend_run>() as u32;
        result.stage[TURBO_EMBED_STAGE_TOKENIZE] = TURBO_STAGE_HOST;
        backend::check(s.backend, "rerank encode run", |err| unsafe { (s.run)(s.raw, &mut result, err) })?;
        check_run(s.backend, &result, true)?;
        allocations += result.host_allocs;
        uploaded += result.h2d_bytes;
        downloaded += result.d2h_bytes;
        device_allocations += result.device_allocs;
        let states: &[f32] = if result.host.is_null() {
            let read = backend::offered!(s.backend, buffer_read)?;
            if staged.is_empty() {
                allocations += 1;
            }
            staged.resize(n * head.hidden, 0.0);
            let bytes = size_of_val(staged.as_slice()) as u64;
            backend::check(s.backend, "rerank encode read", |err| unsafe {
                read(result.output, staged.as_mut_ptr().cast(), bytes, err)
            })?;
            read_back += bytes;
            &staged
        } else {
            unsafe { std::slice::from_raw_parts(result.host.cast::<f32>(), n * head.hidden) }
        };
        for (j, input) in states.chunks_exact(head.hidden).enumerate() {
            if input.iter().any(|v| !v.is_finite()) {
                return Err(Error::new(INTERNAL, "non-finite rerank encoder output"));
            }
            let hidden = if let Some(pooler) = &head.pooler {
                for (dst, (w, &bias)) in
                    state.pooled.iter_mut().zip(pooler.weight.chunks_exact(head.hidden).zip(&pooler.bias))
                {
                    let sum = w.iter().zip(input).fold(f64::from(bias), |a, (&w, &x)| a + f64::from(w) * f64::from(x));
                    *dst = sum.tanh() as f32;
                }
                &state.pooled[..]
            } else {
                input
            };
            let value = head
                .classifier
                .weight
                .iter()
                .zip(hidden)
                .fold(f64::from(head.classifier.bias[0]), |a, (&w, &x)| a + f64::from(w) * f64::from(x))
                as f32;
            if !value.is_finite() {
                return Err(Error::new(INTERNAL, "rerank relevance overflows F32"));
            }
            let score = &mut state.scores[batch.pair[start + j]];
            *score = score.max(value);
        }
    }
    let encode = result.stage[TURBO_EMBED_STAGE_ENCODE];
    if result.host.is_null() {
        // The scores are the session's own host memory, held with the
        // result. No backend object holds them, so the output handle is
        // NULL: a buffer of the result is exported by the core.
        result.placement = TURBO_PLACE_HOST;
        result.output = std::ptr::null_mut();
        result.host = state.scores.as_mut_ptr().cast();
    } else {
        // CPU output reserves max_batch * hidden elements, including when the
        // final window batch is shorter. Compact scores into its leading values.
        unsafe {
            std::ptr::copy_nonoverlapping(state.scores.as_ptr(), result.host.cast::<f32>(), batch.count);
        }
    }
    result.host_allocs = allocations;
    result.h2d_bytes = uploaded;
    result.device_allocs = device_allocations;
    result.d2h_bytes = downloaded + read_back;
    result.stage.fill(TURBO_STAGE_UNUSED);
    result.stage[..TURBO_RERANK_STAGE_COUNT].fill(TURBO_STAGE_HOST);
    // The encoder ran where the backend ran it; the head and the window
    // reduction are the core's, on the host.
    result.stage[TURBO_RERANK_STAGE_ENCODE] = encode;
    Ok(result)
}

/// # Safety
/// Each output pointer holds capacity entries. All inputs are valid for the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_tokenizer_encode_pair(
    tok: *mut turbo_tokenizer,
    pair: turbo_text_pair,
    opts: *const turbo_rerank_options,
    ids: *mut i32,
    types: *mut i32,
    capacity: u32,
    written: *mut u32,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let written = out_ptr(written, "written")?;
            let tok = &tokenizer(tok)?.tok;
            let o = options(opts)?;
            if o.stride != 0 {
                return Err(Error::field(INVALID_ARGUMENT, 3, "single-window encoding requires stride 0"));
            }
            let encoded = tok.encode_pair(
                text(pair.query, "query")?,
                text(pair.passage, "passage")?,
                o.query_tokens,
                o.max_tokens,
                0,
            )?;
            let first = &encoded[0];
            if ids.is_null() || types.is_null() || capacity < first.ids.len() as u32 {
                return Err(Error::new(CAPACITY, "pair output buffers are NULL or too short"));
            }
            std::ptr::copy_nonoverlapping(first.ids.as_ptr(), ids, first.ids.len());
            std::ptr::copy_nonoverlapping(first.types.as_ptr(), types, first.types.len());
            *written = first.ids.len() as u32;
            Ok(())
        })
    }
}
