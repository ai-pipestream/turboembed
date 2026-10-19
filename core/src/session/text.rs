//! Text-batch tokenization and writes into session buffers.

use super::*;

/// A run's Part, or the first error among its texts.
pub(super) type Rows = Result<Part>;

/// Bytes of text a tokenizing task takes at least: a batch of less is
/// tokenized on the caller's thread alone.
pub(super) const BYTES_PER_TASK: u64 = 1024;

/// The caller's text views, which the threads of tokenize_all read.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub(super) struct View(turbo_text);

// SAFETY: a view points at bytes the caller keeps valid and unchanged for
// the call (turbo.h), and tokenize_all's threads end within the call.
unsafe impl Send for View {}
unsafe impl Sync for View {}

/// A text's ids, appended to the buffer it is given.
pub(super) type One<'a> = dyn Fn(usize, turbo_text, &mut Vec<i32>) -> Result<()> + Sync + 'a;

/// `one` over every text, in order, as tasks of about BYTES_PER_TASK on
/// the process's tokenizing threads, which wait between batches, or on
/// threads of its own while another batch has them. The rows, or the
/// error of the first text that fails, are the same as on one thread.
pub(super) fn tokenize_all(texts: &[View], one: &One<'_>) -> Result<Vec<Part>> {
    let bytes = texts.iter().enumerate().try_fold(0u64, |total, (i, t)| {
        let len = text_len(t.0, &format_args!("texts[{i}]"))? as u64;
        Ok::<_, Error>(total.saturating_add(len))
    })?;
    let tasks = (bytes / BYTES_PER_TASK).min(texts.len() as u64) as usize;
    let part = |first: usize, run: &[View]| -> Rows {
        // About a token per three bytes of text, and a few specials.
        let bytes = run.iter().fold(0u64, |n, t| n.saturating_add(t.0.len));
        // This is only an estimate. Bound the initial reservation; the actual
        // encoding grows the buffer as needed, including for repeated views.
        let estimate = (bytes / 3).saturating_add((run.len() as u64).saturating_mul(4)).min(1 << 20);
        let mut p = Part { ids: Vec::with_capacity(estimate as usize), ends: Vec::with_capacity(run.len()) };
        for (j, t) in run.iter().enumerate() {
            one(first + j, t.0, &mut p.ids)?;
            p.ends.push(p.ids.len());
        }
        Ok(p)
    };
    if tasks <= 1 {
        return Ok(vec![part(0, texts)?]);
    }
    let per = texts.len().div_ceil(tasks);
    let chunks: Vec<&[View]> = texts.chunks(per).collect();
    let parts: Vec<Rows> = match tokenizing_pool() {
        Some(mut pool) => {
            let slots: Vec<std::sync::Mutex<Option<Rows>>> =
                chunks.iter().map(|_| std::sync::Mutex::new(None)).collect();
            pool.run(chunks.len(), &|c, _| {
                *slots[c].lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(part(c * per, chunks[c]));
            });
            slots
                .into_iter()
                .map(|s| s.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner).expect("every task ran"))
                .collect()
        }
        None => std::thread::scope(|scope| {
            let part = &part;
            let handles: Vec<_> =
                chunks.iter().enumerate().map(|(c, &run)| scope.spawn(move || part(c * per, run))).collect();
            handles.into_iter().map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p))).collect()
        }),
    };
    parts.into_iter().collect()
}

/// The process's tokenizing threads, one per processor, made by the first
/// batch that needs them; None while another batch has them, or in a
/// build without the cpu backend, whose pool they are.
#[cfg(feature = "cpu")]
pub(super) fn tokenizing_pool() -> Option<std::sync::MutexGuard<'static, crate::cpu::pool::Pool>> {
    pub(super) static POOL: std::sync::OnceLock<std::sync::Mutex<crate::cpu::pool::Pool>> = std::sync::OnceLock::new();
    let pool = POOL.get_or_init(|| {
        // available_parallelism reads the cgroup files on Linux: once.
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        std::sync::Mutex::new(crate::cpu::pool::Pool::with_spin(cpus, crate::cpu::pool::SPIN_BATCH))
    });
    pool.try_lock().ok()
}

#[cfg(not(feature = "cpu"))]
pub(super) fn tokenizing_pool() -> Option<NoPool> {
    None
}

#[cfg(not(feature = "cpu"))]
pub(super) struct NoPool;

#[cfg(not(feature = "cpu"))]
impl NoPool {
    pub(super) fn run(&mut self, _: usize, _: &(dyn Fn(usize, usize) + Sync)) {
        unreachable!("tokenizing_pool gives none")
    }
}

/// # Safety
/// `texts` holds `count` views; other pointers are NULL or valid for the
/// call, as turbo.h says.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn turbo_embed_write_text(
    s: *mut turbo_session,
    texts: *const turbo_text,
    count: u32,
    opts: *const turbo_embed_options,
    err: *mut turbo_error,
) -> i32 {
    unsafe {
        call(err, || {
            let s = &session(s)?.inner;
            let mut state = begin_write(s)?;
            let o = embed_options(s, opts)?;
            if count == 0 {
                return Err(Error::new(INVALID_ARGUMENT, "count is 0: a run needs at least one row"));
            }
            if texts.is_null() {
                return Err(Error::new(INVALID_ARGUMENT, "texts is NULL"));
            }
            if count > s.info.max_batch {
                return Err(Error::new(
                    CAPACITY,
                    format!("{count} texts is over the session's max_batch {}", s.info.max_batch),
                ));
            }
            let tok = &s.model.tokenizer;
            // TURBO_TRUNCATE_MODEL cuts at the bundle's max_seq on every
            // device (docs/bundle.md); a row that is then longer than the
            // session takes is CAPACITY, not cut again.
            let max_tokens = if o.max_tokens == 0 { tok.max_seq } else { o.max_tokens };
            let specials = tok.specials_per_sequence();
            if max_tokens < specials {
                return Err(Error::field(
                    INVALID_ARGUMENT,
                    2,
                    format!("max_tokens {max_tokens} leaves no room beside {specials} special tokens"),
                ));
            }
            let e = Encode { add_special_tokens: true, truncation: o.truncation, max_tokens, prompt: o.prompt };
            let text_count = array_len::<turbo_text>(count as u64, "texts")?;
            let texts = std::slice::from_raw_parts(texts, text_count);
            let one = |i: usize, t: turbo_text, out: &mut Vec<i32>| -> Result<()> {
                let start = out.len();
                tok.encode_into(text(t, &format_args!("texts[{i}]"))?, e, out).map_err(|mut err| {
                    err.message = format!("texts[{i}]: {}", err.message);
                    err
                })?;
                let len = out.len() - start;
                // A static model's template may have no special tokens, so
                // a text can give none: its vector is the zero vector. An
                // encoder has no output for a row of no tokens.
                if len == 0 && !s.model.weights.is_static() {
                    return Err(Error::new(INVALID_ARGUMENT, format!("texts[{i}]: gives no tokens")));
                }
                if len > s.info.max_seq as usize {
                    return Err(Error::new(
                        CAPACITY,
                        format!("texts[{i}]: {len} tokens is over the session's max_seq {}", s.info.max_seq),
                    ));
                }
                Ok(())
            };
            // SAFETY: View is a transparent wrapper of turbo_text.
            let views = std::slice::from_raw_parts(texts.as_ptr().cast::<View>(), texts.len());
            let parts = tokenize_all(views, &one)?;
            let rows = || parts.iter().flat_map(Part::rows);
            // At least one column, padding under mask 0, when every text
            // gave no tokens.
            let seq = rows().map(<[i32]>::len).max().unwrap_or(0).max(1);
            let pad = tok.fill_id();
            let state = &mut *state;
            for (i, row) in rows().enumerate() {
                let (ids, mask) = (&mut state.ids[i * seq..(i + 1) * seq], &mut state.mask[i * seq..(i + 1) * seq]);
                ids[..row.len()].copy_from_slice(row);
                ids[row.len()..].fill(pad);
                mask[..row.len()].fill(1);
                mask[row.len()..].fill(0);
            }
            let mut r = backend_rows(&o, count, seq as u32, seq as u32);
            r.ids = state.ids.as_ptr();
            r.mask = state.mask.as_ptr();
            write_rows(s, state, r, &o, true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oversized_text_is_rejected_before_reservation_or_dispatch() {
        let views = [View(turbo_text { ptr: std::ptr::null(), len: u64::MAX }); 2];
        let result = tokenize_all(&views, &|_, _, _| panic!("invalid text dispatched"));
        assert!(result.is_err());
        assert_eq!(result.err().unwrap().code, INVALID_ARGUMENT);
    }
}
