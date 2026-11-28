use super::*;

/// Rows every [`Encoder`] of this process has forwarded, each window once
/// whatever is read from it: what a request's encoder work is measured by
/// (`forwarded_rows`).
static FORWARDED_ROWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The rows every encoder of this process has forwarded so far.
pub fn forwarded_rows() -> u64 {
    FORWARDED_ROWS.load(std::sync::atomic::Ordering::Relaxed)
}

fn forwarded(rows: &Rows) {
    FORWARDED_ROWS.fetch_add(u64::from(rows.len()), std::sync::atomic::Ordering::Relaxed);
}

impl Encoder {
    #[cfg(test)]
    pub(crate) fn run_counts(&self) -> (usize, usize) {
        self.runs
    }

    /// Reserve output rows for retaining every token of a batch of windows.
    /// The model and numerical settings are retained; replace the session only
    /// when its output buffer is too small. Failure leaves the old session live.
    pub(crate) fn reserve_token_ranges(&mut self) -> Result<(), Error> {
        let needed = self.info.max_batch.saturating_mul(self.info.max_seq);
        if self.info.max_ranges >= needed {
            return Ok(());
        }
        if self.pool.is_some() {
            self.desc.max_ranges = needed;
            self.info.max_ranges = needed;
            return Ok(());
        }
        let mut desc = self.desc;
        desc.max_ranges = needed;
        let mut replacement = ptr::null_mut();
        let mut err = new_error();
        check(unsafe { turbo_session_create(self.model, &desc, &mut replacement, &mut err) }, &err)?;
        unsafe { turbo_session_release(self.session) };
        self.session = replacement;
        self.desc = desc;
        self.info.max_ranges = needed;
        Ok(())
    }

    fn with_session(
        &mut self,
        min_ranges: u32,
        f: impl FnOnce(&mut Self, *mut turbo_session) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match self.pool.clone() {
            Some(pool) => {
                let lease = pool.acquire(min_ranges).map_err(|x| needs_range_write(x, &self.info.execution.backend))?;
                f(self, lease.as_ptr())
            }
            None => f(self, self.session),
        }
    }

    /// Late pooling: `rows` encoded once and, into `out`, the mean of the
    /// encoder's states over each range's tokens (`[ranges.len(), dim]`
    /// when `EncoderOptions::layers` is 0 or 1, or `[layers, ranges.len(),
    /// dim]` when `layers > 1`), neither cut nor normalized. A caller that
    /// wants unit vectors normalizes them with `normalize`, as the core
    /// would.
    pub fn run_ranges(&mut self, rows: &Rows, ranges: &[turbo_token_range], out: &mut Vec<f32>) -> Result<(), Error> {
        let n = ranges.len() as u32;
        let total = n.saturating_mul(self.layers.max(1));
        self.with_session(total, |this, session| {
            let o = turbo_embed_options { layers: this.layers, ..options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE) };
            let mut err = new_error();
            let rc = unsafe { turbo_embed_write_ranges(session, &rows.batch(), ranges.as_ptr(), n, &o, &mut err) };
            check(rc, &err).map_err(|x| needs_range_write(x, &this.info.execution.backend))?;
            forwarded(rows);
            #[cfg(test)]
            {
                this.runs.0 += 1;
                this.runs.1 += rows.len() as usize;
            }
            this.run_on(session, total, out)
        })
    }

    /// `run_ranges` from each of the encoder's last `layers` layers in one
    /// pass (`turbo_embed_write_layer_ranges`): into `out`, `[layers,
    /// ranges.len(), dim]`, block k from layer L - layers + k + 1 of the
    /// model's L, the last block what `run_ranges` gives. `layers` times
    /// the ranges is at most the session's max_ranges.
    pub fn run_layer_ranges(
        &mut self,
        rows: &Rows,
        ranges: &[turbo_token_range],
        layers: u32,
        out: &mut Vec<f32>,
    ) -> Result<(), Error> {
        let n = ranges.len() as u32;
        let total = n.saturating_mul(layers);
        self.with_session(total, |this, session| {
            let o = options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE);
            let mut err = new_error();
            let rc = unsafe {
                turbo_embed_write_layer_ranges(session, &rows.batch(), ranges.as_ptr(), n, layers, &o, &mut err)
            };
            check(rc, &err).map_err(|x| needs_range_write(x, &this.info.execution.backend))?;
            forwarded(rows);
            #[cfg(test)]
            {
                this.runs.0 += 1;
                this.runs.1 += rows.len() as usize;
            }
            this.run_on(session, total, out)
        })
    }

    /// `run_layer_ranges` for the first of those `layers` alone, layer L -
    /// layers + 1 (`turbo_embed_options.layer_alone`): into `out`,
    /// `[ranges.len(), dim]`, block 0 of what `run_layer_ranges` gives.
    /// A device that reads one layer alone stops its encoder there; one
    /// whose cell lacks that field's bit names the field, and the same
    /// vectors are then read from its write of every layer, here and on
    /// each later call. `layers` times the ranges is at most the
    /// session's max_ranges on either.
    pub fn run_layer_alone(
        &mut self,
        rows: &Rows,
        ranges: &[turbo_token_range],
        layers: u32,
        out: &mut Vec<f32>,
    ) -> Result<(), Error> {
        let n = ranges.len() as u32;
        if layers > 1 && !self.every_layer {
            let mut fallback = false;
            self.with_session(n, |this, session| {
                let o = turbo_embed_options { layer_alone: 1, ..options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE) };
                let mut err = new_error();
                let rc = unsafe {
                    turbo_embed_write_layer_ranges(session, &rows.batch(), ranges.as_ptr(), n, layers, &o, &mut err)
                };
                match check(rc, &err) {
                    Ok(()) => {
                        forwarded(rows);
                        this.run_on(session, n, out)
                    }
                    Err(e) if e.code == UNSUPPORTED_OPTION && e.field == 10 => {
                        this.every_layer = true;
                        fallback = true;
                        Ok(())
                    }
                    Err(e) => Err(needs_range_write(e, &this.info.execution.backend)),
                }
            })?;
            if !fallback {
                return Ok(());
            }
        }
        self.run_layer_ranges(rows, ranges, layers, out)?;
        out.truncate(ranges.len() * self.info.dim as usize);
        Ok(())
    }

    /// The row path, for comparing a range with its row: a mean-pooled
    /// vector per row into `out`, normalized when `l2`.
    pub fn run_rows(&mut self, rows: &Rows, l2: bool, out: &mut Vec<f32>) -> Result<(), Error> {
        self.run_rows_pooled(rows, TURBO_POOLING_MEAN, l2, out)
    }

    /// `run_rows` with the pooling given, `TURBO_POOLING_*`:
    /// `TURBO_POOLING_MODEL` pools as the bundle says (the first token for
    /// a CLS-pooled model such as BGE).
    pub fn run_rows_pooled(&mut self, rows: &Rows, pooling: u32, l2: bool, out: &mut Vec<f32>) -> Result<(), Error> {
        self.with_session(0, |this, session| {
            let normalize = if l2 { TURBO_NORMALIZE_L2 } else { TURBO_NORMALIZE_NONE };
            let o = options(pooling, normalize);
            let mut err = new_error();
            check(unsafe { turbo_embed_write_tokens(session, &rows.batch(), &o, &mut err) }, &err)?;
            forwarded(rows);
            this.run_on(session, rows.len(), out)
        })
    }

    /// Sequence-classification CLS path (`#208`): `rows` encoded once and, into
    /// `out`, the first token's (`[CLS]` / `<s>`) final hidden state per row,
    /// unnormalized: `[rows.len(), dim]` values.
    pub fn run_cls(&mut self, rows: &Rows, out: &mut Vec<f32>) -> Result<(), Error> {
        self.run_rows_pooled(rows, TURBO_POOLING_CLS, false, out)
    }

    /// Runs what was written on `session` and reads its `batch` vectors into
    /// `out`, which is resized to them and otherwise reused. A result of
    /// another shape, or a read of fewer bytes, is an error.
    fn run_on(&mut self, session: *mut turbo_session, batch: u32, out: &mut Vec<f32>) -> Result<(), Error> {
        struct Owned(*mut turbo_result);
        impl Drop for Owned {
            fn drop(&mut self) {
                unsafe { turbo_result_release(self.0) };
            }
        }
        let mut err = new_error();
        let mut r = Owned(ptr::null_mut());
        check(unsafe { turbo_session_run(session, &mut r.0, &mut err) }, &err)?;
        let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_result_info>() as u32;
        check(unsafe { turbo_result_get_info(r.0, &mut info, &mut err) }, &err)?;
        if info.dtype != TURBO_DTYPE_F32 {
            return Err(Error::new(UNSUPPORTED_OPTION, format!("vectors of {}, not f32", dtype_name(info.dtype))));
        }
        result_shape(&info, batch, self.info.dim)?;
        out.resize(batch as usize * self.info.dim as usize, 0.0);
        let mut written = 0u64;
        let bytes = size_of_val(out.as_slice()) as u64;
        check(unsafe { turbo_result_read(r.0, out.as_mut_ptr().cast(), bytes, &mut written, &mut err) }, &err)?;
        if written != bytes {
            return Err(Error::new(INTERNAL, format!("the result read {written} bytes of {bytes}")));
        }
        Ok(())
    }
}
