use super::*;
impl Encoder {
    /// Late pooling: `rows` encoded once and, into `out`, the mean of the
    /// encoder's final states over each range's tokens, neither cut nor
    /// normalized: `[ranges.len(), dim]` values. A caller that wants unit
    /// vectors normalizes them with `normalize`, as the core would.
    pub fn run_ranges(&mut self, rows: &Rows, ranges: &[turbo_token_range], out: &mut Vec<f32>) -> Result<(), Error> {
        let o = options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE);
        let mut err = new_error();
        let rc = unsafe {
            turbo_embed_write_ranges(self.session, &rows.batch(), ranges.as_ptr(), ranges.len() as u32, &o, &mut err)
        };
        check(rc, &err).map_err(|x| needs_range_write(x, &self.info.execution.backend))?;
        self.run(ranges.len() as u32, out)
    }

    /// The row path, for comparing a range with its row: a mean-pooled
    /// vector per row into `out`, normalized when `l2`.
    pub fn run_rows(&mut self, rows: &Rows, l2: bool, out: &mut Vec<f32>) -> Result<(), Error> {
        let normalize = if l2 { TURBO_NORMALIZE_L2 } else { TURBO_NORMALIZE_NONE };
        let o = options(TURBO_POOLING_MEAN, normalize);
        let mut err = new_error();
        check(unsafe { turbo_embed_write_tokens(self.session, &rows.batch(), &o, &mut err) }, &err)?;
        self.run(rows.len(), out)
    }

    /// Sequence-classification CLS path (`#208`): `rows` encoded once and, into
    /// `out`, the first token's (`[CLS]` / `<s>`) final hidden state per row,
    /// unnormalized: `[rows.len(), dim]` values.
    pub fn run_cls(&mut self, rows: &Rows, out: &mut Vec<f32>) -> Result<(), Error> {
        let o = options(TURBO_POOLING_CLS, TURBO_NORMALIZE_NONE);
        let mut err = new_error();
        check(unsafe { turbo_embed_write_tokens(self.session, &rows.batch(), &o, &mut err) }, &err)?;
        self.run(rows.len(), out)
    }

    /// Runs what was written and reads its `batch` vectors into `out`,
    /// which is resized to them and otherwise reused. A result of another
    /// shape, or a read of fewer bytes, is an error.
    fn run(&mut self, batch: u32, out: &mut Vec<f32>) -> Result<(), Error> {
        struct Owned(*mut turbo_result);
        impl Drop for Owned {
            fn drop(&mut self) {
                unsafe { turbo_result_release(self.0) };
            }
        }
        let mut err = new_error();
        let mut r = Owned(ptr::null_mut());
        check(unsafe { turbo_session_run(self.session, &mut r.0, &mut err) }, &err)?;
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
