/// A session's desc with tuning OFF, so a developer with TURBO_AUTOTUNE set
/// in their shell still tests the defaults; TURBO_TEST_TUNING=1 turns it
/// ON instead, for a second pass of the suite with tuned sessions.
pub fn session_desc(max_batch: u32, max_seq: u32, precision: u32) -> turbo_session_desc {
    let tuning =
        if std::env::var("TURBO_TEST_TUNING").is_ok_and(|v| v == "1") { TURBO_AUTOTUNE_ON } else { TURBO_AUTOTUNE_OFF };
    turbo_session_desc {
        struct_size: size_of::<turbo_session_desc>() as u32,
        max_batch,
        max_seq,
        precision,
        tuning,
        tuning_budget_ms: 0,
        cpu_threads: 0,
        max_ranges: 0,
    }
}

pub fn embed_options() -> turbo_embed_options {
    turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: 0,
        max_tokens: 0,
        prompt_role: 0,
        normalize: 0,
        pooling: 0,
        output_dim: 0,
    }
}

/// A session made through the C interface, released on drop.
pub struct Session(pub *mut turbo_session);

impl Session {
    pub fn create(m: *mut turbo_model, desc: Option<&turbo_session_desc>) -> Result<Session, Failure> {
        let mut s = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_session_create(m, desc.map_or(ptr::null(), |d| d as *const _), &mut s, &mut err) };
        if rc != 0 {
            assert!(s.is_null(), "a failed create wrote out");
            return Err(failure(rc, &err));
        }
        Ok(Session(s))
    }

    pub fn info(&self) -> turbo_session_info {
        let mut info: turbo_session_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_session_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_session_get_info(self.0, &mut info, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        info
    }

    pub fn write_text(&self, texts: &[&str], opts: Option<&turbo_embed_options>) -> Result<(), Failure> {
        let views: Vec<turbo_text> = texts.iter().map(|s| text(s)).collect();
        let mut err = new_error();
        let o = opts.map_or(ptr::null(), |o| o as *const _);
        let rc = unsafe { turbo_embed_write_text(self.0, views.as_ptr(), views.len() as u32, o, &mut err) };
        if rc != 0 {
            return Err(failure(rc, &err));
        }
        Ok(())
    }

    pub fn write_tokens(&self, b: &turbo_token_batch, opts: Option<&turbo_embed_options>) -> Result<(), Failure> {
        let mut err = new_error();
        let o = opts.map_or(ptr::null(), |o| o as *const _);
        let rc = unsafe { turbo_embed_write_tokens(self.0, b, o, &mut err) };
        if rc != 0 {
            return Err(failure(rc, &err));
        }
        Ok(())
    }

    /// Late pooling: the rows `b` and the `ranges` of their tokens.
    pub fn write_ranges(
        &self,
        b: &turbo_token_batch,
        ranges: &[turbo_token_range],
        opts: Option<&turbo_embed_options>,
    ) -> Result<(), Failure> {
        let mut err = new_error();
        let o = opts.map_or(ptr::null(), |o| o as *const _);
        let rc = unsafe { turbo_embed_write_ranges(self.0, b, ranges.as_ptr(), ranges.len() as u32, o, &mut err) };
        if rc != 0 {
            return Err(failure(rc, &err));
        }
        Ok(())
    }

    pub fn run(&self) -> Result<Outcome, Failure> {
        let mut r = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_session_run(self.0, &mut r, &mut err) };
        if rc != 0 {
            assert!(r.is_null(), "a failed run wrote out");
            return Err(failure(rc, &err));
        }
        Ok(Outcome(r))
    }

    /// Write `texts`, run, and read the vectors.
    pub fn embed(&self, texts: &[&str], opts: Option<&turbo_embed_options>) -> Result<Vec<Vec<f32>>, Failure> {
        self.write_text(texts, opts)?;
        Ok(self.run()?.rows())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        unsafe { turbo_session_release(self.0) };
    }
}

/// The columns [start, end) of row `row`.
pub fn range(row: u32, start: u32, end: u32) -> turbo_token_range {
    turbo_token_range { row, start, end }
}

/// Rows the caller tokenized, [batch, seq] with a row stride, owning their
/// arrays.
pub struct Tokens {
    pub ids: Vec<i32>,
    pub mask: Vec<i32>,
    pub types: Option<Vec<i32>>,
    pub batch: u32,
    pub seq: u32,
    pub stride: u32,
}

impl Tokens {
    /// `rows` padded with `pad` and mask 0 to the longest.
    pub fn new(rows: &[Vec<i32>], pad: i32) -> Tokens {
        let seq = rows.iter().map(Vec::len).max().unwrap_or(0);
        let mut t = Tokens {
            ids: vec![pad; rows.len() * seq],
            mask: vec![0; rows.len() * seq],
            types: None,
            batch: rows.len() as u32,
            seq: seq as u32,
            stride: 0,
        };
        for (i, r) in rows.iter().enumerate() {
            t.ids[i * seq..i * seq + r.len()].copy_from_slice(r);
            t.mask[i * seq..i * seq + r.len()].fill(1);
        }
        t
    }

    pub fn batch(&self) -> turbo_token_batch {
        turbo_token_batch {
            struct_size: size_of::<turbo_token_batch>() as u32,
            batch: self.batch,
            seq: self.seq,
            row_stride: self.stride,
            ids: self.ids.as_ptr(),
            mask: self.mask.as_ptr(),
            types: self.types.as_ref().map_or(ptr::null(), |t| t.as_ptr()),
        }
    }
}

/// A result, released on drop.
pub struct Outcome(pub *mut turbo_result);

impl Outcome {
    pub fn info(&self) -> turbo_result_info {
        let mut info: turbo_result_info = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<turbo_result_info>() as u32;
        let mut err = new_error();
        let rc = unsafe { turbo_result_get_info(self.0, &mut info, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        info
    }

    /// The vectors through turbo_result_read into `flat`, batch x dim
    /// values, as a caller that keeps one buffer for every batch reads
    /// them.
    pub fn read_into(&self, flat: &mut Vec<f32>) {
        let info = self.info();
        flat.resize(info.batch as usize * info.dim as usize, 0.0);
        let mut written = 0u64;
        let mut err = new_error();
        let rc = unsafe {
            turbo_result_read(self.0, flat.as_mut_ptr() as *mut _, (flat.len() * 4) as u64, &mut written, &mut err)
        };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        assert_eq!(written, info.bytes);
    }

    /// The vectors through turbo_result_read, one Vec per row.
    pub fn rows(&self) -> Vec<Vec<f32>> {
        let info = self.info();
        let mut flat = vec![f32::NAN; info.batch as usize * info.dim as usize];
        let mut written = 0u64;
        let mut err = new_error();
        let rc = unsafe {
            turbo_result_read(self.0, flat.as_mut_ptr() as *mut _, (flat.len() * 4) as u64, &mut written, &mut err)
        };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        assert_eq!(written, info.bytes);
        flat.chunks(info.dim as usize).map(<[f32]>::to_vec).collect()
    }
}

impl Drop for Outcome {
    fn drop(&mut self) {
        unsafe { turbo_result_release(self.0) };
    }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).abs()).fold(0.0, f64::max)
}

// ---- Safetensors, read plainly ------------------------------------------------

/// A safetensors file, read without the library, for the tests that check
/// it.
pub struct St {
    header: Value,
    data: Vec<u8>,
}

impl St {
    pub fn read(path: &Path) -> St {
        let bytes = fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        St { header, data: bytes[8 + n..].to_vec() }
    }

    pub fn shape(&self, name: &str) -> Vec<usize> {
        let t = &self.header[name];
        assert!(t.is_object(), "no tensor {name}");
        t["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect()
    }

    fn bytes(&self, name: &str, dtype: &str) -> &[u8] {
        let t = &self.header[name];
        assert_eq!(t["dtype"], dtype, "{name}");
        let o = &t["data_offsets"];
        &self.data[o[0].as_u64().unwrap() as usize..o[1].as_u64().unwrap() as usize]
    }

    pub fn f32s(&self, name: &str) -> Vec<f32> {
        self.bytes(name, "F32").chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    pub fn i32s(&self, name: &str) -> Vec<i32> {
        self.bytes(name, "I32").chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

/// A bundle's reference: each case's ids and vector.
pub struct Reference {
    pub ids: Vec<Vec<i32>>,
    pub embeddings: Vec<Vec<f32>>,
}

impl Reference {
    pub fn read(bundle: &Path, manifest: &Value) -> Reference {
        let st = St::read(&bundle.join(manifest["reference"]["file"].as_str().unwrap()));
        let (width, dim) = (st.shape("ids")[1], st.shape("embeddings")[1]);
        let (ids, lengths, emb) = (st.i32s("ids"), st.i32s("lengths"), st.f32s("embeddings"));
        Reference {
            ids: lengths.iter().enumerate().map(|(i, &n)| ids[i * width..i * width + n as usize].to_vec()).collect(),
            embeddings: emb.chunks(dim).map(<[f32]>::to_vec).collect(),
        }
    }
}
