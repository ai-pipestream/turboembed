use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::PromptRole;
use turbo::record::{ROWS_DENSE, ROWS_MIXED};
use turbo::safetensors::{self, Dtype};
use turbo::{TURBO_PROMPT_DOCUMENT, TURBO_PROMPT_NONE, TURBO_PROMPT_QUERY};

use crate::Result;
use crate::api::{Batch, Tokenizer};

/// Which token rows are measured (`--rows`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowKind {
    /// The reference cases that fit seq, cycled, each padded to seq.
    Mixed,
    /// Every row a reference case of at least seq tokens, cut to seq the
    /// way the bundle truncates: batch x seq live tokens, none padding.
    Dense,
}

impl RowKind {
    pub fn parse(s: &str) -> Result<RowKind> {
        match s {
            "mixed" => Ok(RowKind::Mixed),
            "dense" => Ok(RowKind::Dense),
            _ => Err(format!("--rows {s}: mixed or dense")),
        }
    }

    /// As a record names it.
    pub fn name(self) -> &'static str {
        match self {
            RowKind::Mixed => ROWS_MIXED,
            RowKind::Dense => ROWS_DENSE,
        }
    }
}

/// The bundle's reference: each case's text and prompt role
/// (TURBO_PROMPT_*), ids and fp32 vector.
pub struct Reference {
    pub texts: Vec<(String, u32)>,
    pub ids: Vec<Vec<i32>>,
    pub vectors: Vec<Vec<f32>>,
}

impl Reference {
    /// Read through the core's bundle reader, so every byte is checked
    /// against the manifest's hash first.
    pub fn read(bundle: &Bundle) -> Result<Reference> {
        let file = &bundle.manifest.reference.file;
        let bytes = bundle.read_verified(file).map_err(|e| e.message)?;
        let st = safetensors::File::parse(file, &bytes).map_err(|e| e.message)?;
        let ids = st.get("ids", Dtype::I32, 2).map_err(|e| e.message)?;
        let lengths = st.get("lengths", Dtype::I32, 1).map_err(|e| e.message)?.i32s();
        let emb = st.get("embeddings", Dtype::F32, 2).map_err(|e| e.message)?;
        let (width, dim) = (ids.shape[1] as usize, emb.shape[1] as usize);
        let flat = ids.i32s();
        let texts = bundle.manifest.reference.cases.iter().map(|c| {
            let role = match c.prompt_role {
                PromptRole::None => TURBO_PROMPT_NONE,
                PromptRole::Query => TURBO_PROMPT_QUERY,
                PromptRole::Document => TURBO_PROMPT_DOCUMENT,
            };
            (c.text.clone(), role)
        });
        let mut out = Reference {
            texts: texts.collect(),
            ids: Vec::new(),
            vectors: emb.f32s().chunks(dim).map(<[f32]>::to_vec).collect(),
        };
        for (i, &n) in lengths.iter().enumerate() {
            let n = usize::try_from(n).ok().filter(|&n| n >= 1 && n <= width);
            let n = n.ok_or_else(|| format!("{file}: case {i} has length {}", lengths[i]))?;
            out.ids.push(flat[i * width..i * width + n].to_vec());
        }
        if out.ids.len() != out.vectors.len() || out.ids.len() != out.texts.len() {
            return Err(format!(
                "{file}: {} id rows and {} vectors for {} cases",
                out.ids.len(),
                out.vectors.len(),
                out.texts.len()
            ));
        }
        Ok(out)
    }
}

/// The token rows every program is measured on.
#[derive(Debug, Clone, PartialEq)]
pub struct Rows {
    pub kind: RowKind,
    pub batch: u32,
    pub seq: u32,
    pub ids: Vec<i32>,
    pub mask: Vec<i32>,
    pub types: Vec<i32>,
    /// The reference case each row is.
    pub cases: Vec<u32>,
}

impl Rows {
    /// `batch` rows of `seq` tokens: the reference cases no longer than
    /// `seq`, in order, repeated until the batch is full, each padded with
    /// `pad` and mask 0; types all 0. A frame of one row holds one case;
    /// the npu mixed path uses `cycle` so that frame still covers every
    /// case that fits.
    pub fn build(reference: &Reference, pad: i32, batch: u32, seq: u32) -> Result<Rows> {
        let fit: Vec<u32> =
            (0..reference.ids.len() as u32).filter(|&i| reference.ids[i as usize].len() <= seq as usize).collect();
        if fit.is_empty() || batch == 0 {
            return Err(format!("no reference case fits {seq} tokens, or the batch is empty"));
        }
        let n = batch as usize * seq as usize;
        let mut rows = Rows {
            kind: RowKind::Mixed,
            batch,
            seq,
            ids: vec![pad; n],
            mask: vec![0; n],
            types: vec![0; n],
            cases: Vec::new(),
        };
        for r in 0..batch as usize {
            let case = fit[r % fit.len()];
            let ids = &reference.ids[case as usize];
            rows.ids[r * seq as usize..][..ids.len()].copy_from_slice(ids);
            rows.mask[r * seq as usize..][..ids.len()].fill(1);
            rows.cases.push(case);
        }
        Ok(rows)
    }

    /// Every reference case no longer than `seq`, in case order, each
    /// padded to `seq` in its own frame of one row. `batch` stays 1.
    /// That is the compiled frame: the cases are not stacked into a
    /// wider batch the graph was not built for. Types are 0. One fitting
    /// case is the same rows `build` makes at batch 1.
    pub fn cycle(reference: &Reference, pad: i32, seq: u32) -> Result<Rows> {
        let fit: Vec<u32> =
            (0..reference.ids.len() as u32).filter(|&i| reference.ids[i as usize].len() <= seq as usize).collect();
        if fit.is_empty() || seq == 0 {
            return Err(format!("no reference case fits {seq} tokens"));
        }
        let seq_n = seq as usize;
        let mut ids = vec![pad; fit.len() * seq_n];
        let mut mask = vec![0i32; fit.len() * seq_n];
        let mut cases = Vec::with_capacity(fit.len());
        for (r, &case) in fit.iter().enumerate() {
            let src = &reference.ids[case as usize];
            let at = r * seq_n;
            ids[at..at + src.len()].copy_from_slice(src);
            mask[at..at + src.len()].fill(1);
            cases.push(case);
        }
        Ok(Rows { kind: RowKind::Mixed, batch: 1, seq, ids, mask, types: vec![0; fit.len() * seq_n], cases })
    }

    /// `batch` rows of exactly `seq` live tokens: the reference cases of
    /// `seq` tokens or more, in order, repeated until the batch is full; a
    /// longer case is its text encoded again and cut to `seq` the way the
    /// bundle truncates. Types all 0.
    pub fn dense(reference: &Reference, tokenizer: &Tokenizer, batch: u32, seq: u32) -> Result<Rows> {
        let long: Vec<u32> =
            (0..reference.ids.len() as u32).filter(|&i| reference.ids[i as usize].len() >= seq as usize).collect();
        if long.is_empty() || batch == 0 || seq == 0 {
            return Err(format!("no reference case has {seq} tokens or more, or the batch is empty"));
        }
        if !reference.ids.iter().any(|r| r.len() <= seq as usize) {
            return Err(format!("no reference case fits {seq} tokens without truncation"));
        }
        let mut cut = Vec::new();
        for &case in &long {
            let whole = &reference.ids[case as usize];
            let ids = if whole.len() == seq as usize {
                whole.clone()
            } else {
                let (text, role) = &reference.texts[case as usize];
                tokenizer.encode(text, *role, seq)?
            };
            if ids.len() != seq as usize {
                return Err(format!("reference case {case} cut to {seq} tokens came back as {}", ids.len()));
            }
            cut.push(ids);
        }
        let n = batch as usize * seq as usize;
        let mut ids = Vec::with_capacity(n);
        let mut cases = Vec::new();
        for r in 0..batch as usize {
            ids.extend_from_slice(&cut[r % long.len()]);
            cases.push(long[r % long.len()]);
        }
        Ok(Rows { kind: RowKind::Dense, batch, seq, ids, mask: vec![1; n], types: vec![0; n], cases })
    }

    /// Reference case `case` alone, at its own length.
    pub fn case(reference: &Reference, case: u32) -> Rows {
        let ids = reference.ids[case as usize].clone();
        let n = ids.len();
        Rows {
            kind: RowKind::Mixed,
            batch: 1,
            seq: n as u32,
            ids,
            mask: vec![1; n],
            types: vec![0; n],
            cases: vec![case],
        }
    }

    /// Whether row `r` is its reference case whole, as the reference
    /// vector was made from it, rather than cut to seq.
    pub fn whole(&self, r: usize, reference: &Reference) -> bool {
        self.live(r) == reference.ids[self.cases[r] as usize].as_slice()
    }

    /// Row `r`'s live ids, without padding.
    pub fn live(&self, r: usize) -> &[i32] {
        let row = r * self.seq as usize..(r + 1) * self.seq as usize;
        let n = self.mask[row.clone()].iter().filter(|&&m| m == 1).count();
        &self.ids[row][..n]
    }

    pub fn live_tokens(&self) -> u64 {
        self.mask.iter().filter(|&&m| m == 1).count() as u64
    }

    /// Measured rows. One frame has `batch` of them. A batch-1 cycle has
    /// one per fitting case.
    pub fn row_count(&self) -> usize {
        self.cases.len()
    }

    /// Compiled frames in this pass. A cycle of batch 1 is one frame per
    /// case. Every other set is one frame.
    pub fn frames(&self) -> usize {
        self.row_count() / self.batch.max(1) as usize
    }

    /// Frame `frame` alone: `batch` rows of `seq`, its own cases.
    pub fn frame(&self, frame: usize) -> Rows {
        let rows = self.batch as usize;
        let n = rows * self.seq as usize;
        let start = frame * n;
        let case_at = frame * rows;
        Rows {
            kind: self.kind,
            batch: self.batch,
            seq: self.seq,
            ids: self.ids[start..start + n].to_vec(),
            mask: self.mask[start..start + n].to_vec(),
            types: self.types[start..start + n].to_vec(),
            cases: self.cases[case_at..case_at + rows].to_vec(),
        }
    }

    /// Frame `frame` as the batch `turbo_embed_write_tokens` is given.
    pub fn frame_batch(&self, frame: usize) -> Batch<'_> {
        let n = self.batch as usize * self.seq as usize;
        let start = frame * n;
        Batch {
            batch: self.batch,
            seq: self.seq,
            ids: &self.ids[start..start + n],
            mask: &self.mask[start..start + n],
            types: &self.types[start..start + n],
        }
    }

    /// Where frame `frame`'s input files are written, under `work`.
    pub fn frame_dir(work: &std::path::Path, frame: usize) -> std::path::PathBuf {
        work.join(format!("frame-{frame}"))
    }

    /// The positions a packed run computes: each row's through its last
    /// live token.
    pub fn packed_tokens(&self) -> u64 {
        let seq = self.seq as usize;
        (0..self.row_count())
            .map(|r| self.mask[r * seq..(r + 1) * seq].iter().rposition(|&m| m != 0).map_or(0, |p| p + 1) as u64)
            .sum()
    }

    /// The positions a kernel on the padded rows computes. One frame is
    /// batch x seq. A batch-1 cycle is one frame per case, so the count
    /// is every frame.
    pub fn padded_tokens(&self) -> u64 {
        self.row_count() as u64 * self.seq as u64
    }

    /// docs/benchmarks.md, "Token rows": SHA-256 of `turbo-bench rows 1`,
    /// a NUL, batch and seq as little-endian u32, then ids, mask and types
    /// as little-endian i32, row-major. A batch-1 cycle hashes every
    /// frame in order; `batch` in the header stays 1.
    pub fn sha256(&self) -> String {
        let mut b = b"turbo-bench rows 1\0".to_vec();
        b.extend(self.batch.to_le_bytes());
        b.extend(self.seq.to_le_bytes());
        for v in self.ids.iter().chain(&self.mask).chain(&self.types) {
            b.extend(v.to_le_bytes());
        }
        sha256_hex(&b)
    }

    pub fn as_batch(&self) -> Batch<'_> {
        Batch { batch: self.batch, seq: self.seq, ids: &self.ids, mask: &self.mask, types: &self.types }
    }

    /// One row as a batch of one, at its own length.
    pub fn single(&self, r: usize) -> Rows {
        let ids = self.live(r).to_vec();
        let n = ids.len();
        Rows {
            kind: self.kind,
            batch: 1,
            seq: n as u32,
            ids,
            mask: vec![1; n],
            types: vec![0; n],
            cases: vec![self.cases[r]],
        }
    }
}
