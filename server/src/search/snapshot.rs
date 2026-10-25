//! Deterministic binary snapshot persistence and atomic file writes for
//! `HybridIndex` (`#1221`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use prost::Message;
use turbo_analysis::Span;
use turbo_chunker::terms::Bm25PostingIndex;
use turbo_vector_index::{QuantizedIndex, TokenIndex, TurboQuant, VectorIndex};

use crate::api::Result;
use crate::proto::search::{ChunkInput, IndexSpec, SparseEntryProto, TokenVectorInput, VectorStorageKind};
use crate::search::registry::{
    CompiledDense, HybridIndex, RetainedDocument, StoredChunk, bad_arg, corrupt_snapshot, io_err, not_found,
    parse_metric, parse_space,
};

const SNAPSHOT_MAGIC: &[u8; 8] = b"TSRCHIX\x01";
const MAX_SNAPSHOT_BYTES: u64 = 128 << 20;

/// Validates that `file_name` is a single relative file name with no path
/// separators, parent traversal, or root prefix, and resolves it inside
/// `base_dir`.
pub fn resolve_snapshot_path(base_dir: &Path, file_name: &str) -> Result<PathBuf> {
    if file_name.is_empty() {
        return Err(bad_arg("snapshot file name must not be empty"));
    }
    if file_name.contains('/') || file_name.contains('\\') || file_name.contains(':') || file_name.contains('\0') {
        return Err(bad_arg("snapshot file name must be a plain file name without path separators"));
    }
    let rel = Path::new(file_name);
    let mut comps = rel.components();
    match (comps.next(), comps.next()) {
        (Some(Component::Normal(os)), None) if !os.is_empty() && os != "." && os != ".." => Ok(base_dir.join(os)),
        _ => Err(bad_arg("snapshot file name must be a single normal file name")),
    }
}

pub fn persist_index_to_dir(ix: &HybridIndex, base_dir: &Path, file_name: &str) -> Result<u64> {
    let path = resolve_snapshot_path(base_dir, file_name)?;
    let bytes = encode_snapshot(ix);
    std::fs::create_dir_all(base_dir).map_err(|e| io_err(format!("create snapshot dir: {e}")))?;
    let mut tmp_os = path.as_os_str().to_os_string();
    tmp_os.push(".tmp_snapshot");
    let tmp = PathBuf::from(tmp_os);
    std::fs::write(&tmp, &bytes).map_err(|e| io_err(format!("write snapshot temp file: {e}")))?;
    std::fs::rename(&tmp, &path).map_err(|e| io_err(format!("atomic rename snapshot file: {e}")))?;
    Ok(bytes.len() as u64)
}

pub fn load_index_from_dir(base_dir: &Path, file_name: &str, override_name: Option<&str>) -> Result<HybridIndex> {
    let path = resolve_snapshot_path(base_dir, file_name)?;
    let meta = match std::fs::symlink_metadata(&path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(not_found(format!("snapshot file `{file_name}` does not exist")));
        }
        Err(e) => return Err(io_err(format!("stat snapshot file: {e}"))),
    };
    if !meta.is_file() {
        return Err(bad_arg(format!("snapshot `{file_name}` is not a regular file")));
    }
    if meta.len() > MAX_SNAPSHOT_BYTES {
        return Err(bad_arg(format!("snapshot `{file_name}` exceeds maximum allowed size")));
    }
    let bytes = std::fs::read(&path).map_err(|e| io_err(format!("read snapshot file: {e}")))?;
    let mut ix = decode_snapshot(&bytes)?;
    if let Some(name) = override_name
        && !name.is_empty()
    {
        ix.spec.index_name = name.to_owned();
    }
    Ok(ix)
}

pub fn encode_snapshot(ix: &HybridIndex) -> Vec<u8> {
    let mut out = Vec::with_capacity(4096);
    out.extend_from_slice(SNAPSHOT_MAGIC);
    let spec_bytes = ix.spec.encode_to_vec();
    write_bytes(&spec_bytes, &mut out);
    out.push(u8::from(ix.sealed));

    out.extend_from_slice(&(ix.docs.len() as u32).to_le_bytes());
    for d in &ix.docs {
        write_str(&d.doc_id, &mut out);
        write_str(&d.text, &mut out);
        write_bytes(&d.graph, &mut out);
        out.extend_from_slice(&(d.raw_chunks.len() as u32).to_le_bytes());
        for c in &d.raw_chunks {
            write_bytes(&c.encode_to_vec(), &mut out);
        }
        out.extend_from_slice(&(d.chunks.len() as u32).to_le_bytes());
        for c in &d.chunks {
            write_bytes(&c.encode_to_vec(), &mut out);
        }
        write_map(&d.attributes, &mut out);
    }

    out.extend_from_slice(&(ix.chunks.len() as u32).to_le_bytes());
    for c in &ix.chunks {
        write_str(&c.chunk_id, &mut out);
        write_str(&c.doc_id, &mut out);
        write_str(&c.level_kind, &mut out);
        out.extend_from_slice(&c.span.start.to_le_bytes());
        out.extend_from_slice(&c.span.end.to_le_bytes());
        match c.parent_span {
            None => out.push(0),
            Some(ps) => {
                out.push(1);
                out.extend_from_slice(&ps.start.to_le_bytes());
                out.extend_from_slice(&ps.end.to_le_bytes());
            }
        }
        write_str(&c.text, &mut out);
        write_str(&c.parent_text, &mut out);
        out.extend_from_slice(&(c.terms.len() as u32).to_le_bytes());
        for (t, sp) in &c.terms {
            write_str(t, &mut out);
            out.extend_from_slice(&sp.start.to_le_bytes());
            out.extend_from_slice(&sp.end.to_le_bytes());
        }
        match &c.dense_vector {
            None => out.extend_from_slice(&0u32.to_le_bytes()),
            Some(v) => {
                out.extend_from_slice(&(v.len() as u32).to_le_bytes());
                for &f in v {
                    out.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        out.extend_from_slice(&(c.sparse_terms.len() as u32).to_le_bytes());
        for s in &c.sparse_terms {
            write_bytes(&s.encode_to_vec(), &mut out);
        }
        out.extend_from_slice(&(c.token_vectors.len() as u32).to_le_bytes());
        for tv in &c.token_vectors {
            write_bytes(&tv.encode_to_vec(), &mut out);
        }
        write_map(&c.attributes, &mut out);
    }

    let bm25_bytes = ix.bm25.to_bytes();
    write_bytes(&bm25_bytes, &mut out);

    match &ix.dense {
        CompiledDense::Empty => out.push(0),
        CompiledDense::Flat(v_ix) => {
            out.push(1);
            out.extend_from_slice(&(v_ix.len() as u32).to_le_bytes());
            for r in 0..v_ix.len() as u32 {
                out.extend_from_slice(&v_ix.meta(r).to_le_bytes());
            }
            write_bytes(&v_ix.to_bytes(), &mut out);
        }
        CompiledDense::Quantized(q_ix) => {
            out.push(2);
            out.extend_from_slice(&q_ix.scale().to_le_bytes());
            out.extend_from_slice(&(q_ix.len() as u32).to_le_bytes());
            for r in 0..q_ix.len() as u32 {
                out.extend_from_slice(&q_ix.meta(r).to_le_bytes());
            }
            write_bytes(&q_ix.to_bytes(), &mut out);
        }
    }

    match &ix.token_index {
        None => out.push(0),
        Some(t_ix) => {
            out.push(1);
            out.extend_from_slice(&(ix.token_candidates.len() as u32).to_le_bytes());
            for &c in &ix.token_candidates {
                out.extend_from_slice(&c.to_le_bytes());
            }
            write_bytes(&t_ix.to_bytes(), &mut out);
        }
    }
    out
}

pub fn decode_snapshot(bytes: &[u8]) -> Result<HybridIndex> {
    let mut r = SnapReader(bytes);
    if r.take(SNAPSHOT_MAGIC.len())? != SNAPSHOT_MAGIC {
        return Err(corrupt_snapshot("not a valid TurboSearchService snapshot"));
    }
    let spec_bytes = r.bytes()?;
    let spec = IndexSpec::decode(spec_bytes).map_err(|e| corrupt_snapshot(format!("decode IndexSpec: {e}")))?;
    let space = parse_space(spec.space.as_ref()).map_err(|f| corrupt_snapshot(f.message))?;
    let metric = parse_metric(spec.metric).map_err(|f| corrupt_snapshot(f.message))?;
    let storage =
        VectorStorageKind::try_from(spec.storage).map_err(|_| corrupt_snapshot("invalid storage kind in snapshot"))?;
    let quant = match storage {
        VectorStorageKind::FlatF32 => None,
        VectorStorageKind::TurboQuant => {
            let bits = if spec.quant_bits == 0 { 4 } else { spec.quant_bits as u8 };
            Some(TurboQuant::new(bits, spec.quant_calibrate).map_err(|e| corrupt_snapshot(e.to_string()))?)
        }
    };
    let sealed = r.u8()? != 0;

    let n_docs = r.u32()? as usize;
    if n_docs > r.0.len() / 24 {
        return Err(corrupt_snapshot("snapshot document count exceeds remaining bytes"));
    }
    let mut docs = Vec::with_capacity(n_docs);
    for _ in 0..n_docs {
        let doc_id = r.str()?;
        let text = r.str()?;
        let graph = r.bytes()?.to_vec();
        let n_raw = r.u32()? as usize;
        if n_raw > r.0.len() / 4 {
            return Err(corrupt_snapshot("snapshot raw_chunks count exceeds remaining bytes"));
        }
        let mut raw_chunks = Vec::with_capacity(n_raw);
        for _ in 0..n_raw {
            let cb = r.bytes()?;
            raw_chunks.push(ChunkInput::decode(cb).map_err(|e| corrupt_snapshot(format!("decode ChunkInput: {e}")))?);
        }
        let n_ch = r.u32()? as usize;
        if n_ch > r.0.len() / 4 {
            return Err(corrupt_snapshot("snapshot doc chunks count exceeds remaining bytes"));
        }
        let mut chunks = Vec::with_capacity(n_ch);
        for _ in 0..n_ch {
            let cb = r.bytes()?;
            chunks.push(ChunkInput::decode(cb).map_err(|e| corrupt_snapshot(format!("decode ChunkInput: {e}")))?);
        }
        let attributes = r.map()?;
        docs.push(RetainedDocument { doc_id, text, graph, raw_chunks, chunks, attributes });
    }

    let n_chunks = r.u32()? as usize;
    if n_chunks > r.0.len() / 49 {
        return Err(corrupt_snapshot("snapshot chunk count exceeds remaining bytes"));
    }
    let mut chunks = Vec::with_capacity(n_chunks);
    let mut seen_terms = BTreeSet::new();
    let mut term_df = BTreeMap::new();

    for _ in 0..n_chunks {
        let chunk_id = r.str()?;
        let doc_id = r.str()?;
        let level_kind = r.str()?;
        let (s0, s1) = (r.u32()?, r.u32()?);
        if s0 > s1 {
            return Err(corrupt_snapshot("snapshot chunk span start > end"));
        }
        let span = Span::new(s0, s1);
        let parent_span = match r.u8()? {
            0 => None,
            1 => {
                let (p0, p1) = (r.u32()?, r.u32()?);
                if p0 > p1 {
                    return Err(corrupt_snapshot("snapshot parent_span start > end"));
                }
                Some(Span::new(p0, p1))
            }
            b => return Err(corrupt_snapshot(format!("invalid parent_span tag {b}"))),
        };
        let text = r.str()?;
        let parent_text = r.str()?;
        let n_terms = r.u32()? as usize;
        if n_terms > r.0.len() / 12 {
            return Err(corrupt_snapshot("snapshot term count exceeds remaining bytes"));
        }
        let mut terms = Vec::with_capacity(n_terms);
        let mut unique = BTreeSet::new();
        for _ in 0..n_terms {
            let t = r.str()?;
            let (ts0, ts1) = (r.u32()?, r.u32()?);
            if ts0 > ts1 {
                return Err(corrupt_snapshot("snapshot term span start > end"));
            }
            let sp = Span::new(ts0, ts1);
            seen_terms.insert(t.clone());
            unique.insert(t.clone());
            terms.push((t, sp));
        }
        for u in unique {
            *term_df.entry(u).or_insert(0) += 1;
        }

        let vec_len = r.u32()? as usize;
        if vec_len > r.0.len() / 4 {
            return Err(corrupt_snapshot("snapshot dense vector length exceeds remaining bytes"));
        }
        let dense_vector = if vec_len == 0 {
            None
        } else {
            if vec_len != space.dim as usize {
                return Err(corrupt_snapshot("snapshot dense vector dimension mismatch"));
            }
            let mut v = Vec::with_capacity(vec_len);
            for _ in 0..vec_len {
                let f = r.f32()?;
                if !f.is_finite() {
                    return Err(corrupt_snapshot("non-finite float in snapshot dense vector"));
                }
                v.push(f);
            }
            Some(v)
        };
        let n_sparse = r.u32()? as usize;
        if n_sparse > r.0.len() / 4 {
            return Err(corrupt_snapshot("snapshot sparse terms count exceeds remaining bytes"));
        }
        let mut sparse_terms = Vec::with_capacity(n_sparse);
        for _ in 0..n_sparse {
            let sb = r.bytes()?;
            sparse_terms
                .push(SparseEntryProto::decode(sb).map_err(|e| corrupt_snapshot(format!("decode SparseEntry: {e}")))?);
        }
        let n_tv = r.u32()? as usize;
        if n_tv > r.0.len() / 4 {
            return Err(corrupt_snapshot("snapshot token vectors count exceeds remaining bytes"));
        }
        let mut token_vectors = Vec::with_capacity(n_tv);
        for _ in 0..n_tv {
            let tvb = r.bytes()?;
            token_vectors
                .push(TokenVectorInput::decode(tvb).map_err(|e| corrupt_snapshot(format!("decode TokenVec: {e}")))?);
        }
        let attributes = r.map()?;
        chunks.push(StoredChunk {
            chunk_id,
            doc_id,
            level_kind,
            span,
            parent_span,
            text,
            parent_text,
            terms,
            dense_vector,
            sparse_terms,
            token_vectors,
            attributes,
        });
    }

    let bm25_bytes = r.bytes()?;
    let bm25 = Bm25PostingIndex::from_bytes(bm25_bytes).map_err(corrupt_snapshot)?;

    let dense = match r.u8()? {
        0 => CompiledDense::Empty,
        1 => {
            let n = r.u32()? as usize;
            if n > r.0.len() / 4 {
                return Err(corrupt_snapshot("snapshot dense meta count exceeds remaining bytes"));
            }
            let mut meta = Vec::with_capacity(n);
            for _ in 0..n {
                meta.push(r.u32()?);
            }
            let vb = r.bytes()?;
            CompiledDense::Flat(VectorIndex::from_bytes(vb, meta).map_err(|e| corrupt_snapshot(e.to_string()))?)
        }
        2 => {
            let scale = r.f64()?;
            let n = r.u32()? as usize;
            if n > r.0.len() / 4 {
                return Err(corrupt_snapshot("snapshot quantized meta count exceeds remaining bytes"));
            }
            let mut meta = Vec::with_capacity(n);
            for _ in 0..n {
                meta.push(r.u32()?);
            }
            let qb = r.bytes()?;
            let q_cfg = quant.ok_or_else(|| corrupt_snapshot("missing TurboQuant config"))?;
            let q_ix = QuantizedIndex::from_bytes(space.clone(), metric, q_cfg, scale, qb, meta)
                .map_err(|e| corrupt_snapshot(e.to_string()))?;
            CompiledDense::Quantized(Box::new(q_ix))
        }
        b => return Err(corrupt_snapshot(format!("invalid dense snapshot tag {b}"))),
    };

    let (token_index, token_candidates) = match r.u8()? {
        0 => (None, Vec::new()),
        1 => {
            let n = r.u32()? as usize;
            if n > r.0.len() / 4 {
                return Err(corrupt_snapshot("snapshot token candidate count exceeds remaining bytes"));
            }
            let mut cands = Vec::with_capacity(n);
            for _ in 0..n {
                cands.push(r.u32()?);
            }
            let tb = r.bytes()?;
            let t_ix = TokenIndex::from_bytes(tb, cands.clone()).map_err(|e| corrupt_snapshot(e.to_string()))?;
            (Some(t_ix), cands)
        }
        b => return Err(corrupt_snapshot(format!("invalid token_index tag {b}"))),
    };

    if !r.0.is_empty() {
        return Err(corrupt_snapshot("trailing bytes in search index snapshot"));
    }

    Ok(HybridIndex {
        spec,
        space,
        metric,
        storage,
        quant,
        sealed,
        docs,
        chunks,
        bm25,
        dense,
        token_index,
        token_candidates,
        seen_terms,
        term_df,
    })
}

fn write_bytes(b: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(b.len() as u32).to_le_bytes());
    out.extend_from_slice(b);
}

fn write_str(s: &str, out: &mut Vec<u8>) {
    write_bytes(s.as_bytes(), out);
}

fn write_map(m: &BTreeMap<String, String>, out: &mut Vec<u8>) {
    out.extend_from_slice(&(m.len() as u32).to_le_bytes());
    for (k, v) in m {
        write_str(k, out);
        write_str(v, out);
    }
}

struct SnapReader<'a>(&'a [u8]);

impl<'a> SnapReader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            return Err(corrupt_snapshot("truncated search index snapshot"));
        }
        let (h, t) = self.0.split_at(n);
        self.0 = t;
        Ok(h)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn bytes(&mut self) -> Result<&'a [u8]> {
        let len = self.u32()? as usize;
        self.take(len)
    }

    fn str(&mut self) -> Result<String> {
        let b = self.bytes()?;
        std::str::from_utf8(b).map(str::to_owned).map_err(|_| corrupt_snapshot("invalid UTF-8 in snapshot"))
    }

    fn map(&mut self) -> Result<BTreeMap<String, String>> {
        let n = self.u32()? as usize;
        if n > self.0.len() / 8 {
            return Err(corrupt_snapshot("snapshot map entry count exceeds remaining bytes"));
        }
        let mut out = BTreeMap::new();
        for _ in 0..n {
            out.insert(self.str()?, self.str()?);
        }
        Ok(out)
    }
}
