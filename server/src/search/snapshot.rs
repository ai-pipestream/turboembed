//! Deterministic binary snapshot persistence and atomic file writes for
//! `HybridIndex` (`#1221`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use prost::Message;
use turbo_analysis::Span;
use turbo_chunker::terms::{Bm25PostingIndex, Bm25PostingIndexBuilder};
use turbo_vector_index::{QuantizedIndex, TokenIndex, TurboQuant, VectorIndex};

use crate::api::Result;
use crate::proto::search::{ChunkInput, IndexSpec, SparseEntryProto, TokenVectorInput, VectorStorageKind};
use crate::search::registry::{
    CompiledDense, HybridIndex, RetainedDocument, StoredChunk, bad_arg, parse_metric, parse_space,
};

const SNAPSHOT_MAGIC: &[u8; 8] = b"TSRCHIX\x01";

pub fn persist_index_to_path(ix: &HybridIndex, path: &Path) -> Result<u64> {
    let bytes = encode_snapshot(ix);
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).map_err(|e| bad_arg(format!("create snapshot parent dir: {e}")))?;
    }
    let tmp = path.with_extension("tmp_snapshot");
    std::fs::write(&tmp, &bytes).map_err(|e| bad_arg(format!("write snapshot temp file: {e}")))?;
    std::fs::rename(&tmp, path).map_err(|e| bad_arg(format!("atomic rename snapshot file: {e}")))?;
    Ok(bytes.len() as u64)
}

pub fn load_index_from_path(path: &Path, override_name: Option<&str>) -> Result<HybridIndex> {
    let bytes = std::fs::read(path).map_err(|e| bad_arg(format!("read snapshot file: {e}")))?;
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
        return Err(bad_arg("not a valid TurboSearchService snapshot"));
    }
    let spec_bytes = r.bytes()?;
    let spec = IndexSpec::decode(spec_bytes).map_err(|e| bad_arg(format!("decode IndexSpec: {e}")))?;
    let space = parse_space(spec.space.as_ref())?;
    let metric = parse_metric(spec.metric)?;
    let storage = VectorStorageKind::try_from(spec.storage).map_err(|_| bad_arg("invalid storage kind in snapshot"))?;
    let quant = match storage {
        VectorStorageKind::FlatF32 => None,
        VectorStorageKind::TurboQuant => {
            let bits = if spec.quant_bits == 0 { 4 } else { spec.quant_bits as u8 };
            Some(TurboQuant::new(bits, spec.quant_calibrate).map_err(|e| bad_arg(e.to_string()))?)
        }
    };
    let sealed = r.u8()? != 0;

    let n_docs = r.u32()? as usize;
    let mut docs = Vec::with_capacity(n_docs);
    for _ in 0..n_docs {
        let doc_id = r.str()?;
        let text = r.str()?;
        let graph = r.bytes()?.to_vec();
        let n_ch = r.u32()? as usize;
        let mut chunks = Vec::with_capacity(n_ch);
        for _ in 0..n_ch {
            let cb = r.bytes()?;
            chunks.push(ChunkInput::decode(cb).map_err(|e| bad_arg(format!("decode ChunkInput: {e}")))?);
        }
        let attributes = r.map()?;
        docs.push(RetainedDocument { doc_id, text, graph, chunks, attributes });
    }

    let n_chunks = r.u32()? as usize;
    let mut chunks = Vec::with_capacity(n_chunks);
    let mut bm25_builder = Bm25PostingIndexBuilder::new();
    let mut seen_terms = BTreeSet::new();
    let mut term_df = BTreeMap::new();

    for _ in 0..n_chunks {
        let chunk_id = r.str()?;
        let doc_id = r.str()?;
        let level_kind = r.str()?;
        let span = Span::new(r.u32()?, r.u32()?);
        let parent_span = match r.u8()? {
            0 => None,
            1 => Some(Span::new(r.u32()?, r.u32()?)),
            b => return Err(bad_arg(format!("invalid parent_span tag {b}"))),
        };
        let text = r.str()?;
        let parent_text = r.str()?;
        let n_terms = r.u32()? as usize;
        let mut terms = Vec::with_capacity(n_terms);
        let mut unique = BTreeSet::new();
        for _ in 0..n_terms {
            let t = r.str()?;
            let sp = Span::new(r.u32()?, r.u32()?);
            seen_terms.insert(t.clone());
            unique.insert(t.clone());
            terms.push((t, sp));
        }
        for u in unique {
            *term_df.entry(u).or_insert(0) += 1;
        }
        let term_refs: Vec<&str> = terms.iter().map(|(t, _)| t.as_str()).collect();
        bm25_builder.add_row_tokens(&term_refs).map_err(bad_arg)?;

        let vec_len = r.u32()? as usize;
        let dense_vector = if vec_len == 0 {
            None
        } else {
            let mut v = Vec::with_capacity(vec_len);
            for _ in 0..vec_len {
                v.push(r.f32()?);
            }
            Some(v)
        };
        let n_sparse = r.u32()? as usize;
        let mut sparse_terms = Vec::with_capacity(n_sparse);
        for _ in 0..n_sparse {
            let sb = r.bytes()?;
            sparse_terms.push(SparseEntryProto::decode(sb).map_err(|e| bad_arg(format!("decode SparseEntry: {e}")))?);
        }
        let n_tv = r.u32()? as usize;
        let mut token_vectors = Vec::with_capacity(n_tv);
        for _ in 0..n_tv {
            let tvb = r.bytes()?;
            token_vectors.push(TokenVectorInput::decode(tvb).map_err(|e| bad_arg(format!("decode TokenVec: {e}")))?);
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
    let bm25 = Bm25PostingIndex::from_bytes(bm25_bytes).map_err(bad_arg)?;

    let dense = match r.u8()? {
        0 => CompiledDense::Empty,
        1 => {
            let n = r.u32()? as usize;
            let mut meta = Vec::with_capacity(n);
            for _ in 0..n {
                meta.push(r.u32()?);
            }
            let vb = r.bytes()?;
            CompiledDense::Flat(VectorIndex::from_bytes(vb, meta).map_err(|e| bad_arg(e.to_string()))?)
        }
        2 => {
            let scale = r.f64()?;
            let n = r.u32()? as usize;
            let mut meta = Vec::with_capacity(n);
            for _ in 0..n {
                meta.push(r.u32()?);
            }
            let qb = r.bytes()?;
            let q_cfg = quant.ok_or_else(|| bad_arg("missing TurboQuant config"))?;
            let q_ix = QuantizedIndex::from_bytes(space.clone(), metric, q_cfg, scale, qb, meta)
                .map_err(|e| bad_arg(e.to_string()))?;
            CompiledDense::Quantized(Box::new(q_ix))
        }
        b => return Err(bad_arg(format!("invalid dense snapshot tag {b}"))),
    };

    let (token_index, token_candidates) = match r.u8()? {
        0 => (None, Vec::new()),
        1 => {
            let n = r.u32()? as usize;
            let mut cands = Vec::with_capacity(n);
            for _ in 0..n {
                cands.push(r.u32()?);
            }
            let tb = r.bytes()?;
            let t_ix = TokenIndex::from_bytes(tb, cands.clone()).map_err(|e| bad_arg(e.to_string()))?;
            (Some(t_ix), cands)
        }
        b => return Err(bad_arg(format!("invalid token_index tag {b}"))),
    };

    if !r.0.is_empty() {
        return Err(bad_arg("trailing bytes in search index snapshot"));
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
        bm25_builder,
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
            return Err(bad_arg("truncated search index snapshot"));
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
        std::str::from_utf8(b).map(str::to_owned).map_err(|_| bad_arg("invalid UTF-8 in snapshot"))
    }

    fn map(&mut self) -> Result<BTreeMap<String, String>> {
        let n = self.u32()? as usize;
        let mut out = BTreeMap::new();
        for _ in 0..n {
            out.insert(self.str()?, self.str()?);
        }
        Ok(out)
    }
}
