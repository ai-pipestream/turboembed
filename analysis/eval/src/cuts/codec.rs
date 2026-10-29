use super::{Dump, Header, codecs, cuts_path, graph, key};
use crate::Result;
use crate::artifact::{ArtifactKey, EvaluationArtifactStore, PendingArtifact, Publication, TemporaryFileStore};
use crate::sha::Sha256;
use std::io::{BufReader, Read, Write};
use std::path::{Path, PathBuf};
use turbo_analysis::{
    Document,
    export::{Codecs, DocumentGraph, import},
};
use turbo_analysis_layers::{Boundaries, Chunks, Sentences};
#[path = "wire.rs"]
mod wire;

/// Streams a v2 dump into an exclusively owned staging artifact.
pub struct Writer {
    out: Box<dyn PendingArtifact>,
    hash: Sha256,
    used: usize,
    codecs: Codecs,
    docs: u32,
    poisoned: bool,
    local_path: Option<PathBuf>,
}
impl Writer {
    /// Local convenience adapter. The codec itself uses the storage interface.
    pub fn create(dir: &Path, header: &Header) -> Result<Self> {
        let mut writer = Self::create_in(&TemporaryFileStore::new(dir), header)?;
        writer.local_path = Some(cuts_path(dir, &header.set));
        Ok(writer)
    }
    pub fn create_in(store: &dyn EvaluationArtifactStore, header: &Header) -> Result<Self> {
        let key = key(&header.set)?;
        let header = wire::header(header)?;
        if header.len() > wire::HEADER {
            return Err("cut dump: header exceeds limit".into());
        }
        let out = store.begin(&key).map_err(|e| e.to_string())?;
        let mut w =
            Self { out, hash: Sha256::new(), used: 0, codecs: codecs(), docs: 0, poisoned: false, local_path: None };
        w.write(wire::MAGIC)?;
        w.write(&(header.len() as u32).to_le_bytes())?;
        w.write(&header)?;
        Ok(w)
    }
    fn write(&mut self, bytes: &[u8]) -> Result<()> {
        let result = wire::write(&mut *self.out, &mut self.hash, &mut self.used, bytes);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    pub fn document(&mut self, id: &str, doc: &Document) -> Result<()> {
        if self.poisoned {
            return Err("cut dump: writer failed previously".into());
        }
        // Validation errors poison this write too: callers cannot ignore a
        // failed document and publish a deceptively complete subset.
        self.poisoned = true;
        if self.docs >= wire::DOCUMENTS || id.len() > wire::ID {
            return Err("cut dump: document limit exceeded".into());
        }
        let bytes = graph(doc, &self.codecs)?.to_bytes().map_err(|e| format!("document {id}: {e}"))?;
        if bytes.len() > wire::GRAPH {
            return Err("cut dump: graph exceeds limit".into());
        }
        self.write(&[1])?;
        self.write(&(id.len() as u32).to_le_bytes())?;
        self.write(id.as_bytes())?;
        self.write(&(bytes.len() as u32).to_le_bytes())?;
        self.write(&bytes)?;
        self.docs += 1;
        self.poisoned = false;
        Ok(())
    }
    pub fn publish(mut self) -> Result<(Publication, u32)> {
        if self.poisoned {
            return Err("cut dump: cannot publish a failed writer".into());
        }
        self.write(&[0])?;
        self.write(&self.docs.to_le_bytes())?;
        let digest = self.hash.finish();
        self.out.write_all(&digest).map_err(|e| e.to_string())?;
        let receipt = self.out.finalize().map_err(|e| e.to_string())?;
        Ok((receipt, self.docs))
    }
    /// Compatibility adapter for local callers. Prefer `publish` to inspect
    /// the typed publication receipt and any staging cleanup failure.
    pub fn finish(self) -> Result<(PathBuf, u32)> {
        let path = self.local_path.clone().ok_or("use publish for a storage-backed writer")?;
        let (receipt, count) = self.publish()?;
        if let Some(error) = receipt.cleanup_error {
            eprintln!("artifact {} published; staging cleanup failed: {error}", receipt.key);
        }
        Ok((path, count))
    }
}

impl Dump {
    pub fn read(path: &Path) -> Result<Self> {
        let name = path.file_name().and_then(|s| s.to_str()).ok_or("invalid artifact filename")?;
        let key = ArtifactKey::new(name).map_err(|e| e.to_string())?;
        Self::read_from(&TemporaryFileStore::new(path.parent().unwrap_or(Path::new("."))), &key)
    }
    pub fn read_from(store: &dyn EvaluationArtifactStore, name: &ArtifactKey) -> Result<Self> {
        let source = store.open(name).map_err(|e| e.to_string())?;
        let mut input = wire::Input::new(BufReader::new(source));
        let magic = input.bytes(wire::MAGIC.len())?;
        if magic == b"turbo-eval cuts 1\n" {
            return Err("cut dump version 1 has no verifiable completion: regenerate with --dump-cuts".into());
        }
        if magic != wire::MAGIC {
            return Err("not a version 2 cut dump".into());
        }
        let header = wire::read_header(&input.chunk(wire::HEADER)?)?;
        if key(&header.set)? != *name {
            return Err("cut dump: header set disagrees with artifact key".into());
        }
        let mut docs = Vec::new();
        let codecs = codecs();
        loop {
            match input.fixed::<1>()?[0] {
                0 => {
                    if input.number()? as usize != docs.len() {
                        return Err("cut dump: completion count mismatch".into());
                    }
                    let hash = std::mem::take(&mut input.hash).finish();
                    let mut digest = [0; 32];
                    input.source.read_exact(&mut digest).map_err(|e| format!("cut dump: incomplete digest: {e}"))?;
                    if digest != hash {
                        return Err("cut dump: digest mismatch".into());
                    }
                    input.end()?;
                    return Ok(Self { header, docs });
                }
                1 => {
                    if docs.len() >= wire::DOCUMENTS as usize {
                        return Err("cut dump: too many documents".into());
                    }
                    let id = input.string()?;
                    let bytes = input.chunk(wire::GRAPH)?;
                    let graph = DocumentGraph::from_bytes(&bytes).map_err(|e| format!("{id}: {e}"))?;
                    let doc = import(&graph, &codecs).map_err(|e| format!("{id}: {e}"))?;
                    if doc.get::<Sentences>().is_none()
                        || doc.get::<Boundaries>().is_none()
                        || doc.get::<Chunks>().is_none()
                    {
                        return Err(format!("{id}: missing sentences, boundaries or chunks layer"));
                    }
                    docs.push((id, doc));
                }
                _ => return Err("cut dump: invalid frame".into()),
            }
        }
    }
}
