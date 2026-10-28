//! Bounded binary framing; hashes bytes as they cross the codec boundary.
use super::super::Header;
use crate::Result;
use crate::record::Setting;
use std::io::{Read, Write};
use turbo_goldens::sha::Sha256;

pub const MAGIC: &[u8] = b"turbo-eval cuts 2\n";
pub const HEADER: usize = 1024 * 1024;
pub const ID: usize = 64 * 1024;
pub const GRAPH: usize = 64 * 1024 * 1024;
pub const DOCUMENTS: u32 = 100_000;
pub const TOTAL: usize = 1024 * 1024 * 1024;
const SETTINGS: usize = 4096;

pub fn chunk(out: &mut Vec<u8>, bytes: &[u8], limit: usize) -> Result<()> {
    if bytes.len() > limit {
        return Err("cut dump: field exceeds limit".into());
    }
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

pub fn header(h: &Header) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    for value in [&h.set, &h.sha256, &h.backend, &h.device, &h.precision] {
        chunk(&mut out, value.as_bytes(), ID)?;
    }
    for value in [h.max, h.target, h.size_weight, h.overhead] {
        out.extend_from_slice(&value.to_le_bytes());
    }
    if h.chunker.len() > SETTINGS {
        return Err("cut dump: too many settings".into());
    }
    out.extend_from_slice(&(h.chunker.len() as u32).to_le_bytes());
    for setting in &h.chunker {
        chunk(&mut out, setting.name.as_bytes(), ID)?;
        chunk(&mut out, setting.value.as_bytes(), ID)?;
        if out.len() > HEADER {
            return Err("cut dump: header exceeds limit".into());
        }
    }
    Ok(out)
}

pub struct Input<R> {
    pub source: R,
    pub hash: Sha256,
    used: usize,
}
impl<R: Read> Input<R> {
    pub fn new(source: R) -> Self {
        Self { source, hash: Sha256::new(), used: 0 }
    }
    fn reserve(&mut self, n: usize) -> Result<()> {
        self.used = self.used.checked_add(n).filter(|n| *n <= TOTAL - 32).ok_or("cut dump: artifact exceeds limit")?;
        Ok(())
    }
    pub fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.reserve(N)?;
        let mut bytes = [0; N];
        self.source.read_exact(&mut bytes).map_err(|e| format!("cut dump: incomplete artifact: {e}"))?;
        self.hash.update(&bytes);
        Ok(bytes)
    }
    pub fn bytes(&mut self, n: usize) -> Result<Vec<u8>> {
        self.reserve(n)?;
        // Grow only with bytes actually read, not the declared length.
        let mut bytes = Vec::with_capacity(n.min(8192));
        self.source.by_ref().take(n as u64).read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        if bytes.len() != n {
            return Err("cut dump: incomplete artifact; missing bytes or completion footer".into());
        }
        self.hash.update(&bytes);
        Ok(bytes)
    }
    pub fn number(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.fixed()?))
    }
    pub fn chunk(&mut self, limit: usize) -> Result<Vec<u8>> {
        let n = self.number()? as usize;
        if n > limit {
            return Err("cut dump: field exceeds limit".into());
        }
        self.bytes(n)
    }
    pub fn string(&mut self) -> Result<String> {
        String::from_utf8(self.chunk(ID)?).map_err(|e| format!("cut dump: invalid UTF-8: {e}"))
    }
    pub fn end(&mut self) -> Result<()> {
        let mut byte = [0];
        if self.source.read(&mut byte).map_err(|e| e.to_string())? != 0 {
            return Err("cut dump: trailing bytes".into());
        }
        Ok(())
    }
}

pub fn read_header(bytes: &[u8]) -> Result<Header> {
    let mut i = Input::new(bytes);
    let (set, sha256, backend, device, precision) = (i.string()?, i.string()?, i.string()?, i.string()?, i.string()?);
    let (max, target, size_weight, overhead) = (i.number()?, i.number()?, i.number()?, i.number()?);
    let count = i.number()? as usize;
    if count > SETTINGS {
        return Err("cut dump: too many settings".into());
    }
    let mut chunker = Vec::new();
    for _ in 0..count {
        chunker.push(Setting { name: i.string()?, value: i.string()? });
    }
    i.end()?;
    Ok(Header { set, sha256, backend, device, precision, max, target, size_weight, overhead, chunker })
}

pub fn write(out: &mut dyn Write, hash: &mut Sha256, used: &mut usize, bytes: &[u8]) -> Result<()> {
    *used = used.checked_add(bytes.len()).filter(|n| *n <= TOTAL - 32).ok_or("cut dump: artifact exceeds limit")?;
    out.write_all(bytes).map_err(|e| e.to_string())?;
    hash.update(bytes);
    Ok(())
}
