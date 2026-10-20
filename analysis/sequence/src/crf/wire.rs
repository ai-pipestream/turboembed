//! Versioned CRF payload, with bounded counts before allocations.
//! This is a Turbo CRF format, not an existing OpenNLP maxent payload.
use super::{CrfError, CrfSequenceModel, Result, invalid};
use crate::namefind::NameCodec;
use turbo_maxent::JavaString;
impl CrfSequenceModel {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"TCRF\x01".to_vec();
        out.push(u8::from(self.codec == NameCodec::Bilou));
        for n in [self.labels.len(), self.features.len()] {
            out.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for label in &self.labels {
            put(&mut out, label.as_bytes());
        }
        for feature in &self.features {
            put(&mut out, feature.as_bytes());
        }
        for w in &self.weights {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader(bytes);
        if r.take(5)? != b"TCRF\x01" {
            return Err(invalid("format version"));
        }
        let codec = match r.take(1)?[0] {
            0 => NameCodec::Bio,
            1 => NameCodec::Bilou,
            _ => return Err(invalid("codec")),
        };
        let (n, f) = (r.count()?, r.count()?);
        let parameters = Self::parameter_count(n, f)?;
        let minimum = parameters.checked_mul(8).and_then(|v| v.checked_add((n + f) * 4)).ok_or(CrfError::Capacity)?;
        if r.0.len() < minimum {
            return Err(invalid("truncated payload"));
        }
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            labels.push(std::str::from_utf8(r.string()?).map_err(|_| invalid("label UTF-8"))?.to_owned());
        }
        let mut features = Vec::with_capacity(f);
        for _ in 0..f {
            features.push(JavaString::from_modified_utf8(r.string()?).ok_or_else(|| invalid("attribute encoding"))?);
        }
        if r.0.len() != parameters * 8 {
            return Err(invalid("weight bytes"));
        }
        let weights = r.0.as_chunks::<8>().0.iter().map(|v| f64::from_le_bytes(*v)).collect();
        Self::new(labels, features, codec, weights)
    }
}
fn put(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}
struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > self.0.len() {
            return Err(invalid("truncated payload"));
        }
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        Ok(a)
    }
    fn count(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn string(&mut self) -> Result<&'a [u8]> {
        let n = self.count()?;
        self.take(n)
    }
}
