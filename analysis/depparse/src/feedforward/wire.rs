use super::{Activation, FeatureLayout, FeedforwardDependencyModel, NeuralWeights, Vocabulary};
use crate::{Result, Transition, error};
use std::collections::BTreeMap;
use turbo_maxent::JavaString;

pub(super) const MAGIC: &str = "ONLP-FFDP-1";
pub(super) struct Reader<'a> {
    pub data: &'a [u8],
    pub values: usize,
    pub utf8: bool,
}
impl<'a> Reader<'a> {
    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let bytes = self.data.get(..n).ok_or_else(|| error("truncated neural model"))?;
        self.data = &self.data[n..];
        Ok(bytes)
    }
    pub fn int(&mut self) -> Result<usize> {
        let n = i32::from_be_bytes(self.take(4)?.try_into().unwrap());
        usize::try_from(n).map_err(|_| error("negative wire count"))
    }
    pub fn string(&mut self) -> Result<String> {
        if self.utf8 {
            let n = self.int()?;
            if n > 10_000_000 {
                return Err(error("model string exceeds UTF-8 limit"));
            }
            return std::str::from_utf8(self.take(n)?).map(str::to_owned).map_err(error);
        }
        let n = u16::from_be_bytes(self.take(2)?.try_into().unwrap()) as usize;
        JavaString::from_modified_utf8(self.take(n)?)
            .and_then(|s| s.to_str().map(|s| s.into_owned()))
            .ok_or_else(|| error("invalid Unicode model string"))
    }
    pub fn end(self) -> Result<()> {
        if self.data.is_empty() { Ok(()) } else { Err(error("trailing neural model data")) }
    }
    fn vector(&mut self, expected: usize) -> Result<Vec<f32>> {
        if self.int()? != expected {
            return Err(error("wire vector dimension mismatch"));
        }
        self.floats(expected)
    }
    fn matrix(&mut self, rows: usize, cols: usize) -> Result<Vec<f32>> {
        if self.int()? != rows || self.int()? != cols {
            return Err(error("wire matrix dimension mismatch"));
        }
        self.floats(rows.checked_mul(cols).ok_or_else(|| error("matrix size overflow"))?)
    }
    fn floats(&mut self, n: usize) -> Result<Vec<f32>> {
        self.values = self.values.checked_add(n).ok_or_else(|| error("model size overflow"))?;
        if self.values > 100_000_000 {
            return Err(error("neural allocation limit"));
        }
        // Check available bytes before allocating for an untrusted count.
        let bytes = self.take(n.checked_mul(4).ok_or_else(|| error("float byte size overflow"))?)?;
        let mut out = Vec::with_capacity(n);
        for b in bytes.as_chunks::<4>().0 {
            let v = f32::from_be_bytes(*b);
            if !v.is_finite() {
                return Err(error("non-finite wire parameter"));
            }
            out.push(v);
        }
        Ok(out)
    }
    pub fn vocabulary(&mut self) -> Result<Vocabulary> {
        let mut total = 0usize;
        let mut map = || -> Result<BTreeMap<String, usize>> {
            let n = self.int()?;
            total = total.checked_add(n).ok_or_else(|| error("vocabulary count overflow"))?;
            if total > 2_000_000 || n > self.data.len() / 6 {
                return Err(error("invalid vocabulary count"));
            }
            let mut out = BTreeMap::new();
            for _ in 0..n {
                if out.insert(self.string()?, self.int()?).is_some() {
                    return Err(error("duplicate vocabulary symbol"));
                }
            }
            Ok(out)
        };
        let v = Vocabulary { words: map()?, pos: map()?, labels: map()? };
        v.check()?;
        Ok(v)
    }
    pub fn transitions(&mut self) -> Result<Vec<Transition>> {
        let n = self.int()?;
        if n == 0 || n > 100_000 || n > self.data.len() / 2 {
            return Err(error("invalid transition count"));
        }
        (0..n).map(|_| Transition::decode(&self.string()?)).collect()
    }
    pub fn weights(&mut self, v: usize, o: usize, layout: FeatureLayout) -> Result<NeuralWeights> {
        let e = self.int()?;
        if e == 0 || e > 4096 {
            return Err(error("invalid embedding size"));
        }
        let embeddings = self.matrix(v, e)?;
        let h = self.int()?;
        let input = layout.feature_count() * e;
        if h == 0 || h > 65536 || self.int()? != input {
            return Err(error("invalid hidden dimensions"));
        }
        let hidden_weights = self.floats(h.checked_mul(input).ok_or_else(|| error("hidden size overflow"))?)?;
        let hidden_bias = self.vector(h)?;
        let output_weights = self.matrix(o, h)?;
        let output_bias = self.vector(o)?;
        Ok(NeuralWeights { embedding_size: e, embeddings, hidden_weights, hidden_bias, output_weights, output_bias })
    }
}
pub(super) fn int(out: &mut Vec<u8>, n: usize) -> Result<()> {
    out.extend_from_slice(&i32::try_from(n).map_err(|_| error("wire integer overflow"))?.to_be_bytes());
    Ok(())
}
pub(super) fn string(out: &mut Vec<u8>, s: &str) -> Result<()> {
    let s = JavaString::new(s);
    out.extend_from_slice(
        &u16::try_from(s.len()).map_err(|_| error("wire string exceeds writeUTF limit"))?.to_be_bytes(),
    );
    out.extend_from_slice(s.as_bytes());
    Ok(())
}
pub(super) fn vocabulary(out: &mut Vec<u8>, v: &Vocabulary, transitions: &[Transition]) -> Result<()> {
    vocabulary_with(out, v, transitions, string)
}
pub(super) fn vocabulary_utf8(out: &mut Vec<u8>, v: &Vocabulary, transitions: &[Transition]) -> Result<()> {
    vocabulary_with(out, v, transitions, |out, s| {
        if s.len() > 10_000_000 {
            return Err(error("model string exceeds UTF-8 limit"));
        }
        int(out, s.len())?;
        out.extend_from_slice(s.as_bytes());
        Ok(())
    })
}
fn vocabulary_with(
    out: &mut Vec<u8>,
    v: &Vocabulary,
    transitions: &[Transition],
    write_string: impl Fn(&mut Vec<u8>, &str) -> Result<()>,
) -> Result<()> {
    for map in [&v.words, &v.pos, &v.labels] {
        int(out, map.len())?;
        let mut entries: Vec<_> = map.iter().collect();
        entries.sort_by_key(|(_, id)| **id);
        for (s, &id) in entries {
            write_string(out, s)?;
            int(out, id)?;
        }
    }
    int(out, transitions.len())?;
    for t in transitions {
        write_string(out, &t.encode())?;
    }
    Ok(())
}
pub(super) fn weights(out: &mut Vec<u8>, w: &NeuralWeights) -> Result<()> {
    fn vector(out: &mut Vec<u8>, v: &[f32], cols: Option<usize>) -> Result<()> {
        int(out, v.len() / cols.unwrap_or(1))?;
        if let Some(cols) = cols {
            int(out, cols)?;
        }
        for f in v {
            out.extend_from_slice(&f.to_be_bytes());
        }
        Ok(())
    }
    let h = w.hidden_bias.len();
    int(out, w.embedding_size)?;
    vector(out, &w.embeddings, Some(w.embedding_size))?;
    vector(out, &w.hidden_weights, Some(w.hidden_weights.len() / h))?;
    vector(out, &w.hidden_bias, None)?;
    vector(out, &w.output_weights, Some(h))?;
    vector(out, &w.output_bias, None)
}
impl FeedforwardDependencyModel {
    pub fn from_java_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader { data: bytes, values: 0, utf8: false };
        if r.string()? != MAGIC {
            return Err(error("unknown Java neural format"));
        }
        let vocabulary = r.vocabulary()?;
        let transitions = r.transitions()?;
        let weights = r.weights(vocabulary.len(), transitions.len(), FeatureLayout::OpenNlp14)?;
        r.end()?;
        Self::new(vocabulary, transitions, FeatureLayout::OpenNlp14, Activation::Cube, weights)
    }
    pub fn to_java_bytes(&self) -> Result<Vec<u8>> {
        if self.layout != FeatureLayout::OpenNlp14 || self.activation != Activation::Cube || self.default_beam_size != 1
        {
            return Err(error("legacy Java raw format requires OpenNlp14 cube model with default beam 1"));
        }
        let mut out = Vec::new();
        string(&mut out, MAGIC)?;
        vocabulary(&mut out, &self.vocabulary, &self.transitions)?;
        weights(&mut out, &self.weights)?;
        Ok(out)
    }
}
