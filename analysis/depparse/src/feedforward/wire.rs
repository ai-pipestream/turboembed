use std::collections::BTreeMap;
use std::path::Path;

use super::{Activation, FeatureLayout, FeedforwardDependencyModel, NeuralWeights, Vocabulary};
use crate::{Error, LabelEncoding, Result, Transition, error};
use turbo_analysis::ModelRef;

const MAGIC: &[u8] = b"TURBO-DEPPARSE-FF-V1\n";

impl FeedforwardDependencyModel {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path).map_err(error)?)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let bytes = self.to_bytes()?;
        std::fs::write(path, bytes).map_err(error)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.check()?;
        let mut out = Vec::from(MAGIC);
        out.push(match self.layout {
            FeatureLayout::OpenNlp14 => 0,
            FeatureLayout::ChenManning18 => 1,
        });
        out.push(match self.activation {
            Activation::Cube => 0,
            Activation::Relu => 1,
        });
        out.push(match self.label_encoding {
            LabelEncoding::Plain => 0,
            LabelEncoding::EscapedV1 => 1,
        });
        write_len(&mut out, self.default_beam_size)?;
        write_vocab_map(&mut out, &self.vocabulary.words)?;
        write_vocab_map(&mut out, &self.vocabulary.pos)?;
        write_vocab_map(&mut out, &self.vocabulary.labels)?;
        write_len(&mut out, self.transitions.len())?;
        for t in &self.transitions {
            write_str(&mut out, &t.encode())?;
        }
        let w = &self.weights;
        write_len(&mut out, w.embedding_size)?;
        write_f32s(&mut out, &w.embeddings)?;
        write_f32s(&mut out, &w.hidden_bias)?;
        write_f32s(&mut out, &w.hidden_weights)?;
        write_f32s(&mut out, &w.output_weights)?;
        write_f32s(&mut out, &w.output_bias)?;
        Ok(out)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut r = Reader::new(bytes);
        r.expect(MAGIC)?;
        let layout = match r.u8()? {
            0 => FeatureLayout::OpenNlp14,
            1 => FeatureLayout::ChenManning18,
            other => return Err(Error(format!("unknown feedforward layout tag {other}"))),
        };
        let activation = match r.u8()? {
            0 => Activation::Cube,
            1 => Activation::Relu,
            other => return Err(Error(format!("unknown feedforward activation tag {other}"))),
        };
        let label_encoding = match r.u8()? {
            0 => LabelEncoding::Plain,
            1 => LabelEncoding::EscapedV1,
            other => return Err(Error(format!("unknown feedforward label encoding tag {other}"))),
        };
        let default_beam_size = r.len()?;
        if default_beam_size == 0 {
            return Err(error("default beam size must be positive"));
        }
        let words = r.vocab_map()?;
        let pos = r.vocab_map()?;
        let labels = r.vocab_map()?;
        let vocabulary = Vocabulary { words, pos, labels };
        let n_transitions = r.len()?;
        let mut transitions = Vec::with_capacity(n_transitions);
        for _ in 0..n_transitions {
            transitions.push(Transition::decode(r.str()?)?);
        }
        let embedding_size = r.len()?;
        let embeddings = r.f32s()?;
        let hidden_bias = r.f32s()?;
        let hidden_weights = r.f32s()?;
        let output_weights = r.f32s()?;
        let output_bias = r.f32s()?;
        r.finish()?;
        let weights =
            NeuralWeights { embedding_size, embeddings, hidden_bias, hidden_weights, output_weights, output_bias };
        let mut model = Self::new(vocabulary, transitions, layout, activation, weights)?;
        model.default_beam_size = default_beam_size;
        model.label_encoding = label_encoding;
        Ok(model)
    }

    pub fn identity(&self) -> Result<ModelRef> {
        let bytes = self.to_bytes()?;
        Ok(ModelRef {
            id: "turbo:depparse-feedforward".into(),
            revision: "1".into(),
            sha256: turbo_sha256::sha256(&bytes).to_hex(),
        })
    }
}

fn write_len(out: &mut Vec<u8>, n: usize) -> Result<()> {
    let n = u32::try_from(n).map_err(|_| error("count exceeds u32"))?;
    out.extend_from_slice(&n.to_le_bytes());
    Ok(())
}

fn write_str(out: &mut Vec<u8>, s: &str) -> Result<()> {
    write_len(out, s.len())?;
    out.extend_from_slice(s.as_bytes());
    Ok(())
}

fn write_vocab_map(out: &mut Vec<u8>, map: &BTreeMap<String, usize>) -> Result<()> {
    write_len(out, map.len())?;
    for (k, &v) in map {
        write_str(out, k)?;
        write_len(out, v)?;
    }
    Ok(())
}

fn write_f32s(out: &mut Vec<u8>, values: &[f32]) -> Result<()> {
    write_len(out, values.len())?;
    for &v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes }
    }
    fn expect(&mut self, prefix: &[u8]) -> Result<()> {
        if !self.bytes.starts_with(prefix) {
            return Err(error("invalid feedforward dependency model header"));
        }
        self.bytes = &self.bytes[prefix.len()..];
        Ok(())
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.bytes.len() < n {
            return Err(error("truncated feedforward dependency model"));
        }
        let (head, tail) = self.bytes.split_at(n);
        self.bytes = tail;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn len(&mut self) -> Result<usize> {
        Ok(self.u32()? as usize)
    }
    fn str(&mut self) -> Result<&'a str> {
        let n = self.len()?;
        std::str::from_utf8(self.take(n)?).map_err(error)
    }
    fn vocab_map(&mut self) -> Result<BTreeMap<String, usize>> {
        let n = self.len()?;
        if n > 2_000_000 {
            return Err(error("vocabulary limit exceeded"));
        }
        let mut map = BTreeMap::new();
        for _ in 0..n {
            let k = self.str()?.to_owned();
            let v = self.len()?;
            if map.insert(k, v).is_some() {
                return Err(error("duplicate vocabulary key"));
            }
        }
        Ok(map)
    }
    fn f32s(&mut self) -> Result<Vec<f32>> {
        let n = self.len()?;
        if n > 100_000_000 {
            return Err(error("neural parameter count exceeds limit"));
        }
        let raw = self.take(n.checked_mul(4).ok_or_else(|| error("parameter byte length overflow"))?)?;
        Ok(raw.as_chunks::<4>().0.iter().map(|&c| f32::from_le_bytes(c)).collect())
    }
    fn finish(self) -> Result<()> {
        if !self.bytes.is_empty() {
            return Err(error("trailing bytes in feedforward dependency model"));
        }
        Ok(())
    }
}
