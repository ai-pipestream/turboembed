use super::{Activation, FeatureLayout, FeedforwardDependencyModel, NeuralWeights, Vocabulary};
use crate::{Result, Transition, error};
use turbo_maxent::JavaString;

pub(super) const MAGIC: &str = "ONLP-FFDP-1";
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
    pub fn from_java_bytes(_bytes: &[u8]) -> Result<Self> {
        Err(error(turbo_maxent::Error::Disabled))
    }
    pub fn to_java_bytes(&self) -> Result<Vec<u8>> {
        if self.transitions.contains(&Transition::Swap) {
            return Err(error("SWAP models have no supported legacy Java export; use the native format"));
        }
        if self.label_encoding != crate::LabelEncoding::Plain
            || self.layout != FeatureLayout::OpenNlp14
            || self.activation != Activation::Cube
            || self.default_beam_size != 1
        {
            return Err(error("legacy Java raw format requires plain labels, OpenNlp14 cube model and default beam 1"));
        }
        let mut out = Vec::new();
        string(&mut out, MAGIC)?;
        vocabulary(&mut out, &self.vocabulary, &self.transitions)?;
        weights(&mut out, &self.weights)?;
        Ok(out)
    }
}
