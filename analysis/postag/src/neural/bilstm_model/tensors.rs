//! The model as named tensors and vocabularies, the form a bundle stores
//! (`turbo-bundle postag-convert`). The LSTM gate rows stay in the
//! `(i, f, g, o)` order the layers use.

use super::BilstmPosModel;
use crate::neural::NeuralError;
use crate::neural::lstm::LstmLayer;

/// Everything of a model that is not a tensor.
#[derive(Clone, Debug, PartialEq)]
pub struct Vocabularies {
    pub words: Vec<(String, usize)>,
    /// Character keys in modified UTF-8, as the binary format stores them.
    pub chars: Vec<(Vec<u8>, usize)>,
    pub tags: Vec<String>,
    pub pretrained: Vec<(String, usize)>,
    pub max_word_length: usize,
}

/// A tensor's shape and row-major values.
pub type Shaped = (Vec<usize>, Vec<f64>);

/// One tensor, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
}

fn matrix(out: &mut Vec<NamedTensor>, name: &str, rows: &[Vec<f64>]) {
    let cols = rows.first().map_or(0, Vec::len);
    out.push(NamedTensor { name: name.to_owned(), shape: vec![rows.len(), cols], values: rows.concat() });
}

fn vector(out: &mut Vec<NamedTensor>, name: &str, v: &[f64]) {
    out.push(NamedTensor { name: name.to_owned(), shape: vec![v.len()], values: v.to_vec() });
}

fn lstm(out: &mut Vec<NamedTensor>, name: &str, layer: &LstmLayer) {
    matrix(out, &format!("{name}.w"), layer.w());
    matrix(out, &format!("{name}.u"), layer.u());
    vector(out, &format!("{name}.b"), layer.b());
}

fn format_error(name: &str, why: &str) -> NeuralError {
    NeuralError::Format(format!("BiLSTM POS tensors: {name}: {why}"))
}

fn need<T>(name: &str, v: Option<T>) -> Result<T, NeuralError> {
    v.ok_or_else(|| format_error(name, "missing"))
}

/// Reads tensors by name through `get`, checking each one's rank.
struct Reader<F> {
    get: F,
}

impl<F: FnMut(&str) -> Option<Shaped>> Reader<F> {
    fn fetch(&mut self, name: &str, rank: usize) -> Result<Option<Shaped>, NeuralError> {
        let Some((shape, values)) = (self.get)(name) else { return Ok(None) };
        if shape.len() != rank || shape.iter().product::<usize>() != values.len() {
            return Err(format_error(name, &format!("shape {shape:?} does not hold {} values", values.len())));
        }
        Ok(Some((shape, values)))
    }

    fn matrix(&mut self, name: &str) -> Result<Option<Vec<Vec<f64>>>, NeuralError> {
        Ok(self.fetch(name, 2)?.map(|(shape, values)| {
            if shape[1] == 0 {
                vec![Vec::new(); shape[0]]
            } else {
                values.chunks(shape[1]).map(<[f64]>::to_vec).collect()
            }
        }))
    }

    fn vector(&mut self, name: &str) -> Result<Option<Vec<f64>>, NeuralError> {
        Ok(self.fetch(name, 1)?.map(|(_, values)| values))
    }

    fn lstm(&mut self, name: &str) -> Result<Option<LstmLayer>, NeuralError> {
        let (w, u, b) = (
            self.matrix(&format!("{name}.w"))?,
            self.matrix(&format!("{name}.u"))?,
            self.vector(&format!("{name}.b"))?,
        );
        match (w, u, b) {
            (None, None, None) => Ok(None),
            (Some(w), Some(u), Some(b)) => {
                let input = w.first().map_or(0, Vec::len);
                LstmLayer::of_weights(input, b.len() / 4, w, u, b).map(Some)
            }
            _ => Err(format_error(name, "the layer needs w, u and b together")),
        }
    }
}

impl BilstmPosModel {
    /// The vocabularies, tags and word length limit.
    pub fn vocabularies(&self) -> Vocabularies {
        Vocabularies {
            words: self.word_entries.clone(),
            chars: self.char_entries.clone(),
            tags: self.tags.clone(),
            pretrained: self.pretrained_entries.clone(),
            max_word_length: self.max_word_length,
        }
    }

    /// Every tensor by name. The optional ones (second layer, CRF,
    /// pretrained vectors, adapter) are left out when the model has none.
    pub fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut out = Vec::new();
        matrix(&mut out, "word_embeddings", &self.word_embeddings);
        matrix(&mut out, "char_embeddings", &self.char_embeddings);
        lstm(&mut out, "char_forward", &self.char_forward);
        lstm(&mut out, "char_backward", &self.char_backward);
        lstm(&mut out, "word_forward", &self.word_forward);
        lstm(&mut out, "word_backward", &self.word_backward);
        if let (Some(f), Some(b)) = (&self.word_forward2, &self.word_backward2) {
            lstm(&mut out, "word_forward2", f);
            lstm(&mut out, "word_backward2", b);
        }
        matrix(&mut out, "output_weights", &self.output_weights);
        vector(&mut out, "output_bias", &self.output_bias);
        if let (Some(t), Some(s), Some(e)) = (&self.transition_weights, &self.start_weights, &self.end_weights) {
            matrix(&mut out, "transition_weights", t);
            vector(&mut out, "start_weights", s);
            vector(&mut out, "end_weights", e);
        }
        if let Some(p) = &self.pretrained_vectors {
            let rows: Vec<Vec<f64>> = p.iter().map(|r| r.iter().map(|&x| f64::from(x)).collect()).collect();
            matrix(&mut out, "pretrained_vectors", &rows);
        }
        if let (Some(w), Some(b)) = (&self.adapter_weights, &self.adapter_bias) {
            matrix(&mut out, "adapter_weights", w);
            vector(&mut out, "adapter_bias", b);
        }
        out
    }

    /// The model from its vocabularies and the tensors `get` returns by
    /// name (shape and row-major values), the inverse of
    /// [`Self::named_tensors`]. Pretrained vectors are rounded to F32, as
    /// the binary format stores them.
    pub fn from_named(v: Vocabularies, get: impl FnMut(&str) -> Option<Shaped>) -> Result<Self, NeuralError> {
        let mut r = Reader { get };
        let word_embeddings = need("word_embeddings", r.matrix("word_embeddings")?)?;
        let char_embeddings = need("char_embeddings", r.matrix("char_embeddings")?)?;
        let char_forward = need("char_forward", r.lstm("char_forward")?)?;
        let char_backward = need("char_backward", r.lstm("char_backward")?)?;
        let word_forward = need("word_forward", r.lstm("word_forward")?)?;
        let word_backward = need("word_backward", r.lstm("word_backward")?)?;
        let word_forward2 = r.lstm("word_forward2")?;
        let word_backward2 = r.lstm("word_backward2")?;
        let output_weights = need("output_weights", r.matrix("output_weights")?)?;
        let output_bias = need("output_bias", r.vector("output_bias")?)?;
        let transition_weights = r.matrix("transition_weights")?;
        let start_weights = r.vector("start_weights")?;
        let end_weights = r.vector("end_weights")?;
        let pretrained_vectors = r
            .matrix("pretrained_vectors")?
            .map(|rows| rows.into_iter().map(|row| row.into_iter().map(|x| x as f32).collect()).collect());
        let adapter_weights = r.matrix("adapter_weights")?;
        let adapter_bias = r.vector("adapter_bias")?;
        let pretrained =
            if v.pretrained.is_empty() && pretrained_vectors.is_none() { None } else { Some(v.pretrained) };
        Self::new(
            v.words,
            v.chars,
            v.tags,
            word_embeddings,
            char_embeddings,
            char_forward,
            char_backward,
            word_forward,
            word_backward,
            word_forward2,
            word_backward2,
            output_weights,
            output_bias,
            v.max_word_length,
            pretrained,
            pretrained_vectors,
            transition_weights,
            start_weights,
            end_weights,
            adapter_weights,
            adapter_bias,
        )
    }
}
