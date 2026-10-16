//! The token classifier over the encoder's final states: wtpsplit's
//! `classifier`, a linear layer from the hidden size to the labels, of
//! which the boundary score is label 0 (`Constants.NEWLINE_INDEX`).

use turbo::bundle::Bundle;
use turbo::manifest::Format;
use turbo::safetensors::{self, Dtype};
use turbo::status::{Error, INVALID_ARGUMENT, Result};

/// Where the classifier's tensors are in the bundle's weights.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeadNames {
    /// `[labels, hidden]`, F32.
    pub weight: String,
    /// `[labels]`, F32.
    pub bias: String,
    /// The label whose score is the sentence boundary.
    pub label: u32,
}

impl Default for HeadNames {
    /// SaT's: `classifier.weight` and `classifier.bias`, label 0.
    fn default() -> Self {
        HeadNames { weight: "classifier.weight".into(), bias: "classifier.bias".into(), label: 0 }
    }
}

/// One label's row of the classifier: a score is the row's dot product
/// with a token's final state, plus the bias.
pub struct Head {
    row: Vec<f32>,
    bias: f32,
}

impl Head {
    /// The head `names` gives, read from the first raw-weights artifact
    /// of `bundle` whose file holds the weight, after the file's size and
    /// hash are checked. Its width must be `dim`, the encoder's.
    pub fn load(bundle: &Bundle, names: &HeadNames, dim: u32) -> Result<Head> {
        for a in bundle.manifest.artifacts.iter().filter(|a| a.format == Format::Safetensors) {
            for file in &a.files {
                let bytes = bundle.map_verified(file)?;
                let st = safetensors::File::parse(file, &bytes[..])?;
                if st.tensor(&names.weight).is_none() {
                    continue;
                }
                let w = st.get(&names.weight, Dtype::F32, 2)?;
                let b = st.get(&names.bias, Dtype::F32, 1)?;
                let (labels, width) = (w.shape[0], w.shape[1]);
                if width != u64::from(dim) || b.shape[0] != labels {
                    return Err(Error::new(
                        INVALID_ARGUMENT,
                        format!(
                            "{file}: {} {:?} and {} {:?} are not a classifier over {dim} values",
                            names.weight, w.shape, names.bias, b.shape
                        ),
                    ));
                }
                if u64::from(names.label) >= labels {
                    return Err(Error::new(
                        INVALID_ARGUMENT,
                        format!("{file}: label {} of a classifier with {labels} labels", names.label),
                    ));
                }
                let l = names.label as usize;
                let d = dim as usize;
                let row = w.f32s()[l * d..(l + 1) * d].to_vec();
                let bias = b.f32s()[l];
                return Head::new(row, bias);
            }
        }
        Err(Error::new(
            INVALID_ARGUMENT,
            format!("{}: no raw-weights artifact holds {}", bundle.manifest.model.id, names.weight),
        ))
    }

    /// A head from its row and bias, which must be finite.
    pub fn new(row: Vec<f32>, bias: f32) -> Result<Head> {
        if !row.iter().all(|x| x.is_finite()) || !bias.is_finite() {
            return Err(Error::new(INVALID_ARGUMENT, "the classifier holds a value that is not finite"));
        }
        Ok(Head { row, bias })
    }

    pub fn dim(&self) -> usize {
        self.row.len()
    }

    /// The score of one token's final state, `state` as wide as the row.
    pub fn score(&self, state: &[f32]) -> f32 {
        debug_assert_eq!(state.len(), self.row.len());
        self.row.iter().zip(state).map(|(w, h)| w * h).sum::<f32>() + self.bias
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_score_is_the_row_s_dot_product_plus_the_bias() {
        let h = Head::new(vec![1.0, -2.0, 0.5], 0.25).unwrap();
        assert_eq!(h.dim(), 3);
        assert_eq!(h.score(&[2.0, 1.0, 4.0]), 2.0 - 2.0 + 2.0 + 0.25);
    }

    #[test]
    fn a_value_that_is_not_finite_is_refused() {
        assert!(Head::new(vec![1.0, f32::NAN], 0.0).is_err());
        assert!(Head::new(vec![1.0], f32::INFINITY).is_err());
    }
}
