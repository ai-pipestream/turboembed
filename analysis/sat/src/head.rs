//! The token classifier over the encoder's final states: wtpsplit's
//! `classifier`, a linear layer from the hidden size to the labels, of
//! which the boundary score is label 0 (`Constants.NEWLINE_INDEX`). The
//! bundle's manifest declares it as a `CLASSIFIER_TOKEN` head.

use turbo::bundle::Bundle;
use turbo::manifest::ClassifierKind;
use turbo::status::{Error, INVALID_ARGUMENT, Result};

/// wtpsplit's `Constants.NEWLINE_INDEX`: the label whose score is the
/// sentence boundary.
pub const BOUNDARY_LABEL: u32 = 0;

/// One label's row of the classifier: a score is the row's dot product
/// with a token's final state, plus the bias.
pub struct Head {
    row: Vec<f32>,
    bias: f32,
}

impl Head {
    /// Label `label` of the token classifier the manifest of `bundle`
    /// declares, read from its raw-weights artifact after the files' sizes
    /// and hashes are checked. Its width must be `dim`, the encoder's.
    pub fn load(bundle: &Bundle, label: u32, dim: u32) -> Result<Head> {
        let invalid = |m: String| Error::new(INVALID_ARGUMENT, format!("{}: {m}", bundle.manifest.model.id));
        match &bundle.manifest.classifier {
            None => return Err(invalid("the manifest declares no classifier".into())),
            Some(c) if c.kind != ClassifierKind::Token => {
                return Err(invalid("the manifest's classifier is not CLASSIFIER_TOKEN".into()));
            }
            Some(_) => {}
        }
        let h = turbo::head::Head::load(bundle)?;
        if h.hidden != dim as usize {
            return Err(invalid(format!("a classifier over {} values, the encoder gives {dim}", h.hidden)));
        }
        let labels = h.num_labels();
        if label as usize >= labels {
            return Err(invalid(format!("label {label} of a classifier with {labels} labels")));
        }
        let (l, d) = (label as usize, h.hidden);
        let c = h.classifier;
        Head::new(c.weight[l * d..(l + 1) * d].to_vec(), c.bias[l])
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
