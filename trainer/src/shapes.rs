//! Tensor shapes of a head in training, checked before anything is
//! allocated.

use turbo::manifest::{HeadSpec, LanguageInput};
use turbo::status::{Result, invalid};

use crate::real::Real;
use crate::rng::Rng;

pub(crate) type Shapes = Vec<((String, Vec<u64>), usize)>;

/// The tensors `list` of a head for `spec` over features `hidden` wide, each
/// with its element count: the spec as a bundle must carry it, a nonzero
/// width, and every count without overflow.
pub(crate) fn checked_shapes(spec: &HeadSpec, hidden: usize, list: Vec<(String, Vec<u64>)>) -> Result<Shapes> {
    // A head no bundle could carry is never trained.
    spec.validate("head")?;
    if hidden == 0 {
        return Err(invalid("trainer: the encoder width must be at least 1"));
    }
    list.into_iter()
        .map(|(name, shape)| {
            let n = shape
                .iter()
                .try_fold(1u64, |a, &d| a.checked_mul(d))
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n.checked_mul(std::mem::size_of::<f64>()).is_some_and(|b| b <= isize::MAX as usize))
                .ok_or_else(|| invalid(format!("trainer: tensor {name} of head {:?} is too large", spec.name)))?;
            Ok(((name, shape), n))
        })
        .collect()
}

pub(crate) fn fits<R>(spec: &HeadSpec, shapes: &Shapes, params: &[Vec<R>]) -> Result<()> {
    if params.len() != shapes.len() || shapes.iter().zip(params).any(|((_, n), p)| *n != p.len()) {
        return Err(invalid(format!("trainer: weights do not match the tensors of head {:?}", spec.name)));
    }
    Ok(())
}

/// Seeded initial weights for `shapes`: Xavier-uniform matrices, and zero
/// vectors but for each LSTM forget gate's biases (1).
pub(crate) fn seeded<R: Real>(shapes: &Shapes, rng: &mut Rng) -> Vec<Vec<R>> {
    let mut params = Vec::with_capacity(shapes.len());
    for ((suffix, shape), n) in shapes {
        let n = *n;
        let v = if shape.len() == 2 {
            let limit = (6.0 / (shape[0] + shape[1]) as f64).sqrt();
            (0..n).map(|_| R::from_f64(rng.uniform(limit))).collect()
        } else if suffix.starts_with("lstm.") {
            let h = n / 4;
            (0..n).map(|i| if (h..2 * h).contains(&i) { R::ONE } else { R::ZERO }).collect()
        } else {
            vec![R::ZERO; n]
        };
        params.push(v);
    }
    params
}

/// The row of a head's `language` input an example's `row` names: none
/// for a head without one, row 0 for a head of one language when the
/// example names none, and an error for a row out of range, a row given
/// to a head without the input, or none given to a multilingual head.
pub(crate) fn language_row(language: Option<&LanguageInput>, row: Option<u32>) -> Result<Option<usize>> {
    let Some(language) = language else {
        return match row {
            Some(_) => Err(invalid("trainer: a language given to a head without a language input")),
            None => Ok(None),
        };
    };
    let row = match row {
        Some(row) => row as usize,
        None if language.languages.len() == 1 => 0,
        None => return Err(invalid("trainer: a multilingual example needs a language")),
    };
    if row >= language.languages.len() {
        return Err(invalid(format!("trainer: language row {row} is out of range")));
    }
    Ok(Some(row))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_language_row_is_checked_against_the_head() {
        let one = LanguageInput { languages: vec!["en".into()], dim: 4 };
        let two = LanguageInput { languages: vec!["en".into(), "de".into()], dim: 4 };
        assert_eq!(language_row(None, None).unwrap(), None);
        assert!(language_row(None, Some(0)).is_err());
        assert_eq!(language_row(Some(&one), None).unwrap(), Some(0));
        assert!(language_row(Some(&two), None).is_err());
        assert_eq!(language_row(Some(&two), Some(1)).unwrap(), Some(1));
        assert!(language_row(Some(&two), Some(2)).is_err());
    }

    #[test]
    fn seeded_weights_are_xavier_matrices_and_forget_gate_ones() {
        let shapes: Shapes =
            vec![(("w".into(), vec![3, 5]), 15), (("lstm.b".into(), vec![8]), 8), (("b".into(), vec![2]), 2)];
        let p: Vec<Vec<f64>> = seeded(&shapes, &mut Rng::new(7));
        let limit = (6.0f64 / 8.0).sqrt();
        assert!(p[0].iter().all(|x| x.abs() <= limit) && p[0].iter().any(|&x| x != 0.0));
        assert_eq!(p[1], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(p[2], [0.0, 0.0]);
        assert_eq!(seeded::<f64>(&shapes, &mut Rng::new(7)), p, "the same seed, the same weights");
    }
}
