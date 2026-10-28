//! Tensor shapes of a head in training, checked before anything is
//! allocated.

use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

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
