use turbo::manifest::HeadSpec;
use turbo::status::{Result, invalid};

use crate::real::Real;

pub fn dim(spec: &HeadSpec) -> usize {
    spec.language.as_ref().map_or(0, |language| language.dim as usize)
}

pub fn row(spec: &HeadSpec, row: Option<u32>) -> Result<Option<usize>> {
    let Some(language) = &spec.language else {
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

pub fn join<R: Real>(
    spec: &HeadSpec,
    features: &[R],
    language: Option<usize>,
    n: usize,
    embedding: &[R],
    hidden: usize,
    input: &mut Vec<R>,
) {
    let Some(row) = language else { return };
    let d = dim(spec);
    input.clear();
    input.resize(n * (hidden + d), R::ZERO);
    for i in 0..n {
        let row_out = &mut input[i * (hidden + d)..][..hidden + d];
        row_out[..hidden].copy_from_slice(&features[i * hidden..][..hidden]);
        row_out[hidden..].copy_from_slice(&embedding[row * d..][..d]);
    }
}
