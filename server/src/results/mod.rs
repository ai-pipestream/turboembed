//! The Stage 5 analysis result kinds as wire messages
//! (proto/turbo_analysis.proto): each in-process result layer's columns
//! with its model identity, the conversion both ways, and the message
//! limits the capability listing names. Decoding rebuilds the layer and
//! re-proves its own check, so a malformed message is refused the way the
//! layer refuses a malformed column, never a panic.

use turbo_analysis::ModelRef;

use crate::api::{Failure, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result as proto;

pub mod document;
pub mod extraction;
pub mod layers;
pub mod records;
pub mod rerank;
pub mod sparse;
pub mod tokens;
pub mod typed;

/// The most a result message may carry; the capability listing shows
/// these and a larger message is TURBO_E_CAPACITY naming the limit.
/// Nothing is truncated to fit.
pub const MAX_SPARSE_ENTRIES: u64 = 1 << 20;
pub const MAX_SPARSE_OCCURRENCES: u64 = 1 << 22;
pub const MAX_TOKEN_VECTOR_BYTES: u64 = 32 << 20;
pub const MAX_RERANK_ITEMS: u64 = 1 << 20;
pub const MAX_ANSWERS: u64 = 1 << 16;
pub const MAX_ANSWER_FRAGMENTS: u64 = 1 << 20;
pub const MAX_WINDOWS: u64 = 1 << 20;
pub const MAX_CHUNKS: u64 = 1 << 20;
pub const MAX_LEVELS: u64 = 1 << 20;
pub const MAX_FRAGMENTS: u64 = 1 << 22;
pub const MAX_GRAPH_BYTES: u64 = 64 << 20;
/// Record values, records, closed-choice values or support rows in one
/// RecordsResult, each counted on its own.
pub const MAX_RECORD_VALUES: u64 = 1 << 20;

/// `values` as little-endian bytes, in one copy on a little-endian target.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    if cfg!(target_endian = "little") {
        // SAFETY: an f32 is 4 bytes with no padding, each readable as a u8.
        unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 4) }.to_vec()
    } else {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

/// The whole little-endian f32 values of `bytes`, in one copy on a
/// little-endian target; a trailing part of a value is not read.
pub(crate) fn f32s(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 4;
    if cfg!(target_endian = "little") {
        let mut out = Vec::<f32>::with_capacity(n);
        // SAFETY: out has room for n values, written in full before the
        // length is set; every bit pattern is an f32.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), n * 4);
            out.set_len(n);
        }
        out
    } else {
        bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()
    }
}

/// One result kind's wire form: to the message, and back with the
/// message's own validation, whose refusal names what it found.
///
/// This is the shared layer's Rust client surface too: a caller holding
/// an in-process result builds the message with `to_msg`, and reads a
/// received message back with `from_msg`.
pub trait Wire: Sized {
    type Msg;
    fn to_msg(&self) -> Self::Msg;
    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure>;
}

/// A count over a listed limit: TURBO_E_CAPACITY naming the limit, the
/// way the server answers OUT_OF_RANGE for a request past a published
/// limit, and nothing is cut down to fit.
pub(crate) fn over(limit: &str, what: &str, n: u64) -> Failure {
    Failure::new(TURBO_E_CAPACITY, 0, format!("{what} is {n}, over the {limit} limit"))
}

/// A column count that must match another's, or the refusal names both.
fn mismatched(what: &str, got: usize, want: usize) -> Failure {
    Failure::new(
        TURBO_E_INVALID_ARGUMENT,
        0,
        format!("{what} has {got} values and should match the {want} of the column beside it"),
    )
}

/// An index built from a wire `u32`. `u32::MAX` is `Ix`'s absent
/// sentinel and `Ix::new` panics on it, so no wire value reaches
/// `Ix::new` unchecked: the sentinel (and, for `ix_bounded`, any
/// index at or past `n`) refuses naming the field, before any
/// layer is constructed.
pub fn ix_from_wire<L>(what: &str, i: u32) -> Result<turbo_analysis::Ix<L>, Failure> {
    if i == u32::MAX {
        return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("{what}: index {i} is the absent sentinel")));
    }
    Ok(turbo_analysis::Ix::new(i))
}

pub fn ix_bounded<L>(what: &str, i: u32, n: usize) -> Result<turbo_analysis::Ix<L>, Failure> {
    if i as usize >= n {
        return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("{what}: index {i} is outside the {n} items")));
    }
    ix_from_wire(what, i)
}

pub(crate) fn model_to_proto(m: &ModelRef) -> proto::ModelRef {
    proto::ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: m.sha256.clone() }
}

fn model_from_proto(m: &proto::ModelRef) -> ModelRef {
    ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: m.sha256.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_limit_refusal_names_it() {
        let f = over("max_answers", "answers", 5);
        assert_eq!(f.code, TURBO_E_CAPACITY);
        assert!(f.message.contains("max_answers"), "{}", f.message);
        assert!(f.message.contains('5'), "{}", f.message);
        let f = mismatched("span_start", 2, 3);
        assert_eq!(f.code, TURBO_E_INVALID_ARGUMENT);
        assert!(f.message.contains("span_start"), "{}", f.message);
    }

    #[test]
    fn the_model_identity_crosses_both_ways() {
        let m = ModelRef { id: "org/model".into(), revision: "v1".into(), sha256: "abc".into() };
        let p = model_to_proto(&m);
        assert_eq!((p.id.as_str(), p.revision.as_str(), p.sha256.as_str()), ("org/model", "v1", "abc"));
        assert_eq!(model_from_proto(&p), m);
    }
}

#[cfg(test)]
mod f32_tests {
    use super::{f32_bytes, f32s};

    #[test]
    fn f32_values_cross_as_their_little_endian_bits() {
        let values = [0.0f32, -0.0, 1.5, f32::MIN_POSITIVE, f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let bytes = f32_bytes(&values);
        let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(bytes, expected);
        let back = f32s(&bytes);
        assert_eq!(back.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), values.map(f32::to_bits));
        // A trailing part of a value is not read.
        assert_eq!(f32s(&bytes[..9]).len(), 2);
        assert!(f32s(&[]).is_empty());
    }
}
