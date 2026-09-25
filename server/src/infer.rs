//! A ModelInferRequest as the one write it becomes (docs/kserve.md,
//! Inference and Parameters). Everything here is the server's own checking,
//! made before any library call; what the header checks is left to the
//! library.

use std::collections::HashMap;
use std::ffi::c_char;
use std::marker::PhantomData;

use turbo::{turbo_embed_options, turbo_text};

use crate::api::{Failure, Result, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_ENUM, TURBO_E_INVALID_SHAPE, names};
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_request::InferInputTensor;
use crate::proto::{InferParameter, InferTensorContents, ModelInferRequest};
use crate::status::EMBED_OPTIONS;

/// The rows of one write, pointing into the request.
pub enum Rows<'a> {
    /// turbo_embed_write_text: each turbo_text points into the request's
    /// bytes, typed or raw.
    Texts(Vec<turbo_text>, PhantomData<&'a [u8]>),
    /// turbo_embed_write_tokens on the request's own int_contents.
    Typed { batch: u32, seq: u32, ids: &'a [i32], mask: &'a [i32], types: Option<&'a [i32]> },
    /// turbo_embed_write_tokens on the held session's pinned buffers, once
    /// these little-endian rows are copied there.
    Raw { batch: u32, seq: u32, ids: &'a [u8], mask: &'a [u8], types: Option<&'a [u8]> },
}

pub struct Plan<'a> {
    pub rows: Rows<'a>,
    pub opts: turbo_embed_options,
}

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn bad_shape(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_SHAPE, 0, message)
}

/// The contents of an input that has none.
static EMPTY: InferTensorContents = InferTensorContents {
    bool_contents: Vec::new(),
    int_contents: Vec::new(),
    int64_contents: Vec::new(),
    uint_contents: Vec::new(),
    uint64_contents: Vec::new(),
    fp32_contents: Vec::new(),
    fp64_contents: Vec::new(),
    bytes_contents: Vec::new(),
};

/// What the request asks for, or the refusal.
pub fn plan(req: &ModelInferRequest) -> Result<Plan<'_>> {
    for o in &req.outputs {
        if o.name != "vectors" {
            return Err(bad(format!("requested output `{}`: the model's one output is `vectors`", o.name)));
        }
        if !o.parameters.is_empty() {
            return Err(bad("requested output `vectors` has parameters; none are defined"));
        }
    }
    if req.outputs.len() > 1 {
        return Err(bad("requested output `vectors` is named more than once"));
    }
    for i in &req.inputs {
        if !i.parameters.is_empty() {
            return Err(bad(format!("input `{}` has parameters; none are defined", i.name)));
        }
    }

    // Which inputs, each once.
    let find = |name: &str| -> Result<Option<usize>> {
        let mut at = req.inputs.iter().enumerate().filter(|(_, i)| i.name == name).map(|(k, _)| k);
        let first = at.next();
        if at.next().is_some() {
            return Err(bad(format!("input `{name}` is given more than once")));
        }
        Ok(first)
    };
    for i in &req.inputs {
        if !matches!(i.name.as_str(), "texts" | "ids" | "mask" | "types") {
            return Err(bad(format!(
                "input `{}`: the inputs are `texts`, or `ids` and `mask` with `types` optional",
                i.name
            )));
        }
    }
    let (texts, ids, mask, types) = (find("texts")?, find("ids")?, find("mask")?, find("types")?);
    let token_set = ids.is_some() && mask.is_some() && texts.is_none();
    let text_set = texts.is_some() && ids.is_none() && mask.is_none() && types.is_none();
    if !token_set && !text_set {
        let given: Vec<&str> = req.inputs.iter().map(|i| i.name.as_str()).collect();
        return Err(bad(format!(
            "inputs {given:?}: a request gives `texts` alone, or `ids` and `mask` with `types` optional"
        )));
    }

    // Datatypes, then shapes.
    let want = if text_set { "BYTES" } else { "INT32" };
    for i in &req.inputs {
        if i.datatype != want {
            return Err(bad(format!(
                "input `{}` is {}; it takes {want}, and nothing is converted",
                i.name, i.datatype
            )));
        }
    }
    let rank = if text_set { 1 } else { 2 };
    let mut extents = Vec::with_capacity(req.inputs.len());
    for i in &req.inputs {
        extents.push(shape(i, rank)?);
    }

    // Raw or typed, not both.
    let raw = !req.raw_input_contents.is_empty();
    if raw && req.inputs.iter().any(|i| i.contents.is_some()) {
        return Err(bad("the request has both raw_input_contents and typed contents"));
    }
    if raw && req.raw_input_contents.len() != req.inputs.len() {
        return Err(bad(format!(
            "{} raw_input_contents entries for {} inputs",
            req.raw_input_contents.len(),
            req.inputs.len()
        )));
    }
    let typed = |k: usize| req.inputs[k].contents.as_ref().unwrap_or(&EMPTY);

    if let Some(k) = texts {
        let batch = extents[k][0];
        let views = if raw {
            split_bytes(&req.raw_input_contents[k])?
        } else {
            let c = typed(k);
            only(&req.inputs[k], c, "bytes_contents")?;
            c.bytes_contents.iter().map(|b| view(b)).collect()
        };
        if views.len() as u64 != batch as u64 {
            return Err(bad_shape(format!("input `texts` is shaped [{batch}] and has {} elements", views.len())));
        }
        return Ok(Plan { rows: Rows::Texts(views, PhantomData), opts: options(&req.parameters)? });
    }

    let (ids, mask) = (ids.unwrap(), mask.unwrap());
    let [batch, seq] = [extents[ids][0], extents[ids][1]];
    for k in [Some(mask), types].into_iter().flatten() {
        if extents[k] != extents[ids] {
            return Err(bad_shape(format!(
                "input `{}` is shaped {:?} and `ids` {:?}",
                req.inputs[k].name, extents[k], extents[ids]
            )));
        }
    }
    let count = batch as u64 * seq as u64;
    let rows = if raw {
        let bytes = |k: usize| -> Result<&[u8]> {
            let b = &req.raw_input_contents[k];
            if b.len() as u64 != 4 * count {
                return Err(bad_shape(format!(
                    "input `{}` is shaped [{batch}, {seq}] and its raw contents are {} bytes, not {}",
                    req.inputs[k].name,
                    b.len(),
                    4 * count
                )));
            }
            Ok(b)
        };
        Rows::Raw { batch, seq, ids: bytes(ids)?, mask: bytes(mask)?, types: types.map(bytes).transpose()? }
    } else {
        let ints = |k: usize| -> Result<&[i32]> {
            let c = typed(k);
            only(&req.inputs[k], c, "int_contents")?;
            if c.int_contents.len() as u64 != count {
                return Err(bad_shape(format!(
                    "input `{}` is shaped [{batch}, {seq}] and has {} elements",
                    req.inputs[k].name,
                    c.int_contents.len()
                )));
            }
            Ok(&c.int_contents)
        };
        Rows::Typed { batch, seq, ids: ints(ids)?, mask: ints(mask)?, types: types.map(ints).transpose()? }
    };
    Ok(Plan { rows, opts: options(&req.parameters)? })
}

/// The tensor's extents, each within a uint32_t.
fn shape(i: &InferInputTensor, rank: usize) -> Result<Vec<u32>> {
    if i.shape.len() != rank {
        return Err(bad_shape(format!("input `{}` has rank {}; it takes rank {rank}", i.name, i.shape.len())));
    }
    i.shape
        .iter()
        .map(|&e| {
            u32::try_from(e)
                .map_err(|_| bad_shape(format!("input `{}` has extent {e}, outside 0 to 4294967295", i.name)))
        })
        .collect()
}

/// Refuses contents in any typed field but `keep`.
fn only(i: &InferInputTensor, c: &InferTensorContents, keep: &str) -> Result<()> {
    let used = [
        ("bool_contents", !c.bool_contents.is_empty()),
        ("int_contents", !c.int_contents.is_empty()),
        ("int64_contents", !c.int64_contents.is_empty()),
        ("uint_contents", !c.uint_contents.is_empty()),
        ("uint64_contents", !c.uint64_contents.is_empty()),
        ("fp32_contents", !c.fp32_contents.is_empty()),
        ("fp64_contents", !c.fp64_contents.is_empty()),
        ("bytes_contents", !c.bytes_contents.is_empty()),
    ];
    match used.iter().find(|(n, u)| *u && *n != keep) {
        Some((n, _)) => Err(bad(format!("input `{}` has {n}; it takes {keep}", i.name))),
        None => Ok(()),
    }
}

fn view(b: &[u8]) -> turbo_text {
    turbo_text { ptr: b.as_ptr() as *const c_char, len: b.len() as u64 }
}

/// Raw BYTES: each element a 4-byte little-endian length and that many
/// bytes, back to back, with nothing left over.
fn split_bytes(mut b: &[u8]) -> Result<Vec<turbo_text>> {
    let mut out = Vec::new();
    while !b.is_empty() {
        let Some((len, rest)) = b.split_first_chunk::<4>() else {
            return Err(bad(format!(
                "raw `texts` element {}: {} bytes left, too few for a length",
                out.len(),
                b.len()
            )));
        };
        let len = u32::from_le_bytes(*len) as usize;
        if rest.len() < len {
            return Err(bad(format!("raw `texts` element {}: length {len} with {} bytes left", out.len(), rest.len())));
        }
        out.push(view(&rest[..len]));
        b = &rest[len..];
    }
    Ok(out)
}

/// The option fields that take one of their constants, by field number.
fn constants(field: usize) -> Option<&'static [(u32, &'static str)]> {
    match field {
        1 => Some(names::TRUNCATE),
        3 => Some(names::PROMPT),
        4 => Some(names::NORMALIZE),
        5 => Some(names::POOLING),
        _ => None,
    }
}

/// The request's parameters as one turbo_embed_options: struct_size set,
/// every field 0, and the fields the request names set.
pub fn options(params: &HashMap<String, InferParameter>) -> Result<turbo_embed_options> {
    let mut unknown: Vec<&String> = params.keys().filter(|k| !EMBED_OPTIONS.contains(&k.as_str())).collect();
    unknown.sort();
    if let Some(name) = unknown.first() {
        return Err(bad(format!("parameter `{name}` is not a field of turbo_embed_options")));
    }
    let mut v = [0u32; 6];
    for (k, name) in EMBED_OPTIONS.iter().enumerate() {
        let Some(p) = params.get(*name) else { continue };
        let field = k as u32 + 1;
        let choice = p.parameter_choice.as_ref();
        v[k] = match (constants(k + 1), choice) {
            (Some(table), Some(ParameterChoice::StringParam(s))) => {
                table.iter().find(|(_, n)| n == s).map(|(c, _)| *c).ok_or_else(|| {
                    let all: Vec<&str> = table.iter().map(|(_, n)| *n).collect();
                    Failure::new(
                        TURBO_E_INVALID_ENUM,
                        0,
                        format!("parameter {name}: `{s}` is not one of {}", all.join(", ")),
                    )
                })?
            }
            (Some(_), _) => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name} takes a string_param, not {}", kind(choice)),
                ));
            }
            (None, Some(ParameterChoice::Int64Param(n))) => u32::try_from(*n).map_err(|_| {
                Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name}: {n} is outside 0 to 4294967295"),
                )
            })?,
            (None, _) => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name} takes an int64_param, not {}", kind(choice)),
                ));
            }
        };
    }
    Ok(turbo_embed_options {
        struct_size: size_of::<turbo_embed_options>() as u32,
        truncate: v[0],
        max_tokens: v[1],
        prompt_role: v[2],
        normalize: v[3],
        pooling: v[4],
        output_dim: v[5],
    })
}

fn kind(c: Option<&ParameterChoice>) -> &'static str {
    match c {
        None => "no value",
        Some(ParameterChoice::BoolParam(_)) => "a bool_param",
        Some(ParameterChoice::Int64Param(_)) => "an int64_param",
        Some(ParameterChoice::StringParam(_)) => "a string_param",
        Some(ParameterChoice::DoubleParam(_)) => "a double_param",
        Some(ParameterChoice::Uint64Param(_)) => "a uint64_param",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(c: ParameterChoice) -> InferParameter {
        InferParameter { parameter_choice: Some(c) }
    }

    fn one(name: &str, c: ParameterChoice) -> Result<turbo_embed_options> {
        options(&HashMap::from([(name.to_string(), p(c))]))
    }

    #[test]
    fn every_field_by_its_header_name() {
        use ParameterChoice::*;
        let params = HashMap::from([
            ("truncate".to_string(), p(StringParam("TRUNCATE_LEFT".into()))),
            ("max_tokens".to_string(), p(Int64Param(4294967295))),
            ("prompt_role".to_string(), p(StringParam("PROMPT_DOCUMENT".into()))),
            ("normalize".to_string(), p(StringParam("NORMALIZE_NONE".into()))),
            ("pooling".to_string(), p(StringParam("POOLING_LAST".into()))),
            ("output_dim".to_string(), p(Int64Param(0))),
        ]);
        let o = options(&params).unwrap();
        assert_eq!(o.struct_size as usize, size_of::<turbo_embed_options>());
        assert_eq!(
            (o.truncate, o.max_tokens, o.prompt_role, o.normalize, o.pooling, o.output_dim),
            (3, u32::MAX, 2, 1, 3, 0)
        );
        let o = options(&HashMap::new()).unwrap();
        assert_eq!((o.truncate, o.max_tokens, o.prompt_role, o.normalize, o.pooling, o.output_dim), (0, 0, 0, 0, 0, 0));
    }

    #[test]
    fn refusals() {
        use ParameterChoice::*;
        let e = one("precision", StringParam("PRECISION_EXACT".into())).unwrap_err();
        assert_eq!((e.code, e.field), (TURBO_E_INVALID_ARGUMENT, 0));
        assert!(e.message.contains("precision"), "{e:?}");
        for (name, field, c) in [
            ("truncate", 1, Int64Param(2)),
            ("max_tokens", 2, StringParam("8".into())),
            ("max_tokens", 2, BoolParam(true)),
            ("max_tokens", 2, DoubleParam(8.0)),
            ("max_tokens", 2, Uint64Param(8)),
            ("max_tokens", 2, Int64Param(-1)),
            ("max_tokens", 2, Int64Param(4294967296)),
            ("pooling", 5, BoolParam(true)),
            ("output_dim", 6, Int64Param(i64::MIN)),
        ] {
            let e = one(name, c.clone()).unwrap_err();
            assert_eq!((e.code, e.field), (TURBO_E_INVALID_ARGUMENT, field), "{name} {c:?}");
        }
        let e =
            options(&HashMap::from([("pooling".to_string(), InferParameter { parameter_choice: None })])).unwrap_err();
        assert_eq!((e.code, e.field), (TURBO_E_INVALID_ARGUMENT, 5));
        for (name, s) in [
            ("truncate", "truncate_right"),
            ("truncate", "TURBO_TRUNCATE_RIGHT"),
            ("truncate", "POOLING_MEAN"),
            ("pooling", ""),
            ("normalize", "L2"),
            ("prompt_role", "PROMPT_QUERY "),
        ] {
            let e = one(name, StringParam(s.into())).unwrap_err();
            assert_eq!((e.code, e.field), (TURBO_E_INVALID_ENUM, 0), "{name} {s}");
            assert!(e.message.contains(name) && e.message.contains(&format!("`{s}`")), "{e:?}");
        }
    }

    #[test]
    fn raw_bytes() {
        let mut b = Vec::new();
        for s in ["", "ab", "caf\u{e9}"] {
            b.extend((s.len() as u32).to_le_bytes());
            b.extend(s.as_bytes());
        }
        let v = split_bytes(&b).unwrap();
        assert_eq!(v.iter().map(|t| t.len).collect::<Vec<_>>(), [0, 2, 5]);
        assert!(split_bytes(&[]).unwrap().is_empty());
        for bad in [&b[..b.len() - 1], &[1, 0, 0][..], &[5, 0, 0, 0, b'a'][..]] {
            assert_eq!(split_bytes(bad).err().map(|e| e.code), Some(TURBO_E_INVALID_ARGUMENT));
        }
    }
}
