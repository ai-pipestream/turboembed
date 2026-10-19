//! What a compiled graph takes, by input kind.

use super::args::*;
use super::*;
use crate::backend::turbo_backend_model;
use crate::npu::ze;
use crate::status::RUNTIME;

/// INPUT_TOKEN_IDS: ids, mask, and token types where the graph takes them,
/// each I64 or I32 of one `[batch, seq]`, layout NC.
pub(super) fn accept_tokens(
    inputs: Vec<Compiled>,
    output: Compiled,
    desc: &turbo_backend_model,
) -> Res<(GraphInputs, Arg, u32, u32, u32)> {
    if !(2..=3).contains(&inputs.len()) {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph takes {:?}; an encoder takes input_ids, attention_mask and optionally token_type_ids",
                arg_names(&inputs)
            ),
        ));
    }
    let (mut ids, mut mask, mut types) = (None, None, None);
    for entry in inputs {
        let slot = match entry.arg.name.as_str() {
            "input_ids" => &mut ids,
            "attention_mask" => &mut mask,
            "token_type_ids" => &mut types,
            other => {
                return Err(fail(
                    RUNTIME,
                    format!(
                        "npu: the graph takes an input named {other:?}, which this backend does not know; it runs \
                         graphs whose inputs are input_ids, attention_mask and optionally token_type_ids"
                    ),
                ));
            }
        };
        if slot.replace(entry).is_some() {
            return Err(fail(RUNTIME, "npu: the graph takes two inputs of one name"));
        }
    }
    let (Some(ids), Some(mask)) = (ids, mask) else {
        return Err(fail(RUNTIME, "npu: the graph takes no input_ids or no attention_mask"));
    };
    if ids.rank != 2 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; token ids are [batch, seq]", ids.arg.name, ids.rank),
        ));
    }
    expect_packed_layout(&ids.arg.name, ids.rank, ids.layout)?;
    let (frame_batch, seq) = (ids.dims[0], ids.dims[1]);
    if mask.rank != 2 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; the mask is [batch, seq]", mask.arg.name, mask.rank),
        ));
    }
    expect_frame(&mask.arg.name, &mask.dims, frame_batch, seq)?;
    expect_packed_layout(&mask.arg.name, mask.rank, mask.layout)?;
    let types = match types {
        Some(t) => {
            if t.rank != 2 {
                return Err(fail(
                    RUNTIME,
                    format!("npu: {:?} has rank {}; token types are [batch, seq]", t.arg.name, t.rank),
                ));
            }
            expect_frame(&t.arg.name, &t.dims, frame_batch, seq)?;
            expect_packed_layout(&t.arg.name, t.rank, t.layout)?;
            Some(t.arg)
        }
        None => None,
    };
    let hidden = expect_hidden(&output, frame_batch, seq)?;
    let ids = ids.arg;
    let mask = mask.arg;
    for a in [&ids, &mask].into_iter().chain(types.iter()) {
        if !matches!(
            a.precision,
            ze::GRAPH_ARGUMENT_PRECISION_INT64
                | ze::GRAPH_ARGUMENT_PRECISION_INT32
                | ze::GRAPH_ARGUMENT_PRECISION_UINT64
                | ze::GRAPH_ARGUMENT_PRECISION_UINT32
        ) {
            return Err(fail(
                RUNTIME,
                format!(
                    "npu: {:?} is argument precision 0x{:02x}; rows are written as I64 or I32",
                    a.name, a.precision
                ),
            ));
        }
    }
    expect_manifest_frame(frame_batch, seq, hidden, desc)?;
    Ok((GraphInputs::Tokens { ids, mask, types }, output.arg, frame_batch, seq, hidden))
}

/// INPUT_EMBEDDINGS: `word_rows` `[batch, seq, hidden]` and `attn_bias`
/// `[batch, heads, seq, seq]`, the cut hef_compile.py makes, kept at the
/// batch the IR was reshaped to. Layouts are the packed ones the build
/// flags name. The host writes FP32 or FP16, whichever the compiler kept.
pub(super) fn accept_embeddings(
    inputs: Vec<Compiled>,
    output: Compiled,
    desc: &turbo_backend_model,
) -> Res<(Arg, Arg, Arg, u32, u32, u32)> {
    if inputs.len() != 2 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph takes {:?}; an INPUT_EMBEDDINGS graph takes word_rows and attn_bias",
                arg_names(&inputs)
            ),
        ));
    }
    let (mut rows, mut bias) = (None, None);
    for entry in inputs {
        let slot = match entry.arg.name.as_str() {
            "word_rows" => &mut rows,
            "attn_bias" => &mut bias,
            other => {
                return Err(fail(
                    RUNTIME,
                    format!(
                        "npu: the graph takes an input named {other:?}; an INPUT_EMBEDDINGS graph takes word_rows and \
                         attn_bias"
                    ),
                ));
            }
        };
        if slot.replace(entry).is_some() {
            return Err(fail(RUNTIME, "npu: the graph takes two inputs of one name"));
        }
    }
    let (Some(rows), Some(bias)) = (rows, bias) else {
        return Err(fail(RUNTIME, "npu: the graph takes no word_rows or no attn_bias"));
    };
    if rows.rank != 3 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; word rows are [batch, seq, hidden]", rows.arg.name, rows.rank),
        ));
    }
    expect_packed_layout(&rows.arg.name, rows.rank, rows.layout)?;
    let (frame_batch, seq, hidden) = (rows.dims[0], rows.dims[1], rows.dims[2]);
    if bias.rank != 4 {
        return Err(fail(
            RUNTIME,
            format!("npu: {:?} has rank {}; the attention bias is [batch, heads, seq, seq]", bias.arg.name, bias.rank),
        ));
    }
    expect_packed_layout(&bias.arg.name, bias.rank, bias.layout)?;
    if bias.dims[0] != frame_batch || bias.dims[2] != seq || bias.dims[3] != seq || bias.dims[1] != desc.heads {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} is [{}, {}, {}, {}]; the attention bias is [{frame_batch}, {}, {seq}, {seq}]",
                bias.arg.name, bias.dims[0], bias.dims[1], bias.dims[2], bias.dims[3], desc.heads
            ),
        ));
    }
    expect_float(&rows.arg.name, rows.arg.precision)?;
    expect_float(&bias.arg.name, bias.arg.precision)?;
    let out_hidden = expect_hidden(&output, frame_batch, seq)?;
    if out_hidden != hidden {
        return Err(fail(
            RUNTIME,
            format!("npu: word_rows hidden is {hidden}; {:?} hidden is {out_hidden}", output.arg.name),
        ));
    }
    expect_manifest_frame(frame_batch, seq, hidden, desc)?;
    Ok((rows.arg, bias.arg, output.arg, frame_batch, seq, hidden))
}
