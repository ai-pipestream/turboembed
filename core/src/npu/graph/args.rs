//! The compiled graph's arguments, checked against the manifest.

use std::ffi::CStr;

use super::*;
use crate::backend::{TURBO_BERT_EMBEDDING_TENSORS, turbo_backend_model, turbo_backend_tensor};
use crate::npu::ze;
use crate::status::{INVALID_ARGUMENT, RUNTIME, UNSUPPORTED};
use crate::{TURBO_DTYPE_F16, TURBO_DTYPE_F32};

/// One argument of the compiled graph.
pub(super) struct Arg {
    pub(super) index: u32,
    pub(super) name: String,
    pub(super) precision: u32,
    /// Bytes of one element in the argument's device precision.
    pub(super) elem: usize,
}

/// One compiled argument, with the dims, rank and device layout the
/// driver reported. The layout is what the host buffer must be when
/// `pfnSetArgumentValue` is called.
pub(super) struct Compiled {
    pub(super) arg: Arg,
    pub(super) dims: [u32; 5],
    pub(super) rank: u32,
    pub(super) layout: u32,
}

/// What the compiled graph takes, after the boundary check.
pub(super) enum GraphInputs {
    /// Token ids, the mask, and token types where the graph takes them.
    Tokens { ids: Arg, mask: Arg, types: Option<Arg> },
    /// The Hailo-style cut: gathered word rows and an attention bias.
    /// `table` is the core's F32 `word_embeddings`, alive until model_release.
    Embeddings { rows: Arg, bias: Arg, heads: u32, table: *const f32, vocab: u32 },
}

/// The packed device layout the build flags name for a rank: NC for a
/// rank-2 token frame, CHW for rank 3 (word rows, or the hidden states),
/// NCHW for a rank-4 attention bias.
pub(super) fn packed_layout(rank: u32) -> Option<u32> {
    match rank {
        2 => Some(ze::GRAPH_ARGUMENT_LAYOUT_NC),
        3 => Some(ze::GRAPH_ARGUMENT_LAYOUT_CHW),
        4 => Some(ze::GRAPH_ARGUMENT_LAYOUT_NCHW),
        _ => None,
    }
}

pub(super) fn layout_label(layout: u32) -> String {
    match layout {
        ze::GRAPH_ARGUMENT_LAYOUT_NC => "NC".to_string(),
        ze::GRAPH_ARGUMENT_LAYOUT_NCHW => "NCHW".to_string(),
        ze::GRAPH_ARGUMENT_LAYOUT_CHW => "CHW".to_string(),
        0 => "ANY".to_string(),
        0xC8 => "BLOCKED".to_string(),
        n => format!("0x{n:02x}"),
    }
}

/// Refuse unless `layout` is the packed layout the build flags asked for
/// at this rank. BLOCKED and ANY are not that layout: the host writes
/// packed rows.
pub(super) fn expect_packed_layout(name: &str, rank: u32, layout: u32) -> Res<()> {
    let Some(want) = packed_layout(rank) else {
        return Err(fail(
            RUNTIME,
            format!("npu: {name:?} has rank {rank}; the build flags name a packed layout for rank 2, 3 or 4"),
        ));
    };
    if layout != want {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {name:?} has device layout {}; the build flags asked for {}",
                layout_label(layout),
                layout_label(want)
            ),
        ));
    }
    Ok(())
}

/// The leading two dims are the token frame `[batch, seq]`.
pub(super) fn expect_frame(name: &str, dims: &[u32], batch: u32, seq: u32) -> Res<()> {
    let (d0, d1) = (dims.first().copied().unwrap_or(0), dims.get(1).copied().unwrap_or(0));
    if d0 != batch || d1 != seq {
        return Err(fail(RUNTIME, format!("npu: {name:?} is [{d0}, {d1}, ...]; the token frame is [{batch}, {seq}]")));
    }
    Ok(())
}

/// The TURBO_DTYPE_* the compiled output precision is. FP16 is
/// TURBO_DTYPE_F16 and FP32 is TURBO_DTYPE_F32. A manifest dtype of 0
/// takes that; any other value must be it. The session reports this
/// dtype, not a manifest value that disagrees.
pub(super) fn session_dtype(manifest: u32, precision: u32) -> Res<u32> {
    let dtype = match precision {
        ze::GRAPH_ARGUMENT_PRECISION_FP16 => TURBO_DTYPE_F16,
        ze::GRAPH_ARGUMENT_PRECISION_FP32 => TURBO_DTYPE_F32,
        p => {
            return Err(fail(
                RUNTIME,
                format!("npu: the graph computes in argument precision 0x{p:02x}; FP16 and FP32 are a session dtype"),
            ));
        }
    };
    if manifest != 0 && manifest != dtype {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: the graph computes in {}; the manifest's compute_dtype is {}",
                dtype_label(dtype),
                dtype_label(manifest)
            ),
        ));
    }
    Ok(dtype)
}

pub(super) fn dtype_label(d: u32) -> String {
    match d {
        TURBO_DTYPE_F16 => "DTYPE_F16".to_string(),
        TURBO_DTYPE_F32 => "DTYPE_F32".to_string(),
        n => n.to_string(),
    }
}

/// The element size of a graph argument precision this backend moves, or
/// a refusal naming it.
pub(super) fn elem_bytes(what: &str, precision: u32) -> Res<usize> {
    Ok(match precision {
        ze::GRAPH_ARGUMENT_PRECISION_INT64 | ze::GRAPH_ARGUMENT_PRECISION_UINT64 => 8,
        ze::GRAPH_ARGUMENT_PRECISION_FP32
        | ze::GRAPH_ARGUMENT_PRECISION_INT32
        | ze::GRAPH_ARGUMENT_PRECISION_UINT32 => 4,
        ze::GRAPH_ARGUMENT_PRECISION_FP16 | ze::GRAPH_ARGUMENT_PRECISION_BF16 => 2,
        p => {
            return Err(fail(
                RUNTIME,
                format!("npu: {what} is argument precision 0x{p:02x}, which this backend does not move"),
            ));
        }
    })
}

/// The host_weights word table an INPUT_EMBEDDINGS graph gathers from.
/// The core hands it first, TURBO_BERT_WORD_EMBEDDINGS, F32, [vocab, hidden].
pub(super) fn word_table(desc: &turbo_backend_model) -> Res<(*const f32, u32)> {
    if desc.tensors.is_null() || desc.tensor_count == 0 {
        return Err(fail(
            UNSUPPORTED,
            "npu: INPUT_EMBEDDINGS needs the host_weights word_embeddings table, and none was handed over",
        ));
    }
    if desc.tensor_count != TURBO_BERT_EMBEDDING_TENSORS {
        return Err(fail(
            UNSUPPORTED,
            format!(
                "npu: INPUT_EMBEDDINGS needs the {TURBO_BERT_EMBEDDING_TENSORS} host embedding tensors; {} were handed \
                 over",
                desc.tensor_count
            ),
        ));
    }
    if desc.dtype != TURBO_DTYPE_F32 {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: the host gather reads F32 word rows; the weights are {}", dtype_label(desc.dtype)),
        ));
    }
    let t = unsafe { &*desc.tensors };
    let name = tensor_name(t);
    if t.dtype != TURBO_DTYPE_F32 {
        return Err(fail(
            UNSUPPORTED,
            format!("npu: {name} is {}; the host gather reads F32 word rows", dtype_label(t.dtype)),
        ));
    }
    if t.ndim != 2 || t.shape[0] != desc.vocab_size as u64 || t.shape[1] != desc.hidden as u64 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {name} is {:?} of rank {}; word_embeddings is [{}, {}]",
                // A tensor's shape holds two dims; a higher rank names
                // more than it has.
                &t.shape[..(t.ndim as usize).min(t.shape.len())],
                t.ndim,
                desc.vocab_size,
                desc.hidden
            ),
        ));
    }
    // Both dims are u32, so the element count fits u64; its bytes may not.
    let n = desc.vocab_size as u64 * desc.hidden as u64;
    if n == 0 || n.checked_mul(4) != Some(t.bytes) || t.data.is_null() {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {name} is {} bytes; [{}, {}] F32 is {} bytes",
                t.bytes,
                desc.vocab_size,
                desc.hidden,
                n as u128 * 4
            ),
        ));
    }
    if !(t.data as usize).is_multiple_of(4) {
        return Err(fail(INVALID_ARGUMENT, format!("npu: {name} is not aligned to 4 bytes")));
    }
    Ok((t.data as *const f32, desc.vocab_size))
}

pub(super) fn tensor_name(t: &turbo_backend_tensor) -> String {
    if t.name.is_null() {
        return "<unnamed>".to_string();
    }
    unsafe { CStr::from_ptr(t.name).to_string_lossy().into_owned() }
}

pub(super) fn expect_float(name: &str, precision: u32) -> Res<()> {
    if !matches!(precision, ze::GRAPH_ARGUMENT_PRECISION_FP32 | ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        return Err(fail(
            RUNTIME,
            format!("npu: {name:?} is argument precision 0x{precision:02x}; FP32 and FP16 are written"),
        ));
    }
    Ok(())
}

/// The compiled batch and seq against the manifest. A dynamic graph is
/// refused before a zero manifest shape, so the message stays about the graph.
pub(super) fn expect_manifest_frame(frame_batch: u32, seq: u32, hidden: u32, desc: &turbo_backend_model) -> Res<()> {
    if hidden != desc.hidden {
        return Err(fail(RUNTIME, format!("npu: the graph's hidden is {hidden}; the manifest says {}", desc.hidden)));
    }
    if frame_batch == 0 || seq == 0 {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph compiled to [{frame_batch}, {seq}]; a static shape is needed"),
        ));
    }
    if desc.fixed_seq != seq {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph's seq is {seq}; the manifest's fixed_seq is {}", desc.fixed_seq),
        ));
    }
    if desc.fixed_batch != frame_batch {
        return Err(fail(
            RUNTIME,
            format!("npu: the graph's frame is {frame_batch} rows; the manifest's fixed_batch is {}", desc.fixed_batch),
        ));
    }
    Ok(())
}

pub(super) fn take_hidden(mut outputs: Vec<Compiled>) -> Res<Compiled> {
    if outputs.is_empty() {
        return Err(fail(RUNTIME, "npu: the graph returns nothing"));
    }
    let names: Vec<String> = outputs.iter().map(|c| c.arg.name.clone()).collect();
    match outputs.len() {
        1 => Ok(outputs.remove(0)),
        _ => {
            let at = outputs.iter().position(|c| c.arg.name == "last_hidden_state").ok_or_else(|| {
                fail(RUNTIME, format!("npu: the graph returns {names:?} and none is last_hidden_state"))
            })?;
            Ok(outputs.remove(at))
        }
    }
}

/// The hidden states are FP32 or FP16 `[batch, seq, hidden]`, layout CHW.
pub(super) fn expect_hidden(output: &Compiled, frame_batch: u32, seq: u32) -> Res<u32> {
    if output.rank != 3 {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} has rank {}; OUTPUT_HIDDEN_STATES is [batch, seq, hidden]",
                output.arg.name, output.rank
            ),
        ));
    }
    expect_frame(&output.arg.name, &output.dims, frame_batch, seq)?;
    expect_packed_layout(&output.arg.name, output.rank, output.layout)?;
    if !matches!(output.arg.precision, ze::GRAPH_ARGUMENT_PRECISION_FP32 | ze::GRAPH_ARGUMENT_PRECISION_FP16) {
        return Err(fail(
            RUNTIME,
            format!(
                "npu: {:?} is argument precision 0x{:02x}; FP32 and FP16 are read back",
                output.arg.name, output.arg.precision
            ),
        ));
    }
    Ok(output.dims[2])
}

pub(super) fn arg_names(v: &[Compiled]) -> Vec<String> {
    v.iter().map(|c| c.arg.name.clone()).collect()
}
