//! What the reference programs that run the bundle's ONNX file share: the
//! file itself, found by its format, and the token rows written as the raw
//! input files those programs read.

use std::fs;
use std::path::Path;

use turbo::manifest::{Dtype, Format, Manifest};

use crate::Result;
use crate::measure::Rows;

/// The bundle's ONNX file in `dtype`, or why there is none for the
/// program to run: None is the upstream graph, the FORMAT_ONNX artifact
/// with no compute_dtype, and Some a graph converted to that dtype
/// (docs/bundle.md). `purpose` ends the sentence ("for trtexec to build
/// an engine from").
pub fn file(m: &Manifest, dtype: Option<Dtype>, purpose: &str) -> std::result::Result<String, String> {
    let a = m.artifacts.iter().find(|a| a.format == Format::Onnx && a.compute_dtype == dtype);
    match a.map(|a| a.files.as_slice()) {
        None if dtype.is_none() => Err(format!("the bundle carries no FORMAT_ONNX artifact {purpose}")),
        None => Err(format!("the bundle carries no FORMAT_ONNX artifact in {} {purpose}", dtype_name(dtype))),
        Some([one]) => Ok(one.clone()),
        Some(_) => Err("the bundle's FORMAT_ONNX artifact is not one file".into()),
    }
}

fn dtype_name(d: Option<Dtype>) -> &'static str {
    match d {
        None => "the upstream graph's types",
        Some(Dtype::F32) => "DTYPE_F32",
        Some(Dtype::F16) => "DTYPE_F16",
        Some(Dtype::Bf16) => "DTYPE_BF16",
        Some(Dtype::I8) => "DTYPE_I8",
        Some(Dtype::I32) => "DTYPE_I32",
    }
}

/// The ONNX input names, each of `[A-Za-z0-9_.]+` and neither `.` nor
/// `..`: each becomes a file name under the work directory and a part of
/// the program's input lists, so it can hold no path separator and none
/// of the characters those lists are split on. `option` is the command
/// line option they came from.
pub fn check_inputs(option: &str, inputs: &[String]) -> Result<()> {
    for n in inputs {
        let plain = !n.is_empty() && n.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.');
        if !plain || n == "." || n == ".." {
            return Err(format!("{option}: {n:?} is not an input name of [A-Za-z0-9_.]+"));
        }
    }
    Ok(())
}

/// The rows as raw little-endian values in `dtype`, `int64` or `int32`;
/// `option` is the command line option that named the dtype.
pub fn input_bytes(option: &str, values: &[i32], dtype: &str) -> Result<Vec<u8>> {
    match dtype {
        "int64" => Ok(values.iter().flat_map(|&v| (v as i64).to_le_bytes()).collect()),
        "int32" => Ok(values.iter().flat_map(|&v| v.to_le_bytes()).collect()),
        d => Err(format!("{option} {d:?} is not int64 or int32")),
    }
}

/// Write the ids, the mask and the types to `<name>.bin` in `dir`, for
/// the inputs `names` in that order.
pub fn write_inputs(dir: &Path, names: &[String; 3], dtype: &str, option: &str, rows: &Rows) -> Result<()> {
    for (name, values) in names.iter().zip([&rows.ids, &rows.mask, &rows.types]) {
        let path = dir.join(format!("{name}.bin"));
        fs::write(&path, input_bytes(option, values, dtype)?).map_err(|e| format!("{}: {e}", path.display()))?;
    }
    Ok(())
}
