use super::*;
/// Lower-case ASCII letters and digits, anything else one `-`, none at
/// either end.
pub(super) fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.is_empty() && !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_owned()
}

/// A record's file name, from its contents alone:
///
/// `<machine>.<backend>[.<graph format>].<task>.<precision>[-dense].<model>-<manifest>.<commit>.json`
///
/// machine is the arch label, and for a CPU the arch label and the first 8
/// hex of the SHA-256 of the processor's name, since a CPU record is filed
/// under both; an npu record inserts the graph format (`ngraph-lite` or
/// `native`) after the backend, so a later driver that advertises
/// ZE_GRAPH_FORMAT_NATIVE does not share a file with an NGRAPH_LITE
/// record; task and precision are the enum names without their prefix;
/// model is the last part of the model id, at most 32 bytes; manifest is
/// the first 8 hex of the bundle's manifest hash, commit the first 12 of
/// the library's; `-dense` marks dense rows, so a mixed and a dense
/// record of one commit are both kept. Longer than NAME_MAX is an error.
pub fn file_name(r: &Record) -> Result<String, String> {
    let mut machine = slug(&r.machine.arch);
    if r.device.kind == "DEVICE_CPU" {
        machine = format!("{machine}-{}", &sha256_hex(r.device.name.as_bytes())[..8]);
    }
    let bare = |s: &str, prefix: &str| slug(s.strip_prefix(prefix).unwrap_or(s));
    let mut model = slug(r.bundle.model_id.rsplit('/').next().unwrap_or(""));
    model.truncate(32);
    let model = model.trim_end_matches('-');
    let manifest = r.bundle.manifest_sha256.get(..8).unwrap_or("");
    let commit = r.library.commit.get(..12).unwrap_or("");
    let dense = if r.rows.kind == ROWS_DENSE { "-dense" } else { "" };
    let format = npu_format_slug(r)?;
    let name = format!(
        "{machine}.{}{format}.{}.{}{dense}.{model}-{manifest}.{commit}.json",
        slug(&r.device.backend),
        bare(&r.task, "TASK_"),
        bare(&r.precision, "PRECISION_"),
    );
    if name.len() > NAME_MAX {
        return Err(format!("the record's name {name} is {} bytes, over {NAME_MAX}", name.len()));
    }
    Ok(name)
}

/// `.<slug>` for an npu record, from TURBO_NPU_GRAPH_FORMAT. Empty for
/// every other backend.
pub(super) fn npu_format_slug(r: &Record) -> Result<String, String> {
    if r.device.backend != "npu" {
        return Ok(String::new());
    }
    let fmt = setting(r, "TURBO_NPU_GRAPH_FORMAT")
        .ok_or_else(|| "an npu record names TURBO_NPU_GRAPH_FORMAT as NGRAPH_LITE or NATIVE".to_owned())?;
    if !matches!(fmt, "NGRAPH_LITE" | "NATIVE") {
        return Err(format!("TURBO_NPU_GRAPH_FORMAT={fmt} is not NGRAPH_LITE or NATIVE"));
    }
    Ok(format!(".{}", slug(fmt)))
}

/// The value of `NAME=value` in library.settings, when that name is set.
pub(super) fn setting<'a>(r: &'a Record, name: &str) -> Option<&'a str> {
    let prefix = format!("{name}=");
    r.library.settings.iter().find_map(|s| s.strip_prefix(prefix.as_str()))
}

pub(super) fn is_hex(s: &str, len: usize) -> bool {
    s.len() == len && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
