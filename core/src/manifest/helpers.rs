use super::*;
impl Manifest {
    pub(super) fn check_produced_by(
        &self,
        field: &str,
        p: &ProducedBy,
        listed_path: &dyn Fn(&str, &str) -> Result<()>,
    ) -> Result<()> {
        required(&format!("{field}.tool"), &p.tool)?;
        required(&format!("{field}.tool_version"), &p.tool_version)?;
        required(&format!("{field}.container"), &p.container)?;
        for (j, input) in p.inputs.iter().enumerate() {
            listed_path(&format!("{field}.inputs[{j}]"), input)?;
        }
        Ok(())
    }
}

/// The manifest's snake_case name for a tensor role.
pub fn role_name(role: TensorRole) -> String {
    let mut out = String::new();
    for (i, c) in format!("{role:?}").chars().enumerate() {
        if c.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Bundle paths: relative, `/`-separated, no `.` or `..` or empty
/// component, no backslash, no NUL.
pub(super) fn check_path(field: &str, p: &str) -> Result<()> {
    let bad = |why: &str| Err(invalid(format!("manifest.json: {field}: {p:?} {why}")));
    if p.is_empty() {
        return bad("is empty");
    }
    if p.starts_with('/') {
        return bad("starts with /");
    }
    if p.contains('\\') || p.contains('\0') {
        return bad("contains a backslash or NUL");
    }
    for c in p.split('/') {
        match c {
            ".." => return bad("contains .."),
            "." | "" => return bad("has an empty or . component"),
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub(super) fn required(field: &str, v: &str) -> Result<()> {
    if v.is_empty() {
        return Err(invalid(format!("manifest.json: {field}: required")));
    }
    Ok(())
}

pub(super) fn positive(field: &str, v: u32) -> Result<()> {
    if v == 0 {
        return Err(invalid(format!("manifest.json: {field}: must be over 0")));
    }
    Ok(())
}

/// A string must fit its header buffer with the terminating NUL.
pub(super) fn fits(field: &str, v: &str, buffer: usize) -> Result<()> {
    if v.contains('\0') {
        return Err(invalid(format!("manifest.json: {field}: contains NUL")));
    }
    if v.len() >= buffer {
        return Err(invalid(format!(
            "manifest.json: {field}: {} bytes is over the header's {} for this field",
            v.len(),
            buffer - 1
        )));
    }
    Ok(())
}

pub(super) fn is_task_name(s: &str) -> bool {
    s.len() > 5 && s.starts_with("TASK_") && s[5..].bytes().all(|b| b.is_ascii_uppercase() || b == b'_')
}

pub(super) fn unsupported_task(t: &str) -> Error {
    Error::new(UNSUPPORTED_TASK, format!("manifest.json: task: this build has TASK_EMBED only, not {t}"))
}

/// The task, if the manifest is JSON with a string `task`: another task's
/// block is an unknown field to this build, and should read as
/// UNSUPPORTED_TASK rather than BUNDLE_INVALID.
pub(super) fn peek_task(bytes: &[u8]) -> Option<String> {
    let v: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    v.get("task")?.as_str().map(str::to_owned)
}
