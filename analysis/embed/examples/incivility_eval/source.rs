//! Exact training input, distinct from the Google Parquet dataset.
use super::*;
use std::path::Path;

pub(super) fn verify(dir: &Path) -> Result<Value, String> {
    let index = turbo_zoo::Index::builtin();
    let source = index.get("civil-comments-wilds").map_err(|e| e.to_string())?;
    let file =
        source.files.iter().find(|f| f.path.as_str() == "all_data_with_identities.csv").ok_or("missing CSV pin")?;
    let bytes = fs::read(dir.join(file.path.as_str())).map_err(|e| e.to_string())?;
    if bytes.len() as u64 != file.size || Sha256::of(&bytes) != file.sha256 || Sha512::of(&bytes) != file.sha512 {
        return Err("Civil Comments input differs from the pinned WILDS CSV".into());
    }
    Ok(json!({
        "zoo_id": source.id, "revision": source.version, "license": "CC0-1.0",
        "file": file.path, "url": file.url, "size_bytes": file.size,
        "sha256": file.sha256.to_string(), "sha512": file.sha512.to_string(),
    }))
}
