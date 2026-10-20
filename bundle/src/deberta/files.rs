use super::{from_huggingface, write_package};
use crate::Result;
use std::{fs, io::Read, path::Path};
use turbo::deberta::package::MAX_WEIGHT_BYTES;

pub fn convert_files(
    config: &Path,
    checkpoint: &Path,
    prefix: &str,
    source: (&str, &str, &str),
    out: &Path,
) -> Result<()> {
    let json = read(config, 65536)?;
    let value = serde_json::from_slice(&json).map_err(|e| format!("deberta config JSON: {e}"))?;
    let config = from_huggingface(&value)?;
    let bytes = read(checkpoint, MAX_WEIGHT_BYTES)?;
    write_package(config, &bytes, if prefix == "-" { "" } else { prefix }, source, out)
}
fn read(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let f = fs::File::open(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > limit {
        return Err("deberta input byte limit exceeded".into());
    }
    let mut data = Vec::new();
    f.take(limit + 1).read_to_end(&mut data).map_err(|e| e.to_string())?;
    if data.len() as u64 > limit {
        return Err("deberta input byte limit exceeded".into());
    }
    Ok(data)
}
