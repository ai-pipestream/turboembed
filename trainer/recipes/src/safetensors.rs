//! A safetensors file of F32 tensors: an 8-byte little-endian header
//! length, the JSON header with each tensor's dtype, shape and byte
//! offsets in name order, then the data. The header is padded with
//! spaces to a multiple of 8 so the data starts aligned. A file may
//! carry string metadata under `__metadata__`, as the format allows.

use std::fmt::Write as _;

/// The file holding `tensors` (name, shape, values), sorted by name.
///
/// # Panics
///
/// When a tensor's values do not fill its shape.
pub fn write(tensors: &[(String, Vec<u64>, Vec<f32>)]) -> Vec<u8> {
    write_with(tensors, &[])
}

/// [`write`] with `metadata` (key, value) in the header's `__metadata__`,
/// so what describes the tensors is in the same file and replaced with
/// them in one rename.
///
/// # Panics
///
/// When a tensor's values do not fill its shape.
pub fn write_with(tensors: &[(String, Vec<u64>, Vec<f32>)], metadata: &[(&str, String)]) -> Vec<u8> {
    let mut sorted: Vec<&(String, Vec<u64>, Vec<f32>)> = tensors.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = String::from("{");
    if !metadata.is_empty() {
        let m: serde_json::Map<String, serde_json::Value> =
            metadata.iter().map(|(k, v)| ((*k).to_owned(), serde_json::Value::from(v.as_str()))).collect();
        write!(header, "\"__metadata__\":{}", serde_json::Value::Object(m)).expect("writing to a String");
        if !sorted.is_empty() {
            header.push(',');
        }
    }
    let mut at = 0usize;
    for (i, (name, shape, values)) in sorted.iter().enumerate() {
        assert_eq!(shape.iter().product::<u64>() as usize, values.len(), "{name}: values fill the shape");
        let end = at + values.len() * 4;
        let shape: Vec<String> = shape.iter().map(u64::to_string).collect();
        let sep = if i == 0 { "" } else { "," };
        write!(
            header,
            "{sep}{}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{at},{end}]}}",
            json(name),
            shape.join(",")
        )
        .expect("writing to a String");
        at = end;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + at);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, values) in sorted {
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

fn json(s: &str) -> String {
    serde_json::to_string(s).expect("a string serializes")
}
