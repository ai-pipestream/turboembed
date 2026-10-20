use super::windows;
use crate::{
    bundle::Bundle,
    safetensors::{Dtype, File},
    status::{Result, invalid},
    tokenizer::Tokenizer,
};

/// Verify joint-input IDs, segment IDs and finite two-column reference logits.
/// Rows enumerate all windows of each case in case order.
pub fn check_reference(bundle: &Bundle, tok: &Tokenizer) -> Result<()> {
    let m = &bundle.manifest;
    let q = m.qa.as_ref().ok_or_else(|| invalid("missing qa configuration"))?;
    let a = m.architecture.as_ref().ok_or_else(|| invalid("missing QA architecture"))?;
    let mut rows = Vec::new();
    for case in &m.reference.cases {
        let context = case.context.as_deref().ok_or_else(|| invalid("QA reference case needs context"))?;
        rows.extend(windows(tok, a.family, q, &case.text, context)?);
    }
    let bytes = bundle.read_verified(&m.reference.file)?;
    let file = File::parse(&m.reference.file, &bytes)?;
    let ids = file.get("ids", Dtype::I32, 2)?;
    let types = file.get("types", Dtype::I32, 2)?;
    let lengths = file.get("lengths", Dtype::I32, 1)?;
    let logits = file.get("logits", Dtype::F32, 2)?;
    let tokens: usize = rows.iter().map(|r| r.ids.len()).sum();
    if ids.shape[0] != rows.len() as u64
        || types.shape != ids.shape
        || lengths.shape != [rows.len() as u64]
        || logits.shape != [tokens as u64, 2]
        || logits.f32s().iter().any(|v| !v.is_finite())
    {
        return Err(invalid("QA reference shapes or logits are invalid"));
    }
    let width = ids.shape[1] as usize;
    let (ids, types, lengths) = (ids.i32s(), types.i32s(), lengths.i32s());
    for (i, row) in rows.iter().enumerate() {
        let n = row.ids.len();
        if n > width || lengths[i] < 0 || lengths[i] as usize != n {
            return Err(invalid(format!("QA reference window {i}: wrong length")));
        }
        let range = i * width..i * width + n;
        if ids[range.clone()] != row.ids || types[range] != row.types {
            return Err(invalid(format!("QA reference window {i}: IDs or segment IDs disagree")));
        }
        if ids[i * width + n..(i + 1) * width].iter().any(|&v| v != tok.fill_id())
            || types[i * width + n..(i + 1) * width].iter().any(|&v| v != 0)
        {
            return Err(invalid(format!("QA reference window {i}: invalid padding")));
        }
    }
    Ok(())
}
