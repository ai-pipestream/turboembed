//! Synthetic DeBERTa-v3 weights for the speed and thread tests: every
//! tensor a configuration names, with values that do not matter to them.

use turbo::deberta::Config;

/// Canonical safetensors with every tensor the configuration names:
/// LayerNorm weights 1, every other value small and spread.
pub fn weights(c: &Config) -> Vec<u8> {
    let shapes = c.tensor_shapes().unwrap();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    for (name, shape) in &shapes {
        let n: u64 = shape.iter().product();
        let start = data.len();
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let v =
                if name.ends_with("LayerNorm.weight") { 1.0 } else { ((seed >> 40) as f32 / 16_777_216.0 - 0.5) * 0.1 };
            data.extend_from_slice(&f32::to_le_bytes(v));
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    out
}
