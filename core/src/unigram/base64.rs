/// Standard base64 with padding, as the tokenizer file carries the map.
pub(super) fn base64(s: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    };
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for (i, chunk) in bytes.chunks(4).enumerate() {
        let pad = chunk.iter().filter(|&&b| b == b'=').count();
        if chunk.len() != 4 || pad > 2 || chunk[..4 - pad].contains(&b'=') {
            return None;
        }
        if pad != 0 && (i + 1) * 4 != bytes.len() {
            return None;
        }
        let mut v = 0u32;
        for &b in &chunk[..4 - pad] {
            v = (v << 6) | value(b)?;
        }
        if (pad == 2 && v & 15 != 0) || (pad == 1 && v & 3 != 0) {
            return None;
        }
        v <<= 6 * pad as u32;
        out.push((v >> 16) as u8);
        if pad < 2 {
            out.push((v >> 8) as u8);
        }
        if pad < 1 {
            out.push(v as u8);
        }
    }
    Some(out)
}
