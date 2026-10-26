//! The JSON of the `bytes` fields the page reads or writes (build.rs): a
//! document's text as a string, vector data, rows of little-endian f32, as
//! numbers, and document graphs and file contents as base64, as the
//! protobuf JSON mapping writes bytes. Other bytes fields are arrays of
//! byte values.

/// UTF-8 text as a JSON string.
pub(crate) mod utf8 {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S: Serializer>(b: &[u8], s: S) -> Result<S::Ok, S::Error> {
        match std::str::from_utf8(b) {
            Ok(t) => s.serialize_str(t),
            Err(e) => Err(serde::ser::Error::custom(format!("text is not UTF-8: {e}"))),
        }
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        Ok(String::deserialize(d)?.into_bytes())
    }
}

/// Little-endian f32 values as a JSON array of numbers; a value that is
/// not finite is `null`, as serde_json writes it.
pub(crate) mod f32le {
    use serde::{Deserialize, Deserializer, Serializer};

    pub(crate) fn serialize<S: Serializer>(b: &[u8], s: S) -> Result<S::Ok, S::Error> {
        let (values, rest) = b.as_chunks::<4>();
        if !rest.is_empty() {
            return Err(serde::ser::Error::custom(format!("{} bytes are not a multiple of 4", b.len())));
        }
        s.collect_seq(values.iter().map(|v| f32::from_le_bytes(*v)))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        Ok(Vec::<f32>::deserialize(d)?.iter().flat_map(|v| v.to_le_bytes()).collect())
    }
}

/// Bytes as a base64 string: written with the standard alphabet and
/// padding, read with the standard or the URL-safe alphabet, padded or
/// not, as the protobuf JSON mapping allows.
pub(crate) mod base64 {
    use serde::{Deserialize, Deserializer, Serializer};

    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    pub(crate) fn encode(b: &[u8]) -> String {
        let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
        for group in b.chunks(3) {
            let n = group.iter().enumerate().fold(0u32, |n, (i, &v)| n | (v as u32) << (16 - 8 * i));
            for i in 0..4 {
                if i <= group.len() {
                    out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }

    pub(crate) fn decode(s: &str) -> Result<Vec<u8>, String> {
        let text = s.trim_end_matches('=');
        if s.len() - text.len() > 2 || text.len() % 4 == 1 {
            return Err(format!("{} characters are not base64", s.len()));
        }
        let mut out = Vec::with_capacity(text.len() * 3 / 4);
        let (mut n, mut bits) = (0u32, 0u32);
        for c in text.bytes() {
            let v = match c {
                b'A'..=b'Z' => c - b'A',
                b'a'..=b'z' => c - b'a' + 26,
                b'0'..=b'9' => c - b'0' + 52,
                b'+' | b'-' => 62,
                b'/' | b'_' => 63,
                _ => return Err(format!("{:?} is not a base64 character", c as char)),
            };
            n = n << 6 | v as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((n >> bits) as u8);
                n &= (1 << bits) - 1;
            }
        }
        Ok(out)
    }

    pub(crate) fn serialize<S: Serializer>(b: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&encode(b))
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        decode(&String::deserialize(d)?).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use crate::analysis::{AnalyzeDocument, AnalyzeVectors};

    #[test]
    fn text_is_a_string_and_vector_data_is_numbers() {
        let doc: AnalyzeDocument = serde_json::from_str(r#"{"id":"a","model":"m","text":"héllo"}"#).unwrap();
        assert_eq!((doc.text.as_slice(), doc.output, doc.options), ("héllo".as_bytes(), 0, None));
        let v = AnalyzeVectors {
            dim: 2,
            data: [1.5f32, -2.0].iter().flat_map(|f| f.to_le_bytes()).collect(),
            ..Default::default()
        };
        let json = serde_json::to_value(&v).unwrap();
        assert_eq!(json["data"], serde_json::json!([1.5, -2.0]));
        let back: AnalyzeVectors = serde_json::from_value(json).unwrap();
        assert_eq!(back, v);
        let odd = AnalyzeVectors { data: vec![0; 3], ..Default::default() };
        assert!(serde_json::to_string(&odd).unwrap_err().to_string().contains("not a multiple of 4"));
        let bad = AnalyzeDocument { text: vec![0xff], ..Default::default() };
        assert!(serde_json::to_string(&bad).unwrap_err().to_string().contains("not UTF-8"));
    }

    #[test]
    fn bytes_are_base64_both_ways() {
        use super::base64::{decode, encode};
        for (raw, text) in [
            (&b""[..], ""),
            (&b"f"[..], "Zg=="),
            (&b"fo"[..], "Zm8="),
            (&b"foo"[..], "Zm9v"),
            (&b"\xfb\xff"[..], "+/8="),
        ] {
            assert_eq!(encode(raw), text);
            assert_eq!(decode(text).unwrap(), raw);
        }
        assert_eq!(decode("-_8").unwrap(), b"\xfb\xff");
        assert!(decode("Zg===").is_err() && decode("Z").is_err() && decode("Z!==").is_err());
    }
}
