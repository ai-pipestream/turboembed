//! The JSON of the few `bytes` fields the page reads or writes (build.rs):
//! a document's text as a string, and vector data, rows of little-endian
//! f32, as numbers. Other bytes fields are arrays of byte values.

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
}
