use super::*;

#[test]
fn reused_subwords_preserve_word_slice_control_whitespace() {
    let f = nonzero_finder();
    for text in [
        "Ann met Lee",
        "Ann\u{85}Lee",
        "John\u{c}Smith",
        "John\u{b}Smith",
        "John\tSmith",
        "John\nSmith",
        "Zoë\u{c}李",
        "John\rSmith",
        "John\r\nSmith",
        "Zoë\r李",
        "John\u{2028}Smith",
        "John\u{2029}Smith",
    ] {
        let doc = Document::new(text);
        let view = doc.require(&Key::<View>::DEFAULT).unwrap();
        let tokens = tokens(text);
        let subwords = subword_layer(&f, text);
        let (fallback,) = f.annotate((view, &tokens, None, None)).unwrap();
        let (reused,) = f.annotate((view, &tokens, Some(&subwords), None)).unwrap();
        let spans = |e: &Entities| e.spans.iter().map(|(_, s)| (s.start, s.end)).collect::<Vec<_>>();
        assert!(!fallback.spans.is_empty(), "nonzero head must produce entities: {text:?}");
        assert_eq!(reused.kind, fallback.kind);
        assert_eq!(reused.tokens, fallback.tokens);
        assert_eq!(reused.score, fallback.score, "text={text:?}");
        assert_eq!(spans(&reused), spans(&fallback), "text={text:?}, genuine tokenizer ids={:?}", subwords.ids);
    }
}

// The actual tiny encoder with nonzero classifier weights, so output
// probabilities depend on the reused IDs and hidden states.
fn nonzero_finder() -> NameFinderDl {
    let dir = write_bundle("word-slice-nonzero", &["O", "B-PER", "I-PER"]);
    let path = dir.join("weights/model.safetensors");
    let mut bytes = fs::read(&path).unwrap();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let start = 8 + n + header["classifier.weight"]["data_offsets"][0].as_u64().unwrap() as usize;
    for i in 0..96 {
        let weight = ((i % 32) as f32 - 16.0) * 0.003 * (i / 32 + 1) as f32;
        bytes[start + i * 4..start + (i + 1) * 4].copy_from_slice(&weight.to_le_bytes());
    }
    fs::write(path, &bytes).unwrap();
    let path = dir.join("manifest.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let file = manifest["files"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|f| f["path"] == "weights/model.safetensors")
        .unwrap();
    file["sha256"] = json!(sha256_hex(&bytes));
    fs::write(path, serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
    NameFinderDl::open(&dir, &NameFinderDlOptions { decoding: Decoding::Piece, ..Default::default() }).unwrap()
}
