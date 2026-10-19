use super::*;

fn model(name: &str) -> SentencePiece {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece").join(name);
    SentencePiece::parse(&std::fs::read(path).unwrap()).unwrap()
}

/// BPE word by word gives what BPE over the whole text gives, on the
/// model that allows it, for every golden text and their join.
#[test]
fn bpe_word_by_word_is_bpe_over_the_whole_text() {
    let words = model("bpe-bytes.model");
    assert!(words.split_words);
    let mut whole = model("bpe-bytes.model");
    whole.split_words = false;
    // A model with UNUSED pieces merges the whole text at once.
    assert!(!model("bpe-identity.model").split_words);
    let file =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece/bpe-bytes.goldens.jsonl");
    let file = std::fs::read_to_string(file).unwrap();
    let mut texts: Vec<String> = file
        .lines()
        .skip(1)
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().map(str::to_owned))
        .collect();
    texts.push(texts.join("\n"));
    let mut scratch = Scratch::default();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for t in &texts {
        a.clear();
        b.clear();
        words.encode(t, &mut scratch, &mut a);
        whole.encode(t, &mut scratch, &mut b);
        assert_eq!(a, b, "{t:?}");
    }
}
