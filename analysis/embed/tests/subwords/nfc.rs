//! The subword layer of a byte-level BPE that normalizes with NFC, as
//! Qwen2's does, on text NFC composes and reorders.

use super::*;

/// testdata/tiny-bpe with an NFC normalizer, in a bundle that says so.
fn bpe_nfc() -> (TempBundle, PathBuf, SubwordAnnotator) {
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(testdata().join("tiny-bpe/tokenizer.json")).unwrap()).unwrap();
    json["normalizer"] = serde_json::json!({ "type": "NFC" });
    let file = std::env::temp_dir().join(format!("turbo-embed-test-{}-nfc-tokenizer.json", std::process::id()));
    std::fs::write(&file, serde_json::to_vec(&json).unwrap()).unwrap();
    let (dir, tok, bundle) = bundle_with("bpe-nfc", &file, |m| {
        m.model.id = "turbo-testdata/tiny-bpe-nfc".into();
        m.tokenizer.bpe =
            Some(Bpe { add_prefix_space: false, ignore_merges: false, pattern: Default::default(), nfc: true });
        roberta_specials(m, 4);
    });
    (dir, file, SubwordAnnotator::new(tok, &bundle))
}

/// Marks NFC composes past another, or reorders, still give a layer the
/// document takes: spans in text order at both ends, each covering what
/// its pieces were made from.
#[test]
fn nfc_pieces_are_taken_where_marks_compose_and_reorder() {
    let (_dir, file, a) = bpe_nfc();
    let texts =
        ["a\u{327}\u{301}", "x\u{301}\u{327}", "e\u{302}\u{301} a\u{344} q\u{323}\u{307}x", "a\u{301}\u{327} b"];
    let mut docs: Vec<Document> = texts.iter().map(|t| Document::new(*t)).collect();
    for (s, t) in run(a, &mut docs, false).into_iter().zip(texts) {
        s.unwrap_or_else(|e| panic!("{t:?}: {e}"));
    }
    let sub = docs[0].get::<Subwords>().unwrap();
    // The a, its acute and the cedilla between them share one span.
    assert!(pairs(sub).iter().all(|&(_, s)| s == [0, 5]), "{:?}", pairs(sub));
    let sub = docs[1].get::<Subwords>().unwrap();
    assert_eq!(pairs(sub).first().map(|p| p.1), Some([0, 1]), "{:?}", pairs(sub));
    assert!(pairs(sub).iter().skip(1).all(|&(_, s)| s == [1, 5]), "{:?}", pairs(sub));
    let _ = std::fs::remove_file(file);
}
