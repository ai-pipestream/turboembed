use super::*;

#[test]
fn base64_decodes_with_and_without_padding_bytes() {
    assert_eq!(base64("").unwrap(), b"");
    assert_eq!(base64("YQ==").unwrap(), b"a");
    assert_eq!(base64("YWI=").unwrap(), b"ab");
    assert_eq!(base64("YWJj").unwrap(), b"abc");
    assert_eq!(base64("YWJjZA==").unwrap(), b"abcd");
    assert!(base64("YQ=").is_none());
    assert!(base64("Y!==").is_none());
}

/// The trie finds every piece of XLM-RoBERTa's vocabulary (bge-m3)
/// by its bytes, and every piece a text starts with, shortest first,
/// as a search of the whole list does: at nodes with a table of their
/// children and nodes without.
#[test]
fn the_trie_finds_what_the_list_holds() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
    let json: Value = serde_json::from_slice(&std::fs::read(root.join("bge-m3/tokenizer.json")).unwrap()).unwrap();
    let pieces: Vec<Box<str>> =
        json["model"]["vocab"].as_array().unwrap().iter().map(|p| p[0].as_str().unwrap().into()).collect();
    let trie = Trie::new(&pieces).unwrap();
    assert!(trie.wide.len() > 100 && trie.nodes.iter().any(|n| n.count > 1 && n.wide == Trie::NONE));
    for (id, p) in pieces.iter().enumerate() {
        assert_eq!(trie.get(p.as_bytes()), Some(id as u32), "{p:?}");
    }
    let file = std::fs::read_to_string(root.join("tokenizer-texts.jsonl")).unwrap();
    for line in file.lines() {
        let text = serde_json::from_str::<Value>(line).unwrap()["text"].as_str().unwrap().replace(' ', "\u{2581}");
        for (at, _) in text.char_indices() {
            let rest = &text.as_bytes()[at..];
            let mut got = Vec::new();
            trie.prefixes(rest, |len, id| got.push((len, id)));
            let mut want: Vec<(usize, u32)> = pieces
                .iter()
                .enumerate()
                .filter(|(_, p)| rest.starts_with(p.as_bytes()))
                .map(|(id, p)| (p.len(), id as u32))
                .collect();
            want.sort();
            assert_eq!(got, want, "{:?}", &text[at..]);
        }
    }
}

/// The ASCII shortcut gives what the map gives cluster by cluster:
/// on XLM-RoBERTa's map (bge-m3), for the tokenizer texts and for
/// ASCII next to what joins a cluster.
#[test]
fn the_ascii_shortcut_is_the_cluster_by_cluster_map() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
    let json: Value = serde_json::from_slice(&std::fs::read(root.join("bge-m3/tokenizer.json")).unwrap()).unwrap();
    let mut norms = Vec::new();
    fn find<'a>(n: &'a Value, out: &mut Vec<&'a Value>) {
        if n["type"] == "Precompiled" {
            out.push(n);
        }
        for c in n["normalizers"].as_array().into_iter().flatten() {
            find(c, out);
        }
    }
    find(&json["normalizer"], &mut norms);
    let map = Charsmap::parse("bge-m3", norms[0]["precompiled_charsmap"].as_str().unwrap()).unwrap();
    let file = std::fs::read_to_string(root.join("tokenizer-texts.jsonl")).unwrap();
    let mut texts: Vec<String> =
        file.lines().map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned()).collect();
    texts.extend(
        [
            "a\r\nb",
            "\r\n",
            "x\r",
            "e\u{301}a",
            "ae\u{301}",
            "\u{600}1a",
            "a\u{600}1",
            "\u{0}\u{1}a\u{7f}",
            "ＡＢＣ abc",
            "ﬁ ligature ㎏",
            "a\u{200d}b",
            "🇫🇷x",
            "x🇫🇷",
            "e\u{301}\u{302}\u{303}\u{304}xyz",
        ]
        .map(String::from),
    );
    for t in &texts {
        let mut want = String::new();
        map.clusters(t, &mut want);
        assert_eq!(map.normalize(t), want, "{t:?}");
    }
}
