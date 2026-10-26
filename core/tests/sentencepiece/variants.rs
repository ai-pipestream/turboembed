/// The pieces goldens.py's USER_PIECES appends as user-defined: in none
/// of the models, overlapping their pieces and each other.
const USER_PIECES: [&str; 8] = ["hel", "llo", "a b", "xyz", "wor", "orl", "e\u{301}", "\u{2581}\u{2581}"];

/// goldens.py's VARIANT_TEXTS.
const VARIANT_TEXTS: [&str; 12] = [
    "hello world",
    "  hello  world  ",
    "Hello wor ld orl",
    "a b a  b ab",
    "xyz xyzxyz",
    "\u{2581}\u{2581} \u{2581}\u{2581}\u{2581}",
    "e\u{301}e\u{301} e\u{301}",
    "hel lo llo hello",
    " \u{1f642}",
    "\u{1f642} ",
    "\u{6f22} \u{6f22}",
    "\t \t",
];

/// A committed model with its whitespace switches set (None leaves one
/// as it is) and user-defined pieces appended, as goldens.py's
/// variant_model edits it: a later normalizer_spec or trainer_spec is
/// merged into the first, and pieces append in order.
fn variant_model(model: &str, adp: Option<bool>, rew: Option<bool>, suf: Option<bool>, user: &[&str]) -> Vec<u8> {
    let mut m = Message(std::fs::read(dir().join(format!("{model}.model"))).unwrap());
    for p in user {
        m.piece(p, 0.0, 4);
    }
    let mut normalizer = Message::default();
    if let Some(v) = adp {
        normalizer.uint(3, v as u64);
    }
    if let Some(v) = rew {
        normalizer.uint(4, v as u64);
    }
    m.bytes(3, &normalizer.0);
    if let Some(v) = suf {
        let mut trainer = Message::default();
        trainer.uint(24, v as u64);
        m.bytes(2, &trainer.0);
    }
    m.0
}

/// The whitespace switches in every combination and user-defined pieces
/// that overlap the model's own, on five committed models, against the
/// library: each variant's lines (normalized text, norm_to_orig, ids,
/// offsets, decoded text) hash to what goldens.py wrote.
#[test]
fn variants_match_the_library() {
    let file = std::fs::read_to_string(dir().join("variants.goldens.jsonl")).unwrap();
    let mut want = std::collections::HashMap::new();
    for line in file.lines().skip(1) {
        let v: Value = serde_json::from_str(line).unwrap();
        want.insert(
            v["variant"].as_str().unwrap().to_owned(),
            (v["texts"].as_u64().unwrap() as usize, v["sha256"].as_str().unwrap().to_owned()),
        );
    }
    let mut texts: Vec<String> = goldens("t5").lines.into_iter().map(|g| g.text).collect();
    texts.extend(VARIANT_TEXTS.map(str::to_owned));
    let models = ["t5", "unigram-identity", "unigram-bytes", "bpe-identity", "bpe-bytes"];
    let mut variants: Vec<(String, Vec<u8>)> = Vec::new();
    for model in models {
        for adp in [true, false] {
            for rew in [true, false] {
                for suf in [false, true] {
                    let name = format!("{model}-adp{}-rew{}-suf{}", adp as u8, rew as u8, suf as u8);
                    variants.push((name, variant_model(model, Some(adp), Some(rew), Some(suf), &[])));
                }
            }
        }
    }
    for model in models {
        variants.push((format!("{model}-user"), variant_model(model, None, None, None, &USER_PIECES)));
    }
    for model in ["t5", "unigram-bytes", "bpe-bytes"] {
        variants.push((
            format!("{model}-user-adp0-rew0"),
            variant_model(model, Some(false), Some(false), None, &USER_PIECES),
        ));
    }
    assert_eq!(variants.len(), want.len());
    let (mut scratch, mut normalized, mut pieces, mut ids) =
        (Scratch::default(), Normalized::default(), Vec::new(), Vec::new());
    let join = |v: &mut dyn Iterator<Item = String>| v.collect::<Vec<_>>().join(",");
    for (name, bytes) in &variants {
        let sp = SentencePiece::parse(bytes).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        let mut h = Sha256::new();
        for (i, t) in texts.iter().enumerate() {
            sp.normalize(t, &mut normalized);
            ids.clear();
            sp.encode_ids(t, &mut scratch, &mut ids);
            pieces.clear();
            sp.encode(t, &mut scratch, &mut pieces);
            let fields = [
                i.to_string(),
                String::from_utf8(normalized.text.clone()).unwrap(),
                join(&mut normalized.norm_to_orig.iter().map(u32::to_string)),
                join(&mut ids.iter().map(u32::to_string)),
                join(&mut pieces.iter().map(|p| format!("{}:{}", p.begin, p.end))),
                sp.decode(&ids).unwrap(),
            ];
            h.update(format!("{}\n", fields.join("\t")).as_bytes());
        }
        let (count, sha) = &want[name];
        assert_eq!(*count, texts.len(), "{name}");
        assert_eq!(&hex(&h.finish()), sha, "{name}");
    }
}
