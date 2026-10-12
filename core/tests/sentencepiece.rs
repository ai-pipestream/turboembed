//! The core's SentencePiece tokenizer against Google's sentencepiece
//! library: for every model of testdata/sentencepiece, the normalized
//! text, its norm_to_orig, the ids, the pieces' offsets and the decoded
//! text equal what the library gave (testdata/sentencepiece/goldens.py
//! wrote them, testdata/README.md says how).
//!
//! The published models that are not committed are checked when
//! TURBO_SENTENCEPIECE_MODELS names a directory holding them; each
//! must have the SHA-512 listed in FETCHED.

use std::path::{Path, PathBuf};

use serde_json::Value;
use sha2::{Digest, Sha256, Sha512};
use turbo::sentencepiece::{Algorithm, Normalized, Piece, Scratch, SentencePiece};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The models whose files are committed beside their goldens.
const COMMITTED: [&str; 7] =
    ["unigram-identity", "unigram-bytes", "unigram-suffix", "unigram-rules", "bpe-identity", "bpe-bytes", "t5"];

/// The published models checked when TURBO_SENTENCEPIECE_MODELS is set:
/// file, where it is published, SHA-512.
const FETCHED: [(&str, &str, &str); 3] = [
    (
        "mt5.model",
        "https://storage.googleapis.com/t5-data/vocabs/mc4.250000.100extra/sentencepiece.model",
        "e7176838d9109eb8493904825774935e0143d3b6dd5dd4178e2894d9ea0564ee7033e54c8e2aba96d8c3a32ba8433957001346f9cda00df964f28d724221cecc",
    ),
    (
        "umt5.model",
        "https://storage.googleapis.com/t5-data/vocabs/umt5.256000/sentencepiece.model",
        "7321fd207f1d9b711647f1a0c3f666f6aa807f50cb7519d3c6c4b1a9f7f3fa763713799c5414974355a875fbb00144f4cd19c2d501669412e7fc42dc49c8973c",
    ),
    (
        "llama2.model",
        "https://raw.githubusercontent.com/karpathy/llama2.c/350e04fe35433e6d2941dce5a1f53308f87058eb/tokenizer.model",
        "13c9ca779d3bdaba2f690287ba1f508d479a010c1f9122fd2bd523f310c16e2b877e93b42c18ab97b27f8fbfeb99e4f30fd999210e1b264d7d925530ff08255f",
    ),
];

/// Standard base64, for the character map in bge-m3's tokenizer file.
fn base64(s: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("not base64: {c}"),
    };
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in s.bytes().filter(|&c| c != b'=') {
        acc = (acc << 6) | value(c) as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    out
}

/// A protobuf message written field by field, for the models the tests
/// make.
#[derive(Default)]
struct Message(Vec<u8>);

impl Message {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn uint(&mut self, field: u32, v: u64) -> &mut Self {
        self.varint((field as u64) << 3);
        self.varint(v);
        self
    }

    fn bytes(&mut self, field: u32, b: &[u8]) -> &mut Self {
        self.varint(((field as u64) << 3) | 2);
        self.varint(b.len() as u64);
        self.0.extend_from_slice(b);
        self
    }

    fn float(&mut self, field: u32, f: f32) -> &mut Self {
        self.varint(((field as u64) << 3) | 5);
        self.0.extend_from_slice(&f.to_le_bytes());
        self
    }

    fn piece(&mut self, piece: &str, score: f32, kind: u64) -> &mut Self {
        let mut p = Message::default();
        p.bytes(1, piece.as_bytes()).float(2, score).uint(3, kind);
        self.bytes(1, &p.0)
    }
}

/// XLM-RoBERTa's model as goldens.py's xlmr_model rebuilds it from the
/// cut down tokenizer file of testdata/bge-m3.
fn xlmr_model() -> Vec<u8> {
    let json: Value = serde_json::from_slice(&std::fs::read(dir().join("../bge-m3/tokenizer.json")).unwrap()).unwrap();
    let mut m = Message::default();
    m.piece("<unk>", 0.0, 2).piece("<s>", 0.0, 3).piece("</s>", 0.0, 3);
    let vocab = json["model"]["vocab"].as_array().unwrap();
    let mut count = 3;
    for entry in &vocab[4..] {
        let piece = entry[0].as_str().unwrap();
        if piece != "<mask>" {
            m.piece(piece, entry[1].as_f64().unwrap() as f32, 1);
            count += 1;
        }
    }
    let mut trainer = Message::default();
    trainer.uint(3, 1).uint(4, count);
    m.bytes(2, &trainer.0);
    let charsmap = json["normalizer"]["normalizers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["type"] == "Precompiled")
        .unwrap()["precompiled_charsmap"]
        .as_str()
        .unwrap();
    let mut normalizer = Message::default();
    normalizer.bytes(1, b"nmt_nfkc").bytes(2, &base64(charsmap)).uint(3, 1).uint(4, 1).uint(5, 1);
    m.bytes(3, &normalizer.0);
    m.0
}

struct Golden {
    text: String,
    normalized: String,
    norm_to_orig: Vec<u32>,
    ids: Vec<u32>,
    offsets: Vec<[u32; 2]>,
    decoded: String,
}

struct Goldens {
    lines: Vec<Golden>,
    long_repeat: usize,
    long_count: usize,
    long_sha256: String,
}

fn goldens(name: &str) -> Goldens {
    let file = std::fs::read_to_string(dir().join(format!("{name}.goldens.jsonl"))).unwrap();
    let mut lines = Vec::new();
    let mut long = None;
    for line in file.lines().skip(1) {
        let v: Value = serde_json::from_str(line).unwrap();
        if let Some(l) = v.get("long") {
            long = Some((
                l["repeat"].as_u64().unwrap() as usize,
                l["count"].as_u64().unwrap() as usize,
                l["sha256"].as_str().unwrap().to_owned(),
            ));
            continue;
        }
        let nums = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect::<Vec<_>>();
        lines.push(Golden {
            text: v["text"].as_str().unwrap().to_owned(),
            normalized: v["normalized"].as_str().unwrap().to_owned(),
            norm_to_orig: nums("norm_to_orig"),
            ids: nums("ids"),
            offsets: v["offsets"]
                .as_array()
                .unwrap()
                .iter()
                .map(|o| [o[0].as_u64().unwrap() as u32, o[1].as_u64().unwrap() as u32])
                .collect(),
            decoded: v["decoded"].as_str().unwrap().to_owned(),
        });
    }
    let (long_repeat, long_count, long_sha256) = long.expect("a long line");
    Goldens { lines, long_repeat, long_count, long_sha256 }
}

/// Everything the library gave, for every text of a model's goldens.
/// Returns the number of texts and pieces compared.
fn check(name: &str, sp: &SentencePiece) -> (usize, usize) {
    let g = goldens(name);
    let mut scratch = Scratch::default();
    let mut normalized = Normalized::default();
    let mut pieces: Vec<Piece> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut back = Vec::new();
    let mut count = 0;
    for line in &g.lines {
        let t = &line.text;
        sp.normalize(t, &mut normalized);
        assert_eq!(String::from_utf8_lossy(&normalized.text), line.normalized, "{name}: normalized {t:?}");
        assert_eq!(normalized.norm_to_orig, line.norm_to_orig, "{name}: norm_to_orig {t:?}");
        // Bidirectional: each input offset maps to the first normalized
        // byte that came from it or after it, and back to there or after.
        normalized.orig_to_norm(t.len(), &mut back);
        for (o, &i) in back.iter().enumerate() {
            let i = i as usize;
            assert!(i == normalized.text.len() || normalized.norm_to_orig[i] as usize >= o, "{name}: {t:?} at {o}");
            assert!(i == 0 || (normalized.norm_to_orig[i - 1] as usize) < o, "{name}: {t:?} at {o}");
        }
        ids.clear();
        sp.encode_ids(t, &mut scratch, &mut ids);
        assert_eq!(ids, line.ids, "{name}: ids of {t:?}");
        pieces.clear();
        sp.encode(t, &mut scratch, &mut pieces);
        let got: Vec<[u32; 2]> = pieces.iter().map(|p| [p.begin, p.end]).collect();
        assert_eq!(pieces.iter().map(|p| p.id).collect::<Vec<_>>(), line.ids, "{name}: piece ids of {t:?}");
        assert_eq!(got, line.offsets, "{name}: offsets of {t:?}");
        assert_eq!(sp.decode(&ids).unwrap(), line.decoded, "{name}: decode of {t:?}");
        count += ids.len();
    }
    let joined = g.lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    let long = vec![joined.as_str(); g.long_repeat].join("\n");
    pieces.clear();
    sp.encode(&long, &mut scratch, &mut pieces);
    let mut h = Sha256::new();
    for p in &pieces {
        h.update(format!("{} {} {}\n", p.id, p.begin, p.end));
    }
    assert_eq!(pieces.len(), g.long_count, "{name}: pieces of the long text");
    assert_eq!(hex(&h.finalize()), g.long_sha256, "{name}: the long text's pieces");
    (g.lines.len() + 1, count + pieces.len())
}

fn committed(name: &str) -> SentencePiece {
    SentencePiece::parse(&std::fs::read(dir().join(format!("{name}.model"))).unwrap()).unwrap()
}

#[test]
fn committed_models_match_the_library() {
    for name in COMMITTED {
        let sp = committed(name);
        let (texts, pieces) = check(name, &sp);
        eprintln!("{name}: {texts} texts, {pieces} pieces");
    }
}

#[test]
fn xlmr_matches_the_library() {
    let sp = SentencePiece::parse(&xlmr_model()).unwrap();
    assert_eq!(sp.algorithm(), Algorithm::Unigram);
    let (texts, pieces) = check("xlmr", &sp);
    eprintln!("xlmr: {texts} texts, {pieces} pieces");
}

#[test]
fn fetched_models_match_the_library() {
    let Some(models) = std::env::var_os("TURBO_SENTENCEPIECE_MODELS") else {
        eprintln!("TURBO_SENTENCEPIECE_MODELS is not set: the published models are not checked");
        return;
    };
    for (file, url, sha512) in FETCHED {
        let path = Path::new(&models).join(file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (from {url})", path.display()));
        assert_eq!(hex(&Sha512::digest(&bytes)), sha512, "{file}");
        let sp = SentencePiece::parse(&bytes).unwrap();
        let (texts, pieces) = check(file.trim_end_matches(".model"), &sp);
        eprintln!("{file}: {texts} texts, {pieces} pieces");
    }
}

/// A model of `pieces` (text, type) with score 0, of `model_type`
/// (1 unigram, 2 BPE, 3 word, 4 char), the identity normalizer and
/// `samples` as its self-test data.
fn small_model(pieces: &[(&[u8], u64)], model_type: u64, samples: &[(&[u8], &[u8])]) -> Vec<u8> {
    let mut m = Message::default();
    for (i, &(p, kind)) in pieces.iter().enumerate() {
        let mut piece = Message::default();
        piece.bytes(1, p).float(2, -(i as f32)).uint(3, kind);
        m.bytes(1, &piece.0);
    }
    let mut trainer = Message::default();
    trainer.uint(3, model_type);
    m.bytes(2, &trainer.0);
    let mut normalizer = Message::default();
    normalizer.bytes(1, b"identity");
    m.bytes(3, &normalizer.0);
    let mut test = Message::default();
    for &(input, expected) in samples {
        let mut s = Message::default();
        s.bytes(1, input).bytes(2, expected);
        test.bytes(1, &s.0);
    }
    if !samples.is_empty() {
        m.bytes(4, &test.0);
    }
    m.0
}

/// The library runs a model's self-test samples on load and refuses the
/// model when one comes out otherwise; so does the core.
#[test]
fn a_model_whose_self_test_fails_is_refused() {
    let pieces: [(&[u8], u64); 4] = [(b"<unk>", 2), ("\u{2581}".as_bytes(), 1), (b"a", 1), ("\u{2581}a".as_bytes(), 1)];
    for model_type in [1, 2] {
        let good = small_model(&pieces, model_type, &[(b"a", "\u{2581}a".as_bytes())]);
        SentencePiece::parse(&good).unwrap();
        let bad = small_model(&pieces, model_type, &[(b"a", b"a")]);
        let e = SentencePiece::parse(&bad).err().unwrap();
        assert!(e.message.contains("self-test"), "{}", e.message);
    }
}

/// Differences from the library, each on purpose:
///
/// - A piece that is not UTF-8 is refused: the core's pieces are text.
///   The library takes any bytes.
/// - A self-test sample whose input is not UTF-8 counts as failed: the
///   core encodes text. The library encodes the bytes.
/// - Word and character models are refused: only unigram and BPE are
///   read.
/// - A BPE model with a one-character CONTROL piece gives that piece's
///   id for the character, as the library's encode into ids does, and an
///   empty offset where it stands; the library's encode with offsets
///   fails on such a text.
#[test]
fn the_differences_from_the_library() {
    let e = SentencePiece::parse(&small_model(&[(b"<unk>", 2), (b"\xff", 1)], 1, &[])).err().unwrap();
    assert!(e.message.contains("not UTF-8"), "{}", e.message);

    let pieces: [(&[u8], u64); 2] = [(b"<unk>", 2), (b"a", 1)];
    let e = SentencePiece::parse(&small_model(&pieces, 1, &[(b"\xff", b"<unk>")])).err().unwrap();
    assert!(e.message.contains("self-test"), "{}", e.message);

    for model_type in [3, 4] {
        let e = SentencePiece::parse(&small_model(&pieces, model_type, &[])).err().unwrap();
        assert!(e.message.contains("only unigram and BPE"), "{}", e.message);
    }

    let pieces: [(&[u8], u64); 4] = [(b"<unk>", 2), (b"x", 3), ("\u{2581}".as_bytes(), 1), (b"a", 1)];
    let sp = SentencePiece::parse(&small_model(&pieces, 2, &[])).unwrap();
    let (mut scratch, mut out) = (Scratch::default(), Vec::new());
    sp.encode("ax", &mut scratch, &mut out);
    assert_eq!(
        out,
        [Piece { id: 2, begin: 0, end: 0 }, Piece { id: 3, begin: 0, end: 1 }, Piece { id: 1, begin: 1, end: 1 }],
    );
}

/// Encoding into a scratch that has grown allocates nothing more: the
/// buffers are kept between calls.
#[test]
fn a_grown_scratch_is_reused() {
    let sp = committed("t5");
    let (mut scratch, mut out) = (Scratch::default(), Vec::new());
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(20);
    sp.encode(&text, &mut scratch, &mut out);
    let first = out.clone();
    out.clear();
    let cap = out.capacity();
    sp.encode(&text, &mut scratch, &mut out);
    assert_eq!(out, first);
    assert_eq!(out.capacity(), cap);
}
