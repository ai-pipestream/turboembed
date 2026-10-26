use super::*;
use std::path::PathBuf;

pub fn tokenizer_json() -> PathBuf {
    testdata().join("tiny-bpe/tokenizer.json")
}

/// What the tokenizer was trained on: the texts the tests encode, so
/// that their words merge deep, twice over.
fn corpus() -> Vec<String> {
    let mut t = unigram::texts();
    t.push(long_text());
    t.extend(t.clone());
    t
}

/// testdata/tiny-bpe/corpus.jsonl: corpus(), one JSON string per line,
/// so that a trainer outside these tests can train on what upstream
/// trained on.
fn corpus_file() -> PathBuf {
    testdata().join("tiny-bpe/corpus.jsonl")
}

#[test]
fn the_corpus_file_is_the_corpus() {
    let file = std::fs::read_to_string(corpus_file()).unwrap();
    let read: Vec<String> = file.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(read, corpus());
}

/// Writes testdata/tiny-bpe/corpus.jsonl.
#[test]
#[ignore = "writes the test data; run by hand"]
fn write_corpus_file() {
    let lines: Vec<String> = corpus().iter().map(|t| serde_json::to_string(t).unwrap() + "\n").collect();
    std::fs::write(corpus_file(), lines.concat()).unwrap();
}

/// Writes testdata/tiny-bpe/tokenizer.json: upstream's BPE trainer on
/// corpus() with GPT-2's byte-level alphabet, 1200 pieces, RoBERTa's
/// special tokens, pre-tokenizer, post-processor and decoder.
#[test]
#[ignore = "writes the test data; run by hand"]
fn make_tiny_bpe() {
    use tokenizers::models::TrainerWrapper;
    use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::processors::roberta::RobertaProcessing;
    use tokenizers::{AddedToken, Tokenizer};
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(ByteLevel::new(false, true, true)));
    tok.with_decoder(Some(ByteLevel::default()));
    tok.with_post_processor(Some(
        RobertaProcessing::new(("</s>".into(), 2), ("<s>".into(), 0)).trim_offsets(true).add_prefix_space(false),
    ));
    let mut specials = ["<s>", "<pad>", "</s>", "<unk>"].map(|s| AddedToken::from(s, true)).to_vec();
    specials.push(AddedToken::from("<mask>", true).lstrip(true));
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(1200)
        .min_frequency(1)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();
    tok.train(&mut trainer, corpus().into_iter()).unwrap();
    std::fs::create_dir_all(tokenizer_json().parent().unwrap()).unwrap();
    tok.save(tokenizer_json(), false).unwrap();
}
/// The MiniLM fixture's manifest with this tokenizer: RoBERTa's
/// special tokens and template.
pub fn manifest_bpe() -> serde_json::Value {
    let mut m = manifest();
    m["tokenizer"] = json!({
        "file": "tokenizer.json",
        "bpe": { "add_prefix_space": false },
        "special_tokens": [
            { "role": "SPECIAL_BOS",  "content": "<s>",    "id": 0 },
            { "role": "SPECIAL_PAD",  "content": "<pad>",  "id": 1 },
            { "role": "SPECIAL_EOS",  "content": "</s>",   "id": 2 },
            { "role": "SPECIAL_UNK",  "content": "<unk>",  "id": 3 },
            { "role": "SPECIAL_MASK", "content": "<mask>", "id": 4, "lstrip": true }
        ],
        "template": ["<s>", "$TEXT", "</s>"],
        "truncation": "TRUNCATE_RIGHT"
    });
    m
}

/// The test texts, and texts the tokenizer was not trained on: its
/// characters drawn at random, so merges stop part way, and the
/// corners of GPT-2's pattern.
pub fn texts() -> Vec<String> {
    let mut t = unigram::texts();
    let pool: Vec<char> = corpus().concat().chars().collect();
    let mut x: u64 = 0x5851_f42d_4c95_7f2d;
    for _ in 0..400 {
        let mut s = String::new();
        for _ in 0..(x >> 58) + 1 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.push(pool[(x % pool.len() as u64) as usize]);
        }
        t.push(s);
    }
    t.extend(
        [
            "it's they're we've I'm you'll he'd don't 'S 'RE 'tis '' ' s",
            "a  b   c\t\td \n\ne \u{3000}f \u{a0}g",
            "trailing   ",
            "   ",
            " x",
            "x ",
            "1234567890 12,345.67 ①②③ Ⅻ ½ ٣٤٥ ௧௨",
            "!!! ??? ... --- ''' \"\"\" (((",
            "a!b?c.d,e;f:g",
            "e\u{301}\u{302} \u{0} \u{1} \u{7f} \u{fffd} \u{feff}",
            "<mask> <s>x</s> <pad><unk>",
            "before   <mask>   after",
        ]
        .map(str::to_owned),
    );
    t
}

fn fixture(name: &str, m: serde_json::Value, path: &std::path::Path) -> Fixture {
    Fixture::with_tokenizer(&format!("bpe-{name}"), m, path)
}

/// The core against upstream on the tokenizer file at `path`: ids,
/// spans and decode.
fn check(name: &str, m: serde_json::Value, path: &std::path::Path) {
    let prefixed = m["tokenizer"]["bpe"]["add_prefix_space"] == true;
    let specials: Vec<i32> =
        m["tokenizer"]["special_tokens"].as_array().unwrap().iter().map(|s| s["id"].as_i64().unwrap() as i32).collect();
    let f = fixture(name, m, path);
    let tok = core_tokenizer(&f);
    let mut up = upstream_at(path);
    up.with_truncation(None).unwrap();
    for text in texts() {
        let (ours, theirs) = both_spans(&tok, &up, &text);
        spans_near_upstream(&text, &ours, &theirs, prefixed, &specials);
        check_well_formed(&text, &ours);
        decode_matches_upstream(&tok, &up, &text);
    }
}

/// The core's spans are upstream's offsets, but where upstream's are
/// wrong: a token of whitespace alone is that whitespace, not the
/// empty offset at its end that upstream's trim leaves; and the first
/// token of a text that a space was put in front of starts at the
/// text's start, where upstream's trim counts the space put in front
/// as one of the text's and starts it a byte late (inside a character
/// that is more than one).
fn spans_near_upstream(
    text: &str,
    ours: &[(i32, [u32; 2])],
    theirs: &[(i32, [u32; 2])],
    prefixed: bool,
    specials: &[i32],
) {
    assert_eq!(ours.len(), theirs.len(), "{text:?}");
    let first = ours.iter().position(|o| o.1 != [0, 0]);
    for (i, (o, t)) in ours.iter().zip(theirs).enumerate() {
        assert_eq!(o.0, t.0, "{text:?}");
        let (a, b) = (o.1, t.1);
        let blank = text[a[0] as usize..a[1] as usize].trim().is_empty();
        // The first token of a piece of text a space was put in front
        // of (the text, or what follows a special token in it); the
        // space alone comes from the character it is put before.
        let starts_piece = Some(i) == first || (i > 0 && specials.contains(&ours[i - 1].0));
        let lead = text[a[0] as usize..].chars().next().map_or(0, char::len_utf8) as u32;
        let late_start = prefixed
            && starts_piece
            && !text[..a[0] as usize].ends_with(' ')
            && (b == [a[0] + 1, a[1]] || (a == [a[0], a[0] + lead] && b[0] > a[0] && b[1] <= a[1]));
        assert!(
            a == b || (blank && b[0] == b[1] && b[1] == a[1]) || late_start,
            "{text:?}: token {i} ({}): ours {a:?} {:?}, upstream {b:?} {:?}",
            o.0,
            &text[a[0] as usize..a[1] as usize],
            text.get(b[0] as usize..b[1] as usize)
        );
    }
}

#[test]
fn matches_upstream() {
    check("plain", manifest_bpe(), &tokenizer_json());
}

/// The same with a space put in front, merges written as "left right",
/// whole words taken as pieces, no model type named, special tokens
/// marked normalized, and raw and normalized ones overlapping.
#[test]
fn matches_upstream_with_each_option() {
    let base: serde_json::Value = serde_json::from_slice(&std::fs::read(tokenizer_json()).unwrap()).unwrap();
    for variant in ["prefix", "legacy", "ignore_merges", "untyped", "normalized", "mixed"] {
        let mut json = base.clone();
        let mut m = manifest_bpe();
        match variant {
            "prefix" => {
                json["pre_tokenizer"]["add_prefix_space"] = json!(true);
                m["tokenizer"]["bpe"]["add_prefix_space"] = json!(true);
            }
            "legacy" => {
                let merges = json["model"]["merges"].as_array().unwrap();
                let legacy: Vec<String> =
                    merges.iter().map(|p| format!("{} {}", p[0].as_str().unwrap(), p[1].as_str().unwrap())).collect();
                json["model"]["merges"] = json!(legacy);
            }
            // GPT-2's and RoBERTa's own files name no model type, and
            // upstream takes a model with merges for BPE.
            "untyped" => {
                json["model"].as_object_mut().unwrap().remove("type");
            }
            // A raw token inside a normalized one is matched first,
            // as upstream matches raw added tokens before normalized
            // ones: "</s>" is "<" and "/" around "s>".
            "mixed" => mixed(&mut json, &mut m),
            // Theirs also mark the special tokens normalized, which
            // with no normalizer matches them in the text as it is.
            "normalized" => {
                for a in json["added_tokens"].as_array_mut().unwrap() {
                    a["normalized"] = json!(true);
                }
                for s in m["tokenizer"]["special_tokens"].as_array_mut().unwrap() {
                    s["normalized"] = json!(true);
                }
            }
            _ => {
                json["model"]["ignore_merges"] = json!(true);
                m["tokenizer"]["bpe"]["ignore_merges"] = json!(true);
            }
        }
        let path = std::env::temp_dir().join(format!("turbo-bpe-{variant}-{}.json", std::process::id()));
        std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        check(variant, m, &path);
        std::fs::remove_file(path).unwrap();
    }
}

/// "<s>" (id 0) renamed "s>" and kept raw, and "</s>" (id 2) marked
/// normalized, so the raw token sits inside the normalized one.
fn mixed(json: &mut serde_json::Value, m: &mut serde_json::Value) {
    let v = json["model"]["vocab"].as_object_mut().unwrap();
    let id = v.remove("<s>").unwrap();
    v.insert("s>".into(), id);
    json["added_tokens"][0]["content"] = json!("s>");
    json["added_tokens"][2]["normalized"] = json!(true);
    let t = &mut m["tokenizer"];
    t["special_tokens"][0]["content"] = json!("s>");
    t["special_tokens"][2]["normalized"] = json!(true);
    t["template"] = json!(["s>", "$TEXT", "</s>"]);
}

/// The text "</s>" alone: the raw "s>" is taken first and "</" is
/// encoded, then the template's own tokens go around it, as upstream
/// gives it; the normalized "</s>" whole would be [0, 2, 2].
#[test]
fn a_raw_token_inside_a_normalized_one_is_taken_first() {
    let mut json: serde_json::Value = serde_json::from_slice(&std::fs::read(tokenizer_json()).unwrap()).unwrap();
    let mut m = manifest_bpe();
    mixed(&mut json, &mut m);
    let path = std::env::temp_dir().join(format!("turbo-bpe-raw-first-{}.json", std::process::id()));
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let tok = core_tokenizer(&fixture("raw-first", m, &path));
    let mut up = upstream_at(&path);
    up.with_truncation(None).unwrap();
    let (ours, theirs) = both_spans(&tok, &up, "</s>");
    std::fs::remove_file(path).unwrap();
    let ids = |t: &[(i32, [u32; 2])]| t.iter().map(|x| x.0).collect::<Vec<_>>();
    assert_eq!(ids(&theirs), [0, 1137, 0, 2]);
    assert_eq!(ids(&ours), ids(&theirs));
}

/// GPT-2's and RoBERTa's own tokenizer files, from the directory
/// TURBO_BPE_FILES names: gpt2/tokenizer.json and
/// roberta-base/tokenizer.json.
#[test]
#[ignore = "needs TURBO_BPE_FILES"]
fn matches_upstream_on_gpt2_and_roberta() {
    let dir = std::path::PathBuf::from(std::env::var("TURBO_BPE_FILES").expect("TURBO_BPE_FILES"));
    let mut gpt2 = manifest_bpe();
    gpt2["tokenizer"]["special_tokens"] =
        json!([{ "role": "SPECIAL_UNK", "content": "<|endoftext|>", "id": 50256, "normalized": true }]);
    gpt2["tokenizer"]["template"] = json!(["$TEXT"]);
    gpt2["architecture"]["vocab_size"] = json!(50257);
    check("gpt2", gpt2, &dir.join("gpt2/tokenizer.json"));
    let mut roberta = manifest_bpe();
    roberta["tokenizer"]["special_tokens"][4]["id"] = json!(50264);
    for s in roberta["tokenizer"]["special_tokens"].as_array_mut().unwrap() {
        s["normalized"] = json!(true);
    }
    roberta["architecture"]["vocab_size"] = json!(50265);
    check("roberta", roberta, &dir.join("roberta-base/tokenizer.json"));
}

/// A file that is not what the manifest says, or not what the core
/// runs, is refused when the tokenizer is made.
#[test]
fn the_manifest_must_agree_with_the_file() {
    let base: serde_json::Value = serde_json::from_slice(&std::fs::read(tokenizer_json()).unwrap()).unwrap();
    type Edit = fn(&mut serde_json::Value);
    let cases: [(&str, Edit); 5] = [
        ("add_prefix_space", |j| j["pre_tokenizer"]["add_prefix_space"] = json!(true)),
        ("ByteLevel", |j| j["pre_tokenizer"]["use_regex"] = json!(false)),
        ("normalizer", |j| j["normalizer"] = json!({ "type": "NFC" })),
        ("dropout", |j| j["model"]["dropout"] = json!(0.1)),
        ("no piece", |j| j["model"]["merges"].as_array_mut().unwrap().push(json!(["zzzz", "qqqq"]))),
    ];
    for (want, edit) in cases {
        let mut json = base.clone();
        edit(&mut json);
        let mut f = fixture("bad", manifest_bpe(), &tokenizer_json());
        std::fs::write(f.dir.join("tokenizer.json"), serde_json::to_vec(&json).unwrap()).unwrap();
        f.list("tokenizer.json");
        let e = f.open().unwrap_err();
        assert!(e.message.contains(want), "{want}: {e:?}");
    }
}

#[test]
fn info_names_the_kind() {
    let f = fixture("info", manifest_bpe(), &tokenizer_json());
    let info = f.open().unwrap().info();
    assert_eq!(field(&info.kind), "bpe");
    assert_eq!(info.vocab_size, 1200);
}
