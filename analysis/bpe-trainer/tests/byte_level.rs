//! The byte-level trainer against upstream `tokenizers`' `BpeTrainer`:
//! the committed tiny-bpe fixture, which upstream trained, byte for byte,
//! and upstream run here on other corpora and settings, file for file.
//! Then the trained file through the core's tokenizer: it loads, its ids
//! are upstream's for the same file, decoding gives the text back, and
//! each word of the corpus encodes to the pieces training left it as.

mod common;

use common::*;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{Bpe, Manifest, PromptRole, SpecialRole, SpecialToken as Special, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_bpe_trainer::{ByteLevelBpe, ByteLevelConfig, PostProcessor, SpecialToken};

fn roberta_specials() -> Vec<SpecialToken> {
    let mut s: Vec<SpecialToken> = ["<s>", "<pad>", "</s>", "<unk>"].map(SpecialToken::new).to_vec();
    s.push(SpecialToken { content: "<mask>".into(), lstrip: true });
    s
}

fn config(vocab_size: usize) -> ByteLevelConfig {
    ByteLevelConfig {
        vocab_size,
        min_frequency: 0,
        special_tokens: roberta_specials(),
        add_prefix_space: false,
        post_processor: PostProcessor::None,
    }
}

/// Upstream's trainer on `texts` with `c`, as testdata/tiny-bpe was made.
fn upstream(texts: &[String], c: &ByteLevelConfig) -> tokenizers::Tokenizer {
    use tokenizers::models::TrainerWrapper;
    use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::processors::roberta::RobertaProcessing;
    use tokenizers::{AddedToken, Tokenizer};
    let mut tok = Tokenizer::new(BPE::default());
    tok.with_pre_tokenizer(Some(ByteLevel::new(c.add_prefix_space, true, true)));
    tok.with_decoder(Some(ByteLevel::default()));
    if let PostProcessor::Roberta { cls, sep } = &c.post_processor {
        let id = |t: &str| c.special_tokens.iter().position(|s| s.content == t).unwrap() as u32;
        tok.with_post_processor(Some(
            RobertaProcessing::new((sep.clone(), id(sep)), (cls.clone(), id(cls)))
                .trim_offsets(true)
                .add_prefix_space(c.add_prefix_space),
        ));
    }
    let specials =
        c.special_tokens.iter().map(|s| AddedToken::from(s.content.clone(), true).lstrip(s.lstrip)).collect();
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .show_progress(false)
        .vocab_size(c.vocab_size)
        .min_frequency(c.min_frequency)
        .special_tokens(specials)
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .build()
        .into();
    tok.train(&mut trainer, texts.iter().cloned()).unwrap();
    tok
}

/// The tiny-bpe fixture is upstream's trainer on the core tests' corpus
/// with RoBERTa's settings; the trainer writes the same 28901 bytes.
#[test]
fn the_tiny_bpe_fixture_is_trained_again_byte_for_byte() {
    let mut c = config(1200);
    c.min_frequency = 1;
    c.post_processor = PostProcessor::Roberta { cls: "<s>".into(), sep: "</s>".into() };
    let corpus = jsonl_strings(&testdata().join("tiny-bpe/corpus.jsonl"));
    let m = ByteLevelBpe::train(corpus.iter().map(String::as_str), c).unwrap();
    let want = std::fs::read_to_string(testdata().join("tiny-bpe/tokenizer.json")).unwrap();
    assert_eq!(m.merges().len(), 939);
    assert_eq!(m.vocab().len(), 1200);
    assert!(m.to_tokenizer_json() == want, "the written file differs from testdata/tiny-bpe/tokenizer.json");
}

/// The corpora the live comparisons run on.
fn corpora() -> Vec<(&'static str, Vec<String>)> {
    let analysis = std::fs::read_to_string(repo().join("docs/analysis.md")).unwrap();
    vec![
        ("tokenizer-texts", tokenizer_texts()),
        ("bpe-inputs", bpe_inputs().into_iter().map(|(_, t)| t).collect()),
        ("docs-analysis", analysis.lines().map(str::to_owned).collect()),
    ]
}

/// The whole file, as upstream writes it, for every corpus and setting:
/// the vocabulary in id order and the merges in rank order.
#[test]
fn every_setting_writes_upstreams_file() {
    let mut runs = 0;
    for (name, texts) in corpora() {
        for vocab_size in [261, 300, 1000, 4000] {
            for (min_frequency, add_prefix_space, specials) in
                [(0, false, true), (3, false, true), (0, true, false), (2, true, true)]
            {
                let mut c = config(vocab_size);
                c.min_frequency = min_frequency;
                c.add_prefix_space = add_prefix_space;
                if !specials {
                    c.special_tokens.clear();
                }
                let up = upstream(&texts, &c).to_string(false).unwrap();
                let m = ByteLevelBpe::train(texts.iter().map(String::as_str), c.clone()).unwrap();
                let ours = m.to_tokenizer_json();
                if ours != up {
                    let a: serde_json::Value = serde_json::from_str(&ours).unwrap();
                    let b: serde_json::Value = serde_json::from_str(&up).unwrap();
                    let merges = |v: &serde_json::Value| v["model"]["merges"].as_array().unwrap().clone();
                    let (ma, mb) = (merges(&a), merges(&b));
                    let at = ma.iter().zip(&mb).position(|(x, y)| x != y);
                    panic!(
                        "{name} with {c:?}: ours has {} merges, upstream {}; first difference at {at:?}: {:?} {:?}",
                        ma.len(),
                        mb.len(),
                        at.map(|i| &ma[i]),
                        at.map(|i| &mb[i]),
                    );
                }
                runs += 1;
            }
        }
    }
    assert_eq!(runs, 48);
}

/// Training twice gives the same model, and so does the corpus in
/// another order: ties are broken by the pieces, not by the order the
/// words were seen in.
#[test]
fn training_is_deterministic_and_ignores_the_order_of_the_texts() {
    let texts = tokenizer_texts();
    let a = ByteLevelBpe::train(texts.iter().map(String::as_str), config(2000)).unwrap();
    let b = ByteLevelBpe::train(texts.iter().map(String::as_str), config(2000)).unwrap();
    let c = ByteLevelBpe::train(texts.iter().rev().map(String::as_str), config(2000)).unwrap();
    assert_eq!(a.to_tokenizer_json(), b.to_tokenizer_json());
    assert_eq!(a.to_tokenizer_json(), c.to_tokenizer_json());
}

/// Small random corpora, many with one or two letters repeated, so
/// that merges overlap and pairs tie: upstream writes the same file.
#[test]
fn random_corpora_train_as_upstream_trains() {
    let mut state: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    for round in 0..1000 {
        let letters = 1 + next(3);
        let mut texts = vec![];
        for _ in 0..1 + next(6) {
            let word: String = (0..1 + next(18)).map(|_| char::from(b'a' + next(letters) as u8)).collect();
            for _ in 0..1 + next(4) {
                texts.push(word.clone());
            }
        }
        let c = ByteLevelConfig { special_tokens: vec![], ..config(300) };
        let m = ByteLevelBpe::train(texts.iter().map(String::as_str), c.clone()).unwrap();
        assert!(m.to_tokenizer_json() == upstream(&texts, &c).to_string(false).unwrap(), "round {round}: {texts:?}");
    }
}

/// A bundle with the trained tokenizer.json, on the small BERT's
/// manifest, RoBERTa's special tokens and template.
fn bundle(name: &str, json: &str, add_prefix_space: bool) -> (TempDir, Tokenizer) {
    let dir = TempDir::new(name);
    std::fs::write(dir.0.join("tokenizer.json"), json).unwrap();
    let mut m = Manifest::parse(&std::fs::read(testdata().join("tiny-bert-bundle/manifest.json")).unwrap()).unwrap();
    let entry = m.files.iter_mut().find(|f| f.path == "tokenizer.json").unwrap();
    entry.size = json.len() as u64;
    entry.sha256 = sha256_hex(json.as_bytes());
    m.model.id = format!("turbo-testdata/{name}");
    m.tokenizer.normalizer = None;
    m.tokenizer.wordpiece = None;
    m.tokenizer.bpe = Some(Bpe { add_prefix_space, ignore_merges: false, pattern: Default::default(), nfc: false });
    let special = |role, content: &str, id, lstrip| Special {
        role,
        content: content.into(),
        id,
        lstrip,
        normalized: false,
        single_word: false,
        rstrip: false,
    };
    m.tokenizer.special_tokens = vec![
        special(SpecialRole::Bos, "<s>", 0, false),
        special(SpecialRole::Pad, "<pad>", 1, false),
        special(SpecialRole::Eos, "</s>", 2, false),
        special(SpecialRole::Unk, "<unk>", 3, false),
        special(SpecialRole::Mask, "<mask>", 4, true),
    ];
    m.tokenizer.template = vec!["<s>".into(), "$TEXT".into(), "</s>".into()];
    std::fs::write(dir.0.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let tok = Tokenizer::unchecked(&Bundle::open(&dir.0).unwrap()).unwrap();
    (dir, tok)
}

fn bare() -> Encode {
    Encode { add_special_tokens: false, truncation: Truncation::None, max_tokens: u32::MAX, prompt: PromptRole::None }
}

/// The trained file loads in the core; every text encodes to upstream's
/// ids for the same file and decodes as upstream decodes it, which is the
/// text itself unless a space was put in front or `<mask>` took the
/// whitespace before it; and each word of the corpus, encoded alone, is
/// the pieces training left it as.
#[test]
fn the_trained_file_loads_in_the_core_and_round_trips() {
    for add_prefix_space in [false, true] {
        let texts = tokenizer_texts();
        let mut c = config(1500);
        c.add_prefix_space = add_prefix_space;
        let m = ByteLevelBpe::train(texts.iter().map(String::as_str), c).unwrap();
        let json = m.to_tokenizer_json();
        let (_dir, tok) = bundle(&format!("trained-{add_prefix_space}"), &json, add_prefix_space);
        assert_eq!(tok.kind_name(), "bpe");
        assert_eq!(tok.vocab_size(), 1500);
        let up = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        let mut all = texts.clone();
        all.extend(bpe_inputs().into_iter().map(|(_, t)| t));
        all.extend(["", " ", "unseen: qzxj \u{2713} \u{1d518} \u{0}", "<mask> and <s> in the text"].map(str::to_owned));
        for text in &all {
            let ids = tok.encode(text, bare()).unwrap();
            let up_ids: Vec<u32> = up.encode(text.as_str(), false).unwrap().get_ids().to_vec();
            assert_eq!(ids, up_ids.iter().map(|&i| i as i32).collect::<Vec<_>>(), "ids of {text:?}");
            let back = tok.decode(&ids, false).unwrap();
            assert_eq!(back, up.decode(&up_ids, false).unwrap(), "decode of {text:?}");
            if !add_prefix_space && !text.contains("<mask>") {
                assert_eq!(&back, text);
            }
        }

        let pieces: Vec<&str> = m.vocab().iter().map(String::as_str).collect();
        let mut checked = 0;
        for (word, _) in turbo_bpe_trainer::byte_level::words(texts.iter().map(String::as_str), add_prefix_space) {
            let byte = |c: char| turbo::bpe::byte_chars().iter().position(|&b| b == c).unwrap() as u8;
            let bytes: Vec<u8> = word.chars().map(byte).collect();
            let text = String::from_utf8(bytes).expect("a word is whole characters");
            if add_prefix_space && !text.starts_with(' ') {
                // Alone, it would get a space in front.
                continue;
            }
            let ids = tok.encode(&text, bare()).unwrap();
            let got: Vec<&str> = ids.iter().map(|&i| pieces[i as usize]).collect();
            assert_eq!(got, trained_pieces(&word, m.merges()), "word {text:?}");
            checked += 1;
        }
        assert!(checked > 300, "{checked} words");
    }
}

/// The pieces training leaves `word` as: each merge in rank order, made
/// left to right everywhere in the word.
fn trained_pieces(word: &str, merges: &[turbo_bpe_trainer::Merge]) -> Vec<String> {
    let mut s: Vec<String> = word.chars().map(String::from).collect();
    for m in merges {
        let mut i = 0;
        while i + 1 < s.len() {
            if s[i] == m.left && s[i + 1] == m.right {
                let right = s.remove(i + 1);
                s[i].push_str(&right);
            }
            i += 1;
        }
    }
    s
}
