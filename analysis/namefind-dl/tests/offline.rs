//! The finder end to end on the small BERT in testdata, given a token
//! classification head whose weights are zero and whose bias favours
//! `B-PER`: every piece is labelled `B-PER` with the same probability, so
//! every piece is its own entity. What the head decides is then known
//! without a reference, and the tests check what the finder does around
//! it: chunks and their overlap, rows split at the row length, special
//! tokens, batches, sentences and the token ranges.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::{Analyzer, Bound, Document, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_embed::EncoderOptions;
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_namefind_dl::decode::label_probability;
use turbo_namefind_dl::{NameFinderDl, NameFinderDlOptions};

const BIAS: [f32; 3] = [0.0, 2.0, 0.0];

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

/// The tiny bundle with a `[O, B-PER, I-PER]` token head appended.
fn bundle() -> PathBuf {
    static DIR: OnceLock<PathBuf> = OnceLock::new();
    DIR.get_or_init(write_bundle).clone()
}

fn write_bundle() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-namefind-dl-{}", std::process::id()));
    let tmp = dir.clone();
    let _ = fs::remove_dir_all(&tmp);
    for f in ["tokenizer.json", "reference/reference.safetensors"] {
        fs::create_dir_all(tmp.join(f).parent().unwrap()).unwrap();
        fs::copy(tiny().join(f), tmp.join(f)).unwrap();
    }
    let src = fs::read(tiny().join("weights/model.safetensors")).unwrap();
    let n = u64::from_le_bytes(src[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> = serde_json::from_slice(&src[8..8 + n]).unwrap();
    header.remove("__metadata__");
    let mut data = src[8 + n..].to_vec();
    let mut add = |name: &str, shape: &[usize], values: &[f32]| {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(name.into(), json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}));
    };
    add("classifier.weight", &[3, 32], &[0.0; 96]);
    add("classifier.bias", &[3], &BIAS);
    let mut head = serde_json::to_vec(&header).unwrap();
    head.resize(head.len().next_multiple_of(8), b' ');
    let mut weights = (head.len() as u64).to_le_bytes().to_vec();
    weights.extend(head);
    weights.extend(data);
    fs::create_dir_all(tmp.join("weights")).unwrap();
    fs::write(tmp.join("weights/model.safetensors"), &weights).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(tiny().join("manifest.json")).unwrap()).unwrap();
    m["classifier"] = json!({"kind": "CLASSIFIER_TOKEN", "labels": ["O", "B-PER", "I-PER"]});
    let names = m["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    names.insert("classifier_weight".into(), json!("classifier.weight"));
    names.insert("classifier_bias".into(), json!("classifier.bias"));
    let files = m["files"].as_array_mut().unwrap();
    let w = files.iter_mut().find(|f| f["path"] == "weights/model.safetensors").unwrap();
    w["size"] = json!(weights.len());
    w["sha256"] = json!(sha256_hex(&weights));
    fs::write(tmp.join("manifest.json"), serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    dir
}

/// Each piece's bytes in `text`, the tokenizer read once over the text.
fn pieces(text: &str) -> Vec<(usize, usize)> {
    let t = Tokenizer::load(&Bundle::open(&bundle()).unwrap()).unwrap();
    let bare = Encode {
        add_special_tokens: false,
        truncation: Truncation::None,
        max_tokens: u32::MAX,
        prompt: PromptRole::None,
    };
    let (mut ids, mut spans) = (Vec::new(), Vec::new());
    t.encode_spans(text, bare, &mut ids, &mut spans).unwrap();
    spans.iter().filter(|s| s[0] < s[1]).map(|s| (s[0] as usize, s[1] as usize)).collect()
}

fn finder(split: usize, overlap: usize, max_seq: u32, max_batch: u32) -> NameFinderDl {
    let encoder = EncoderOptions { max_seq, max_batch, ..EncoderOptions::default() };
    NameFinderDl::open(&bundle(), &NameFinderDlOptions { encoder, split, overlap }).unwrap()
}

const TEXT: &str = "Hello world, this is an unbelievable test of the finder's chunks.";

#[test]
fn every_piece_once_however_the_text_is_cut() {
    let want = pieces(TEXT);
    let p = label_probability(&BIAS, 1);
    // Whole chunks in one row; chunks of 3 words sharing 1; rows of 2
    // pieces; and one row per run.
    for (split, overlap, max_seq, max_batch) in [(250, 50, 0, 0), (3, 1, 0, 0), (3, 2, 4, 0), (4, 1, 5, 1)] {
        let f = finder(split, overlap, max_seq, max_batch);
        let got = f.find(&[(TEXT, vec![(0, TEXT.len())]), ("", vec![(0, 0)]), (TEXT, vec![(6, 11)])]).unwrap();
        let spans: Vec<(usize, usize)> = got[0].iter().map(|c| (c.start, c.end)).collect();
        assert_eq!(spans, want, "split {split}, overlap {overlap}, max_seq {max_seq}");
        assert!(got[0].iter().all(|c| c.kind == 0 && (c.probability - p).abs() < 1e-12));
        assert!(got[1].is_empty());
        assert_eq!(got[2].iter().map(|c| (c.start, c.end)).collect::<Vec<_>>(), [(6, 11)]);
    }
    assert_eq!(finder(250, 50, 0, 0).kinds().label(turbo_analysis_layers::Tag(0)), "PER");
}

#[test]
fn options_that_cannot_cut_are_errors() {
    let open = |split, overlap, max_seq| {
        let encoder = EncoderOptions { max_seq, ..EncoderOptions::default() };
        NameFinderDl::open(&bundle(), &NameFinderDlOptions { encoder, split, overlap })
    };
    assert!(open(5, 5, 0).is_err());
    assert!(open(0, 0, 0).is_err());
    // A row of only `[CLS] [SEP]` has no room for a piece.
    assert!(open(5, 1, 2).is_err());
    let f = finder(250, 50, 0, 0);
    assert!(f.find(&[(TEXT, vec![(0, TEXT.len() + 1)])]).is_err());
    assert!(f.find(&[("é", vec![(0, 1)])]).is_err());
}

#[test]
fn as_a_step_it_reads_sentences_and_maps_tokens() {
    let text = "Ann met Bob . Carl stayed";
    let mut doc = Document::new(text);
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut start = 0;
    for word in text.split(' ') {
        tokens.push(Span::new(start as u32, (start + word.len()) as u32));
        start += word.len() + 1;
    }
    let mut sentences = Spans::new(ViewId::SOURCE);
    sentences.push(Span::new(0, 12));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let analyzer =
        Analyzer::builder().given(Key::<Tokens>::DEFAULT).step(Bound::new(finder(250, 50, 0, 0))).build().unwrap();
    let mut docs = vec![doc];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let e = docs[0].get::<Entities>().unwrap();
    // Only the first sentence is read; each piece is an entity, and each
    // maps to the token holding it.
    let want = pieces(&text[..12]);
    let got: Vec<(usize, usize, usize, usize)> = e
        .spans
        .iter()
        .map(|(i, s)| (s.start as usize, s.end as usize, e.tokens[i.get()].start.get(), e.tokens[i.get()].end.get()))
        .collect();
    let word = |b: usize| text[..b].matches(" ").count();
    let expect: Vec<(usize, usize, usize, usize)> = want.iter().map(|&(a, b)| (a, b, word(a), word(a) + 1)).collect();
    assert_eq!(got, expect);
}
