//! The sentence vector step against OpenNLP's SentenceVectorsDL, on the
//! goldens `turbo-goldens` writes under `sentvectors/all-minilm-l6-v2-mean`
//! (analysis/goldens-tool/README.md): for each sentence the English sentence
//! model found in an `en` input, the vector SentenceVectorsDL gave it through
//! a mean-pooled copy of the all-MiniLM-L6-v2 ONNX export.
//!
//! Each sentence is embedded here as a document of its own with one
//! sentence over all of it, so its row holds that sentence alone, as
//! SentenceVectorsDL's does. The two still differ by what they average:
//! SentenceVectorsDL's graph takes the mean over every position the mask
//! keeps, [CLS] and [SEP] included, where the step pools the sentence's own
//! pieces only. So the step's vector is held to a cosine, [`MIN_COSINE`],
//! set by the shortest sentences, where [CLS] and [SEP] are a large part of
//! the row. The mean over the whole row, the specials included, is
//! computed here as well, with the encoder, and held to
//! [`ROW_MIN_COSINE`]: that one checks the tokenizer and the encoder
//! against OpenNLP's and ONNX Runtime's.
//!
//! `TURBO_TEST_MINILM_BUNDLE` is an all-MiniLM-L6-v2 bundle made from the
//! same upstream revision. Without it the test says it was skipped and passes.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::Encode;
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Sentences, Vectors};

/// The lowest cosine allowed between a sentence's vector here and the one
/// SentenceVectorsDL wrote: [CLS] vs mean, not a numerical tolerance.
const MIN_COSINE: f64 = 0.91;

/// The lowest cosine allowed between the mean over a sentence's whole row
/// here and SentenceVectorsDL's vector: the same tokens averaged the same
/// way, so only the two runtimes' float arithmetic is between them.
const ROW_MIN_COSINE: f64 = 0.999999;

fn data() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../goldens/data")
}

/// The sentences of a golden, as byte spans, and each one's vector.
fn read_golden(text: &str) -> Vec<((usize, usize), Vec<f32>)> {
    let mut spans = Vec::new();
    let mut vectors = Vec::new();
    let mut layer = "";
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("layer\t") {
            layer = rest.split('\t').next().unwrap();
            continue;
        }
        if line.starts_with("end\t") {
            layer = "";
            continue;
        }
        match layer {
            "opennlp:sentences" => {
                let mut f = line.split('\t').map(|x| x.parse::<usize>().unwrap());
                spans.push((f.next().unwrap(), f.next().unwrap()));
            }
            "turbo:vector<opennlp:sentences>" => {
                vectors.push(line.split(' ').map(|h| f32::from_bits(u32::from_str_radix(h, 16).unwrap())).collect());
            }
            _ => {}
        }
    }
    assert_eq!(spans.len(), vectors.len());
    spans.into_iter().zip(vectors).collect()
}

fn cosine(g: &[f32], w: &[f32]) -> f64 {
    let dot: f64 = g.iter().zip(w).map(|(a, b)| *a as f64 * *b as f64).sum();
    let n = |v: &[f32]| v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
    dot / (n(g) * n(w))
}

#[test]
fn sentence_vectors_match_sentence_vectors_dl() {
    let Ok(bundle) = std::env::var("TURBO_TEST_MINILM_BUNDLE") else {
        println!("skipped: TURBO_TEST_MINILM_BUNDLE is not set");
        return;
    };
    let bundle = PathBuf::from(bundle);
    let vectors = Arc::new(VectorAnnotator::open(&bundle, &VectorOptions::default()).unwrap());
    assert_eq!(vectors.dim(), 384, "TURBO_TEST_MINILM_BUNDLE must be an all-MiniLM-L6-v2 bundle");
    let a = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SubwordAnnotator::open(&bundle).unwrap())
        .add(SentenceVectors(vectors))
        .build()
        .unwrap();
    let dir = data().join("sentvectors/all-minilm-l6-v2-mean/en");
    let mut goldens: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .collect();
    goldens.sort();
    assert!(!goldens.is_empty());
    let sub = SubwordAnnotator::open(&bundle).unwrap();
    let tok = sub.tokenizer();
    let mut enc = Encoder::open(&bundle, &EncoderOptions { max_batch: 1, ..EncoderOptions::default() }).unwrap();
    let mut rows = Rows::new(enc.info().max_seq, 0);
    let mut row = Vec::new();
    let (mut worst, mut sum, mut count, mut row_worst) = (1f64, 0f64, 0usize, 1f64);
    for golden in goldens {
        let id = golden.file_stem().unwrap().to_str().unwrap().to_owned();
        let input = std::fs::read_to_string(data().join(format!("inputs/en/{id}.txt"))).unwrap();
        for ((start, end), want) in read_golden(&std::fs::read_to_string(&golden).unwrap()) {
            let sentence = &input[start..end];
            let mut d = Document::new(sentence);
            let mut s = Spans::new(ViewId::SOURCE);
            s.push(Span::new(0, sentence.len() as u32));
            d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
            let mut docs = [d];
            a.run(&mut docs).remove(0).unwrap();
            let got = &docs[0].get::<Vectors<Sentences>>().expect("sentence vectors").data;
            assert_eq!(got.len(), want.len(), "{id} {start}..{end}");
            let c = cosine(got, &want);
            assert!(c >= MIN_COSINE, "{id} {start}..{end} {sentence:?}: cosine {c} under {MIN_COSINE}");
            // The whole row, [CLS] and [SEP] included, is what the graph
            // averages: that one is held to the encoder's own agreement.
            let ids = tok
                .encode(
                    sentence,
                    Encode {
                        add_special_tokens: true,
                        truncation: Truncation::None,
                        max_tokens: u32::MAX,
                        prompt: PromptRole::None,
                    },
                )
                .unwrap();
            rows.clear();
            rows.push(&[&ids]);
            enc.run_rows(&rows, false, &mut row).unwrap();
            let r = cosine(&row, &want);
            assert!(
                r >= ROW_MIN_COSINE,
                "{id} {start}..{end} {sentence:?}: whole row cosine {r} under {ROW_MIN_COSINE}"
            );
            row_worst = row_worst.min(r);
            worst = worst.min(c);
            sum += c;
            count += 1;
        }
    }
    println!(
        "{count} sentences: lowest cosine {worst}, mean {}; whole rows: lowest cosine {row_worst}",
        sum / count as f64
    );
}
