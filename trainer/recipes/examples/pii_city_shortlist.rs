//! Dev-only shortlist from the archived encoder features. Real serving
//! gates are checked separately by the tagger's additive city audit.

use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use serde_json::json;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::tagger::{TokenTagger, Work};
use turbo_eval::read::parquet::Parquet;
use turbo_tokenize::{TokenizerOptions, simple_spans};
use turbo_trainer_recipes::cache::CacheRecord;
use turbo_trainer_recipes::corpus::{check_pin, read_file};
use turbo_trainer_recipes::nemotron_pii::{TRAIN, spans};

fn model(dir: &Path, hash: &str) -> TokenTagger {
    let b = Bundle::open(dir).unwrap();
    assert_eq!(b.manifest_sha256, hash);
    TokenTagger::load(&b, "pii").unwrap()
}

fn proposals(t: &TokenTagger, floats: &[f32], words: &[String], temp: f64, floor: f64) -> Vec<(usize, usize)> {
    let forms: Vec<_> = words.iter().map(String::as_str).collect();
    let ids: Vec<_> = (0..words.len() as u32).collect();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    t.scores_forms(floats, &ids, Some(&forms), &mut scores, &mut work).unwrap();
    t.decode(&scores, words.len(), &mut tags).unwrap();
    let types = t.types();
    let labels = t.labels().len();
    let score = |i: usize| {
        let logits = &scores[i * labels..(i + 1) * labels];
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponent = |k: f32| (f64::from(k - max) / temp).exp();
        exponent(logits[tags[i] as usize]) / logits.iter().map(|&k| exponent(k)).sum::<f64>()
    };
    t.spans(&tags)
        .unwrap()
        .iter()
        .filter_map(|w| {
            let (a, b) = (w.start as usize, w.end as usize);
            (types[w.kind as usize] == "city" && (a..b).map(score).sum::<f64>() / ((b - a) as f64) >= floor)
                .then_some((a, b))
        })
        .collect()
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(
        args.len(),
        4,
        "pii_city_shortlist <publisher corpus> <publisher cache> <catalog bundle> <refit bundle>"
    );
    let cache = Path::new(&args[1]);
    let info = fs::read(cache.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&info), "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9");
    let info: CacheRecord = serde_json::from_slice(&info).unwrap();
    assert_eq!(info.layers, [22]);
    let a = model(Path::new(&args[2]), "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d");
    let b = model(Path::new(&args[3]), "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7");

    let (rec, parquet) = read_file(Path::new(&args[0]), TRAIN.path).unwrap();
    check_pin(&rec, &TRAIN).unwrap();
    let src = Parquet::open(TRAIN.path, &parquet).unwrap();
    let uid = src.strings("uid").unwrap();
    let text = src.strings("text").unwrap();
    let tags = src.strings("spans").unwrap();
    assert_eq!(src.rows(), 100_000);
    let mut reader = BufReader::new(File::open(cache.join("dev.layer22.f32")).unwrap());
    let mut bytes = Vec::new();
    let mut entries = Vec::new();
    let (mut total, mut off, mut gold_count, mut cached_options) = (0, 0, 0, 0);
    for i in (9..src.rows()).step_by(10) {
        if i >= 50_000 {
            assert_eq!(uid.at(i).unwrap(), uid.at(i - 50_000).unwrap());
        }
        let text = text.at(i).unwrap();
        let r = spans::to_sentence(TRAIN.path, i, text, tags.at(i).unwrap()).unwrap().unwrap();
        let mut word_spans = Vec::new();
        simple_spans(text, TokenizerOptions::default(), &mut word_spans);
        assert!(word_spans.iter().map(|s| &text[s.range()]).eq(r.words.iter().map(String::as_str)));
        bytes.resize(r.words.len() * info.encoder.hidden * 4, 0);
        reader.read_exact(&mut bytes).unwrap();
        off += bytes.len();
        total += 1;
        let byte_range = |a: usize, b: usize| [word_spans[a].start, word_spans[b - 1].end];
        let gold: Vec<_> = turbo_trainer_recipes::eval::spans(&r.labels)
            .into_iter()
            .filter_map(|(a, b, k)| (k == "city").then(|| byte_range(a, b)))
            .collect();
        gold_count += gold.len();
        let features: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|x| f32::from_le_bytes(*x)).collect();
        // Wide guards keep borderline candidates for an authoritative CPU
        // serving rerun and omit only clearly admitted catalog duplicates.
        let native = proposals(&a, &features, &r.words, 1.093_758, 0.55);
        let maybe = proposals(&b, &features, &r.words, 1.080_714_2, 0.49);
        let ranges: Vec<_> = maybe.into_iter().filter(|s| !native.contains(s)).map(|(a, b)| byte_range(a, b)).collect();
        cached_options += ranges.len();
        if !ranges.is_empty() {
            entries.push(json!({"row":i,"text":text,"gold":gold,"ranges":ranges}));
        }
    }
    assert_eq!(total, 10_000);
    assert_eq!(gold_count, 2005);
    assert_eq!(reader.stream_position().unwrap(), off as u64);
    println!(
        "{}",
        serde_json::to_string(
            &json!({"rows":total,"gold":gold_count,"cached_options":cached_options,"entries":entries})
        )
        .unwrap()
    );
}
