//! #2596: dev-only token and span replay of the two rejected-pilot heads.

use serde_json::json;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use turbo::bundle::sha256_hex;
use turbo::head::tagger::{TokenTagger, Work};
use turbo::manifest::WordPooling;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_tokenize::{TokenizerOptions, simple_spans};
use turbo_trainer_recipes::cache::SplitRecord;
use turbo_trainer_recipes::eval::{self, Span};
use turbo_trainer_recipes::nemotron_pii::context_contrasts;
use turbo_trainer_recipes::{prior, receipt::EncoderRecord};

fn core_spans(tagger: &TokenTagger, tags: &[u32]) -> Vec<Span> {
    tagger
        .spans(tags)
        .unwrap()
        .into_iter()
        .map(|s| (s.start as usize, s.end as usize, tagger.types()[s.kind as usize].clone()))
        .collect()
}

fn predict(tagger: &TokenTagger, words: &[String], features: &[f32]) -> (Vec<u32>, Vec<Span>) {
    let ids: Vec<u32> = (0..words.len() as u32).collect();
    let forms: Vec<&str> = words.iter().map(String::as_str).collect();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    tagger.scores_forms(features, &ids, Some(&forms), &mut scores, &mut work).unwrap();
    tagger.decode(&scores, words.len(), &mut tags).unwrap();
    let core = core_spans(tagger, &tags);
    let emitted: Vec<_> = tags.iter().map(|t| tagger.labels()[*t as usize].clone()).collect();
    assert_eq!(core, eval::spans(&emitted), "production decoder and receipt scorer disagree");
    (tags, core)
}

fn audit(
    root: &Path,
    enc: &EncoderRecord,
    rec: &SplitRecord,
    name: &str,
    expected: &str,
    encoder: &mut WordEncoder,
) -> serde_json::Value {
    let prior = prior::read(&root.join(name), enc).unwrap();
    assert_eq!(sha256_hex(&prior.weights), expected);
    let tagger = prior.head(|get| TokenTagger::from_tensors(&prior.spec, enc.hidden, get)).unwrap();
    assert_eq!(rec.name, "dev");
    assert_eq!(rec.sentences, 600);
    let rows = context_contrasts::documents(true).unwrap();
    let label_ids: BTreeMap<_, _> = tagger.labels().iter().enumerate().map(|(i, l)| (l.as_str(), i as u32)).collect();
    let mut reader = BufReader::new(File::open(root.join("contrast").join(&rec.files[0])).unwrap());
    assert_eq!(reader.get_ref().metadata().unwrap().len(), rec.bytes_per_file);
    let mut by_role: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut per_surname_position: BTreeMap<&str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut per_particle: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
    let mut examples = Vec::new();
    let mut found_last_lengths = BTreeMap::<usize, usize>::new();
    for row in rows {
        let mut bytes = vec![0; row.words.len() * enc.hidden * 4];
        reader.read_exact(&mut bytes).unwrap();
        let mat: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|r| f32::from_le_bytes(*r)).collect();
        let (tags, pred) = predict(&tagger, &row.words, &mat);
        for &(s, e, ref kind) in &pred {
            if kind == "last_name" {
                *found_last_lengths.entry(e - s).or_default() += 1;
            }
        }
        let gold = eval::spans(&row.labels);
        assert_eq!(gold.len(), 4);
        let gold_ids: Vec<_> = row.labels.iter().map(|l| label_ids[l.as_str()]).collect();
        assert_eq!(core_spans(&tagger, &gold_ids), gold, "gold projection differs by decoder");
        for (a, b, kind) in gold {
            let overlapping: Vec<_> = pred.iter().filter(|(s, e, _)| *s < b && a < *e).collect();
            let class = if overlapping.iter().any(|(s, e, k)| *s == a && *e == b && *k == kind) {
                "exact"
            } else if overlapping.iter().any(|(s, e, k)| *s == a && *e == b && *k != kind) {
                "exact_boundary_wrong_type"
            } else if overlapping.iter().any(|(_, _, k)| *k == kind) {
                "same_type_wrong_boundary"
            } else if !overlapping.is_empty() {
                "wrong_type_and_boundary"
            } else {
                "uncovered"
            };
            *by_role.entry(kind.clone()).or_default().entry(class.into()).or_default() += 1;
            if kind != "last_name" {
                continue;
            }
            assert_eq!(b - a, 2);
            assert_eq!((&*row.labels[a], &*row.labels[a + 1]), ("B-last_name", "I-last_name"));
            let prefix_label = &tagger.labels()[tags[a] as usize];
            let name_label = &tagger.labels()[tags[a + 1] as usize];
            for (key, label) in [("particle", prefix_label), ("name", name_label)] {
                *per_surname_position.entry(key).or_default().entry(label.clone()).or_default() += 1;
            }
            *per_particle
                .entry(row.words[a].clone())
                .or_default()
                .entry(format!("{prefix_label} / {name_label}"))
                .or_default() += 1;
            if examples.len() < 10 {
                let from = a.saturating_sub(2);
                let end = (b + 2).min(row.words.len());
                examples.push(json!({"snippet":row.words[from..end].join(" "), "gold":row.words[a..b].join(" "), "prefix_tag":prefix_label,"name_tag":name_label}));
            }
        }
    }
    assert_eq!(reader.stream_position().unwrap(), rec.bytes_per_file);
    let mut probes = Vec::new();
    for context in [
        "The signed voucher from Dr. {} Armitage arrived.",
        "Jo handed the spare badge to nurse {} Thurston.",
        "Ms. {} Pollard filed the corrected list.",
    ] {
        for particle in ["de", "del", "Van", "von"] {
            let text = context.replace("{}", particle);
            let mut spans = Vec::new();
            simple_spans(&text, TokenizerOptions::default(), &mut spans);
            let words: Vec<_> = spans.iter().map(|w| text[w.range()].to_owned()).collect();
            let forms: Vec<_> = words.iter().map(String::as_str).collect();
            let layers = encoder.layer_features(&[forms], WordPooling::First, 1).unwrap();
            let (tags, pred) = predict(&tagger, &words, &layers[0][0]);
            let i = words.iter().position(|w| w == particle).unwrap();
            let got: Vec<_> = pred
                .into_iter()
                .filter(|(s, e, _)| *s < i + 2 && *e > i)
                .map(|(s, e, k)| (words[s..e].join(" "), k))
                .collect();
            probes.push(json!({"text":text,"particle_tag":tagger.labels()[tags[i] as usize],"name_tag":tagger.labels()[tags[i+1] as usize],"spans":got}));
        }
    }
    json!({"by_role":by_role,"last_name_predicted_lengths":found_last_lengths,"surname_token_tags":per_surname_position,"surname_pairs_by_particle":per_particle,"examples":examples,"fresh_cpu_probes":probes})
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 2);
    let root = Path::new(&args[0]);
    let (enc, rec): (EncoderRecord, SplitRecord) =
        serde_json::from_slice(&fs::read(root.join("contrast/dev.json")).unwrap()).unwrap();
    let bundle = turbo_pins::require(&args[1]);
    let options = EncoderOptions { cpu_threads: 4, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(&bundle, &options).unwrap();
    println!("{}",serde_json::to_string_pretty(&json!({
        "control":audit(root,&enc,&rec,"control","3130d49ed775838a4107f0fb463ec63904a4189ebcde8dec5e10d53b282e71c6",&mut encoder),
        "candidate":audit(root,&enc,&rec,"candidate","cdf1b3cbb4bb5819611042a758bf2dcada5dba184592e251d4a79b660fb4d063",&mut encoder),
    })).unwrap());
}
