//! #2596: dev-only token and span replay of the two rejected-pilot heads.

use serde_json::json;
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::Path;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::tagger::{TokenTagger, Work};
use turbo::manifest::WordPooling;
use turbo_analysis_embed::EncoderOptions;
use turbo_tagger::WordEncoder;
use turbo_tokenize::{TokenizerOptions, simple_spans};
use turbo_trainer_recipes::cache::{CacheRecord, SplitRecord};
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::eval::{self, Span};
use turbo_trainer_recipes::nemotron_pii::{context_contrasts, read_pinned_train_dev};
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
    let (mut strict_left, mut any_left) = (0, 0);
    let mut role_types: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();
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
            let shared = if kind == "last_name" { b - 1 } else { a };
            let picked = tagger.labels()[tags[shared] as usize].as_str();
            let picked = picked.split_once('-').map_or(picked, |(_, t)| t);
            *role_types.entry(kind.clone()).or_default().entry(picked.into()).or_default() += 1;
            if kind != "last_name" {
                continue;
            }
            assert_eq!(b - a, 2);
            assert_eq!((&*row.labels[a], &*row.labels[a + 1]), ("B-last_name", "I-last_name"));
            let prefix_label = &tagger.labels()[tags[a] as usize];
            let name_label = &tagger.labels()[tags[a + 1] as usize];
            if class != "exact" && overlapping.iter().any(|(s, e, k)| *s == a + 1 && *e == b && k == "last_name") {
                any_left += 1;
                strict_left += usize::from(prefix_label == "O");
            }
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
    json!({"one_word_left_oracle":{"untyped_only":strict_left,"ignoring_other_types":any_left},"type_at_shared_word":role_types,"by_role":by_role,"last_name_predicted_lengths":found_last_lengths,"surname_token_tags":per_surname_position,"surname_pairs_by_particle":per_particle,"examples":examples,"fresh_cpu_probes":probes})
}

fn publisher_oracle(rows: &[Sentence], cache: &Path, hidden: usize, tagger: &TokenTagger) -> serde_json::Value {
    let mut cache = BufReader::new(File::open(cache.join("dev.layer22.f32")).unwrap());
    let (mut gold, mut exact, mut multi_exact, mut left, mut right, mut strict_union) = (0, 0, 0, 0, 0, 0);
    let (mut relaxed_left, mut relaxed_right, mut predicted) = (0, 0, 0);
    let mut bytes = Vec::new();
    let mut offset = 0;
    for row in rows {
        bytes.resize(row.words.len() * hidden * 4, 0);
        cache.read_exact(&mut bytes).unwrap();
        offset += bytes.len();
        let expected: Vec<_> = eval::spans(&row.labels).into_iter().filter(|s| s.2 == "last_name").collect();
        let mat: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|f| f32::from_le_bytes(*f)).collect();
        let (tags, pred) = predict(tagger, &row.words, &mat);
        predicted += pred.iter().filter(|s| s.2 == "last_name").count();
        let outside = |i: usize| tagger.labels()[tags[i] as usize] == "O";
        for (a, b, k) in expected {
            gold += 1;
            if pred.iter().any(|(s, e, t)| *s == a && *e == b && t == &k) {
                exact += 1;
                multi_exact += usize::from(b - a > 1);
                continue;
            }
            if b - a == 1 {
                continue;
            }
            let clipped_left = pred.iter().any(|(s, e, t)| *s == a + 1 && *e == b && t == &k);
            let clipped_right = pred.iter().any(|(s, e, t)| *s == a && *e == b - 1 && t == &k);
            relaxed_left += usize::from(clipped_left);
            relaxed_right += usize::from(clipped_right);
            let safe_left = clipped_left && outside(a);
            let safe_right = clipped_right && outside(b - 1);
            left += usize::from(safe_left);
            right += usize::from(safe_right);
            strict_union += usize::from(safe_left || safe_right);
        }
    }
    assert_eq!(cache.stream_position().unwrap(), offset as u64);
    assert_eq!(gold, 4207);
    let f1 = |correct: usize| 2.0 * correct as f64 / (gold + predicted) as f64;
    json!({"gold":gold,"predicted":predicted,"exact":exact,"multi_exact":multi_exact,
        "raw_f1":f1(exact),"left":{"strict_gain":left,"relaxed_gain":relaxed_left,"strict_f1_bound":f1(exact+left),
            "relaxed_f1_bound":f1(exact+relaxed_left)},
        "right":{"strict_gain":right,"relaxed_gain":relaxed_right},
        "either":{"strict_gain":strict_union,"strict_f1_bound":f1(exact+strict_union)} })
}

fn publisher_report(root: &Path, codec: &EncoderRecord, args: &[std::ffi::OsString]) -> serde_json::Value {
    let (dataset, cache) = (Path::new(&args[0]), Path::new(&args[1]));
    let bytes = fs::read(cache.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&bytes), "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9");
    let metadata: CacheRecord = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(metadata.encoder.manifest_sha256, codec.manifest_sha256);
    assert_eq!(metadata.layers, [22]);
    let (_, dev) = read_pinned_train_dev(dataset).unwrap();
    assert_eq!(dev.len(), 10_000);
    let mut arms = BTreeMap::new();
    for (arm, digest) in [
        ("control", "3130d49ed775838a4107f0fb463ec63904a4189ebcde8dec5e10d53b282e71c6"),
        ("candidate", "cdf1b3cbb4bb5819611042a758bf2dcada5dba184592e251d4a79b660fb4d063"),
        ("overlay-head", "b75a793608ca52e2b8815361c2186c5648e86f58537857e8f31fa7c694bfa981"),
    ] {
        let prior = prior::read(&root.join(arm), codec).unwrap();
        assert_eq!(sha256_hex(&prior.weights), digest);
        let tagger = prior.head(|get| TokenTagger::from_tensors(&prior.spec, codec.hidden, get)).unwrap();
        arms.insert(arm, publisher_oracle(&dev, cache, codec.hidden, &tagger));
    }
    for (arm, bundle_path, digest) in [
        ("catalog", &args[2], "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d"),
        ("refit", &args[3], "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7"),
    ] {
        let bundle = Bundle::open(Path::new(bundle_path)).unwrap();
        assert_eq!(bundle.manifest_sha256, digest);
        let tagger = TokenTagger::load(&bundle, "pii").unwrap();
        arms.insert(arm, publisher_oracle(&dev, cache, codec.hidden, &tagger));
    }
    json!(arms)
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!(
        matches!(args.len(), 2 | 6),
        "pii_surname_audit <experiment> <encoder> [<publisher corpus> <publisher cache> <catalog bundle> <refit bundle>]"
    );
    let root = Path::new(&args[0]);
    let (enc, rec): (EncoderRecord, SplitRecord) =
        serde_json::from_slice(&fs::read(root.join("contrast/dev.json")).unwrap()).unwrap();
    if args.len() == 6 {
        println!("{}", serde_json::to_string_pretty(&publisher_report(root, &enc, &args[2..])).unwrap());
        return;
    }
    let bundle = turbo_pins::require(&args[1]);
    let options = EncoderOptions { cpu_threads: 4, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(&bundle, &options).unwrap();
    println!("{}",serde_json::to_string_pretty(&json!({
        "control":audit(root,&enc,&rec,"control","3130d49ed775838a4107f0fb463ec63904a4189ebcde8dec5e10d53b282e71c6",&mut encoder),
        "candidate":audit(root,&enc,&rec,"candidate","cdf1b3cbb4bb5819611042a758bf2dcada5dba184592e251d4a79b660fb4d063",&mut encoder),
        "overlay":audit(root,&enc,&rec,"overlay-head","b75a793608ca52e2b8815361c2186c5648e86f58537857e8f31fa7c694bfa981",&mut encoder),
    })).unwrap());
}
