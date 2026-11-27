//! Frozen-catalog warm-start probe, with independently verified streamed
//! batches and exact supplementary loss dose. No serving or holdout path.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::tagger::{TokenTagger, tensors};
use turbo::manifest::HeadSpec;
use turbo::safetensors::File as Tensors;
use turbo_trainer::adamw::Settings;
use turbo_trainer::replay::Replay;
use turbo_trainer::rng::Rng;
use turbo_trainer::{Example, Tagger};
use turbo_trainer_recipes::cache::CacheRecord;
use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval::{Metrics, Seen, evaluate, spans};
use turbo_trainer_recipes::nemotron_pii::read_pinned_train_dev;
use turbo_trainer_recipes::{run, safetensors};

mod pii_replay;
use pii_replay::{HIDDEN, MIN, Pair, REPEATS, Reader, TARGETS, persona, predict, signature};

const CATALOG: &str = "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d";
const PIN: &str = "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9";
const LR: f64 = 8e-5;
const ROLE: [&str; 4] = ["city", "first_name", "last_name", "company_name"];

fn reader(cache: &Path, split: &str, rows: &[Sentence], meta: &CacheRecord) -> Reader<BufReader<File>> {
    assert!(matches!(split, "train" | "dev"));
    let size = meta.splits.iter().find(|s| s.name == split).unwrap().bytes_per_file;
    let file = File::open(cache.join(format!("{split}.layer22.f32"))).unwrap();
    assert_eq!(file.metadata().unwrap().len(), size);
    let r = Reader::new(BufReader::new(file), rows.iter().map(|s| s.words.len()), HIDDEN);
    assert!(r.input_bytes() < size); // excludes the synthetic tail of the archive
    r
}

fn digest_check(bundle: &Bundle) -> (HeadSpec, Vec<Vec<f32>>) {
    assert_eq!(bundle.manifest_sha256, CATALOG);
    let head = bundle.manifest.heads.iter().find(|h| h.name == "pii").unwrap().clone();
    let weights = bundle.map_verified("weights/model.safetensors").unwrap();
    let raw = Tensors::parse("catalog", weights.as_ref()).unwrap();
    let params = tensors(&head, HIDDEN)
        .iter()
        .map(|(k, _)| raw.tensor(&format!("{}.{}", head.tensor_prefix, k)).unwrap().f32s())
        .collect();
    (head, params)
}

fn schedule(n: usize, extra: Vec<usize>) -> Vec<usize> {
    let blocks: Vec<_> = (0..n).collect::<Vec<_>>().chunks(128).map(<[usize]>::to_vec).collect();
    let mut order: Vec<_> = (0..blocks.len()).collect();
    Rng::new(2596).shuffle(&mut order);
    let mut out = Vec::with_capacity(n + extra.len());
    let mut j = 0;
    let (mut base_idx, mut extra_idx) = (0, 0);
    for i in order {
        for &f in &blocks[i] {
            out.push(f);
            base_idx += 1;
            while j < extra.len() && base_idx * extra.len() >= (j + 1) * n {
                out.push(extra[j]);
                j += 1;
                extra_idx += 1;
            }
        }
    }
    assert_eq!(extra_idx, extra.len());
    assert_eq!(base_idx, n);
    assert_eq!(out.len(), n + extra.len());
    out
}

fn fit_one(
    rows: &[Sentence],
    read: &mut Reader<BufReader<File>>,
    spec: &HeadSpec,
    params: Vec<Vec<f32>>,
    ids: &[usize],
    name: &str,
) -> Tagger<f32> {
    let labels: BTreeMap<_, _> = spec.labels.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let optim = Settings {
        lr: LR,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        clip: 5.0,
        warmup: 0,
        total: ids.len().div_ceil(128) as u64,
    };
    let mut replay = Replay::new(Tagger::with_params(spec, HIDDEN, params).unwrap(), optim, 6).unwrap();
    for (i, ids) in ids.chunks(128).enumerate() {
        let batch: Vec<_> = ids
            .iter()
            .map(|&idx| {
                let row = &rows[idx];
                Example {
                    features: read.read(idx, row.words.len()),
                    labels: row.labels.iter().map(|s| labels[s.as_str()]).collect(),
                    forms: row.words.clone(),
                    language: None,
                }
            })
            .collect();
        replay.update(&batch).unwrap();
        if (i + 1) % 64 == 0 {
            eprintln!("{name}: {} steps and {} words", replay.steps(), replay.words());
        }
    }
    assert_eq!(replay.steps(), optim.total);
    assert_eq!(replay.words(), ids.iter().map(|&i| rows[i].words.len() as u64).sum::<u64>());
    replay.into_head()
}

fn score(tagger: &TokenTagger, rows: &[Sentence], read: &mut Reader<BufReader<File>>) -> (Metrics, [usize; 2]) {
    let (mut false_al, mut false_company) = (0, 0);
    let predicted: Vec<_> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let floats = read.read(i, r.words.len());
            let pred = predict(tagger, r, &floats);
            for (j, label) in r.labels.iter().enumerate() {
                if label == "O"
                    && r.words[j].eq_ignore_ascii_case("al")
                    && pred.iter().any(|(a, b, k)| *a <= j && j < *b && k == "last_name")
                {
                    false_al += 1;
                }
                if label == "B-first_name" && pred.iter().any(|(a, b, k)| *a <= j && j < *b && k == "company_name") {
                    false_company += 1;
                }
            }
            let mut labels = vec!["O".to_owned(); r.words.len()];
            for (a, b, k) in pred {
                labels[a] = format!("B-{k}");
                for t in &mut labels[a + 1..b] {
                    *t = format!("I-{k}");
                }
            }
            assert_eq!(
                spans(&labels).iter().map(|s| s.1 - s.0).sum::<usize>(),
                labels.iter().filter(|l| *l != "O").count()
            );
            labels
        })
        .collect();
    (evaluate(Task::Bio, rows, &predicted, &Seen::default(), None), [false_al, false_company])
}

fn gate(catalog: &Metrics, control: &Metrics, variant: &Metrics) -> Value {
    let extract = |m: &Metrics| {
        let Metrics::Spans { all, per_type, .. } = m else { panic!("BIO") };
        [*all, per_type[ROLE[0]], per_type[ROLE[1]], per_type[ROLE[2]], per_type[ROLE[3]]]
    };
    let (a, b, c) = (extract(catalog), extract(control), extract(variant));
    let reports: Vec<_> = ["all", "city", "first_name", "last_name", "company_name"]
        .into_iter()
        .enumerate()
        .map(|(i, k)| {
            let ok =
                c[i].precision >= a[i].precision.max(b[i].precision) && c[i].recall >= a[i].recall.max(b[i].recall);
            json!({"kind":k,"pass":ok,"catalog":a[i],"control":b[i],"candidate":c[i]})
        })
        .collect();
    json!(reports)
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!(
        matches!(args.len(), 5 | 6),
        "pii_warm_replay <corpus> <cache> <catalog bundle> <plan-path> --plan|--fit [output]"
    );
    let cache = Path::new(&args[1]);
    let mode = &args[4];
    let plan = Path::new(&args[3]);
    let meta = fs::read(cache.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&meta), PIN);
    let meta: CacheRecord = serde_json::from_slice(&meta).unwrap();
    assert_eq!(meta.layers, [22]);
    assert_eq!(meta.encoder.hidden, HIDDEN);
    let bundle = Bundle::open(Path::new(&args[2])).unwrap();
    assert_eq!(bundle.manifest_sha256, CATALOG);
    let baseline = TokenTagger::load(&bundle, "pii").unwrap();
    let (train, dev) = read_pinned_train_dev(Path::new(&args[0])).unwrap();
    assert_eq!((train.len(), dev.len()), (90_000, 10_000));
    if mode == "--plan" {
        let mut trainfile = reader(cache, "train", &train, &meta);
        let pairs = pii_replay::plan(&train, &mut trainfile, &baseline);
        let counts = std::array::from_fn::<_, 6, _>(|i| pairs.iter().filter(|p| p.target == i).count());
        let eligible = counts.iter().all(|&n| n >= MIN);
        let info = json!({"catalog":CATALOG,"cache":PIN,"pairs":pairs,"counts":counts,
            "names":TARGETS,"preflight_passed":eligible});
        fs::write(plan, serde_json::to_vec_pretty(&info).unwrap()).unwrap();
        println!("{}", json!({"matched_by_case":counts,"preflight_passed":eligible}));
        if !eligible {
            std::process::exit(2);
        }
        return;
    }
    assert_eq!(mode, "--fit");
    let output = Path::new(&args[5]);
    let selected: Value = serde_json::from_slice(&fs::read(plan).unwrap()).unwrap();
    assert_eq!(selected["catalog"], CATALOG);
    assert_eq!(selected["cache"], PIN);
    assert_eq!(selected["preflight_passed"], true);
    let pairs: Vec<Pair> = serde_json::from_value(selected["pairs"].clone()).unwrap();
    let mut used = BTreeSet::new();
    for item in &pairs {
        assert!(used.insert(persona(item.control)));
        assert!(used.insert(persona(item.case)));
        assert_eq!(signature(&train[item.control]), signature(&train[item.case]));
    }
    assert!((0..6).all(|i| pairs.iter().filter(|p| p.target == i).count() >= MIN));
    let extra = |variant| {
        (0..REPEATS).flat_map(|_| pairs.iter().map(move |p| if variant { p.case } else { p.control })).collect()
    };
    let control_ids = schedule(train.len(), extra(false));
    let variant_ids = schedule(train.len(), extra(true));
    assert_eq!(control_ids.len(), variant_ids.len());
    // Stronger than total counts: every corresponding row has
    // identical counts, hence every update has identical dose.
    assert!(control_ids.iter().zip(&variant_ids).all(|(&u, &v)| signature(&train[u]) == signature(&train[v])));
    eprintln!(
        "matched {} original rows + {} supplemental repetitions; {} identical-dose updates",
        train.len(),
        pairs.len() * REPEATS,
        control_ids.len().div_ceil(128)
    );
    let (spec, params) = digest_check(&bundle);
    assert_eq!(spec, *baseline.spec());
    let mut trainfile = reader(cache, "train", &train, &meta);
    let control = fit_one(&train, &mut trainfile, &spec, params.clone(), &control_ids, "control");
    let candidate = fit_one(&train, &mut trainfile, &spec, params, &variant_ids, "candidate");
    let mut devfile = reader(cache, "dev", &dev, &meta);
    let (a, an) = score(&baseline, &dev, &mut devfile);
    let (b, bn) = score(&control.to_token_tagger().unwrap(), &dev, &mut devfile);
    let (c, cn) = score(&candidate.to_token_tagger().unwrap(), &dev, &mut devfile);
    let rows = gate(&a, &b, &c);
    let protected = (0..cn.len()).all(|i| cn[i] <= an[i].min(bn[i]));
    let roles = rows.as_array().unwrap();
    let city_gain = roles[1]["candidate"]["correct"].as_u64() > roles[1]["control"]["correct"].as_u64();
    let passed = roles.iter().all(|r| r["pass"] == true) && protected && city_gain;
    let info = json!({"dev_gate_passed":passed,"holdout_scored":false,"rows":rows,"city_improved_over_control":city_gain,
        "neutral":{"catalog":an,"control":bn,"candidate":cn,"pass":protected},"plan_counts":selected["counts"]});
    fs::create_dir(output).unwrap();
    let weights = |model: Tagger<f32>| safetensors::write_with(&run::named(&spec, HIDDEN, model.tensors()), &[]);
    let a = run::write(output, "control.safetensors", &weights(control)).unwrap();
    let b = run::write(output, "candidate.safetensors", &weights(candidate)).unwrap();
    run::write(output, "dev.json", &serde_json::to_vec_pretty(&info).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&info).unwrap());
    println!("head hashes {} {}", a.sha256, b.sha256);
    if !passed {
        std::process::exit(2);
    }
}
