//! Development-only score-support ceilings for two pinned PII heads.
//! No selection policy, score threshold search, or held-out run.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use serde::Serialize;
use serde_json::json;
use turbo::bundle::{Bundle, sha256_hex};
use turbo::head::tagger::{TokenTagger, Work};
use turbo_trainer_recipes::cache::CacheRecord;
use turbo_trainer_recipes::eval::Span;
use turbo_trainer_recipes::nemotron_pii::read_pinned_train_dev;

mod role_probe;

const TYPES: [&str; 4] = ["city", "first_name", "last_name", "company_name"];

#[derive(Default)]
struct Proposals {
    native: BTreeSet<Span>,
    top2: BTreeSet<Span>,
    top3: BTreeSet<Span>,
    by_boundary: BTreeMap<(usize, usize), String>,
    flips: Vec<role_probe::Flip>,
}

fn open(path: &Path, digest: &str) -> TokenTagger {
    let bundle = Bundle::open(path).unwrap();
    assert_eq!(bundle.manifest_sha256, digest);
    TokenTagger::load(&bundle, "pii").unwrap()
}

fn label_groups(tagger: &TokenTagger) -> Vec<usize> {
    let types = tagger.types();
    tagger
        .labels()
        .iter()
        .map(|l| l.split_once('-').map_or(types.len(), |(_, kind)| types.iter().position(|s| s == kind).unwrap()))
        .collect()
}

fn score(tagger: &TokenTagger, words: &[String], features: &[f32], groups: &[usize], temperature: f32) -> Proposals {
    let ids: Vec<_> = (0..words.len() as u32).collect();
    let forms: Vec<_> = words.iter().map(String::as_str).collect();
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    tagger.scores_forms(features, &ids, Some(&forms), &mut scores, &mut work).unwrap();
    tagger.decode(&scores, words.len(), &mut tags).unwrap();
    let mut out = Proposals::default();
    let label_count = tagger.labels().len();
    let types = tagger.types();
    let mut mean = vec![0.0_f32; types.len() + 1];
    let mut weights = vec![0.0_f32; label_count];
    for sp in tagger.spans(&tags).unwrap() {
        let (a, b) = (sp.start as usize, sp.end as usize);
        let native = types[sp.kind as usize].clone();
        out.by_boundary.insert((a, b), native.clone());
        out.native.insert((a, b, native.clone()));
        out.top2.insert((a, b, native.clone()));
        out.top3.insert((a, b, native.clone()));
        mean.fill(0.0);
        for token in a..b {
            let logits = &scores[token * label_count..(token + 1) * label_count];
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut scale = 0.0;
            for (w, p) in weights.iter_mut().zip(logits) {
                *w = ((*p - max) / temperature).exp();
                scale += *w;
            }
            for (&w, &group) in weights.iter().zip(groups) {
                mean[group] += w / scale / (b - a) as f32;
            }
        }
        // Include O as an alternative for rank, but don't emit O as a span.
        let mut ranking: Vec<_> = (0..mean.len()).collect();
        ranking.sort_by(|&i, &j| mean[j].total_cmp(&mean[i]).then(i.cmp(&j)));
        if matches!(native.as_str(), "first_name" | "city") {
            let city = types.iter().position(|k| k == "city").unwrap();
            let first = types.iter().position(|k| k == "first_name").unwrap();
            let to_city = native == "first_name";
            let opposite = if to_city { city } else { first };
            if ranking[..2].contains(&opposite) {
                out.flips.push(role_probe::Flip {
                    a,
                    b,
                    to_city,
                    features: role_probe::features(a, b, features, mean[city], mean[first], mean[types.len()]),
                });
            }
        }
        for (i, &kind) in ranking.iter().take(3).enumerate() {
            if let Some(k) = types.get(kind) {
                out.top3.insert((a, b, k.clone()));
                if i < 2 {
                    out.top2.insert((a, b, k.clone()));
                }
            }
        }
    }
    out
}

#[derive(Default, PartialEq, Serialize)]
struct Burden {
    true_positive: usize,
    false_positive: usize,
    missed: usize,
}

impl Burden {
    fn f1(&self) -> f64 {
        2.0 * self.true_positive as f64 / (2 * self.true_positive + self.false_positive + self.missed) as f64
    }
}

fn burden(gold: &BTreeSet<Span>, predicted: &BTreeSet<Span>, by_type: &mut BTreeMap<String, Burden>) {
    for kind in TYPES {
        let row = by_type.entry(kind.into()).or_default();
        row.true_positive += predicted.intersection(gold).filter(|s| s.2 == kind).count();
        row.false_positive += predicted.difference(gold).filter(|s| s.2 == kind).count();
        row.missed += gold.difference(predicted).filter(|s| s.2 == kind).count();
    }
}

#[derive(Default, Serialize)]
struct Recovery {
    native_refit: usize,
    native_catalog_only: usize,
    same_refit_boundary_wrong_type: usize,
    refit_boundary_missing: usize,
    conflict_catalog_wins: usize,
    conflict_refit_wins: usize,
    conflict_neither_wins: usize,
    wrong_type_supported_by_refit_top2: usize,
    wrong_type_supported_by_refit_top3: usize,
    wrong_type_not_native_to_either_but_supported_by_union_top3: usize,
}

fn recovery(gold: &BTreeSet<Span>, catalog: &Proposals, refit: &Proposals, out: &mut BTreeMap<String, Recovery>) {
    for g in gold.iter().filter(|g| TYPES.contains(&g.2.as_str())) {
        let row = out.entry(g.2.clone()).or_default();
        let c = catalog.native.contains(g);
        let r = refit.native.contains(g);
        let a = catalog.by_boundary.get(&(g.0, g.1));
        let b = refit.by_boundary.get(&(g.0, g.1));
        row.native_refit += usize::from(r);
        row.native_catalog_only += usize::from(c && !r);
        if a.is_some() && b.is_some() && a != b {
            row.conflict_catalog_wins += usize::from(c);
            row.conflict_refit_wins += usize::from(r);
            row.conflict_neither_wins += usize::from(!c && !r);
        }
        if !r {
            row.refit_boundary_missing += usize::from(b.is_none());
            row.same_refit_boundary_wrong_type += usize::from(b.is_some());
            row.wrong_type_supported_by_refit_top2 += usize::from(b.is_some() && refit.top2.contains(g));
            row.wrong_type_supported_by_refit_top3 += usize::from(b.is_some() && refit.top3.contains(g));
            row.wrong_type_not_native_to_either_but_supported_by_union_top3 +=
                usize::from(!c && b.is_some() && (refit.top3.contains(g) || catalog.top3.contains(g)));
        }
    }
}

#[derive(Default, Serialize)]
struct FlipSupport {
    eligible: usize,
    keeps_correct: usize,
    flip_correct: usize,
    neither: usize,
    both_native_conflict: usize,
    both_native_conflict_flip_correct: usize,
}

fn flip_support(
    gold: &BTreeSet<Span>,
    catalog: &Proposals,
    refit: &Proposals,
    out: &mut BTreeMap<String, FlipSupport>,
) {
    for &(a, b, ref surface) in &refit.native {
        let (other, direction) = match surface.as_str() {
            "city" => ("first_name", "city_to_first_name"),
            "first_name" => ("city", "first_name_to_city"),
            _ => continue,
        };
        let choice = (a, b, other.to_owned());
        if !refit.top2.contains(&choice) {
            continue;
        }
        let stats = out.entry(direction.into()).or_default();
        let old_right = gold.contains(&(a, b, surface.clone()));
        let new_right = gold.contains(&choice);
        let disagrees = catalog.native.contains(&choice);
        stats.eligible += 1;
        stats.keeps_correct += usize::from(old_right);
        stats.flip_correct += usize::from(new_right);
        stats.neither += usize::from(!old_right && !new_right);
        stats.both_native_conflict += usize::from(disagrees);
        stats.both_native_conflict_flip_correct += usize::from(disagrees && new_right);
    }
}

fn union(a: &BTreeSet<Span>, b: &BTreeSet<Span>) -> BTreeSet<Span> {
    a.union(b).cloned().collect()
}

fn fit_probe(
    tagger: &TokenTagger,
    groups: &[usize],
    rows: &[turbo_trainer_recipes::corpus::Sentence],
    archive: &Path,
    hidden: usize,
) -> [role_probe::Fit; 2] {
    let mut file = BufReader::new(File::open(archive.join("train.layer22.f32")).unwrap());
    let mut bytes = Vec::new();
    let mut offset = 0;
    let (mut fit, mut cal): ([Vec<role_probe::Sample>; 2], [Vec<role_probe::Sample>; 2]) = Default::default();
    assert_eq!(rows.len(), 90_000);
    for (i, row) in rows.iter().enumerate() {
        let len = row.words.len() * hidden * 4;
        offset += len;
        // The pinned release carries the same 45,000 train personas in
        // the same order in US then international blocks.
        let persona = i % 45_000;
        let choice = persona % 6;
        if choice > 2 {
            file.seek_relative(len as i64).unwrap();
            continue;
        }
        bytes.resize(len, 0);
        file.read_exact(&mut bytes).unwrap();
        let f32s: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let got = score(tagger, &row.words, &f32s, groups, 1.080_714_2);
        let gold = turbo_trainer_recipes::eval::spans(&row.labels);
        for step in got.flips {
            let correct = gold.iter().any(|(a, b, k)| *a == step.a && *b == step.b && k == step.other());
            let dest = if choice == 2 { &mut cal } else { &mut fit };
            dest[step.direction()].push((step.features, correct));
        }
    }
    assert_eq!(file.stream_position().unwrap(), offset as u64);
    std::array::from_fn(|i| role_probe::fit(&fit[i], &cal[i]))
}

fn apply_probe(
    reference: &Proposals,
    fits: &[role_probe::Fit; 2],
    gold: &BTreeSet<Span>,
    decisions: &mut [[usize; 2]; 2],
) -> BTreeSet<Span> {
    let mut proposed = reference.native.clone();
    for f in &reference.flips {
        let d = f.direction();
        if fits[d].admits(&f.features) {
            assert!(proposed.remove(&(f.a, f.b, f.native().into())));
            let next = (f.a, f.b, f.other().into());
            decisions[d][0] += 1;
            decisions[d][1] += usize::from(gold.contains(&next));
            assert!(proposed.insert(next));
        }
    }
    assert_eq!(reference.native.len(), proposed.len());
    let bnd = |s: &BTreeSet<Span>| s.iter().map(|p| (p.0, p.1)).collect::<Vec<_>>();
    assert_eq!(bnd(&reference.native), bnd(&proposed));
    let unaffected = |s: &BTreeSet<Span>| {
        s.iter().filter(|p| !matches!(p.2.as_str(), "city" | "first_name")).cloned().collect::<Vec<_>>()
    };
    assert_eq!(unaffected(&reference.native), unaffected(&proposed));
    proposed
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert!(
        matches!(args.len(), 4 | 5),
        "pii_role_oracle <publisher corpus> <publisher cache> <catalog bundle> <refit bundle> [--probe]"
    );
    let probe = args.get(4).is_some_and(|flag| {
        assert_eq!(flag, "--probe");
        true
    });
    let dataset = Path::new(&args[0]);
    let cache = Path::new(&args[1]);
    let cache_meta = fs::read(cache.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&cache_meta), "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9");
    let metadata: CacheRecord = serde_json::from_slice(&cache_meta).unwrap();
    assert_eq!(metadata.layers, [22]);
    assert_eq!(metadata.encoder.hidden, 768);
    let catalog = open(Path::new(&args[2]), "19d3fb87daac08f0361390809b747af638bc2dbb93e447695c3f5609fc735b6d");
    let refit = open(Path::new(&args[3]), "dd73330bc8bef5bc0f1b9c7b6c004cfbf6713fa1dadcc03a6238a159b6ddc3b7");
    assert_eq!(catalog.labels(), refit.labels());
    let groups = label_groups(&catalog);
    let (train, gold) = read_pinned_train_dev(dataset).unwrap();
    let model = probe.then(|| fit_probe(&refit, &groups, &train, cache, metadata.encoder.hidden));
    if let Some(fitted) = &model {
        eprintln!("locked train-only calibration receipts: {}", json!([&fitted[0].receipt, &fitted[1].receipt]));
    }
    drop(train);
    let mut decisions = [[0usize; 2]; 2];
    assert_eq!(gold.len(), 10_000);
    let mut features = BufReader::new(File::open(cache.join("dev.layer22.f32")).unwrap());
    let mut bytes = Vec::new();
    let mut offset = 0;
    let mut burdens = BTreeMap::<&str, BTreeMap<String, Burden>>::new();
    let mut recoveries = BTreeMap::new();
    let mut city_first_only = BTreeMap::new();
    for row in gold {
        bytes.resize(row.words.len() * metadata.encoder.hidden * 4, 0);
        features.read_exact(&mut bytes).unwrap();
        offset += bytes.len();
        let f32s: Vec<_> = bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let a = score(&catalog, &row.words, &f32s, &groups, 1.093_758);
        let b = score(&refit, &row.words, &f32s, &groups, 1.080_714_2);
        let expected: BTreeSet<_> = turbo_trainer_recipes::eval::spans(&row.labels).into_iter().collect();
        for (key, set) in [
            ("catalog_native", &a.native),
            ("refit_native", &b.native),
            ("either_native", &union(&a.native, &b.native)),
            ("refit_with_top2", &b.top2),
            ("refit_with_top3", &b.top3),
            ("either_with_top3", &union(&a.top3, &b.top3)),
        ] {
            burden(&expected, set, burdens.entry(key).or_default());
        }
        if let Some(model) = &model {
            let probe = apply_probe(&b, model, &expected, &mut decisions);
            burden(&expected, &probe, burdens.entry("bounded_probe").or_default());
        }
        recovery(&expected, &a, &b, &mut recoveries);
        flip_support(&expected, &a, &b, &mut city_first_only);
    }
    assert_eq!(features.stream_position().unwrap(), offset as u64);
    let summary = model.map(|m| {
        let candidate = &burdens["bounded_probe"];
        let reference = &burdens["refit_native"];
        let catalog = &burdens["catalog_native"];
        let city_person_pass = ["city", "first_name"]
            .into_iter()
            .all(|kind| candidate[kind].f1() >= catalog[kind].f1().max(reference[kind].f1()));
        assert!(["last_name", "company_name"].into_iter().all(|kind| candidate[kind] == reference[kind]));
        json!({
            "directions":["first_name_to_city","city_to_first_name"],
            "calibration":[m[0].receipt,m[1].receipt],
            "dev_flips_and_exact":decisions,
            "city_and_first_name_pass":city_person_pass,
            "unrelated_metrics_identical_to_refit":true,
        })
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "split":"publisher_dev","native_and_grouped_hypotheses":burdens,"refit_target_opportunities":recoveries,
            "city_first_opposite_in_refit_top2":city_first_only, "probe":summary,"holdout_scored":false
        }))
        .unwrap()
    );
}
