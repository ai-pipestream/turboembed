//! The measurement behind `turbo-bench postag`, built with the `postag`
//! feature, which links the tagger.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};

use turbo::bundle::sha256_hex;
use turbo::record::{Device, Library, Machine};
use turbo_analysis::Annotator;
use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, Numerics};

use super::{Accuracy, Drift, LENGTHS, POSTAG_DIR, PosModel, PosRecord, PosRun, SENTENCES_PER_CALL, Side, Treebank};
use crate::api::{Runtime, field};
use crate::cpus::{Cpus, SYSFS_CPUS, pin};
use crate::measure::percentile;
use crate::{BUILD_CHANGES, BUILD_COMMIT, Result, check_build, git, utc};

const USAGE: &str = "turbo-bench postag --model <ONLP-BLPT file> --treebank <file.conllu> [--cpus <list>] [--rounds <n>] \
                     [--repo <dir>] [--out <dir>]";

/// A sentence of the treebank: its words and their UPOS tags.
struct Sentence {
    words: Vec<String>,
    tags: Vec<String>,
}

/// The syntactic words of a CoNLL-U file, by sentence: a multiword
/// token's range line and an empty node are not words.
fn conllu(text: &str) -> Vec<Sentence> {
    let mut out = Vec::new();
    let (mut words, mut tags) = (Vec::new(), Vec::new());
    for line in text.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            if !words.is_empty() {
                out.push(Sentence { words: std::mem::take(&mut words), tags: std::mem::take(&mut tags) });
            }
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if line.starts_with('#') || cols.len() < 4 || cols[0].contains(['-', '.']) {
            continue;
        }
        words.push(cols[1].to_owned());
        tags.push(cols[3].to_owned());
    }
    out
}

/// The options, each `--name value`, once.
fn options(args: &[String]) -> Result<BTreeMap<String, String>> {
    let mut o = BTreeMap::new();
    let mut it = args.iter();
    while let Some(name) = it.next() {
        let value = it.next().ok_or_else(|| format!("{name}: no value\n{USAGE}"))?;
        if !matches!(name.as_str(), "--model" | "--treebank" | "--cpus" | "--rounds" | "--repo" | "--out") {
            return Err(format!("{name}: not a postag option\n{USAGE}"));
        }
        if o.insert(name.clone(), value.clone()).is_some() {
            return Err(format!("{name} given twice"));
        }
    }
    Ok(o)
}

/// One timed pass: every sentence of `set` through `tagger`, `per_call`
/// at a time. Its seconds, and each call's milliseconds.
fn pass(tagger: &BilstmPosTagger, set: &[&Sentence], per_call: usize) -> Result<(f64, Vec<f64>)> {
    let mut calls = Vec::with_capacity(set.len().div_ceil(per_call));
    let start = Instant::now();
    for chunk in set.chunks(per_call) {
        let sentences: Vec<&[String]> = chunk.iter().map(|s| s.words.as_slice()).collect();
        let t = Instant::now();
        std::hint::black_box(tagger.best_sequences(&sentences).map_err(|e| e.to_string())?);
        calls.push(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok((start.elapsed().as_secs_f64(), calls))
}

/// One configuration's passes on one numerics: each pass's seconds and
/// every call's milliseconds.
#[derive(Default)]
struct Times {
    passes: Vec<f64>,
    calls: Vec<f64>,
}

impl Times {
    fn side(mut self, words: usize) -> Side {
        let mut rates: Vec<f64> = self.passes.iter().map(|s| words as f64 / s).collect();
        rates.sort_by(f64::total_cmp);
        self.calls.sort_by(f64::total_cmp);
        Side {
            words_per_second: rates[rates.len() / 2],
            call_p50_ms: percentile(&self.calls, 50.0),
            call_p95_ms: percentile(&self.calls, 95.0),
        }
    }
}

/// What each numerics gets right, where they agree, and how far apart
/// their emissions and marginals are, over the whole treebank.
fn compare(packed: &BilstmPosTagger, parity: &BilstmPosTagger, test: &[Sentence]) -> Result<(Accuracy, Drift)> {
    let err = |e: turbo_postag::neural::NeuralError| e.to_string();
    let (mut a, mut d) = (
        Accuracy { packed_correct: 0, parity_correct: 0, tags_agree: 0 },
        Drift { max_emission_abs_diff: 0.0, max_marginal_abs_diff: 0.0 },
    );
    for s in test {
        let (p, r) = (packed.emissions(&s.words).map_err(err)?, parity.emissions(&s.words).map_err(err)?);
        for (p, r) in p.iter().flatten().zip(r.iter().flatten()) {
            d.max_emission_abs_diff = d.max_emission_abs_diff.max((p - r).abs());
        }
        let p = packed.best_sequences(&[&s.words]).map_err(err)?.remove(0);
        let r = parity.best_sequences(&[&s.words]).map_err(err)?.remove(0);
        for (i, gold) in s.tags.iter().enumerate() {
            let (pt, rt) = (&p.outcomes()[i], &r.outcomes()[i]);
            a.packed_correct += u32::from(pt == gold);
            a.parity_correct += u32::from(rt == gold);
            if pt == rt {
                a.tags_agree += 1;
                d.max_marginal_abs_diff = d.max_marginal_abs_diff.max((p.probs()[i] - r.probs()[i]).abs());
            }
        }
    }
    Ok((a, d))
}

pub fn cmd(args: &[String]) -> Result<()> {
    let mut o = options(args)?;
    let model_path = PathBuf::from(o.remove("--model").ok_or(format!("--model is required\n{USAGE}"))?);
    let treebank_path = PathBuf::from(o.remove("--treebank").ok_or(format!("--treebank is required\n{USAGE}"))?);
    let cpus = o.remove("--cpus").map(|l| Cpus::parse(&l, Path::new(SYSFS_CPUS))).transpose()?;
    let rounds: u32 = o.remove("--rounds").map_or(Ok(3), |v| v.parse().map_err(|e| format!("--rounds {v}: {e}")))?;
    if rounds == 0 {
        return Err("--rounds 0: nothing would be timed".into());
    }
    let repo = o.remove("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.remove("--out").map(PathBuf::from);
    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    check_build(&before.commit, BUILD_COMMIT, BUILD_CHANGES)?;
    if let Some(c) = &cpus {
        pin(c)?;
    }
    let recorded_at = utc(SystemTime::now());

    let read = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    let model_bytes = read(&model_path)?;
    let treebank_bytes = read(&treebank_path)?;
    let text = std::str::from_utf8(&treebank_bytes).map_err(|e| format!("{}: {e}", treebank_path.display()))?;
    let test = conllu(text);
    let words: usize = test.iter().map(|s| s.words.len()).sum();
    if test.is_empty() {
        return Err(format!("{}: no sentence", treebank_path.display()));
    }
    let load = |n| {
        let m = BilstmPosModel::load(model_bytes.as_slice()).map_err(|e| format!("{}: {e}", model_path.display()))?;
        BilstmPosTagger::with_numerics(m, n).map_err(|e| format!("{n:?}: {e}"))
    };
    let (packed, parity) = (load(Numerics::Packed)?, load(Numerics::JvmParity)?);
    let (accuracy, drift) = compare(&packed, &parity, &test)?;
    eprintln!(
        "{} sentences, {words} words: packed {} correct, parity {} correct, {} agree; max emission diff {:e}, max \
         marginal diff {:e}",
        test.len(),
        accuracy.packed_correct,
        accuracy.parity_correct,
        accuracy.tags_agree,
        drift.max_emission_abs_diff,
        drift.max_marginal_abs_diff
    );

    // The configurations: (sentences per call, lengths, sentences).
    let all: Vec<&Sentence> = test.iter().collect();
    let mut configs: Vec<(u32, String, Vec<&Sentence>)> =
        SENTENCES_PER_CALL.iter().map(|&n| (n, "all".to_owned(), all.clone())).collect();
    for (min, max) in LENGTHS {
        let set: Vec<&Sentence> = test
            .iter()
            .filter(|s| s.words.len() as u32 >= min && max.is_none_or(|m| s.words.len() as u32 <= m))
            .collect();
        if !set.is_empty() {
            configs.push((32, super::lengths_name(min, max), set));
        }
    }
    // Warm each tagger once before anything is timed.
    for t in [&packed, &parity] {
        pass(t, &all[..all.len().min(64)], 32)?;
    }
    let mut times: Vec<[Times; 2]> = configs.iter().map(|_| Default::default()).collect();
    for round in 0..rounds {
        for ((n, lengths, set), t) in configs.iter().zip(&mut times) {
            // Packed, parity, parity, packed.
            for (i, tagger) in [(0, &packed), (1, &parity), (1, &parity), (0, &packed)] {
                let (secs, calls) = pass(tagger, set, *n as usize)?;
                t[i].passes.push(secs);
                t[i].calls.extend(calls);
            }
            eprintln!("round {} of {rounds}: {n} per call, lengths {lengths} done", round + 1);
        }
    }
    let runs: Vec<PosRun> = configs
        .iter()
        .zip(times)
        .map(|((n, lengths, set), [p, r])| {
            let w: usize = set.iter().map(|s| s.words.len()).sum();
            let (packed, parity) = (p.side(w), r.side(w));
            PosRun {
                sentences_per_call: *n,
                lengths: lengths.clone(),
                sentences: set.len() as u32,
                words: w as u32,
                packed_over_parity: packed.words_per_second / parity.words_per_second,
                packed,
                parity,
            }
        })
        .collect();

    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let rt = Runtime::create()?;
    let d = rt.cpu()?.ok_or("this build lists no CPU device")?;
    let file = treebank_path.file_name().and_then(|f| f.to_str()).unwrap_or("").to_owned();
    let model_ref = packed.model().model_ref();
    let r = PosRecord {
        record_version: super::POS_RECORD_VERSION,
        recorded_at,
        machine: Machine { arch: field(&d.arch), host_cpu: field(&d.name), os: std::env::consts::OS.into() },
        device: Device {
            backend: field(&d.backend),
            kind: crate::kind_name(d.kind).into(),
            name: field(&d.name),
            vendor: field(&d.vendor),
            driver_version: field(&d.driver_version),
            runtime_version: field(&d.runtime_version),
            memory_total: d.memory_total,
        },
        library: Library {
            version: crate::api::version().split(' ').next().unwrap_or("").into(),
            build: crate::api::version(),
            commit: after.commit.clone(),
            pushed_to: after.pushed_to.clone(),
            settings: Vec::new(),
        },
        procedure: match &cpus {
            Some(c) => format!("the tool ran pinned to CPUs {}, the tagger on one thread", c.list),
            None => "the tool ran unpinned, the tagger on one thread".into(),
        },
        command: format!(
            "cargo run --release -p turbo-bench --features postag -- postag --model <model> --treebank \
             <dir>/{file}{} --rounds {rounds}",
            cpus.as_ref().map_or(String::new(), |c| format!(" --cpus {}", c.list))
        ),
        model: PosModel {
            id: model_ref.id.to_string(),
            revision: model_ref.revision.to_string(),
            sha256: model_ref.sha256.to_string(),
            file_sha256: sha256_hex(&model_bytes),
            file_bytes: model_bytes.len() as u64,
            tags: packed.model().tags().len() as u32,
            packed_config_hash: format!("{:016x}", packed.config_hash()),
            parity_config_hash: format!("{:016x}", parity.config_hash()),
        },
        treebank: Treebank {
            file,
            sha256: sha256_hex(&treebank_bytes),
            sentences: test.len() as u32,
            words: words as u32,
        },
        rounds,
        accuracy,
        drift,
        runs,
    };
    let dir = out.unwrap_or_else(|| Path::new(&after.top).join(POSTAG_DIR));
    let path = super::write(&r, &dir)?;
    for run in &r.runs {
        eprintln!(
            "{:>3} per call, lengths {:>5}: packed {:.0} words/s (p50 {:.3} ms), parity {:.0} words/s (p50 {:.3} ms), \
             {:.2}x",
            run.sentences_per_call,
            run.lengths,
            run.packed.words_per_second,
            run.packed.call_p50_ms,
            run.parity.words_per_second,
            run.parity.call_p50_ms,
            run.packed_over_parity
        );
    }
    println!("{}", path.display());
    Ok(())
}
