//! The measurement behind `turbo-bench tag`, built with the `postag`
//! feature, which links the tagger and the bundle reader.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use turbo::bundle::sha256_hex;
use turbo::record::{Device, Library, Machine};
use turbo_analysis::Annotator;
use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, Numerics};

use super::{CPU_TIERS, Speed, TAG_DIR, TagModel, TagRecord, Tier};
use crate::api::{Runtime, field};
use crate::cpus::{Cpus, SYSFS_CPUS, pin};
use crate::measure::percentile;
use crate::postag::Treebank;
use crate::postag::measure::{Sentence, conllu, pass};
use crate::{BUILD_CHANGES, BUILD_COMMIT, Result, check_build, git, utc};

const USAGE: &str = "turbo-bench tag --model <bundle-dir | ONLP-BLPT file> --corpus <file.conllu> [--device cpu] \
                     [--precision exact|model|fastest] [--sentences-per-call <n>] [--rounds <n>] [--cpus <list>] \
                     [--repo <dir>] [--out <dir>]";

const OPTIONS: [&str; 9] =
    ["--model", "--corpus", "--device", "--precision", "--sentences-per-call", "--rounds", "--cpus", "--repo", "--out"];

fn options(args: &[String]) -> Result<BTreeMap<String, String>> {
    let mut o = BTreeMap::new();
    let mut it = args.iter();
    while let Some(name) = it.next() {
        let value = it.next().ok_or_else(|| format!("{name}: no value\n{USAGE}"))?;
        if !OPTIONS.contains(&name.as_str()) {
            return Err(format!("{name}: not a tag option\n{USAGE}"));
        }
        if o.insert(name.clone(), value.clone()).is_some() {
            return Err(format!("{name} given twice"));
        }
    }
    Ok(o)
}

/// The model of `path` for a precision: a bundle directory's F32 or F16
/// weights, or an `ONLP-BLPT` file's f64. Its source SHA-256, the
/// bundle manifest's SHA-256 (empty for a file) and the weights' name.
fn load(path: &Path, weights: &[&str]) -> Result<(BilstmPosModel, String, String, &'static str)> {
    let read = |p: &Path| std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()));
    if path.is_dir() {
        let m = turbo_bundle::postag::verify(path, None)?;
        let manifest = sha256_hex(&read(&path.join(turbo_bundle::postag::MANIFEST))?);
        let f16 = weights == ["f16"];
        let model = turbo_bundle::postag::load(path, f16)?;
        return Ok((model, m.source_sha256, manifest, if f16 { "f16" } else { "f32" }));
    }
    if !weights.contains(&"f64") {
        return Err(format!("{}: this precision runs a bundle's F16 weights; give a bundle directory", path.display()));
    }
    let bytes = read(path)?;
    let model = BilstmPosModel::load(bytes.as_slice()).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok((model, sha256_hex(&bytes), String::new(), "f64"))
}

fn speed(passes: &[f64], mut calls: Vec<f64>, words: usize) -> Speed {
    let mut rates: Vec<f64> = passes.iter().map(|s| words as f64 / s).collect();
    rates.sort_by(f64::total_cmp);
    calls.sort_by(f64::total_cmp);
    Speed {
        words_per_second: rates[rates.len() / 2],
        call_p50_ms: percentile(&calls, 50.0),
        call_p99_ms: percentile(&calls, 99.0),
    }
}

pub fn cmd(args: &[String]) -> Result<()> {
    let mut o = options(args)?;
    let model_path = PathBuf::from(o.remove("--model").ok_or(format!("--model is required\n{USAGE}"))?);
    let corpus_path = PathBuf::from(o.remove("--corpus").ok_or(format!("--corpus is required\n{USAGE}"))?);
    let device = o.remove("--device").unwrap_or_else(|| "cpu".into());
    let precision = o.remove("--precision").unwrap_or_else(|| "model".into());
    let num = |name: &str, default: u32, o: &mut BTreeMap<String, String>| -> Result<u32> {
        let v = o.remove(name).map_or(Ok(default), |v| v.parse().map_err(|e| format!("{name} {v}: {e}")))?;
        if v == 0 { Err(format!("{name} 0: nothing would be timed")) } else { Ok(v) }
    };
    let per_call = num("--sentences-per-call", 32, &mut o)?;
    let rounds = num("--rounds", 3, &mut o)?;
    let cpus = o.remove("--cpus").map(|l| Cpus::parse(&l, Path::new(SYSFS_CPUS))).transpose()?;
    let repo = o.remove("--repo").map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join(".."), PathBuf::from);
    let out = o.remove("--out").map(PathBuf::from);
    if device != "cpu" {
        return Err(format!("--device {device}: no {device} tier of the tagger exists yet; the cpu tiers do"));
    }
    let &(_, numerics_name, weights) = CPU_TIERS
        .iter()
        .find(|(p, _, _)| *p == precision)
        .ok_or(format!("--precision {precision}: not exact, model or fastest"))?;
    let numerics = if numerics_name == "Packed" { Numerics::Packed } else { Numerics::JvmParity };
    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    check_build(&before.commit, BUILD_COMMIT, BUILD_CHANGES)?;
    if let Some(c) = &cpus {
        pin(c)?;
    }
    let recorded_at = utc(SystemTime::now());

    let corpus_bytes = std::fs::read(&corpus_path).map_err(|e| format!("{}: {e}", corpus_path.display()))?;
    let text = std::str::from_utf8(&corpus_bytes).map_err(|e| format!("{}: {e}", corpus_path.display()))?;
    let test = conllu(text);
    if test.is_empty() {
        return Err(format!("{}: no sentence", corpus_path.display()));
    }
    let words: usize = test.iter().map(|s| s.words.len()).sum();
    let all: Vec<&Sentence> = test.iter().collect();

    // Cold: the first pass of a tagger loaded just before it.
    let (model, source_sha256, bundle_sha256, weights_name) = load(&model_path, weights)?;
    let tagger = BilstmPosTagger::with_numerics(model, numerics).map_err(|e| format!("{numerics:?}: {e}"))?;
    let (cold_secs, cold_calls) = pass(&tagger, &all, per_call as usize)?;
    let cold = speed(&[cold_secs], cold_calls, words);
    let (mut passes, mut calls) = (Vec::new(), Vec::new());
    for round in 0..rounds {
        let (secs, c) = pass(&tagger, &all, per_call as usize)?;
        passes.push(secs);
        calls.extend(c);
        eprintln!("warm pass {} of {rounds}: {:.0} words/s", round + 1, words as f64 / secs);
    }
    let warm = speed(&passes, calls, words);
    let mut correct = 0u32;
    for s in &test {
        let tags = tagger.tag(&s.words).map_err(|e| e.to_string())?;
        correct += tags.iter().zip(&s.tags).filter(|(a, b)| a == b).count() as u32;
    }

    let after = git::provenance(&repo)?;
    if after != before {
        return Err(format!("the working tree moved from {} to {} during the run", before.commit, after.commit));
    }
    let rt = Runtime::create()?;
    let d = rt.cpu()?.ok_or("this build lists no CPU device")?;
    let file = corpus_path.file_name().and_then(|f| f.to_str()).unwrap_or("").to_owned();
    let model_ref = tagger.model().model_ref();
    let r = TagRecord {
        record_version: super::TAG_RECORD_VERSION,
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
            "cargo run --release -p turbo-bench --features postag -- tag --model <{}> --corpus <dir>/{file} --device \
             {device} --precision {precision} --sentences-per-call {per_call} --rounds {rounds}{}",
            if bundle_sha256.is_empty() { "model" } else { "bundle" },
            cpus.as_ref().map_or(String::new(), |c| format!(" --cpus {}", c.list))
        ),
        model: TagModel {
            id: model_ref.id.to_string(),
            revision: model_ref.revision.to_string(),
            sha256: model_ref.sha256.to_string(),
            source_sha256,
            bundle_sha256,
            weights: weights_name.into(),
            tags: tagger.model().tags().len() as u32,
        },
        tier: Tier {
            device,
            precision,
            numerics: numerics_name.into(),
            config_hash: format!("{:016x}", tagger.config_hash()),
        },
        corpus: Treebank { file, sha256: sha256_hex(&corpus_bytes), sentences: test.len() as u32, words: words as u32 },
        sentences_per_call: per_call,
        rounds,
        correct,
        cold,
        warm,
    };
    let dir = out.unwrap_or_else(|| Path::new(&after.top).join(TAG_DIR));
    let path = super::write(&r, &dir)?;
    eprintln!(
        "{} {}: {correct}/{words} correct; cold {:.0} words/s (p50 {:.3} ms, p99 {:.3} ms); warm {:.0} words/s (p50 \
         {:.3} ms, p99 {:.3} ms)",
        r.tier.device,
        r.tier.precision,
        r.cold.words_per_second,
        r.cold.call_p50_ms,
        r.cold.call_p99_ms,
        r.warm.words_per_second,
        r.warm.call_p50_ms,
        r.warm.call_p99_ms
    );
    println!("{}", path.display());
    Ok(())
}
