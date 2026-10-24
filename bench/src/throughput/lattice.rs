//! The measurement behind `turbo-bench lattice`, built with the
//! `lattice` feature: the lattice tokenizer over the Ja/Ko
//! goldens inputs with the published MeCab dictionaries, as the
//! `lattice_throughput` example times them.

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::Tokens;
use turbo_tokenize::lattice::{DictionaryFormat, Scratch, UnidicLayout};
use turbo_tokenize::{LatticeAnalyzer, LatticeTokenizer, MecabDictionary, WhitespaceMode};

use super::measure::{Options, goldens_text, median_pass, now, out_dir, record, run, same_tree};
use super::{Subject, write};
use crate::cpus::{Cpus, SYSFS_CPUS, pin};
use crate::{BUILD_CHANGES, BUILD_COMMIT, Result, check_build, git};

const USAGE: &str = "turbo-bench lattice [--dictionaries <dir>] [--rounds <n>] [--readings] [--cpus <list>] \
                     [--repo <dir>] [--out <dir>] [ipadic|ko-dic|unidic ...]";

const TEN_MIB: usize = 10 << 20;

struct Setup {
    name: &'static str,
    dir: &'static str,
    inputs: &'static str,
    format: DictionaryFormat,
}

const SETUPS: [Setup; 3] = [
    Setup { name: "ipadic", dir: "mecab-ipadic-2.7.0-20070801.utf8", inputs: "ja", format: DictionaryFormat::Ipadic },
    Setup { name: "ko-dic", dir: "mecab-ko-dic-2.1.1-20180720", inputs: "ko", format: DictionaryFormat::MecabKoDic },
    Setup {
        name: "unidic",
        dir: "unidic-mecab-2.1.2",
        inputs: "ja",
        format: DictionaryFormat::Unidic(UnidicLayout::Source17),
    },
];

pub fn cmd(args: &[String]) -> Result<()> {
    let mut o =
        Options::parse(args, &["--dictionaries", "--rounds", "--cpus", "--repo", "--out"], &["--readings"], USAGE)?;
    let dictionaries = o.take("--dictionaries").map(PathBuf::from);
    let rounds = o.rounds()?;
    let readings = o.flag("--readings");
    let cpus = o.take("--cpus").map(|l| Cpus::parse(&l, Path::new(SYSFS_CPUS))).transpose()?;
    let repo = o.repo();
    let out = o.out();
    let named = std::mem::take(&mut o.positional);
    for n in &named {
        if !SETUPS.iter().any(|s| s.name == n) {
            return Err(format!("{n}: not ipadic, ko-dic or unidic\n{USAGE}"));
        }
    }
    // Refused before anything is measured, and checked again after.
    let before = git::provenance(&repo)?;
    check_build(&before.commit, BUILD_COMMIT, BUILD_CHANGES)?;
    if let Some(c) = &cpus {
        pin(c)?;
    }
    let recorded_at = now();
    let top = Path::new(&before.top).to_owned();
    let dir = dictionaries
        .or_else(|| std::env::var_os("TURBO_MECAB_DICTIONARIES").map(PathBuf::from))
        .unwrap_or_else(|| top.join("target/turbo-goldens/models"));
    let procedure = match &cpus {
        Some(c) => format!("the tool ran pinned to CPUs {}, the tokenizer on one thread", c.list),
        None => "the tool ran unpinned, the tokenizer on one thread".into(),
    };

    for setup in SETUPS.iter().filter(|s| named.is_empty() || named.iter().any(|n| n == s.name)) {
        let path = dir.join(setup.dir);
        let t = Instant::now();
        let dictionary = MecabDictionary::load(&path, WhitespaceMode::Unicode)
            .map_err(|e| format!("{}: {e} (run turbo-goldens fetch-models)", path.display()))?;
        let load_seconds = t.elapsed().as_secs_f64();
        let tokenizer = LatticeTokenizer::new(dictionary, WhitespaceMode::Unicode);
        let subject_hash = tokenizer.dictionary().sha256().to_hex();
        let (text, source) = goldens_text(&top, &[setup.inputs], TEN_MIB)?;
        eprintln!("{} ({}): {} bytes from inputs/{}", setup.name, setup.dir, text.len(), setup.inputs);

        let mut scratch = Scratch::new();
        let mut morphemes = Vec::new();
        let (units, secs) = median_pass(rounds, || {
            morphemes.clear();
            tokenizer.analyze_with(black_box(&text), &mut scratch, &mut morphemes).map_err(|e| e.to_string())?;
            Ok(morphemes.len() as u64)
        })?;
        let mut runs = vec![run("analyze", units, secs, &text)];
        eprintln!("analyze: {units} morphemes in {secs:.4} s (median)");

        if readings {
            let analyzer = LatticeAnalyzer::new(tokenizer, setup.format).map_err(|e| e.to_string())?;
            let pipeline = Analyzer::builder().add(analyzer).build().map_err(|e| e.to_string())?;
            let (units, secs) = median_pass(rounds, || {
                let mut docs = vec![Document::new(text.as_str())];
                pipeline.run(&mut docs)[0].clone().map_err(|e| format!("{e:?}"))?;
                black_box(&docs);
                Ok(docs[0].get::<Tokens>().map_or(0, |t| t.spans.len()) as u64)
            })?;
            runs.push(run("analyzer-readings", units, secs, &text));
            eprintln!("analyzer-readings: {units} tokens in {secs:.4} s (median)");
        }

        let command = format!(
            "cargo run --release -p turbo-bench --features lattice -- lattice --dictionaries <dir> {}\
             {}{} --rounds {rounds}",
            setup.name,
            cpus.as_ref().map_or(String::new(), |c| format!(" --cpus {}", c.list)),
            if readings { " --readings" } else { "" },
        );
        let after = git::provenance(&repo)?;
        same_tree(&before, &after)?;
        let r = record(
            "lattice",
            Subject { name: setup.name.into(), sha256: subject_hash, load_seconds: Some(load_seconds) },
            source,
            &text,
            procedure.clone(),
            command,
            rounds,
            runs,
            &after,
            recorded_at.clone(),
        )?;
        let path = write(&r, &out_dir(out.clone(), &after))?;
        println!("{}", path.display());
    }
    Ok(())
}
