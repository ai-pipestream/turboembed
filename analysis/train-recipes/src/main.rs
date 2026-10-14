//! `turbo-train-recipes`: runs the MASC recipes.
//!
//! ```text
//! turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus]
//! turbo-train-recipes fetch-masc --home <dir>
//! ```
//!
//! `masc` checks the extracted corpus against the recorded release, splits
//! it, trains each model, gates it against the OpenNLP model of the same
//! task (read from `--references`, by the file name the zoo gives it, and
//! checked against the zoo's digest), and writes `<out>/<model id>/` with
//! the model, `card.json` and `gate.json`. `--unrecorded-corpus` runs on a
//! copy whose digest is not recorded yet, prints that digest, and marks
//! the cards `release_recorded: false`. The exit status is 1 when a gate
//! fails and 2 on an error.
//!
//! `fetch-masc` fetches the release archive through the zoo into
//! `<home>/artifacts/`, once its digests are recorded.

use std::path::PathBuf;
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::masc::{self, RELEASE, Release};
use turbo_train_recipes::{Error, RECIPES, run};

const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus]
  turbo-train-recipes fetch-masc --home <dir>";

struct Args {
    command: String,
    corpus: Option<PathBuf>,
    references: Option<PathBuf>,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
    unrecorded: bool,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or(USAGE)?;
    let mut a = Args { command, corpus: None, references: None, out: None, home: None, unrecorded: false };
    while let Some(flag) = it.next() {
        let mut value = || it.next().map(PathBuf::from).ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => a.corpus = Some(value()?),
            "--references" => a.references = Some(value()?),
            "--out" => a.out = Some(value()?),
            "--home" => a.home = Some(value()?),
            "--unrecorded-corpus" => a.unrecorded = true,
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    Ok(a)
}

fn need(v: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    v.ok_or_else(|| format!("{flag} is required\n{USAGE}"))
}

fn masc_command(a: Args) -> Result<bool, String> {
    let corpus_dir = need(a.corpus, "--corpus")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    let digest = masc::check_corpus(&corpus_dir, RELEASE, a.unrecorded).map_err(|e| e.to_string())?;
    let recorded = matches!(RELEASE, Release::Recorded(_));
    println!("corpus {}: {} files, digest {}", corpus_dir.display(), digest.files, digest.sha256);
    if !recorded {
        println!("the release is not recorded: this run is not a release of the models");
    }
    let corpus = MascCorpus::read(&corpus_dir).map_err(|e| Error::from(e).to_string())?;
    let split = masc::SPLIT.apply(corpus, &corpus_dir).map_err(|e| e.to_string())?;
    println!(
        "{} training documents, {} held out, {} skipped by the reader",
        split.train.documents().len(),
        split.held_out.documents().len(),
        split.skipped_paths.len()
    );
    let mut all_pass = true;
    for recipe in &RECIPES {
        let file = recipe.reference_file().map_err(|e| e.to_string())?;
        let path = references.join(&file);
        let reference = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        recipe.check_reference(&reference).map_err(|e| e.to_string())?;
        let outcome =
            run(recipe, &split, &digest, recorded, recipe.reference_id(), &reference).map_err(|e| e.to_string())?;
        write_model_dir(&out, &outcome.card, &outcome.trained.bytes, &outcome.gate).map_err(|e| e.to_string())?;
        println!(
            "{}: {} samples, {} sentences skipped, {} events",
            recipe.model_id(),
            outcome.trained.samples,
            outcome.trained.skipped_sentences,
            outcome.trained.events
        );
        println!("{}", outcome.gate);
        all_pass &= outcome.gate.result.pass;
    }
    Ok(all_pass)
}

fn fetch_command(a: Args) -> Result<bool, String> {
    let home = need(a.home, "--home")?;
    let user = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .map_err(|_| "USER is not set".to_owned())?
        .parse()
        .map_err(|e| format!("USER: {e}"))?;
    let path = masc::fetch_release(RELEASE, &home, &user).map_err(|e| e.to_string())?;
    println!("{}", path.display());
    Ok(true)
}

fn main() -> ExitCode {
    let result = parse().and_then(|a| match a.command.as_str() {
        "masc" => masc_command(a),
        "fetch-masc" => fetch_command(a),
        _ => Err(USAGE.to_owned()),
    });
    match result {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(2)
        }
    }
}
