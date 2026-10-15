//! `turbo-train-recipes`: runs the MASC recipes.
//!
//! ```text
//! turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
//! turbo-train-recipes fetch-masc --home <dir>
//! ```
//!
//! `masc` checks the extracted corpus against the recorded release, splits
//! it, trains each model, gates it against the OpenNLP model of the same
//! task (read from `--references` by its file name, the zoo's or the
//! recorded one, and checked against its digest), and writes
//! `<out>/<model id>/` with the model, `card.json` and `gate.json`.
//! `--unrecorded-corpus` runs on a corpus directory that is not the
//! recorded release (`masc::RELEASE`), and marks the cards
//! `release_recorded: false`. The run prints the directory's digest.
//! `--recipe` runs only the recipes named, by the trained model's id;
//! without it every recipe runs. The recipes run in the order of
//! `RECIPES`, whatever the order they are named in. Before anything is
//! trained, every reference the run needs is read and checked, every
//! model directory it would write is checked not to exist, and every
//! recipe's training and held-out samples are read and checked
//! (`Recipe::check_data`: skipped sentences, and the held-out tags the
//! tagger's reference cannot give), so a run that fails on one writes
//! nothing. The exit status is 1 when a gate fails
//! and 2 on an error.
//!
//! `fetch-masc` fetches the release archive through the zoo into
//! `<home>/artifacts/`, checked against its recorded size and digests.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::masc::{self, RELEASE, SplitCorpus};
use turbo_train_recipes::{Error, RECIPES, Recipe, run};

const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
  turbo-train-recipes fetch-masc --home <dir>";

struct Args {
    command: String,
    corpus: Option<PathBuf>,
    references: Option<PathBuf>,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
    unrecorded: bool,
    recipes: Vec<Recipe>,
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or(USAGE)?;
    let mut a =
        Args { command, corpus: None, references: None, out: None, home: None, unrecorded: false, recipes: Vec::new() };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => a.corpus = Some(value()?.into()),
            "--references" => a.references = Some(value()?.into()),
            "--out" => a.out = Some(value()?.into()),
            "--home" => a.home = Some(value()?.into()),
            "--unrecorded-corpus" => a.unrecorded = true,
            "--recipe" => {
                let id = value()?;
                let recipe = Recipe::trains(&id).ok_or_else(|| {
                    let ids: Vec<&str> = RECIPES.iter().map(Recipe::model_id).collect();
                    format!("no recipe trains {id}; the recipes train {}", ids.join(", "))
                })?;
                if !a.recipes.contains(&recipe) {
                    a.recipes.push(recipe);
                }
            }
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
    let recorded = RELEASE.is(&digest);
    println!("corpus {}: {} files, digest {}", corpus_dir.display(), digest.files, digest.sha256);
    if !recorded {
        println!("the corpus is not the recorded release: this run is not a release of the models");
    }
    let recipes: Vec<Recipe> = RECIPES.into_iter().filter(|r| a.recipes.is_empty() || a.recipes.contains(r)).collect();
    let corpus = MascCorpus::read(&corpus_dir).map_err(|e| Error::from(e).to_string())?;
    let split = masc::SPLIT.apply(corpus, &corpus_dir).map_err(|e| e.to_string())?;
    println!(
        "{} training documents, {} held out, {} excluded",
        split.train.documents().len(),
        split.held_out.documents().len(),
        split.excluded.len()
    );
    for e in &split.excluded {
        println!("excluded {} ({}: {})", e.path, e.exclusion, e.exclusion.reason());
    }
    let reference_bytes = check_before_training(&recipes, &references, &out, &split)?;
    let mut all_pass = true;
    for (recipe, reference) in recipes.iter().zip(&reference_bytes) {
        let outcome =
            run(recipe, &split, &digest, recorded, recipe.reference_id(), reference).map_err(|e| e.to_string())?;
        write_model_dir(&out, &outcome.card, &outcome.trained.bytes, &outcome.gate).map_err(|e| e.to_string())?;
        println!("{}", outcome.card.summary());
        println!("{}", outcome.gate);
        all_pass &= outcome.gate.result.pass;
    }
    Ok(all_pass)
}

/// Reads and checks the reference of each recipe from `references`,
/// checks that `out` holds none of their model directories, and checks
/// each recipe's data in `split` (`Recipe::check_data`, with the
/// reference when it was read). Returns the references' bytes, in the
/// recipes' order, or every problem found.
fn check_before_training(
    recipes: &[Recipe],
    references: &Path,
    out: &Path,
    split: &SplitCorpus,
) -> Result<Vec<Vec<u8>>, String> {
    let mut bytes = Vec::with_capacity(recipes.len());
    let mut problems = Vec::new();
    for recipe in recipes {
        let dir = out.join(recipe.model_id());
        if dir.exists() {
            problems.push(format!("{}: the model directory exists", dir.display()));
        }
        let read = recipe.reference_file().map_err(|e| e.to_string()).and_then(|file| {
            let path = references.join(file);
            let reference = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            recipe.check_reference(&reference).map_err(|e| e.to_string())?;
            Ok(reference)
        });
        let reference = match read {
            Ok(reference) => Some(reference),
            Err(problem) => {
                problems.push(problem);
                None
            }
        };
        if let Err(e) = recipe.check_data(split, recipe.reference_id(), reference.as_deref()) {
            problems.push(format!("{}: {e}", recipe.model_id()));
        }
        bytes.extend(reference);
    }
    if problems.is_empty() { Ok(bytes) } else { Err(problems.join("\n")) }
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
