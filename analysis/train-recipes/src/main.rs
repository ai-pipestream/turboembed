//! `turbo-train-recipes`: runs the MASC recipes and the UD recipes.
//!
//! ```text
//! turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
//! turbo-train-recipes fetch-masc --home <dir>
//! turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
//! turbo-train-recipes fetch-ud --home <dir>
//! ```
//!
//! `masc` checks the extracted corpus against the recorded release, splits
//! it, trains each model, gates it against the OpenNLP model of the same
//! task (read from `--references` by its file name, the zoo's or the
//! recorded one, and checked against its digest), and writes
//! `<out>/<model id>/` with the model, `card.json` and `gate.json`.
//! `--unrecorded-corpus` runs on a copy whose digest is not recorded yet,
//! prints that digest, and marks the cards `release_recorded: false`.
//! `--recipe` runs only the recipes named, by the trained model's id;
//! without it every recipe runs. The exit status is 1 when a gate fails
//! and 2 on an error.
//!
//! `fetch-masc` fetches the release archive through the zoo into
//! `<home>/artifacts/`, once its digests are recorded.
//!
//! `ud` reads every reference model the chosen recipes are gated against
//! from `--references` and checks it against the zoo's digest, then reads
//! each treebank from `<treebanks>/<dataset id>/` (the layout `fetch-ud`
//! writes, `<home>/artifacts`) with its licence check and file digests,
//! all before any training; it then trains each model, gates it, and
//! writes `<out>/<model id>/` with the model, `card.json` and `gate.json`.
//! A model with no reference is scored alone. The exit status is 1 when a
//! gate fails and 2 on an error. `fetch-ud` fetches the treebanks' files
//! at the release through the zoo into `<home>/artifacts/<dataset id>/`.

use std::path::PathBuf;
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::masc::{self, RELEASE, Release};
use turbo_train_recipes::treebank::{TREEBANKS, Treebank};
use turbo_train_recipes::ud::{UD_RECIPES, UdRecipe, run_ud};
use turbo_train_recipes::ud_card::write_ud_model_dir;
use turbo_train_recipes::{Error, RECIPES, Recipe, run};

const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
  turbo-train-recipes fetch-masc --home <dir>
  turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
  turbo-train-recipes fetch-ud --home <dir>";

struct Args {
    command: String,
    corpus: Option<PathBuf>,
    treebanks: Option<PathBuf>,
    references: Option<PathBuf>,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
    unrecorded: bool,
    /// The model ids `--recipe` names, resolved by the command.
    recipe_ids: Vec<String>,
}

/// The recipes of `all` that `ids` name, each once, in the order named;
/// all of them when `ids` is empty.
fn chosen<R: Copy + PartialEq>(
    all: &[R],
    ids: &[String],
    model_id: impl Fn(&R) -> &'static str,
) -> Result<Vec<R>, String> {
    if ids.is_empty() {
        return Ok(all.to_vec());
    }
    let mut out = Vec::new();
    for id in ids {
        let recipe = all.iter().find(|r| model_id(r) == id).ok_or_else(|| {
            let known: Vec<&str> = all.iter().map(&model_id).collect();
            format!("no recipe trains {id}; the recipes train {}", known.join(", "))
        })?;
        if !out.contains(recipe) {
            out.push(*recipe);
        }
    }
    Ok(out)
}

fn parse() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let command = it.next().ok_or(USAGE)?;
    let mut a = Args {
        command,
        corpus: None,
        treebanks: None,
        references: None,
        out: None,
        home: None,
        unrecorded: false,
        recipe_ids: Vec::new(),
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--corpus" => a.corpus = Some(value()?.into()),
            "--treebanks" => a.treebanks = Some(value()?.into()),
            "--references" => a.references = Some(value()?.into()),
            "--out" => a.out = Some(value()?.into()),
            "--home" => a.home = Some(value()?.into()),
            "--unrecorded-corpus" => a.unrecorded = true,
            "--recipe" => a.recipe_ids.push(value()?),
            _ => return Err(format!("unknown argument {flag}\n{USAGE}")),
        }
    }
    Ok(a)
}

fn need(v: Option<PathBuf>, flag: &str) -> Result<PathBuf, String> {
    v.ok_or_else(|| format!("{flag} is required\n{USAGE}"))
}

fn masc_command(a: Args) -> Result<bool, String> {
    let recipes = chosen(&RECIPES, &a.recipe_ids, Recipe::model_id)?;
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
    for recipe in &recipes {
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

fn user() -> Result<turbo_zoo::UserId, String> {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .map_err(|_| "USER is not set".to_owned())?
        .parse()
        .map_err(|e| format!("USER: {e}"))
}

fn fetch_command(a: Args) -> Result<bool, String> {
    let home = need(a.home, "--home")?;
    let path = masc::fetch_release(RELEASE, &home, &user()?).map_err(|e| e.to_string())?;
    println!("{}", path.display());
    Ok(true)
}

fn ud_command(a: Args) -> Result<bool, String> {
    let recipes = chosen(&UD_RECIPES, &a.recipe_ids, |r: &UdRecipe| r.model_id)?;
    let treebanks = need(a.treebanks, "--treebanks")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    // Every reference and every treebank is read and checked before any
    // model is trained.
    let mut reference_files = Vec::with_capacity(recipes.len());
    for recipe in &recipes {
        let bytes = match recipe.reference_model() {
            Some(reference) => {
                let path = references.join(reference.file().map_err(|e| e.to_string())?);
                let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                reference.check(&bytes).map_err(|e| e.to_string())?;
                Some(bytes)
            }
            None => None,
        };
        reference_files.push(bytes);
    }
    let mut texts = Vec::new();
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        let dir = treebanks.join(t.dataset_id());
        let text = t.read(&dir).map_err(|e| e.to_string())?;
        println!("{} {}: licence and digests checked", t.repository, t.dataset_id());
        texts.push((*t, text));
    }
    let text_of = |t: &Treebank| &texts.iter().find(|(u, _)| u == t).expect("every treebank was read").1;
    let mut all_pass = true;
    for (recipe, reference) in recipes.iter().zip(&reference_files) {
        let started = std::time::Instant::now();
        let outcome = run_ud(recipe, text_of(&recipe.treebank), reference.as_deref()).map_err(|e| e.to_string())?;
        write_ud_model_dir(&out, &outcome.card, &outcome.trained.bytes, &outcome.gate).map_err(|e| e.to_string())?;
        println!(
            "{}: {} samples, {} sentences skipped, {} events, {} held-out samples, {:.0} s",
            recipe.model_id,
            outcome.trained.samples,
            outcome.trained.skipped_sentences,
            outcome.trained.events,
            outcome.card.treebank.held_out_samples,
            started.elapsed().as_secs_f64()
        );
        println!("{}", outcome.gate);
        // A model scored alone has no gate to fail.
        all_pass &= outcome.gate.passed() || recipe.reference_model().is_none();
    }
    Ok(all_pass)
}

fn fetch_ud_command(a: Args) -> Result<bool, String> {
    let home = need(a.home, "--home")?;
    let user = user()?;
    for t in TREEBANKS {
        let dir = t.fetch(&home, &user).map_err(|e| e.to_string())?;
        println!("{}", dir.display());
    }
    Ok(true)
}

fn main() -> ExitCode {
    let result = parse().and_then(|a| match a.command.as_str() {
        "masc" => masc_command(a),
        "fetch-masc" => fetch_command(a),
        "ud" => ud_command(a),
        "fetch-ud" => fetch_ud_command(a),
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
