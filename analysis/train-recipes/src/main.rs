//! `turbo-train-recipes`: runs the MASC recipes and the UD recipes.
//!
//! ```text
//! turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
//! turbo-train-recipes fetch-masc --home <dir>
//! turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
//! turbo-train-recipes fetch-ud --home <dir>
//! turbo-train-recipes zoo-index --models <dir> --release <tag> [--check]
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
//!
//! `ud` runs the UD recipes the same way, in the order of `UD_RECIPES`.
//! Before anything is trained, it reads every reference model the chosen
//! recipes are gated against from `--references` and checks it against
//! the zoo's digest or the recorded one, checks that none of their model
//! directories exists, reads each treebank from `<treebanks>/<dataset
//! id>/` (the layout `fetch-ud` writes, `<home>/artifacts`) with its
//! licence check and file digests, and checks every recipe's training and
//! held-out samples (`UdRecipe::check_data`, as `Recipe::check_data`). It
//! then trains each model, gates it, writes `<out>/<model id>/` with the
//! model, `card.json` and `gate.json`, and prints the card's summary, the
//! gate record and the seconds the recipe took. The exit status is 1 when
//! a gate fails and 2 on an error. `fetch-ud` fetches the treebanks'
//! files at the release through the zoo into
//! `<home>/artifacts/<dataset id>/`.
//!
//! `zoo-index` reads the model directories `masc` wrote into `--models`,
//! checks each model file against its card, and prints the zoo's
//! built-in index with each model's entry at its address in release
//! `--release` (`release::index_with`). With `--check` it prints nothing
//! of the index and exits with 1 unless the built-in index is already
//! that one, naming the entries that differ.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::masc::{self, RELEASE, SplitCorpus};
use turbo_train_recipes::release::{self, ReleaseTag};
use turbo_train_recipes::treebank::{TREEBANKS, Treebank, TreebankText};
use turbo_train_recipes::ud::{UD_RECIPES, UdRecipe, run_ud};
use turbo_train_recipes::ud_card::write_ud_model_dir;
use turbo_train_recipes::{Error, RECIPES, Recipe, Reference, run};
use turbo_zoo::Index;

const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
  turbo-train-recipes fetch-masc --home <dir>
  turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
  turbo-train-recipes fetch-ud --home <dir>
  turbo-train-recipes zoo-index --models <dir> --release <tag> [--check]";

struct Args {
    command: String,
    corpus: Option<PathBuf>,
    treebanks: Option<PathBuf>,
    references: Option<PathBuf>,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
    models: Option<PathBuf>,
    release: Option<ReleaseTag>,
    unrecorded: bool,
    check: bool,
    /// The model ids `--recipe` names, resolved by the command.
    recipe_ids: Vec<String>,
}

/// The recipes of `all` that `ids` name, in the order of `all` whatever
/// the order of `ids`; all of them when `ids` is empty.
fn chosen<R: Copy>(all: &[R], ids: &[String], model_id: impl Fn(&R) -> &'static str) -> Result<Vec<R>, String> {
    if let Some(id) = ids.iter().find(|id| !all.iter().any(|r| model_id(r) == id.as_str())) {
        let known: Vec<&str> = all.iter().map(&model_id).collect();
        return Err(format!("no recipe trains {id}; the recipes train {}", known.join(", ")));
    }
    Ok(all.iter().filter(|r| ids.is_empty() || ids.iter().any(|id| model_id(r) == id.as_str())).copied().collect())
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
        models: None,
        release: None,
        unrecorded: false,
        check: false,
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
            "--models" => a.models = Some(value()?.into()),
            "--release" => a.release = Some(value()?.parse().map_err(|e: Error| e.to_string())?),
            "--unrecorded-corpus" => a.unrecorded = true,
            "--check" => a.check = true,
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
    let recorded = RELEASE.is(&digest);
    println!("corpus {}: {} files, digest {}", corpus_dir.display(), digest.files, digest.sha256);
    if !recorded {
        println!("the corpus is not the recorded release: this run is not a release of the models");
    }
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

/// Reads `reference` from `references` and checks it, adding what is
/// wrong to `problems`.
fn read_reference(reference: Reference, references: &Path, problems: &mut Vec<String>) -> Option<Vec<u8>> {
    let read = reference.file().map_err(|e| e.to_string()).and_then(|file| {
        let path = references.join(file);
        let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        reference.check(&bytes).map_err(|e| e.to_string())?;
        Ok(bytes)
    });
    read.map_err(|problem| problems.push(problem)).ok()
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
        let reference = read_reference(recipe.reference(), references, &mut problems);
        if let Err(e) = recipe.check_data(split, recipe.reference_id(), reference.as_deref()) {
            problems.push(format!("{}: {e}", recipe.model_id()));
        }
        bytes.extend(reference);
    }
    if problems.is_empty() { Ok(bytes) } else { Err(problems.join("\n")) }
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

/// Reads and checks the reference of each UD recipe from `references`,
/// and checks that `out` holds none of their model directories. Returns
/// the references' bytes, in the recipes' order, or every problem found.
fn check_ud_references(recipes: &[UdRecipe], references: &Path, out: &Path) -> Result<Vec<Vec<u8>>, String> {
    let mut bytes = Vec::with_capacity(recipes.len());
    let mut problems = Vec::new();
    for recipe in recipes {
        let dir = out.join(recipe.model_id);
        if dir.exists() {
            problems.push(format!("{}: the model directory exists", dir.display()));
        }
        bytes.extend(read_reference(recipe.reference, references, &mut problems));
    }
    if problems.is_empty() { Ok(bytes) } else { Err(problems.join("\n")) }
}

/// Checks each UD recipe's data (`UdRecipe::check_data`) in the text of
/// its treebank, with its reference. Returns every problem found.
fn check_ud_data<'t>(
    recipes: &[UdRecipe],
    references: &[Vec<u8>],
    text_of: impl Fn(&Treebank) -> &'t TreebankText,
) -> Result<(), String> {
    let problems: Vec<String> = recipes
        .iter()
        .zip(references)
        .filter_map(|(r, reference)| {
            r.check_data(text_of(&r.treebank), reference).err().map(|e| format!("{}: {e}", r.model_id))
        })
        .collect();
    if problems.is_empty() { Ok(()) } else { Err(problems.join("\n")) }
}

fn ud_command(a: Args) -> Result<bool, String> {
    let recipes = chosen(&UD_RECIPES, &a.recipe_ids, |r: &UdRecipe| r.model_id)?;
    let treebanks = need(a.treebanks, "--treebanks")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    let reference_bytes = check_ud_references(&recipes, &references, &out)?;
    let mut texts = Vec::new();
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        let dir = treebanks.join(t.dataset_id());
        let text = t.read(&dir).map_err(|e| e.to_string())?;
        println!("{} {}: licence and digests checked", t.repository, t.dataset_id());
        texts.push((*t, text));
    }
    let text_of = |t: &Treebank| &texts.iter().find(|(u, _)| u == t).expect("every treebank was read").1;
    check_ud_data(&recipes, &reference_bytes, text_of)?;
    let mut all_pass = true;
    for (recipe, reference) in recipes.iter().zip(&reference_bytes) {
        let started = std::time::Instant::now();
        let outcome = run_ud(recipe, text_of(&recipe.treebank), reference).map_err(|e| e.to_string())?;
        write_ud_model_dir(&out, &outcome.card, &outcome.trained.bytes, &outcome.gate).map_err(|e| e.to_string())?;
        println!("{}", outcome.card.summary());
        println!("{}", outcome.gate);
        println!("{}: {:.0} s", recipe.model_id, started.elapsed().as_secs_f64());
        all_pass &= outcome.gate.result.pass;
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

fn zoo_index_command(a: Args) -> Result<bool, String> {
    let models = need(a.models, "--models")?;
    let tag = a.release.ok_or_else(|| format!("--release is required\n{USAGE}"))?;
    let entries = release::entries(&models, &tag).map_err(|e| e.to_string())?;
    let index = release::index_with(&Index::builtin(), &entries);
    if !a.check {
        print!("{index}");
        return Ok(true);
    }
    if index == turbo_zoo::index::BUILTIN {
        println!("the zoo's built-in index lists the {} models at their addresses in release {tag}", entries.len());
        return Ok(true);
    }
    let builtin = Index::builtin();
    for entry in &entries {
        match builtin.get(entry.id.as_str()) {
            Ok(listed) if listed == entry => {}
            Ok(_) => println!("{}: the built-in index lists another entry", entry.id),
            Err(_) => println!("{}: the built-in index does not list it", entry.id),
        }
    }
    println!("the zoo's built-in index is not the one these models and release {tag} give");
    Ok(false)
}

fn main() -> ExitCode {
    let result = parse().and_then(|a| match a.command.as_str() {
        "masc" => masc_command(a),
        "fetch-masc" => fetch_command(a),
        "ud" => ud_command(a),
        "fetch-ud" => fetch_ud_command(a),
        "zoo-index" => zoo_index_command(a),
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
