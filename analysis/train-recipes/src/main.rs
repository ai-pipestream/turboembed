//! `turbo-train-recipes`: runs the MASC recipes, the UD recipes and the
//! language detector recipe.
//!
//! ```text
//! turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
//! turbo-train-recipes fetch-masc --home <dir>
//! turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
//! turbo-train-recipes fetch-ud --home <dir> [--recipe <model id>]...
//! turbo-train-recipes zoo-index --models <dir> --release <tag> [--check]
//! turbo-train-recipes tatoeba --export <dir> --references <dir> --out <dir> [--unrecorded-export]
//! turbo-train-recipes fineweb-ranges (--paths | --web <dir> [--files <path>] (--need | --record))
//! turbo-train-recipes tatoeba-fineweb --export <dir> --web <dir> --references <dir> --out <dir>
//!     [--unrecorded-export] [--files <path> --ranges <path> --unrecorded-web]
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
//! a gate fails and 2 on an error. `fetch-ud` fetches the files at the
//! release of the treebanks the recipes `--recipe` names read (of every
//! treebank without `--recipe`) through the zoo into
//! `<home>/artifacts/<dataset id>/`.
//!
//! `zoo-index` reads the directories `masc`, `ud`, `tatoeba` or
//! `tatoeba-fineweb` wrote into `--models` for the models the release
//! `--release` publishes (`release::MODELS_RELEASES`), checks each file
//! (the model, the language detectors' `training-sentences.txt` and the
//! web one's `training-documents.txt`) and each gate and comparison
//! record against its card, and prints the zoo's built-in index with each model's entry
//! at its address in the release (`release::index_with`). With `--check`
//! it prints nothing of the index and exits with 1 unless the built-in
//! index is already that one and lists nothing else at the release's
//! addresses, naming each entry that differs with the size and SHA-256 of
//! each file the index lists and the model has, and each other model
//! listed there (`release::differences`).
//!
//! `tatoeba` runs the language detector recipe (`langdetect::LANGDETECT`)
//! on the Tatoeba export extracted into `--export` (`sentences_detailed.csv`
//! and `sentences_CC0.csv`), gates it against `langdetect-183.bin` read
//! from `--references` and checked against its recorded size and digest,
//! and writes `<out>/<model id>/` with the model,
//! `training-sentences.txt` (the ids of the sentences trained on),
//! `card.json` and `gate.json`. It refuses any export but the recorded
//! one (`langdetect::EXPORT`) unless given `--unrecorded-export`; the
//! card of another export is marked `recorded: false`, so its model is
//! not a release. The exit status is 1 when the gate fails and 2 on an
//! error.
//!
//! `fineweb-ranges` fetches nothing itself; it says what to fetch of the
//! FineWeb-2 and FineWeb files the web language detector reads
//! (`fineweb`). `--paths` prints each file's repository, revision and
//! path, a line each, for the files list to be made from the hub's API.
//! `--need` prints, for each file of the files list (`--files`, or the
//! committed `fineweb-files.txt`), the next range the bytes already in
//! `--web` give and it does not hold: its address, the file to save it
//! as, its first byte and its end, tab separated; nothing once every
//! range is there. `--record` prints the ranges list of what `--web`
//! holds, each range with its SHA-256.
//!
//! `tatoeba-fineweb` runs the web language detector recipe
//! (`langdetect_web::LANGDETECT_WEB`): the Tatoeba export as `tatoeba`
//! reads it, and the web ranges fetched into `--web`, checked against the
//! committed lists, or against `--files` and `--ranges`, given together,
//! with `--unrecorded-web` (a model trained on those is not a release).
//! The web ranges are checked before the export is read. It
//! reads `langdetect-183.bin` and the Tatoeba recipe's model as the zoo
//! lists it (`turbo-mul-tatoeba-langdetect-1.bin`) from `--references`,
//! each checked against its digest, gates the model against the first on
//! both held-out sets, scores it beside the second, and writes
//! `<out>/<model id>/` with the model, `training-sentences.txt`,
//! `training-documents.txt`, `card.json`, `gate.json`, `gate-web.json`,
//! `compare-tatoeba.json` and `compare-web.json`. The exit status is 1
//! when a gate fails and 2 on an error.

mod cli;
mod commands;

use std::path::Path;
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::masc::{self, RELEASE, SplitCorpus};
use turbo_train_recipes::release;
use turbo_train_recipes::{Error, RECIPES, Recipe, Reference, run};
use turbo_zoo::Index;

use cli::{Args, USAGE, chosen, need, only, parse};
use commands::{fetch_ud_command, fineweb_ranges_command, tatoeba_command, tatoeba_fineweb_command, ud_command};

fn masc_command(a: Args) -> Result<bool, String> {
    only(&a, "masc", &["--corpus", "--references", "--out", "--unrecorded-corpus", "--recipe"])?;
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
    let read = if recipes.iter().any(Recipe::reads_chunks) { MascCorpus::read_with_chunks } else { MascCorpus::read };
    let corpus = read(&corpus_dir).map_err(|e| Error::from(e).to_string())?;
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
    let reference_bytes = check_before_training(&recipes, &references, &out, &split, recorded)?;
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
pub(crate) fn read_reference(reference: Reference, references: &Path, problems: &mut Vec<String>) -> Option<Vec<u8>> {
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
/// reference when it was read, and `recorded` when the corpus is the
/// recorded release). Returns the references' bytes, in the recipes'
/// order, or every problem found.
fn check_before_training(
    recipes: &[Recipe],
    references: &Path,
    out: &Path,
    split: &SplitCorpus,
    recorded: bool,
) -> Result<Vec<Vec<u8>>, String> {
    let mut bytes = Vec::with_capacity(recipes.len());
    let mut problems = Vec::new();
    for recipe in recipes {
        let dir = out.join(recipe.model_id());
        if dir.exists() {
            problems.push(format!("{}: the model directory exists", dir.display()));
        }
        let reference = read_reference(recipe.reference(), references, &mut problems);
        if let Err(e) = recipe.check_data(split, recorded, recipe.reference_id(), reference.as_deref()) {
            problems.push(format!("{}: {e}", recipe.model_id()));
        }
        bytes.extend(reference);
    }
    if problems.is_empty() { Ok(bytes) } else { Err(problems.join("\n")) }
}

pub(crate) fn user() -> Result<turbo_zoo::UserId, String> {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .map_err(|_| "USER is not set".to_owned())?
        .parse()
        .map_err(|e| format!("USER: {e}"))
}

fn fetch_command(a: Args) -> Result<bool, String> {
    only(&a, "fetch-masc", &["--home"])?;
    let home = need(a.home, "--home")?;
    let path = masc::fetch_release(RELEASE, &home, &user()?).map_err(|e| e.to_string())?;
    println!("{}", path.display());
    Ok(true)
}

fn zoo_index_command(a: Args) -> Result<bool, String> {
    only(&a, "zoo-index", &["--models", "--release", "--check"])?;
    let models = need(a.models, "--models")?;
    let release = a.release.ok_or_else(|| format!("--release is required\n{USAGE}"))?;
    let tag = release.tag;
    let entries = release::entries(&models, &release).map_err(|e| e.to_string())?;
    let index = release::index_with(&Index::builtin(), &entries);
    if !a.check {
        print!("{index}");
        return Ok(true);
    }
    let lines = release::differences(&Index::builtin(), &entries, &release);
    if lines.is_empty() && index == turbo_zoo::index::BUILTIN {
        println!("the zoo's built-in index lists the {} models at their addresses in release {tag}", entries.len());
        return Ok(true);
    }
    for line in lines {
        println!("{line}");
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
        "tatoeba" => tatoeba_command(a),
        "fineweb-ranges" => fineweb_ranges_command(a),
        "tatoeba-fineweb" => tatoeba_fineweb_command(a),
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
