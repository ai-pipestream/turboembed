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
//! `zoo-index` reads the directories `masc`, `ud` or `tatoeba` wrote into
//! `--models` for the models the release `--release` publishes
//! (`release::MODELS_RELEASES`), checks each file (the model, and the
//! language detector's `training-sentences.txt`) and gate record against
//! its card, and prints the zoo's built-in index with each model's entry
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
//! committed lists, or against `--files` and `--ranges` with
//! `--unrecorded-web` (a model trained on those is not a release). It
//! reads `langdetect-183.bin` and the Tatoeba recipe's model as the zoo
//! lists it (`turbo-mul-tatoeba-langdetect-1.bin`) from `--references`,
//! each checked against its digest, gates the model against the first on
//! both held-out sets, scores it beside the second, and writes
//! `<out>/<model id>/` with the model, `training-sentences.txt`,
//! `training-documents.txt`, `card.json`, `gate.json`, `gate-web.json`,
//! `compare-tatoeba.json` and `compare-web.json`. The exit status is 1
//! when a gate fails and 2 on an error.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::card::write_model_dir;
use turbo_train_recipes::fineweb::{self, Store, WEB, WebFile, WebRange};
use turbo_train_recipes::langdetect::{self, LANGDETECT, write_language_dir};
use turbo_train_recipes::langdetect_web::{self, LANGDETECT_WEB, WebInput, write_web_language_dir};
use turbo_train_recipes::masc::{self, RELEASE, SplitCorpus};
use turbo_train_recipes::release::{self, ModelsRelease};
use turbo_train_recipes::tatoeba::Export;
use turbo_train_recipes::treebank::{TREEBANKS, Treebank, TreebankText};
use turbo_train_recipes::ud::{UD_RECIPES, UdRecipe, run_ud};
use turbo_train_recipes::ud_card::write_ud_model_dir;
use turbo_train_recipes::{Error, RECIPES, Recipe, Reference, run};
use turbo_zoo::Index;

const USAGE: &str = "usage:
  turbo-train-recipes masc --corpus <dir> --references <dir> --out <dir> [--unrecorded-corpus] [--recipe <model id>]...
  turbo-train-recipes fetch-masc --home <dir>
  turbo-train-recipes ud --treebanks <dir> --references <dir> --out <dir> [--recipe <model id>]...
  turbo-train-recipes fetch-ud --home <dir> [--recipe <model id>]...
  turbo-train-recipes zoo-index --models <dir> --release <tag> [--check]
  turbo-train-recipes tatoeba --export <dir> --references <dir> --out <dir> [--unrecorded-export]
  turbo-train-recipes fineweb-ranges (--paths | --web <dir> [--files <path>] (--need | --record))
  turbo-train-recipes tatoeba-fineweb --export <dir> --web <dir> --references <dir> --out <dir>
      [--unrecorded-export] [--files <path> --ranges <path> --unrecorded-web]";

struct Args {
    command: String,
    corpus: Option<PathBuf>,
    treebanks: Option<PathBuf>,
    references: Option<PathBuf>,
    out: Option<PathBuf>,
    home: Option<PathBuf>,
    models: Option<PathBuf>,
    release: Option<ModelsRelease>,
    export: Option<PathBuf>,
    unrecorded: bool,
    unrecorded_export: bool,
    check: bool,
    web: Option<PathBuf>,
    files: Option<PathBuf>,
    ranges: Option<PathBuf>,
    unrecorded_web: bool,
    paths: bool,
    need: bool,
    record: bool,
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
        export: None,
        unrecorded: false,
        unrecorded_export: false,
        check: false,
        web: None,
        files: None,
        ranges: None,
        unrecorded_web: false,
        paths: false,
        need: false,
        record: false,
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
            "--release" => a.release = Some(ModelsRelease::tagged(&value()?).map_err(|e| e.to_string())?),
            "--unrecorded-corpus" => a.unrecorded = true,
            "--check" => a.check = true,
            "--export" => a.export = Some(value()?.into()),
            "--unrecorded-export" => a.unrecorded_export = true,
            "--recipe" => a.recipe_ids.push(value()?),
            "--web" => a.web = Some(value()?.into()),
            "--files" => a.files = Some(value()?.into()),
            "--ranges" => a.ranges = Some(value()?.into()),
            "--unrecorded-web" => a.unrecorded_web = true,
            "--paths" => a.paths = true,
            "--need" => a.need = true,
            "--record" => a.record = true,
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
    let recipes = chosen(&UD_RECIPES, &a.recipe_ids, |r: &UdRecipe| r.model_id)?;
    let home = need(a.home, "--home")?;
    let user = user()?;
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        let dir = t.fetch(&home, &user).map_err(|e| e.to_string())?;
        println!("{}", dir.display());
    }
    Ok(true)
}

fn zoo_index_command(a: Args) -> Result<bool, String> {
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

fn tatoeba_command(a: Args) -> Result<bool, String> {
    let export_dir = need(a.export, "--export")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    let recipe = LANGDETECT;
    let model_dir = out.join(recipe.model_id);
    if model_dir.exists() {
        return Err(format!("{}: the model directory exists", model_dir.display()));
    }
    let path = references.join(recipe.reference.file().map_err(|e| e.to_string())?);
    let reference = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    recipe.reference.check(&reference).map_err(|e| e.to_string())?;
    let export = Export::read(&export_dir).map_err(|e| e.to_string())?;
    println!(
        "export {}: {} {} bytes, SHA-256 {}; {} {} bytes, SHA-256 {}; digest {}",
        export_dir.display(),
        export.detailed.name,
        export.detailed.size,
        export.detailed.sha256,
        export.cc0.name,
        export.cc0.size,
        export.cc0.sha256,
        export.digest()
    );
    let recorded =
        langdetect::check_export(&export, langdetect::EXPORT, a.unrecorded_export).map_err(|e| e.to_string())?;
    if !recorded {
        println!("the export is not the recorded one: this run is not a release of the model");
    }
    let outcome = langdetect::run(&recipe, &export, recorded, &reference).map_err(|e| e.to_string())?;
    write_language_dir(&out, &outcome).map_err(|e| e.to_string())?;
    let e = &outcome.card.export;
    println!(
        "{}: {} records, {} of them with another language or text than {} gives, {} not in it",
        export.cc0.name, e.cc0_records, e.cc0_differing, export.detailed.name, e.cc0_not_in_detailed
    );
    println!("{}", outcome.card.summary());
    println!("{}", outcome.gate);
    Ok(outcome.gate.result.pass)
}

/// The files list at `path`, or the committed one.
fn files_list(path: Option<&Path>) -> Result<Vec<WebFile>, String> {
    match path {
        None => fineweb::parse_files(fineweb::FILES_NAME, fineweb::FILES).map_err(|e| e.to_string()),
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            fineweb::parse_files(fineweb::FILES_NAME, &text).map_err(|e| e.to_string())
        }
    }
}

/// The ranges list at `path`, or the committed one.
fn ranges_list(path: Option<&Path>) -> Result<Vec<WebRange>, String> {
    match path {
        None => fineweb::parse_ranges(fineweb::RANGES_NAME, fineweb::RANGES).map_err(|e| e.to_string()),
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            fineweb::parse_ranges(fineweb::RANGES_NAME, &text).map_err(|e| e.to_string())
        }
    }
}

fn fineweb_ranges_command(a: Args) -> Result<bool, String> {
    if [a.paths, a.need, a.record].iter().filter(|x| **x).count() != 1 {
        return Err(format!("one of --paths, --need and --record is required\n{USAGE}"));
    }
    if a.paths {
        for w in WEB.wants() {
            println!("{}\t{}\t{}", w.repo(), w.source.dataset().revision, w.path());
        }
        return Ok(true);
    }
    let web = need(a.web, "--web")?;
    let files = files_list(a.files.as_deref())?;
    let store = Store::new(&web, None);
    if a.need {
        for (f, span) in fineweb::needed(&files, &WEB, &store).map_err(|e| e.to_string())? {
            let dataset = [fineweb::FINEWEB_2, fineweb::FINEWEB].into_iter().find(|d| d.repo == f.repo);
            let url = dataset.expect("the files list names the two datasets").url(&f.path);
            println!("{url}\t{}\t{}\t{}", store.file(&f.repo, &f.path, span).display(), span.start, span.end);
        }
    } else {
        let ranges = fineweb::record(&files, &WEB, &store).map_err(|e| e.to_string())?;
        print!("{}", fineweb::write_ranges(&ranges));
    }
    Ok(true)
}

fn tatoeba_fineweb_command(a: Args) -> Result<bool, String> {
    let export_dir = need(a.export, "--export")?;
    let web_dir = need(a.web, "--web")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    let recipe = LANGDETECT_WEB;
    let mut problems = Vec::new();
    let model_dir = out.join(recipe.model_id);
    if model_dir.exists() {
        problems.push(format!("{}: the model directory exists", model_dir.display()));
    }
    let reference = read_reference(recipe.tatoeba.reference, &references, &mut problems);
    let compared = read_reference(recipe.compared, &references, &mut problems);
    let files = files_list(a.files.as_deref());
    let ranges = ranges_list(a.ranges.as_deref());
    let committed = files_list(None).and_then(|f| ranges_list(None).map(|r| (f, r)));
    let (files, ranges, committed) = match (files, ranges, committed) {
        (Ok(f), Ok(r), Ok(c)) => (f, r, c),
        (f, r, c) => {
            problems.extend([f.err(), r.err(), c.err()].into_iter().flatten());
            return Err(problems.join("\n"));
        }
    };
    let recorded_web = !committed.0.is_empty() && files == committed.0 && ranges == committed.1;
    if !recorded_web && !a.unrecorded_web {
        problems.push(
            "the web lists are not the committed ones (fineweb-files.txt and fineweb-ranges.txt); --unrecorded-web \
             runs on them, and the model is not a release"
                .to_owned(),
        );
    }
    let (Some(reference), Some(compared), true) = (reference, compared, problems.is_empty()) else {
        return Err(problems.join("\n"));
    };
    let export = Export::read(&export_dir).map_err(|e| e.to_string())?;
    println!("export {}: digest {}", export_dir.display(), export.digest());
    let recorded_export =
        langdetect::check_export(&export, langdetect::EXPORT, a.unrecorded_export).map_err(|e| e.to_string())?;
    if !recorded_export {
        println!("the export is not the recorded one: this run is not a release of the model");
    }
    let lists = fineweb::lists_digest(&files, &ranges);
    println!("web lists: {} files, {} ranges, digest {lists}", files.len(), ranges.len());
    if !recorded_web {
        println!("the web lists are not the committed ones: this run is not a release of the model");
    }
    let store = Store::new(&web_dir, Some(&ranges));
    let web = WebInput { files: &files, ranges: &ranges, recorded: recorded_web, store: &store };
    let outcome = langdetect_web::run(&recipe, &export, recorded_export, &web, &reference, &compared)
        .map_err(|e| e.to_string())?;
    write_web_language_dir(&out, &outcome).map_err(|e| e.to_string())?;
    println!("{}", outcome.card.summary());
    println!("{}", outcome.scored.tatoeba_gate);
    println!("{}", outcome.scored.web_gate);
    Ok(outcome.scored.tatoeba_gate.result.pass && outcome.scored.web_gate.result.pass)
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
