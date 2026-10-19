use std::path::Path;

use turbo_train_recipes::fineweb::{self, Store, WEB, WebFile, WebRange};
use turbo_train_recipes::langdetect::{self, LANGDETECT, write_language_dir};
use turbo_train_recipes::langdetect_web::{self, LANGDETECT_WEB, WebInput, write_web_language_dir};
use turbo_train_recipes::tatoeba::Export;
use turbo_train_recipes::treebank::{TREEBANKS, Treebank, TreebankText};
use turbo_train_recipes::ud::{UD_RECIPES, UdRecipe, run_ud};
use turbo_train_recipes::ud_card::write_ud_model_dir;

use crate::cli::{Args, USAGE, chosen, need, only};
use crate::{read_reference, user};

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

pub(crate) fn ud_command(a: Args) -> Result<bool, String> {
    only(&a, "ud", &["--treebanks", "--references", "--out", "--recipe"])?;
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

pub(crate) fn fetch_ud_command(a: Args) -> Result<bool, String> {
    only(&a, "fetch-ud", &["--home", "--recipe"])?;
    let recipes = chosen(&UD_RECIPES, &a.recipe_ids, |r: &UdRecipe| r.model_id)?;
    let home = need(a.home, "--home")?;
    let user = user()?;
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        let dir = t.fetch(&home, &user).map_err(|e| e.to_string())?;
        println!("{}", dir.display());
    }
    Ok(true)
}

pub(crate) fn tatoeba_command(a: Args) -> Result<bool, String> {
    only(&a, "tatoeba", &["--export", "--references", "--out", "--unrecorded-export"])?;
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
            fineweb::parse_files(&p.display().to_string(), &text).map_err(|e| e.to_string())
        }
    }
}

/// The ranges list at `path`, or the committed one.
fn ranges_list(path: Option<&Path>) -> Result<Vec<WebRange>, String> {
    match path {
        None => fineweb::parse_ranges(fineweb::RANGES_NAME, fineweb::RANGES).map_err(|e| e.to_string()),
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
            fineweb::parse_ranges(&p.display().to_string(), &text).map_err(|e| e.to_string())
        }
    }
}

pub(crate) fn fineweb_ranges_command(a: Args) -> Result<bool, String> {
    if [a.paths, a.need, a.record].iter().filter(|x| **x).count() != 1 {
        return Err(format!("one of --paths, --need and --record is required\n{USAGE}"));
    }
    if a.paths {
        only(&a, "fineweb-ranges --paths", &["--paths"])?;
        for w in WEB.wants() {
            println!("{}\t{}\t{}", w.repo(), w.source.dataset().revision, w.path());
        }
        return Ok(true);
    }
    only(&a, "fineweb-ranges", &["--web", "--files", "--need", "--record"])?;
    let web = need(a.web, "--web")?;
    let files = files_list(a.files.as_deref())?;
    let store = Store::new(&web, None);
    if a.need {
        for (f, span) in fineweb::needed(&files, &WEB, &store).map_err(|e| e.to_string())? {
            let dataset = [fineweb::FINEWEB_2, fineweb::FINEWEB].into_iter().find(|d| d.repo == f.repo);
            let url = dataset.expect("the files list names the two datasets").url(&f.path);
            println!("{url}\t{}\t{}\t{}", store.file(f, span).display(), span.start, span.end);
        }
    } else {
        let ranges = fineweb::record(&files, &WEB, &store).map_err(|e| e.to_string())?;
        print!("{}", fineweb::write_ranges(&ranges));
    }
    Ok(true)
}

pub(crate) fn tatoeba_fineweb_command(a: Args) -> Result<bool, String> {
    only(
        &a,
        "tatoeba-fineweb",
        &[
            "--export",
            "--web",
            "--references",
            "--out",
            "--unrecorded-export",
            "--files",
            "--ranges",
            "--unrecorded-web",
        ],
    )?;
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
    if a.files.is_some() != a.ranges.is_some() {
        problems.push("--files and --ranges are given together".to_owned());
    }
    if !web_dir.is_dir() {
        problems.push(format!("{}: no such directory", web_dir.display()));
    }
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
    // The web ranges are checked before the export is read.
    let store = Store::new(&web_dir, Some(&ranges));
    let web = WebInput::checked(recipe.web, &files, &ranges, recorded_web, &store).map_err(|e| e.to_string())?;
    let lists = fineweb::lists_digest(&files, &ranges);
    println!("web lists: {} files, {} ranges, digest {lists}", files.len(), ranges.len());
    if !recorded_web {
        println!("the web lists are not the committed ones: this run is not a release of the model");
    }
    let export = Export::read(&export_dir).map_err(|e| e.to_string())?;
    println!("export {}: digest {}", export_dir.display(), export.digest());
    let recorded_export =
        langdetect::check_export(&export, langdetect::EXPORT, a.unrecorded_export).map_err(|e| e.to_string())?;
    if !recorded_export {
        println!("the export is not the recorded one: this run is not a release of the model");
    }
    let outcome = langdetect_web::run(&recipe, &export, recorded_export, &web, &reference, &compared)
        .map_err(|e| e.to_string())?;
    write_web_language_dir(&out, &outcome).map_err(|e| e.to_string())?;
    println!("{}", outcome.card.summary());
    println!("{}", outcome.scored.tatoeba_gate);
    println!("{}", outcome.scored.web_gate);
    Ok(outcome.scored.tatoeba_gate.result.pass && outcome.scored.web_gate.result.pass)
}
