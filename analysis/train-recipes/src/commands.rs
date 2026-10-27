use std::path::Path;

use turbo_train_recipes::depparse::{DEPPARSE_RECIPES, DepparseRecipe, run_depparse, write_depparse_model_dir};
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

pub(crate) fn depparse_command(a: Args) -> Result<bool, String> {
    only(&a, "depparse", &["--treebanks", "--out", "--recipe"])?;
    let recipes = chosen(&DEPPARSE_RECIPES, &a.recipe_ids, |r: &DepparseRecipe| r.model_id)?;
    let treebanks = need(a.treebanks, "--treebanks")?;
    let out = need(a.out, "--out")?;
    let mut problems = Vec::new();
    for recipe in &recipes {
        let dir = out.join(recipe.model_id);
        if dir.exists() {
            problems.push(format!("{}: the model directory exists", dir.display()));
        }
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    let mut texts = Vec::new();
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        let dir = treebanks.join(t.dataset_id());
        let text = t.read(&dir).map_err(|e| e.to_string())?;
        println!("{} {}: licence and digests checked", t.repository, t.dataset_id());
        texts.push((*t, text));
    }
    let text_of = |t: &Treebank| &texts.iter().find(|(u, _)| u == t).expect("every treebank was read").1;
    let data_problems: Vec<String> = recipes
        .iter()
        .filter_map(|r| r.check_data(text_of(&r.treebank)).err().map(|e| format!("{}: {e}", r.model_id)))
        .collect();
    if !data_problems.is_empty() {
        return Err(data_problems.join("\n"));
    }
    let mut all_pass = true;
    for recipe in &recipes {
        let started = std::time::Instant::now();
        let outcome = run_depparse(recipe, text_of(&recipe.treebank)).map_err(|e| e.to_string())?;
        write_depparse_model_dir(&out, &outcome).map_err(|e| e.to_string())?;
        println!("{}", outcome.card.summary());
        println!("{}", outcome.gate);
        println!("{}", outcome.uas_gate);
        println!("{}: {:.1} s", recipe.model_id, started.elapsed().as_secs_f64());
        all_pass &= outcome.gate.result.pass && outcome.uas_gate.result.pass;
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

/// `static-embed`: initialize a static table from the teacher's
/// embedding matrix, seal it so the core tokenizer loads, tokenize
/// the corpora, contrastive-train the table, and seal the trained
/// bundle at `--out` with its training card beside it.
pub(crate) fn static_embed_command(a: Args) -> Result<bool, String> {
    use turbo_train_recipes::static_embed::{self, bundle::Identity, bundle::Paths, data};
    only(
        &a,
        "static-embed",
        &[
            "--teacher",
            "--tokenizer",
            "--recipes",
            "--sentences",
            "--parquet",
            "--out",
            "--commit",
            "--steps",
            "--lr",
            "--init-only",
        ],
    )?;
    let teacher_path = need(a.teacher, "--teacher")?;
    let tokenizer = need(a.tokenizer, "--tokenizer")?;
    let recipes = need(a.recipes, "--recipes")?;
    let out = need(a.out, "--out")?;
    let commit = a.commit.ok_or("--commit is required")?;
    if commit.len() != 40 || !commit.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(format!("--commit {commit} is not a 40-char lowercase hex commit"));
    }
    if out.join("manifest.json").exists() {
        return Err(format!("{}: already holds a bundle", out.display()));
    }
    let mut cfg = static_embed::Config::default();
    if let Some(s) = &a.steps {
        cfg.steps = s.parse().map_err(|_| format!("--steps {s} is not a number"))?;
    }
    if let Some(s) = &a.lr {
        cfg.lr = s.parse().map_err(|_| format!("--lr {s} is not a number"))?;
    }
    let err = |e: turbo_train_recipes::error::Error| e.to_string();
    let bytes = std::fs::read(&teacher_path).map_err(|e| format!("{}: {e}", teacher_path.display()))?;
    let (teacher, shape) = ["bert.embeddings.word_embeddings.weight", "embeddings.word_embeddings.weight"]
        .iter()
        .find_map(|name| static_embed::read_tensor(&bytes, name).ok())
        .ok_or_else(|| "teacher safetensors: no word embeddings tensor".to_owned())?;
    let [vocab, in_dim] = shape[..] else {
        return Err(format!("teacher embeddings shape {shape:?} is not [vocab, dim]"));
    };
    println!("teacher embeddings {vocab}x{in_dim}; initializing table (center, PCA {}, SIF)", cfg.dim);
    let mut table = static_embed::initial_table(&teacher, vocab, in_dim, cfg.dim, cfg.sif_coefficient);
    drop(teacher);
    let who = Identity {
        id: "turbo/static-en-bert-fineweb-256",
        repository: "https://github.com/ai-pipestream/turboembed",
        commit: &commit,
    };
    let stage = out.with_extension("stage");
    let init_bundle = out.with_extension("init");
    // The init bundle is this command's own scratch (the tokenizer
    // vehicle); a previous run's copy is replaced. The final `--out`
    // above stays guarded.
    if init_bundle.join("manifest.json").exists() {
        std::fs::remove_dir_all(&init_bundle).map_err(|e| format!("{}: {e}", init_bundle.display()))?;
    }
    let teacher_sha = turbo_sha256::sha256(&bytes).to_hex();
    let init_manifest = static_embed::bundle::make_bundle(
        &table,
        vocab,
        cfg.dim,
        &Paths { tokenizer_json: &tokenizer, recipes: &recipes, stage: &stage, out: &init_bundle },
        &who,
    )
    .map_err(err)?;
    println!("init bundle sealed at {} (manifest sha256 {init_manifest})", init_bundle.display());
    if a.init_only {
        return Ok(true);
    }
    let tok = static_embed::tokenizer_of(&init_bundle).map_err(err)?;
    let mut docs: Vec<Vec<String>> = Vec::new();
    for path in &a.sentences {
        let d = data::read_sentences(path).map_err(err)?;
        println!("{}: {} documents", path.display(), d.len());
        docs.extend(d);
    }
    if let Some(p) = &a.parquet {
        let d = data::read_parquet(p, 60_000).map_err(err)?;
        println!("{}: {} documents", p.display(), d.len());
        docs.extend(d);
    }
    if docs.is_empty() {
        return Err("no training documents: pass --sentences or --parquet".into());
    }
    let sentences: usize = docs.iter().map(|d| d.len()).sum();
    let ids = static_embed::tokenize_docs(&tok, &docs, cfg.max_tokens).map_err(err)?;
    let kept: usize = ids.iter().map(|d| d.len()).sum();
    println!("training on {} documents, {sentences} sentences ({kept} tokenized), {} steps", ids.len(), cfg.steps);
    let started = std::time::Instant::now();
    let (first, last) = static_embed::train(&mut table, vocab, &ids, &cfg);
    println!("trained in {:.1}s: loss {first:.4} -> {last:.4}", started.elapsed().as_secs_f64());
    let manifest = static_embed::bundle::make_bundle(
        &table,
        vocab,
        cfg.dim,
        &Paths { tokenizer_json: &tokenizer, recipes: &recipes, stage: &stage, out: &out },
        &who,
    )
    .map_err(err)?;
    let card = serde_json::json!({
        "model_id": who.id,
        "kind": "static-embed",
        "teacher": {"file": teacher_path.file_name().map(|n| n.to_string_lossy().into_owned()), "sha256": teacher_sha, "tensor": "embeddings.word_embeddings.weight", "vocab": vocab, "in_dim": in_dim},
        "config": {"dim": cfg.dim, "batch": cfg.batch, "steps": cfg.steps, "lr": cfg.lr, "temperature": cfg.temperature, "max_tokens": cfg.max_tokens, "seed": cfg.seed, "sif_coefficient": cfg.sif_coefficient},
        "data": {"documents": ids.len(), "sentences": kept, "sentences_files": a.sentences.iter().map(|p| p.file_name().map(|n| n.to_string_lossy().into_owned())).collect::<Vec<_>>(), "parquet": a.parquet.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().into_owned())},
        "loss_first_window": first,
        "loss_last_window": last,
        "init_manifest_sha256": init_manifest,
        "manifest_sha256": manifest,
        "seconds": started.elapsed().as_secs_f64(),
    });
    let card_path = out.with_extension("card.json");
    std::fs::write(&card_path, serde_json::to_vec_pretty(&card).unwrap())
        .map_err(|e| format!("{}: {e}", card_path.display()))?;
    println!("trained bundle sealed at {} (manifest sha256 {manifest}); card {}", out.display(), card_path.display());
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
