use crate::cli::{Args, USAGE, chosen, need, only};
use crate::user;
use std::path::Path;
use turbo_train_recipes::fineweb::{self, Store, WEB, WebFile};
use turbo_train_recipes::treebank::TREEBANKS;
use turbo_train_recipes::ud::{UD_RECIPES, UdRecipe};

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
    let (stage, init_bundle) = static_embed::bundle::reserve_scratch(&out).map_err(err)?;
    println!("teacher embeddings {vocab}x{in_dim}; initializing table (center, PCA {}, SIF)", cfg.dim);
    let mut table = static_embed::initial_table(&teacher, vocab, in_dim, cfg.dim, cfg.sif_coefficient);
    drop(teacher);
    let who = Identity {
        id: "turbo/static-en-bert-fineweb-256",
        repository: "https://github.com/ai-pipestream/turboembed",
        commit: &commit,
    };
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
