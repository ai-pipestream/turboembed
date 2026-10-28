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
