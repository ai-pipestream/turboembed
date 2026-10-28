use turbo_formats::masc::MascCorpus;
use turbo_train_recipes::masc;
use turbo_train_recipes::namefind::selection::{RECIPES, write_attempt};

use crate::cli::{Args, chosen, need, only};

pub(crate) fn run(a: Args) -> Result<bool, String> {
    only(&a, "masc-namefind-select", &["--corpus", "--references", "--out", "--recipe"])?;
    let recipes = chosen(&RECIPES, &a.recipe_ids, |r| r.model_id)?;
    let corpus = need(a.corpus, "--corpus")?;
    let references = need(a.references, "--references")?;
    let out = need(a.out, "--out")?;
    let digest = masc::check_corpus(&corpus, masc::RELEASE, false).map_err(|e| e.to_string())?;
    let corpus_data = MascCorpus::read(&corpus).map_err(|e| e.to_string())?;
    let split = masc::SPLIT.apply(corpus_data, &corpus).map_err(|e| e.to_string())?;
    let mut problems = Vec::new();
    let mut models = Vec::new();
    for recipe in &recipes {
        if out.join(recipe.model_id).exists() {
            problems.push(format!("{} already has an attempt directory", recipe.model_id));
        }
        models.extend(crate::read_reference(recipe.reference, &references, &mut problems));
        if let Err(e) = recipe.check_training(&split.train) {
            problems.push(e.to_string());
        }
    }
    if !problems.is_empty() {
        return Err(problems.join("\n"));
    }
    let mut pass = true;
    for (recipe, reference) in recipes.iter().zip(models) {
        let result = write_attempt(recipe, &split, &digest, true, &reference, &out).map_err(|e| e.to_string())?;
        println!("{}: gate {}", recipe.model_id, if result { "PASS" } else { "FAIL" });
        pass &= result;
    }
    Ok(pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_cannot_change_grid_or_use_unrecorded_data_or_old_ids() {
        let args =
            crate::cli::parse_from(["masc-namefind-select", "--unrecorded-corpus"].into_iter().map(str::to_owned))
                .unwrap();
        assert!(run(args).unwrap_err().contains("does not read --unrecorded-corpus"));
        let args = crate::cli::parse_from(
            ["masc-namefind-select", "--recipe", "turbo-en-masc-3.0.0-ner-person-1"].into_iter().map(str::to_owned),
        )
        .unwrap();
        assert!(run(args).unwrap_err().contains("no recipe trains"));
        assert!(
            crate::cli::parse_from(["masc-namefind-select", "--iterations", "5"].into_iter().map(str::to_owned),)
                .is_err()
        );
    }
}
