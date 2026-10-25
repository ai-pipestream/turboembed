//! Train and gate local native-v1 lemma models without altering released ids.
//! Usage: ud_native_lemmas TREEBANKS REFERENCES OUT SOURCE_RECIPE_ID...
use std::error::Error;
use std::path::Path;

use turbo_train_recipes::treebank::TREEBANKS;
use turbo_train_recipes::ud::{UdRecipe, native};
use turbo_train_recipes::ud_card::write_ud_model_dir;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() < 4 {
        return Err("usage: ud_native_lemmas TREEBANKS REFERENCES OUT SOURCE_RECIPE_ID...".into());
    }
    let (treebanks, references, out) = (Path::new(&args[0]), Path::new(&args[1]), Path::new(&args[2]));
    let mut recipes = Vec::new();
    for id in &args[3..] {
        let id = id.to_str().ok_or("recipe id is not UTF-8")?;
        let r = UdRecipe::trains(id).ok_or_else(|| format!("unknown source recipe {id}"))?;
        let local_id = native::model_id(&r)?;
        if recipes.contains(&r) {
            return Err(format!("duplicate source recipe {id}").into());
        }
        if out.join(local_id).exists() {
            return Err(format!("output {local_id} already exists").into());
        }
        recipes.push(r);
    }
    // Validate every selected source/reference before writing any model.
    let mut texts = Vec::new();
    for t in TREEBANKS.iter().filter(|t| recipes.iter().any(|r| r.treebank == **t)) {
        texts.push((*t, t.read(&treebanks.join(t.dataset_id()))?));
    }
    let mut checked = Vec::new();
    for r in &recipes {
        let bytes = std::fs::read(references.join(r.reference.file()?))?;
        r.reference.check(&bytes)?;
        let text = &texts.iter().find(|(t, _)| *t == r.treebank).expect("selected treebank").1;
        r.check_data(text, &bytes)?;
        checked.push(bytes);
    }
    let mut passed = true;
    for (r, reference) in recipes.iter().zip(checked) {
        let text = &texts.iter().find(|(t, _)| *t == r.treebank).expect("selected treebank").1;
        let result = native::run(r, text, &reference)?;
        write_ud_model_dir(out, &result.card, &result.trained.bytes, &result.gate)?;
        println!("{}", result.card.summary());
        println!("{}", result.gate);
        passed &= result.gate.result.pass;
    }
    if !passed {
        return Err("one or more unchanged native lemma gates failed; reports were retained".into());
    }
    Ok(())
}
