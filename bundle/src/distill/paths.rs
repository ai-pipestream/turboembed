use super::{EMBEDDINGS, Recipe, Result, json};

/// Where the recipe puts things.
pub(super) struct Paths {
    pub(super) weights: String,
    pub(super) tokenizer: String,
    pub(super) texts: String,
    pub(super) reference: String,
}

pub(super) fn paths(recipe: &Recipe) -> Result<Paths> {
    let m = &recipe.manifest;
    let arts = m["artifacts"].as_array().ok_or("manifest.artifacts: missing")?;
    let [art] = arts.as_slice() else {
        return Err("a static recipe has one artifact, its table".into());
    };
    let files = art["files"].as_array().ok_or("manifest.artifacts[0].files: missing")?;
    let [weights] = files.as_slice() else {
        return Err("a static recipe's artifact is one file".into());
    };
    let names = &art["tensor_names"];
    if *names != json!({ "static_embeddings": EMBEDDINGS }) {
        return Err(format!(
            "manifest.artifacts[0].tensor_names: the tool writes static_embeddings {EMBEDDINGS:?} alone"
        ));
    }
    if art.get("produced_by").is_some() {
        return Err("manifest.artifacts[0].produced_by: written by the tool, not the recipe".into());
    }
    let s = |p: &str| recipe.str_at(p).map(str::to_owned);
    let p = Paths {
        weights: weights.as_str().ok_or("manifest.artifacts[0].files[0]: not a string")?.to_owned(),
        tokenizer: s("/tokenizer/file")?,
        texts: s("/static_embedding/quality/texts")?,
        reference: s("/reference/file")?,
    };
    for f in [&p.weights, &p.tokenizer, &p.texts, &p.reference] {
        crate::check_rel(f)?;
    }
    Ok(p)
}
