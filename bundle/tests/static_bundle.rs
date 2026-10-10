//! Static bundles made with no container and no other program: from a
//! table and a tokenizer in Model2Vec's layout, and distilled from the
//! small BERT bundle. Both are checked against testdata/tiny-static-bundle.

use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo_bundle::recipe::Recipe;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-bundle-static-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn testdata() -> PathBuf {
    root().join("testdata/tiny-static-bundle")
}

/// The test bundle's table and tokenizer as a Model2Vec model's upstream
/// files, with a config.json saying `config`, and a recipe that makes a
/// bundle from them: the test bundle's manifest without what distillation
/// wrote.
fn model2vec_layout(dir: &Path, config: Value) -> PathBuf {
    let up = dir.join("upstream");
    fs::create_dir_all(&up).unwrap();
    fs::copy(testdata().join("tokenizer.json"), up.join("tokenizer.json")).unwrap();
    fs::copy(testdata().join("weights/static.safetensors"), up.join("model.safetensors")).unwrap();
    fs::write(up.join("config.json"), serde_json::to_vec(&config).unwrap()).unwrap();
    let mut m: Value = serde_json::from_slice(&fs::read(testdata().join("manifest.json")).unwrap()).unwrap();
    let o = m.as_object_mut().unwrap();
    o.remove("files");
    let st = o["static_embedding"].as_object_mut().unwrap();
    st.remove("distilled_from");
    st.remove("quality");
    o["artifacts"][0].as_object_mut().unwrap().remove("produced_by");
    o["reference"].as_object_mut().unwrap().remove("produced_by");
    let sha = |p: &str| turbo::bundle::sha256_hex(&fs::read(up.join(p)).unwrap());
    let recipe = json!({
        "manifest": m,
        "upstream": [
            { "path": "tokenizer.json", "to": "tokenizer.json", "sha256": sha("tokenizer.json") },
            { "path": "model.safetensors", "to": "weights/static.safetensors", "sha256": sha("model.safetensors") },
            { "path": "config.json", "sha256": sha("config.json") }
        ]
    });
    let p = dir.join("recipe.json");
    fs::write(&p, serde_json::to_vec_pretty(&recipe).unwrap()).unwrap();
    up
}

/// The reference the tool writes is the test bundle's, to the byte: the
/// same ids and the same vectors, which Model2Vec's StaticModel wrote
/// when the test bundle was first made. The library gives them to the
/// bit, alone and in a batch.
#[test]
fn a_static_bundle_is_made_from_a_table_and_a_tokenizer() {
    let d = scratch("make");
    let up = model2vec_layout(&d, json!({ "normalize": true, "max_length": 32 }));
    let r = Recipe::load(&d.join("recipe.json")).unwrap();
    let bundle = d.join("bundle");
    turbo_bundle::make(&r, &up, &bundle).unwrap();
    let file = "reference/reference.safetensors";
    assert_eq!(fs::read(bundle.join(file)).unwrap(), fs::read(testdata().join(file)).unwrap());
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["reference"]["produced_by"]["tool"], "turbo-bundle");
    assert_eq!(m["reference"]["produced_by"]["reproducible"], true);
    fs::remove_dir_all(d).unwrap();
}

/// A config.json that says otherwise than the manifest is refused.
#[test]
fn a_config_that_disagrees_with_the_manifest_is_refused() {
    for (config, word) in
        [(json!({ "normalize": false, "max_length": 32 }), "normalize"), (json!({ "normalize": true }), "max_length")]
    {
        let d = scratch(word);
        let up = model2vec_layout(&d, config);
        let r = Recipe::load(&d.join("recipe.json")).unwrap();
        let e = turbo_bundle::make(&r, &up, &d.join("bundle")).unwrap_err();
        assert!(e.contains(word), "{e}");
        fs::remove_dir_all(d).unwrap();
    }
}

/// Distillation from the small BERT bundle seals and checks: the library
/// gives the reference to the bit and the quality texts within tolerance
/// of the tool's arithmetic (distill::check).
#[test]
fn a_static_model_is_distilled_and_sealed() {
    let d = scratch("distill");
    let bundle = d.join("bundle");
    let mut r = Recipe::load(&root().join("testdata/tiny-static-recipe/recipe.json")).unwrap();
    turbo_bundle::distill::stage(&r, &root().join("testdata/tiny-bert-bundle"), &bundle).unwrap();
    turbo_bundle::distill::seal(&mut r, &bundle).unwrap();
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(m["reference"]["produced_by"]["tool"], "turbo-bundle");
    fs::remove_dir_all(d).unwrap();
}
