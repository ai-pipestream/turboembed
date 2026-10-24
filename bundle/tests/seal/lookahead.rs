use super::*;

#[test]
fn staging_refuses_a_partial_layer_lookahead_before_copying_files() {
    let d = scratch("partial-layer-lookahead");
    let mut recipe = Recipe::load(&tiny_recipe(&d)).unwrap();
    recipe.manifest["architecture"]["layers"] = json!(12);
    recipe.manifest["architecture"]["lookahead"] = json!(8);
    let up = upstream(&d);
    let bundle = d.join("bundle");
    fs::write(
        up.join("config.json"),
        serde_json::to_vec(&json!({ "lookahead": 48, "num_hidden_layers": 12, "lookahead_split_layers": 6 })).unwrap(),
    )
    .unwrap();
    let e = seal::stage(&recipe, &up, &bundle).unwrap_err();
    assert!(
        e.contains("lookahead_split_layers 6") && e.contains("per-layer lookahead") && e.contains("not supported"),
        "{e}"
    );
    assert!(!bundle.exists(), "an unsupported schedule must be refused before staging any files");
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn staging_accepts_a_uniform_layer_lookahead_and_checks_its_value() {
    let d = scratch("uniform-layer-lookahead");
    let mut recipe = Recipe::load(&tiny_recipe(&d)).unwrap();
    recipe.manifest["architecture"]["lookahead"] = json!(4);
    let up = upstream(&d);
    let bundle = d.join("bundle");
    for split in [Value::Null, json!(2)] {
        fs::write(
            up.join("config.json"),
            serde_json::to_vec(&json!({ "lookahead": 8, "num_hidden_layers": 2, "lookahead_split_layers": split }))
                .unwrap(),
        )
        .unwrap();
        seal::stage(&recipe, &up, &bundle).unwrap();
        assert!(bundle.join("tokenizer.json").is_file());
        recipe.manifest["architecture"]["lookahead"] = json!(8);
        let e = seal::stage(&recipe, &up, &bundle).unwrap_err();
        assert!(e.contains("lookahead: 8, the upstream config.json trains 4 a layer"), "{e}");
        recipe.manifest["architecture"]["lookahead"] = json!(4);
    }
    fs::remove_dir_all(d).unwrap();
}
