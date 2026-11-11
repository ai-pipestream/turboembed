use super::*;

/// Exact config.json at the recipe's pinned sat-3l revision, not a
/// synthetic one-row classifier. The checkpoint has 111 output rows;
/// SaT consumes row zero for sentence boundaries.
#[test]
fn sat_3l_accepts_its_pinned_classifier_configuration() {
    let recipe = Recipe::load(&root().join("bundle/recipes/sat-3l.json")).unwrap();
    let config = include_bytes!("../data/sat-3l-config.json");
    let pin = recipe.upstream.iter().find(|u| u.path == "config.json").unwrap();
    assert_eq!(turbo::bundle::sha256_hex(config), pin.sha256.as_deref().unwrap());
    let parsed: Value = serde_json::from_slice(config).unwrap();
    assert_eq!(parsed["id2label"].as_object().unwrap().len(), 111);
    assert_eq!(parsed["id2label"]["0"], "LABEL_0");
    let upstream = scratch("sat-3l-pinned-config");
    fs::write(upstream.join("config.json"), config).unwrap();
    seal::check_classifier_config(&recipe, &upstream, true).unwrap();
    fs::remove_dir_all(upstream).unwrap();
}

#[test]
fn sat_3l_fetches_bge_m3_tokenizer_converts_weights_in_turbo_reference_and_runs_sat_reference() {
    let recipe = Recipe::load(&root().join("bundle/recipes/sat-3l.json")).unwrap();
    let plan = turbo_bundle::fetch::plan(&recipe).unwrap();
    assert_eq!(plan.len(), 3);
    assert_eq!(plan[0].path, "tokenizer.json");
    assert_eq!(
        plan[0].url,
        "https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
    );
    assert_eq!(plan[1].path, "pytorch_model.bin");
    assert_eq!(
        plan[1].url,
        "https://huggingface.co/segment-any-text/sat-3l/resolve/a3b7ba6b61881d619989757aea971cde7d3d6a98/pytorch_model.bin"
    );
    assert_eq!(plan[2].path, "config.json");
    assert_eq!(
        plan[2].url,
        "https://huggingface.co/segment-any-text/sat-3l/resolve/a3b7ba6b61881d619989757aea971cde7d3d6a98/config.json"
    );

    let convs = convert::conversions(&recipe).unwrap();
    assert_eq!(convs.len(), 1);
    assert_eq!(convs[0].name, "weights-f32");
    assert_eq!(convs[0].from_file, "pytorch_model.bin");
    assert!(convs[0].from_upstream);
    assert_eq!(convs[0].script, convert::BIN_TO_SAFETENSORS);
    assert_eq!(
        convs[0].container.as_deref(),
        Some("turbo-reference@sha256:0e153c1db87c62f55ff6f918c7d8a0120c3d2e5d45d3f5a0336e81153c12dd6b")
    );

    let sat_pin = "turbo-sat-reference@sha256:71f05fa32779f77c99d93cdfc9b8bc99b74d307bd1e55b13af2c5050346f4a77";
    assert!(reference::is_sat(sat_pin));
    assert!(!reference::is_sat(CONTAINER));
    let args =
        reference::docker_args(&recipe, "sha256:71f05fa3", Path::new("/up"), Path::new("/wk"), Some("1000:1000"))
            .unwrap();
    assert_eq!(
        args,
        [
            "run",
            "--rm",
            "--network",
            "none",
            "--mount",
            "type=bind,src=/up,dst=/model,readonly",
            "--mount",
            "type=bind,src=/wk,dst=/work",
            "--user",
            "1000:1000",
            "sha256:71f05fa3",
            "reference",
            "/model",
            "/model",
            "/work/recipe.json",
            "/work/reference.safetensors",
            "/work/produced_by.json",
            sat_pin,
        ]
    );
}

#[test]
fn sat_1l_sm_fetches_xlm_roberta_base_tokenizer_and_runs_sat_reference() {
    let recipe = Recipe::load(&root().join("bundle/recipes/sat-1l-sm.json")).unwrap();
    let plan = turbo_bundle::fetch::plan(&recipe).unwrap();
    assert_eq!(plan.len(), 3);
    assert_eq!(
        plan[0].url,
        "https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089/tokenizer.json"
    );
    assert_eq!(
        plan[1].url,
        "https://huggingface.co/segment-any-text/sat-1l-sm/resolve/00a783439c201b20ee91ee1624dc817f1bf0e578/model.safetensors"
    );
    assert_eq!(
        plan[2].url,
        "https://huggingface.co/segment-any-text/sat-1l-sm/resolve/00a783439c201b20ee91ee1624dc817f1bf0e578/config.json"
    );
    assert!(convert::conversions(&recipe).unwrap().is_empty());
    let args = reference::docker_args(&recipe, "sha256:71f05fa3", Path::new("/up"), Path::new("/wk"), None).unwrap();
    assert_eq!(
        &args[args.len() - 7..],
        [
            "reference",
            "/model",
            "/model",
            "/work/recipe.json",
            "/work/reference.safetensors",
            "/work/produced_by.json",
            "turbo-sat-reference@sha256:71f05fa32779f77c99d93cdfc9b8bc99b74d307bd1e55b13af2c5050346f4a77",
        ]
    );
}

#[test]
fn tiny_sat_recipe_fetches_from_second_repo_stages_and_seals() {
    let d = scratch("tiny-sat-make");
    let src_bundle = root().join("testdata/tiny-sat-bundle");
    let mut raw: Value =
        serde_json::from_slice(&fs::read(root().join("testdata/tiny-sat-recipe/recipe.json")).unwrap()).unwrap();
    let weights_bytes = fs::read(src_bundle.join("weights/model.safetensors")).unwrap();
    raw["upstream"][0]["repository"] = json!("https://huggingface.co/BAAI/bge-m3");
    raw["upstream"][0]["commit"] = json!("5617a9f61b028005a4858fdac845db406aefb181");
    raw["upstream"][1]["sha256"] = json!(turbo::bundle::sha256_hex(&weights_bytes));
    let sat_pin = "turbo-sat-reference@sha256:0000000000000000000000000000000000000000000000000000000000000000";
    raw["manifest"]["reference"]["produced_by"]["container"] = json!(sat_pin);
    let recipe_path = d.join("recipe.json");
    fs::write(&recipe_path, serde_json::to_vec_pretty(&raw).unwrap()).unwrap();

    let mut recipe = Recipe::load(&recipe_path).unwrap();
    let plan = turbo_bundle::fetch::plan(&recipe).unwrap();
    assert_eq!(
        plan[0].url,
        "https://huggingface.co/BAAI/bge-m3/resolve/5617a9f61b028005a4858fdac845db406aefb181/tokenizer.json"
    );
    assert_eq!(
        plan[1].url,
        "https://github.com/ai-pipestream/turboembed/resolve/b02014e9a3e174ffc26ebf63fe934d5b2bb8080b/model.safetensors"
    );

    let up = d.join("upstream");
    fs::create_dir_all(&up).unwrap();
    fs::copy(src_bundle.join("tokenizer.json"), up.join("tokenizer.json")).unwrap();
    fs::copy(src_bundle.join("weights/model.safetensors"), up.join("model.safetensors")).unwrap();
    turbo_bundle::fetch::fetch(&recipe, &up, false).unwrap();
    assert!(up.join(turbo_bundle::fetch::PROVENANCE).is_file());

    let bundle = d.join("bundle");
    let err = turbo_bundle::make(&recipe, &up, &bundle).unwrap_err();
    assert!(err.contains(sat_pin), "{err}");
    assert!(bundle.join("tokenizer.json").is_file(), "make staged tokenizer.json before reference::run");
    assert!(bundle.join("weights/model.safetensors").is_file(), "make staged weights before reference::run");

    fs::create_dir_all(bundle.join("reference")).unwrap();
    let ref_bytes = fs::read(src_bundle.join("reference/reference.safetensors")).unwrap();
    let st = turbo::safetensors::File::parse("reference.safetensors", &ref_bytes).unwrap();
    let total_tokens: u64 =
        st.get("lengths", turbo::safetensors::Dtype::I32, 1).unwrap().i32s().into_iter().map(|l| l.max(0) as u64).sum();
    let ref_with_logits = super::catalogue::append_safetensors(
        &ref_bytes,
        &[("logits", &[total_tokens, 1], &vec![0.0f32; total_tokens as usize])],
    );
    fs::write(bundle.join("reference/reference.safetensors"), &ref_with_logits).unwrap();
    let report = json!({
        "tool": "wtpsplit",
        "tool_version": "wtpsplit 2.1.1, torch 2.2.2+cpu",
        "container": sat_pin,
        "args": [
            "/model",
            "/model",
            "/work/recipe.json",
            "/work/reference.safetensors",
            "/work/produced_by.json",
            sat_pin
        ]
    });
    fs::write(bundle.join("reference/report.json"), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    seal::seal_staged(&mut recipe, &up, &bundle).unwrap();
    seal::verify(&bundle).unwrap();
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn upstream_source_and_weights_container_pins_are_checked() {
    let d = scratch("sat-recipe-pins");
    let base: Value = serde_json::from_slice(&fs::read(root().join("bundle/recipes/sat-3l.json")).unwrap()).unwrap();
    let check = |edit: &dyn Fn(&mut Value)| -> Result<Recipe, String> {
        let mut r = base.clone();
        edit(&mut r);
        let p = d.join("edited.json");
        fs::write(&p, serde_json::to_vec(&r).unwrap()).unwrap();
        Recipe::load(&p)
    };
    let e = check(&|r| {
        r["upstream"][0].as_object_mut().unwrap().remove("commit");
    })
    .unwrap_err();
    assert!(e.contains("repository and commit must be given together"), "{e}");

    let e = check(&|r| {
        r["upstream"][0].as_object_mut().unwrap().remove("repository");
    })
    .unwrap_err();
    assert!(e.contains("repository and commit must be given together"), "{e}");

    let e = check(&|r| {
        r["upstream"][0]["repository"] = json!("http://huggingface.co/BAAI/bge-m3");
    })
    .unwrap_err();
    assert!(e.contains("https://"), "{e}");

    let e = check(&|r| {
        r["upstream"][0]["commit"] = json!("main");
    })
    .unwrap_err();
    assert!(e.contains("40-hex"), "{e}");

    let e = check(&|r| {
        r["manifest"]["artifacts"][0]["produced_by"]["container"] = json!("turbo-reference:latest");
    })
    .unwrap_err();
    assert!(e.contains("name@sha256:"), "{e}");

    let mut bad_role = check(&|_| {}).unwrap();
    bad_role.manifest["reference"]["cases"][0]["prompt_role"] = json!("PROMPT_QUERY");
    let e = reference::docker_args(&bad_role, "sha256:71f05fa3", Path::new("/up"), Path::new("/wk"), None).unwrap_err();
    assert!(e.contains("PROMPT_NONE"), "{e}");

    fs::remove_dir_all(d).unwrap();
}
