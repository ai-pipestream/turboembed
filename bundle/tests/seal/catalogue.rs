use super::*;

/// Catalogue entries pin every upstream file and task-specific settings.

#[test]
fn the_catalogue_names_pinned_recipes() {
    for (id, _) in turbo_bundle::catalogue::MODELS {
        let r = Recipe::load(Path::new(id)).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(r.str_at("/model/id").unwrap(), *id);
        if r.manifest["task"] == "TASK_RERANK" {
            assert!(r.manifest.get("rerank").is_some(), "{id}");
            assert_eq!(r.manifest["classifier"]["labels"].as_array().unwrap().len(), 1);
            assert!(r.manifest["reference"]["produced_by"]["container"].as_str().unwrap().contains("@sha256:"));
        } else {
            assert!(r.manifest.get("static_embedding").is_some(), "{id}");
            assert!(r.upstream.iter().all(|u| u.sha256.is_some()), "{id}: every file pinned");
            assert_eq!(r.manifest["static_embedding"]["max_length"], 512, "{id}");
            assert_eq!(r.manifest["embed"]["normalize"], "NORMALIZE_L2", "{id}");
            assert!(r.manifest["reference"].get("produced_by").is_none(), "{id}");
        }
        assert!(r.upstream.iter().all(|u| u.sha256.is_some()), "{id}: every file pinned");
    }
    assert!(Recipe::load(Path::new("minishlab/not-a-model")).is_err());
}

/// Files already in the upstream directory with their pinned hashes are
/// kept, so this fetch reads no network: it records where they came from,
/// and a model with terms beyond its licence is fetched only once they
/// are accepted.
#[test]
fn a_fetch_records_its_provenance_and_asks_for_terms() {
    let d = scratch("fetch-terms");
    let up = upstream(&d);
    let p = tiny_recipe(&d);
    let mut r: Value = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
    for u in r["upstream"].as_array_mut().unwrap() {
        let bytes = fs::read(up.join(u["path"].as_str().unwrap())).unwrap();
        u["sha256"] = json!(turbo::bundle::sha256_hex(&bytes));
    }
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    turbo_bundle::fetch::fetch(&Recipe::load(&p).unwrap(), &up, false).unwrap();
    let got: Value = serde_json::from_slice(&fs::read(up.join(turbo_bundle::fetch::PROVENANCE)).unwrap()).unwrap();
    let (repository, revision) = Recipe::load(&p).unwrap().source().map(|(a, b)| (a.to_owned(), b.to_owned())).unwrap();
    assert_eq!(got["repository"], json!(repository));
    assert_eq!(got["revision"], json!(revision));
    assert_eq!(got["terms_accepted"], json!(false));
    assert_eq!(got["sha256"].as_object().unwrap().len(), 3);
    assert_eq!(got["sha256"]["tokenizer.json"], r["upstream"][0]["sha256"]);

    r["notice"] = json!("Some terms.");
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();
    fs::remove_file(up.join(turbo_bundle::fetch::PROVENANCE)).unwrap();
    let e = turbo_bundle::fetch::fetch(&Recipe::load(&p).unwrap(), &up, false).unwrap_err();
    assert!(e.contains("--accept-terms"), "{e}");
    assert!(!up.join(turbo_bundle::fetch::PROVENANCE).exists(), "refused before anything is written");
    turbo_bundle::fetch::fetch(&Recipe::load(&p).unwrap(), &up, true).unwrap();
    let got: Value = serde_json::from_slice(&fs::read(up.join(turbo_bundle::fetch::PROVENANCE)).unwrap()).unwrap();
    assert_eq!(got["notice"], json!("Some terms."));
    assert_eq!(got["terms_accepted"], json!(true));
}

fn append_safetensors(bytes: &[u8], extra: &[(&str, &[u64], &[f32])]) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let raw_header: serde_json::Map<String, Value> = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let data = &bytes[8 + n..];
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!({ "format": "pt" }));
    let mut payload = Vec::new();
    for (k, v) in raw_header {
        if k == "__metadata__" {
            continue;
        }
        let offs = v["data_offsets"].as_array().unwrap();
        let (a, b) = (offs[0].as_u64().unwrap() as usize, offs[1].as_u64().unwrap() as usize);
        let begin = payload.len();
        payload.extend_from_slice(&data[a..b]);
        let end = payload.len();
        header.insert(
            k,
            json!({
                "dtype": v["dtype"],
                "shape": v["shape"],
                "data_offsets": [begin, end]
            }),
        );
    }
    for (name, shape, values) in extra {
        let begin = payload.len();
        for &v in *values {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let end = payload.len();
        header.insert((*name).to_owned(), json!({ "dtype": "F32", "shape": shape, "data_offsets": [begin, end] }));
    }
    let mut h = serde_json::to_vec(&header).unwrap();
    while !h.len().is_multiple_of(8) {
        h.push(b' ');
    }
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(payload);
    out
}

#[test]
fn a_classifier_bundle_checks_config_id2label_and_verifies_head_and_reference_logits() {
    let d = scratch("classifier-seal");
    let mut r = Recipe::load(&tiny_recipe(&d)).unwrap();
    r.manifest["classifier"] = json!({
        "kind": "CLASSIFIER_SEQUENCE",
        "labels": ["NEG", "POS"]
    });
    let tn = r.manifest["artifacts"][0]["tensor_names"].as_object_mut().unwrap();
    tn.insert("pooler_weight".into(), json!("pooler.dense.weight"));
    tn.insert("pooler_bias".into(), json!("pooler.dense.bias"));
    tn.insert("classifier_weight".into(), json!("classifier.weight"));
    tn.insert("classifier_bias".into(), json!("classifier.bias"));

    // reference::cases forwards the classifier block into cases.json.
    let spec = reference::cases(&r).unwrap();
    assert_eq!(spec["classifier"], r.manifest["classifier"]);

    let up = upstream(&d);
    let real_weights = fs::read(root().join("testdata/tiny-bert-bundle/weights/model.safetensors")).unwrap();
    let weights_with_head = append_safetensors(
        &real_weights,
        &[("classifier.weight", &[2, 32], &[0.1f32; 64]), ("classifier.bias", &[2], &[-0.05f32, 0.15f32])],
    );
    fs::write(up.join("model.safetensors"), &weights_with_head).unwrap();

    // Upstream config.json with mismatched id2label is refused by stage.
    fs::write(up.join("config.json"), br#"{"id2label":{"0":"NEG","1":"WRONG"}}"#).unwrap();
    let bundle = d.join("bundle");
    let e = seal::stage(&r, &up, &bundle).unwrap_err();
    assert!(e.contains("id2label[\"1\"] is \"WRONG\", manifest.classifier.labels[1] is \"POS\""), "{e}");

    // Matching id2label passes stage.
    fs::write(up.join("config.json"), br#"{"id2label":{"0":"NEG","1":"POS"}}"#).unwrap();
    seal::stage(&r, &up, &bundle).unwrap();

    let ref_bytes = fs::read(root().join("testdata/tiny-bert-reference/reference.safetensors")).unwrap();
    let n_cases = r.manifest["reference"]["cases"].as_array().unwrap().len() as u64;
    let ref_with_logits =
        append_safetensors(&ref_bytes, &[("logits", &[n_cases, 2], &vec![0.25f32; (n_cases * 2) as usize])]);
    fs::create_dir_all(bundle.join("reference")).unwrap();
    fs::write(bundle.join("reference/reference.safetensors"), &ref_with_logits).unwrap();

    let pb = reference::produced_by(&reported(), CONTAINER).unwrap();
    let made = converted(&r, &bundle);
    seal::seal(&r, &bundle, pb, made).expect("classifier bundle sealed and verified");

    // Wrong logits shape in reference.safetensors is refused by verify.
    let bad_ref = append_safetensors(&ref_bytes, &[("logits", &[n_cases, 3], &vec![0.25f32; (n_cases * 3) as usize])]);
    fs::write(bundle.join("reference/reference.safetensors"), &bad_ref).unwrap();
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    let e = seal::seal_manifest(m, &bundle).unwrap_err();
    assert!(e.contains(&format!("logits are [{n_cases}, 3], not [{n_cases}, 2]")), "{e}");

    fs::remove_dir_all(d).unwrap();
}
