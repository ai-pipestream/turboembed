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
}
