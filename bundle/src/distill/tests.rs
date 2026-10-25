//! Static tokenization, term planning and progress through real tiny bundles.

use super::*;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), target).unwrap();
        }
    }
}

#[test]
fn staged_quality_ids_keep_static_cut_and_special_token_rules() {
    let base = Bundle::open(&root().join("testdata/tiny-bert-bundle")).unwrap();
    let tok = Tokenizer::load(&base).unwrap();
    let bytes = base.read_verified(&base.manifest.tokenizer.file).unwrap();
    let vocab = vocabulary(&bytes, base.manifest.architecture.as_ref().unwrap().vocab_size as usize).unwrap();
    let median = median_chars(&vocab);
    let source = root().join("testdata/tiny-static-bundle");
    assert_eq!(bytes, fs::read(source.join("tokenizer.json")).unwrap());
    let work = std::env::temp_dir().join(format!("turbo-bundle-quality-{}", std::process::id()));
    copy_tree(&source, &work);
    let mut manifest: Value = serde_json::from_slice(&fs::read(work.join("manifest.json")).unwrap()).unwrap();
    manifest.as_object_mut().unwrap().remove("files");
    manifest["static_embedding"].as_object_mut().unwrap().remove("distilled_from");
    fs::remove_file(work.join("manifest.json")).unwrap();
    let cases = [
        "".into(),
        "hello".into(),
        "[CLS] hello [SEP]".into(),
        "🙂".into(),
        "🙂 hello".into(),
        "é café 😀 world".into(),
        " ".repeat(400) + "hello",
        "hello ".repeat(400),
        "q".repeat(200),
    ];
    let texts: Vec<_> = cases.into_iter().map(|text| quality::QualityText { text, group: 0 }).collect();
    for max in [1, 2, 8, 32] {
        manifest["static_embedding"]["max_length"] = json!(max);
        let staged = quality::staged_ids(&manifest, &work, &texts).unwrap();
        for (case, got) in texts.iter().zip(&staged) {
            assert_eq!(*got, static_ids(&tok, median, &case.text, max).unwrap(), "max={max}, text={:?}", case.text);
        }
        assert!(staged[0].is_empty(), "static template must not add CLS/SEP");
        assert!(staged[6].is_empty(), "character cut must precede the far-away hello");
        assert!(!work.join("manifest.json").exists());
    }
    fs::remove_dir_all(work).unwrap();
}

#[test]
fn real_wordpiece_planner_handles_long_and_normalized_duplicate_terms() {
    let base = Bundle::open(&root().join("testdata/tiny-bert-bundle")).unwrap();
    let tok = Tokenizer::load(&base).unwrap();
    let terms = ["neuroimaging".into(), "Néuroimaging".into(), "q".repeat(64), "  \t\n".into()];
    let plan = extra::plan(&tok, &base.manifest, &terms).unwrap();
    assert_eq!(plan.entries, ["neuroimaging"]);
    assert_eq!(
        plan.refused,
        [
            (terms[1].clone(), extra::Refusal::Duplicate),
            (terms[2].clone(), extra::Refusal::Long),
            (terms[3].clone(), extra::Refusal::Empty),
        ]
    );
}

#[test]
fn base_embedding_progress_finishes_exactly_at_actual_rows() {
    let path = root().join("testdata/tiny-bert-bundle");
    let base = Bundle::open(&path).unwrap();
    let runtime = api::Runtime::create().unwrap();
    let context = runtime.cpu().unwrap();
    let model = context.load(&path).unwrap();
    let info = model.info().unwrap();
    let h = base.manifest.architecture.as_ref().unwrap().hidden as usize;
    let batch = info.max_batch as usize;
    let mut failures = Vec::new();
    for n in [1, 63 * batch + 1, 64 * batch, 64 * batch + 1] {
        let kept: Vec<_> = (0..n).collect();
        let mut events = Vec::new();
        let rows = embed_entries(&model, &info, &base.manifest, &kept, h, &mut |done| events.push(done)).unwrap();
        assert_eq!(rows.len(), n * h);
        assert!(rows.iter().all(|x| x.is_finite()));
        let bytes: Vec<_> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
        eprintln!("actual base rows={n}, progress={events:?}, vectors_sha256={}", turbo::bundle::sha256_hex(&bytes));
        if events.last() != Some(&n) || events.iter().any(|&done| done > n) || !events.windows(2).all(|w| w[0] < w[1]) {
            failures.push((n, events));
        }
    }
    assert!(failures.is_empty(), "progress omits or overstates actual completed rows: {failures:?}");
}
