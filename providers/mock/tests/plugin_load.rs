//! Load the mock provider through the plugin ABI and run every task through it.

use std::path::PathBuf;

use turbo::abi;
use turbo::mock::{write_mock_bundle, MockBundleKind};
use turbo::{
    ClassifyOptions, Context, ContextDesc, DeviceSelector, EmbedOptions, GenerateDesc, Message, ModelDesc,
    RerankOptions, RuntimeDesc, SelectPolicy, SessionDesc,
};

/// Path of the built cdylib. Cargo builds the package's own library target
/// before its integration tests, so the artifact exists when this runs.
fn plugin_path() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from).unwrap_or_else(|| dir.join("target"));
    let name = format!("{}turbo_provider_mock{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX);
    // `cargo test` compiles the cdylib into `deps/` (plain name on macOS,
    // hash suffix on Linux) and does not always uplift it to the profile
    // directory; the uplifted copy can also be stale from an older
    // `cargo build`, so `deps/` is consulted first.
    let deps = target.join(profile).join("deps");
    if deps.join(&name).is_file() {
        return deps.join(&name);
    }
    let path = target.join(profile).join(&name);
    let stem = format!("{}turbo_provider_mock-", std::env::consts::DLL_PREFIX);
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&deps)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| {
                    let f = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    f.starts_with(&stem) && f.ends_with(std::env::consts::DLL_SUFFIX)
                })
                .collect()
        })
        .unwrap_or_default();
    if path.is_file() {
        candidates.push(path.clone());
    }
    candidates.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    candidates.pop().unwrap_or_else(|| {
        panic!(
            "provider library {} was not built and {} holds no {}*{}; run `cargo build -p turbo-provider-mock`",
            path.display(),
            deps.display(),
            stem,
            std::env::consts::DLL_SUFFIX
        )
    })
}

fn runtime_with_plugin() -> std::sync::Arc<turbo::Runtime> {
    turbo::create_runtime(RuntimeDesc {
        no_default_providers: true,
        provider_paths: vec![plugin_path().to_string_lossy().into_owned()],
        log: None,
    })
    .expect("runtime with the mock plugin")
}

#[test]
fn plugin_loads_and_enumerates_devices() {
    let rt = runtime_with_plugin();
    assert_eq!(rt.device_count(), 2);
    let providers = rt.providers();
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].id(), "mock");
    assert_eq!(providers[0].version(), "2.0.0-alpha.0");
    assert!(rt.failures().is_empty(), "{:?}", rt.failures());
    let idx = rt.select(&DeviceSelector::default()).unwrap();
    assert_eq!(rt.device(idx).unwrap().info.kind, turbo::DeviceKind::Accel);
}

#[test]
fn loading_the_same_provider_twice_is_rejected() {
    let rt = runtime_with_plugin();
    let err = rt.load_provider(&plugin_path()).unwrap_err();
    assert_eq!(err.code(), abi::TURBO_E_PROVIDER_LOAD);
    assert!(err.message().contains("already registered"));
}

#[test]
fn builtin_and_plugin_mock_produce_identical_embeddings() {
    let tmp = tempfile::tempdir().unwrap();
    write_mock_bundle(tmp.path(), MockBundleKind::Embedding).unwrap();
    let texts = ["hello world", "the quick brown fox", ""];
    let mut vectors = Vec::new();
    for plugin in [false, true] {
        let rt = if plugin { runtime_with_plugin() } else { turbo::create_runtime(RuntimeDesc::default()).unwrap() };
        let idx = rt.select(&DeviceSelector::default()).unwrap();
        let ctx = Context::create(rt, idx, &ContextDesc::default()).unwrap();
        let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
        assert_eq!(model.info().dim, 8);
        assert_eq!(model.info().provider_id, "mock");
        let session = model.create_session(&SessionDesc::default()).unwrap();
        session
            .write_text(&texts, &EmbedOptions { prompt_role: turbo::PromptRole::Query, ..Default::default() })
            .unwrap();
        let result = session.run(&Default::default()).unwrap();
        let out = result.output(0).unwrap();
        assert_eq!(out.shape, vec![3, 8]);
        let mut bytes = vec![0u8; out.logical_bytes().unwrap() as usize];
        result.read(0, &mut bytes).unwrap();
        vectors.push(bytes);
        let stats = session.stats().unwrap();
        assert_eq!(stats.runs, 1);
    }
    assert_eq!(vectors[0], vectors[1], "plugin path must be bit-identical to the built-in path");
}

#[test]
fn plugin_rerank_classify_and_generic_run() {
    let tmp = tempfile::tempdir().unwrap();
    let rt = runtime_with_plugin();
    let idx = rt
        // Ordinal 1, the accelerator: the mock's CPU device does not
        // advertise TURBO_CAP_OPT_TOP_N (`MOCK_CPU_CAPS`), and this case
        // reranks with `top_n` and `return_sorted`.
        .select(&DeviceSelector {
            policy: SelectPolicy::Explicit,
            provider_id: "mock".into(),
            ordinal: 1,
            ..Default::default()
        })
        .unwrap();
    let ctx = Context::create(rt, idx, &ContextDesc::default()).unwrap();

    let rr = tmp.path().join("rerank");
    std::fs::create_dir(&rr).unwrap();
    write_mock_bundle(&rr, MockBundleKind::Reranker).unwrap();
    let model = ctx.load_model(&rr, &ModelDesc::default()).unwrap();
    let session = model.create_session(&SessionDesc::default()).unwrap();
    session
        .write_pairs(
            "cats",
            &["cats are great", "dogs bark", "cats cats"],
            &RerankOptions { top_n: 2, return_sorted: true, ..Default::default() },
        )
        .unwrap();
    let result = session.run(&Default::default()).unwrap();
    assert_eq!(result.outputs().len(), 2);
    assert_eq!(&*result.outputs()[1].name, "sorted");
    assert_eq!(result.outputs()[1].shape, vec![2], "top_n 2 bounds the sorted output");
    let mut sorted = [0u8; 8];
    result.read(1, &mut sorted).unwrap();
    let ranked: Vec<i32> = sorted.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(ranked[0], 2, "the document with the most query overlap ranks first");
    assert_ne!(ranked[1], ranked[0], "the sorted output must not repeat a document");
    // The whole ranking is checkable against the scores, which stay in
    // input order and cover every document.
    let mut scores = [0u8; 12];
    result.read(0, &mut scores).unwrap();
    let scores: Vec<f32> = scores.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(scores.len(), 3, "top_n cuts the ranking, not the scores");
    assert!(scores.iter().all(|s| (0.0..=1.0).contains(s)), "activated scores are in 0..=1: {scores:?}");
    assert!(scores[ranked[0] as usize] >= scores[ranked[1] as usize], "{scores:?} / {ranked:?}");
    let left_out = (0..3).find(|i| !ranked.contains(i)).expect("one document is outside the top 2");
    assert!(
        scores[left_out as usize] <= scores[ranked[1] as usize],
        "document {left_out} scored {} but was left out of the top 2: {scores:?}",
        scores[left_out as usize]
    );
    drop(result);

    let tc = tmp.path().join("tc");
    std::fs::create_dir(&tc).unwrap();
    write_mock_bundle(&tc, MockBundleKind::TokenClassifier).unwrap();
    let model = ctx.load_model(&tc, &ModelDesc::default()).unwrap();
    assert_eq!(model.info().labels, vec!["O", "PER", "LOC"]);
    let session = model.create_session(&SessionDesc::default()).unwrap();
    let text = "Ada visited Berlin";
    session
        .write_text_classify(&[text], &ClassifyOptions { aggregation: turbo::Aggregation::None, ..Default::default() })
        .unwrap();
    let result = session.run(&Default::default()).unwrap();
    // One span per word whose hash-derived label is not the outside tag.
    assert!(!result.spans().is_empty() && result.spans().len() <= 3, "{:?}", result.spans());
    for s in result.spans() {
        let word = &text[s.byte_start as usize..s.byte_end as usize];
        assert!(!word.contains(' '));
        assert_ne!(s.label, 0, "outside words yield no span");
    }
    drop(result);

    let gen = tmp.path().join("gen");
    std::fs::create_dir(&gen).unwrap();
    write_mock_bundle(&gen, MockBundleKind::Generic).unwrap();
    let model = ctx.load_model(&gen, &ModelDesc::default()).unwrap();
    assert_eq!(model.info().inputs[0].name, "x");
    let session = model.create_session(&SessionDesc::default()).unwrap();
    let x = ctx.alloc(&turbo::BufferDesc::packed(turbo::Placement::Host, turbo::DType::F32, &[2, 3]).unwrap()).unwrap();
    let p = x.host_ptr().unwrap().as_ptr().cast::<f32>();
    for i in 0..6 {
        unsafe { *p.add(i) = i as f32 };
    }
    session.bind("x", &x).unwrap();
    let result = session.run(&Default::default()).unwrap();
    let mut y = vec![0u8; 24];
    result.read(0, &mut y).unwrap();
    let ys: Vec<f32> = y.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(ys, vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
}

#[test]
fn plugin_generation_streams_and_cancels() {
    let tmp = tempfile::tempdir().unwrap();
    write_mock_bundle(tmp.path(), MockBundleKind::Generative).unwrap();
    let rt = runtime_with_plugin();
    let idx = rt.select(&DeviceSelector::default()).unwrap();
    let ctx = Context::create(rt, idx, &ContextDesc::default()).unwrap();
    let model = ctx.load_model(tmp.path(), &ModelDesc::default()).unwrap();
    let g = model.create_generation(&GenerateDesc { max_new_tokens: 6, logprobs: 2, ..Default::default() }).unwrap();
    g.prompt(&[Message { role: "user", content: "hi there" }]).unwrap();
    let mut n = 0;
    loop {
        let chunk = g.step().unwrap();
        n += chunk.tokens.len();
        assert_eq!(chunk.logprobs.len(), chunk.tokens.len() * 2);
        if n == 3 {
            drop(chunk);
            g.cancel().unwrap();
            let last = g.step().unwrap();
            assert!(last.done);
            assert_eq!(last.finish_reason, turbo::FinishReason::Cancelled);
            break;
        }
        assert!(!chunk.done);
    }
    assert_eq!(g.step().unwrap_err().code(), abi::TURBO_E_INVALID_STATE);
}
