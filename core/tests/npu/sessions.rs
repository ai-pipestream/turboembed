use super::*;

// ---- A bundle with an OpenVINO IR ------------------------------------------------

/// TURBO_TEST_BUNDLE: a bundle with an OpenVINO IR whose artifacts[]
/// backends lists "npu".
fn ir_bundle() -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var_os("TURBO_TEST_BUNDLE").expect("TURBO_TEST_BUNDLE is not set"))
}

fn on_npu() -> Loaded {
    Loaded::load_on(&ir_bundle(), |rt| first_of(rt, "npu").expect("an npu device")).unwrap_or_else(|e| panic!("{e:?}"))
}

/// LOOKUP for the artifact this load reported. The hash is the one
/// `turbo_model_info.artifact_sha256` carries, so the expectation
/// follows the file the device compiled: host for INPUT_EMBEDDINGS,
/// device for INPUT_TOKEN_IDS.
fn lookup_stage_for(bundle: &std::path::Path, artifact_sha: &str) -> u32 {
    let opened = turbo::bundle::Bundle::open(bundle).unwrap_or_else(|e| panic!("{}: {e}", bundle.display()));
    let art = opened
        .manifest
        .artifacts
        .iter()
        .find(|a| turbo::model::artifact_sha256(&opened.manifest, a) == artifact_sha)
        .unwrap_or_else(|| panic!("no artifact hashes to {artifact_sha}"));
    match art.graph_input {
        turbo::manifest::GraphInput::Embeddings => TURBO_STAGE_HOST,
        turbo::manifest::GraphInput::TokenIds => TURBO_STAGE_DEVICE,
    }
}

#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_session_computes_in_the_compiled_dtype_and_exact_is_refused() {
    let l = on_npu();
    let want = l.info().dtype;
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        let s = Session::create(l.m, Some(&session_desc(0, 0, p))).unwrap();
        assert_eq!(s.info().compute_dtype, want, "precision {p}");
    }
    let f = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).err().expect("refused");
    assert!(f.is(UNSUPPORTED_OPTION, "EXACT") && f.field == 3, "{f:?}");
}

/// The rows go to the device as the graph's own input precisions, the
/// encoder runs on the NPU, and pooling and normalize run on the host:
/// the vectors stay in host memory, unit length by default. Lookup is
/// on the host when the loaded artifact is INPUT_EMBEDDINGS, because
/// the host gathers the word rows, and on the device for a token-id IR.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn a_run_reports_its_frames_and_where_each_stage_ran() {
    let l = on_npu();
    let lookup = lookup_stage_for(&ir_bundle(), &field(&l.info().artifact_sha256));
    let s = Session::create(l.m, Some(&session_desc(4, 0, TURBO_PRECISION_MODEL))).unwrap();
    let seq = s.info().max_seq as u64;
    let dim = l.info().dim as u64;
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let n = texts.len() as u64;
    s.write_text(&texts, None).unwrap();
    let r = s.run().unwrap();
    let info = r.info();
    assert_eq!((info.batch as u64, info.dim as u64), (n, dim));
    assert_eq!(info.placement, TURBO_PLACE_HOST);
    assert_eq!(field(&info.backend), "npu");
    // Per frame: the token rows over, at least a byte an element, and
    // the hidden states back.
    assert!(info.h2d_bytes >= n * seq, "{}", info.h2d_bytes);
    assert!(info.d2h_bytes >= seq * dim, "{}", info.d2h_bytes);
    assert_eq!((info.host_allocs, info.device_allocs), (0, 0));
    let st = &info.stage;
    assert_eq!(st[TURBO_EMBED_STAGE_TOKENIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_UPLOAD], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_LOOKUP], lookup);
    assert_eq!(st[TURBO_EMBED_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    assert_eq!(st[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_NORMALIZE], TURBO_STAGE_HOST);
    assert_eq!(st[TURBO_EMBED_STAGE_DOWNLOAD], TURBO_STAGE_DEVICE);
    for v in r.rows() {
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "unit length: {norm}");
    }
}

/// Two models on one context, their sessions run at once from two
/// threads: an execute and the synchronization that waits for it are
/// one critical section on the context's immediate command list, and
/// runs on one model take turns on its graph. Interleaved runs must
/// give each thread its own rows' vectors, the same from both models.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn two_models_on_one_context_run_at_once() {
    struct SendModel(*mut turbo_model);
    unsafe impl Send for SendModel {}

    let rt = Rt::new();
    let d = rt.npu().first().copied().expect("an npu device");
    let ctx = Ctx::create(&rt, d);
    let path = ir_bundle();
    let load = || {
        let mut m = ptr::null_mut();
        let mut err = new_error();
        let rc = unsafe { turbo_model_load(ctx.0, text(path.to_str().unwrap()), &mut m, &mut err) };
        assert_eq!(rc, 0, "{:?}", failure(rc, &err));
        m
    };
    let models = [load(), load()];
    let texts = [
        ["threads share one context", "and two compiled graphs"],
        ["each run keeps its own rows", "whatever the other is doing"],
    ];
    let vectors: Vec<Vec<Vec<f32>>> = std::thread::scope(|scope| {
        let handles: Vec<_> = models
            .iter()
            .map(|&m| {
                let m = SendModel(m);
                scope.spawn(move || {
                    // The whole wrapper moves in, not its raw field.
                    let m = m;
                    let s = Session::create(m.0, Some(&session_desc(2, 0, TURBO_PRECISION_MODEL))).unwrap();
                    let mut last = Vec::new();
                    for row in texts.iter().cycle().take(8) {
                        last = s.embed(row, None).unwrap();
                    }
                    last
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    // Both threads ended on texts[1]: one model's vectors match the
    // other's, so no run read another run's hidden states.
    for (a, b) in vectors[0].iter().zip(&vectors[1]) {
        let cos: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        assert!(cos > 0.9999, "the two models agree on the same rows: cosine {cos}");
    }
    for m in models {
        unsafe { turbo_model_release(m) };
    }
}

/// Pooling and normalize are the backend's, on the host: CLS differs
/// from mean, and NONE leaves the vector at its own length.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn pooling_and_normalize_follow_the_options() {
    let l = on_npu();
    let s = Session::create(l.m, Some(&session_desc(1, 0, TURBO_PRECISION_MODEL))).unwrap();
    let text = ["Embedding models turn text into vectors."];
    let mut o = embed_options();
    o.pooling = TURBO_POOLING_MEAN;
    let mean = s.embed(&text, Some(&o)).unwrap().remove(0);
    o.pooling = TURBO_POOLING_CLS;
    let cls = s.embed(&text, Some(&o)).unwrap().remove(0);
    let cos: f64 = mean.iter().zip(&cls).map(|(a, b)| *a as f64 * *b as f64).sum();
    assert!(cos < 0.999, "CLS and mean pooling give different vectors: cosine {cos}");
    o.pooling = TURBO_POOLING_MEAN;
    o.normalize = TURBO_NORMALIZE_NONE;
    let raw = s.embed(&text, Some(&o)).unwrap().remove(0);
    let norm = raw.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() > 1e-3, "NONE leaves the length as pooled: {norm}");
    let back: f64 = raw.iter().zip(&mean).map(|(a, b)| *a as f64 / norm * *b as f64).sum();
    assert!(back > 0.99999, "the same direction as the normalized vector: {back}");
}

/// Late pooling on the NPU: the device encodes the rows, and the host
/// pools each range from the hidden states the device hands back, in
/// range order. A range over a whole row sums the same tokens in the
/// same order as the row's mean, so its vector is the row's bit for
/// bit, including a range that runs into the row's padding. A range is
/// pooled the same way wherever it sits in the list. The session takes
/// no CPU thread count.
#[test]
#[ignore = "needs an Intel NPU and TURBO_TEST_BUNDLE with an OpenVINO IR for it"]
fn ranges_match_rows_pooled_on_the_host() {
    let l = on_npu();
    let mut d = session_desc(3, 0, TURBO_PRECISION_MODEL);
    d.cpu_threads = 2;
    let f = Session::create(l.m, Some(&d)).err().expect("refused");
    assert!(f.is(UNSUPPORTED_OPTION, "cpu_threads 2") && f.field == 6, "{f:?}");
    d.cpu_threads = 0;
    d.max_ranges = 8;
    let s = Session::create(l.m, Some(&d)).unwrap();

    let tok = Tok::create(&ir_bundle()).unwrap_or_else(|e| panic!("{e:?}"));
    let texts = [
        "Late pooling encodes a document once and pools each chunk from it.",
        "a short row",
        "Each range is the mean of its own tokens' final hidden states.",
    ];
    let rows: Vec<Vec<i32>> = texts.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let len: Vec<u32> = rows.iter().map(|r| r.len() as u32).collect();
    let t = Tokens::new(&rows, 0);
    assert!(len[1] < t.seq, "row 1 is padded, so a range over the whole of seq covers padding");
    let mut o = embed_options();
    o.pooling = TURBO_POOLING_MEAN;

    s.write_tokens(&t.batch(), Some(&o)).unwrap();
    let by_row = s.run().unwrap().rows();
    let dim = l.info().dim as usize;

    // Out of row order, a range over row 1's padding, a sub-range twice
    // with another between, and one sub-range of the last row.
    let ranges = [
        range(2, 0, len[2]),
        range(0, 1, len[0] - 1),
        range(1, 0, t.seq),
        range(0, 0, len[0]),
        range(2, 1, 4),
        range(0, 1, len[0] - 1),
    ];
    s.write_ranges(&t.batch(), &ranges, Some(&o)).unwrap();
    let r = s.run().unwrap();
    let info = r.info();
    assert_eq!((info.batch as usize, info.dim as usize), (ranges.len(), dim));
    assert_eq!((info.host_allocs, info.device_allocs), (0, 0));
    assert_eq!(info.stage[TURBO_EMBED_STAGE_ENCODE], TURBO_STAGE_DEVICE);
    assert_eq!(info.stage[TURBO_EMBED_STAGE_POOL], TURBO_STAGE_HOST);
    let by_range = r.rows();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for (i, row) in [(0, 2), (2, 1), (3, 0)] {
        assert_eq!(bits(&by_range[i]), bits(&by_row[row]), "range {i} over the whole of row {row}");
    }
    assert_eq!(bits(&by_range[1]), bits(&by_range[5]), "the same range gives the same bits wherever it is");
    for (i, v) in by_range.iter().enumerate() {
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "range {i} is unit length: {norm}");
    }
    assert_ne!(bits(&by_range[1]), bits(&by_row[0]), "a sub-range is not its row's mean");
}
