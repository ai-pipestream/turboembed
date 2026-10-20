//! The Open Inference Protocol against the server in-process on the CPU
//! with testdata/tiny-bert-bundle: every RPC, against the bundle's
//! reference vectors and the statuses docs/kserve.md gives.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;

#[tokio::test]
async fn readiness_before_and_after_load() {
    let s =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), 1)], MAX, Serve::default()).await.unwrap();
    let mut c = client(&s).await;
    let ready = |name: &str, version: &str| ModelReadyRequest { name: name.into(), version: version.into() };

    // Answering, nothing loaded.
    assert!(c.server_live(ServerLiveRequest {}).await.unwrap().into_inner().live);
    assert!(!c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    assert!(!c.model_ready(ready(NAME, "")).await.unwrap().into_inner().ready);
    let e = c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, 261, 0));
    assert!(e.message().starts_with("TURBO_E_INVALID_STATE: "), "{e:?}");
    assert_eq!(refusal(&refused(&mut c, texts(&["a"])).await), (Code::FailedPrecondition, 261, 0));
    let e = c.model_ready(ready("nope", "")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    assert!(c.server_metadata(ServerMetadataRequest {}).await.is_ok());

    s.load().await.unwrap();
    assert!(c.server_live(ServerLiveRequest {}).await.unwrap().into_inner().live);
    assert!(c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    assert!(c.model_ready(ready(NAME, "")).await.unwrap().into_inner().ready);
    assert!(c.model_ready(ready(NAME, "1")).await.unwrap().into_inner().ready);
    let e = c.model_ready(ready(NAME, "2")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    s.stop().await;
}

#[tokio::test]
async fn server_metadata() {
    let (s, mut c) = serve(1).await;
    let m = c.server_metadata(ServerMetadataRequest {}).await.unwrap().into_inner();
    assert_eq!(m.name, "turboembed");
    assert_eq!(m.version, version());
    assert!(m.version.starts_with("0.1.0") && m.version.contains("cpu"), "{}", m.version);
    assert!(m.extensions.is_empty());
    s.stop().await;
}

#[tokio::test]
async fn model_metadata() {
    let (s, mut c) = serve(2).await;
    let m =
        c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: "1".into() }).await.unwrap().into_inner();
    assert_eq!(m.name, NAME);
    assert_eq!(m.versions, ["1"]);
    assert_eq!(m.platform, "");
    let t = |v: &[model_metadata_response::TensorMetadata]| {
        v.iter().map(|t| (t.name.clone(), t.datatype.clone(), t.shape.clone())).collect::<Vec<_>>()
    };
    let st = |a: &str, b: &str, c: &[i64]| (a.to_string(), b.to_string(), c.to_vec());
    assert_eq!(
        t(&m.inputs),
        [
            st("texts", "BYTES", &[-1]),
            st("ids", "INT32", &[-1, -1]),
            st("mask", "INT32", &[-1, -1]),
            st("types", "INT32", &[-1, -1])
        ]
    );
    assert_eq!(t(&m.outputs), [st("vectors", "FP32", &[-1, -1])]);

    // Every key the doc lists, and no other.
    let mut keys: Vec<&str> = m.properties.keys().map(String::as_str).collect();
    keys.sort();
    let mut want: Vec<String> = Vec::new();
    for (s, fields) in [
        (
            "model_info",
            &[
                "task",
                "dim",
                "pooling",
                "normalize",
                "max_seq",
                "max_batch",
                "dtype",
                "model_id",
                "revision",
                "manifest_sha256",
                "artifact_sha256",
                "tokenizer_sha256",
                "prefix_query",
                "prefix_document",
                "output_dims_count",
                "output_dims",
            ][..],
        ),
        ("session_info", &["max_batch", "max_seq", "precision", "compute_dtype"][..]),
        (
            "device_info",
            &[
                "kind",
                "ordinal",
                "unified_memory",
                "memory_total",
                "arch",
                "name",
                "vendor",
                "backend",
                "runtime_version",
                "driver_version",
            ][..],
        ),
        (
            "capability",
            &["status", "dtype", "options_honored", "cosine_floor", "speed_ratio", "benchmark", "reason"][..],
        ),
    ] {
        want.extend(fields.iter().map(|f| format!("{s}.{f}")));
    }
    want.sort();
    assert_eq!(keys, want);

    // The values, against the library read directly.
    let p = |k: &str| m.properties[k].as_str();
    assert_eq!(p("model_info.task"), "TASK_EMBED");
    assert_eq!(p("model_info.dim"), "32");
    assert_eq!(p("model_info.pooling"), "POOLING_MEAN");
    assert_eq!(p("model_info.normalize"), "NORMALIZE_L2");
    assert_eq!(p("model_info.max_seq"), "64");
    assert_eq!(p("model_info.max_batch"), "64");
    assert_eq!(p("model_info.dtype"), "DTYPE_F32");
    assert_eq!(p("model_info.model_id"), "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(p("model_info.revision"), "1");
    assert_eq!(p("model_info.prefix_query"), "query: ");
    assert_eq!(p("model_info.prefix_document"), "");
    assert_eq!(p("model_info.output_dims_count"), "0");
    assert_eq!(p("model_info.output_dims"), "");
    assert_eq!(p("model_info.manifest_sha256").len(), 64);
    assert_eq!(p("session_info.max_batch"), "64");
    assert_eq!(p("session_info.max_seq"), "64");
    assert_eq!(p("session_info.precision"), "PRECISION_MODEL");
    assert_eq!(p("session_info.compute_dtype"), "DTYPE_F32");
    let rt = Runtime::create().unwrap();
    let di = rt.device_info(cpu()).unwrap();
    assert_eq!(p("device_info.kind"), "DEVICE_CPU");
    assert_eq!(p("device_info.ordinal"), di.ordinal.to_string());
    assert_eq!(p("device_info.unified_memory"), di.unified_memory.to_string());
    assert_eq!(p("device_info.memory_total"), di.memory_total.to_string());
    assert_eq!(p("device_info.arch"), field(&di.arch));
    assert_eq!(p("device_info.name"), field(&di.name));
    assert_eq!(p("device_info.backend"), "cpu");
    let cap = rt.capability(cpu(), turbo::TURBO_TASK_EMBED, 0).unwrap();
    assert_eq!(p("capability.options_honored"), cap.options_honored.to_string());
    assert_eq!(p("capability.cosine_floor").parse::<f32>().unwrap(), cap.cosine_floor);
    assert_eq!(p("capability.speed_ratio").parse::<f32>().unwrap(), cap.speed_ratio);
    assert_eq!(p("capability.benchmark"), field(&cap.benchmark));
    assert_eq!(p("capability.reason"), field(&cap.reason));
    assert!(p("capability.status").starts_with("CAP_"));
    s.stop().await;
}

#[tokio::test]
async fn texts_match_the_reference() {
    let (s, mut c) = serve(1).await;
    let cases = reference();
    for role in ["PROMPT_NONE", "PROMPT_QUERY", "PROMPT_DOCUMENT"] {
        let group: Vec<&Case> = cases.iter().filter(|k| k.role == role).collect();
        assert!(!group.is_empty(), "{role}");
        let t: Vec<&str> = group.iter().map(|k| k.text.as_str()).collect();
        let b: Vec<&[u8]> = t.iter().map(|x| x.as_bytes()).collect();
        for (form, mut r) in [("typed", texts(&t)), ("raw", raw_texts(&b))] {
            if role != "PROMPT_NONE" {
                r = with(r, "prompt_role", st(role));
            }
            r.id = format!("{role}-{form}");
            let resp = c.model_infer(r).await.unwrap().into_inner();
            assert_eq!(resp.model_name, NAME);
            assert_eq!(resp.model_version, "1");
            assert_eq!(resp.id, format!("{role}-{form}"));
            assert_eq!(resp.outputs[0].shape, [group.len() as i64, 32]);
            for (k, v) in group.iter().zip(vectors(&resp)) {
                matches(&format!("{form} {role} `{}`", k.text), &v, &k.vector);
            }

            // The run's summary.
            let p =
                |k: &str| resp.parameters.get(k).unwrap_or_else(|| panic!("no {k}")).parameter_choice.clone().unwrap();
            assert_eq!(p("task"), st("TASK_EMBED"));
            assert_eq!(p("batch"), int(group.len() as i64));
            assert_eq!(p("dim"), int(32));
            assert_eq!(p("dtype"), st("DTYPE_F32"));
            assert_eq!(p("compute_dtype"), st("DTYPE_F32"));
            assert_eq!(p("bytes"), int(group.len() as i64 * 32 * 4));
            assert_eq!(p("backend"), st("cpu"));
            assert_eq!(p("stage_count"), int(7));
            assert_eq!(p("stage.EMBED_STAGE_TOKENIZE"), st("STAGE_HOST"));
            assert!(matches!(p("placement"), ParameterChoice::StringParam(ref x) if x.starts_with("PLACE_")));
            let mut keys: Vec<&str> = resp.parameters.keys().map(String::as_str).collect();
            keys.sort();
            let mut want = vec![
                "task",
                "batch",
                "dim",
                "dtype",
                "compute_dtype",
                "placement",
                "device",
                "bytes",
                "h2d_bytes",
                "d2h_bytes",
                "host_allocs",
                "device_allocs",
                "backend",
                "arch",
                "runtime_version",
                "manifest_sha256",
                "artifact_sha256",
                "tokenizer_sha256",
                "stage_count",
                "stage.EMBED_STAGE_TOKENIZE",
                "stage.EMBED_STAGE_UPLOAD",
                "stage.EMBED_STAGE_LOOKUP",
                "stage.EMBED_STAGE_ENCODE",
                "stage.EMBED_STAGE_POOL",
                "stage.EMBED_STAGE_NORMALIZE",
                "stage.EMBED_STAGE_DOWNLOAD",
            ];
            want.sort();
            assert_eq!(keys, want);
        }
    }

    // Every case in one request, each text carrying its own prefix.
    let prefixed: Vec<String> =
        cases.iter().map(|k| format!("{}{}", if k.role == "PROMPT_QUERY" { "query: " } else { "" }, k.text)).collect();
    let t: Vec<&str> = prefixed.iter().map(String::as_str).collect();
    let mut r = texts(&t);
    r.outputs = vec![InferRequestedOutputTensor { name: "vectors".into(), parameters: HashMap::new() }];
    for (k, v) in cases.iter().zip(embed(&mut c, r).await) {
        matches(&format!("one request, `{}`", k.text), &v, &k.vector);
    }
    s.stop().await;
}

#[tokio::test]
async fn token_rows_match_the_reference() {
    let (s, mut c) = serve(1).await;
    let cases = reference();
    let rows: Vec<Vec<i32>> = cases.iter().map(|k| k.ids.clone()).collect();
    for with_types in [false, true] {
        for (form, r) in [("typed", typed_tokens(&rows, with_types)), ("raw", raw_tokens(&rows, with_types))] {
            let resp = c.model_infer(r).await.unwrap().into_inner();
            for (k, v) in cases.iter().zip(vectors(&resp)) {
                matches(&format!("{form} tokens (types {with_types}) `{}`", k.text), &v, &k.vector);
            }
            let tok = &resp.parameters["stage.EMBED_STAGE_TOKENIZE"];
            assert_eq!(tok.parameter_choice, Some(st("STAGE_UNUSED")));
        }
    }
    // One row at a time through the same session and its buffers.
    for k in &cases {
        let v = embed(&mut c, raw_tokens(std::slice::from_ref(&k.ids), true)).await;
        matches("raw, one row", &v[0], &k.vector);
    }
    s.stop().await;
}

#[tokio::test]
async fn every_parameter() {
    let (s, mut c) = serve(1).await;
    let cases = reference();
    let long = cases.iter().max_by_key(|k| k.text.len()).unwrap();
    let short = &cases[1];

    // truncate
    let e = refused(&mut c, with(texts(&[&long.text]), "truncate", st("TRUNCATE_NONE"))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    let v = embed(&mut c, with(texts(&[&long.text]), "truncate", st("TRUNCATE_RIGHT"))).await;
    matches("TRUNCATE_RIGHT", &v[0], &long.vector);
    let v = embed(&mut c, with(texts(&[&long.text]), "truncate", st("TRUNCATE_MODEL"))).await;
    matches("TRUNCATE_MODEL", &v[0], &long.vector);
    let v = embed(&mut c, with(texts(&[&long.text]), "truncate", st("TRUNCATE_LEFT"))).await;
    assert!(cosine(&v[0], &long.vector) < 0.9999, "TRUNCATE_LEFT keeps the end");
    let e =
        refused(&mut c, with(typed_tokens(std::slice::from_ref(&short.ids), false), "truncate", st("TRUNCATE_RIGHT")))
            .await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 1));
    assert!(e.message().starts_with("TURBO_E_INVALID_ARGUMENT field 1 (truncate): "), "{}", e.message());

    // max_tokens
    let v = embed(&mut c, with(texts(&[&short.text]), "max_tokens", int(4))).await;
    assert!(cosine(&v[0], &short.vector) < 0.9999, "max_tokens 4 cuts the text");
    let v = embed(&mut c, with(texts(&[&short.text]), "max_tokens", int(64))).await;
    matches("max_tokens 64", &v[0], &short.vector);
    let e = refused(&mut c, with(texts(&[&short.text]), "max_tokens", int(65))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    let e = refused(&mut c, with(raw_tokens(std::slice::from_ref(&short.ids), false), "max_tokens", int(4))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    let v = embed(
        &mut c,
        with(raw_tokens(std::slice::from_ref(&short.ids), false), "max_tokens", int(short.ids.len() as i64)),
    )
    .await;
    matches("max_tokens checked", &v[0], &short.vector);

    // prompt_role
    let q = cases.iter().find(|k| k.role == "PROMPT_QUERY").unwrap();
    let v = embed(&mut c, with(texts(&[&q.text]), "prompt_role", st("PROMPT_QUERY"))).await;
    matches("PROMPT_QUERY", &v[0], &q.vector);
    let v = embed(&mut c, with(texts(&[&q.text]), "prompt_role", st("PROMPT_NONE"))).await;
    assert!(cosine(&v[0], &q.vector) < 0.9999, "PROMPT_NONE has no prefix");
    let e =
        refused(&mut c, with(raw_tokens(std::slice::from_ref(&q.ids), false), "prompt_role", st("PROMPT_QUERY"))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 3));

    // normalize
    let v = embed(&mut c, with(texts(&[&short.text]), "normalize", st("NORMALIZE_NONE"))).await;
    assert!((norm(&v[0]) - 1.0).abs() > 1e-3, "NORMALIZE_NONE: norm {}", norm(&v[0]));
    assert!(cosine(&v[0], &short.vector) >= 0.9999);
    let v = embed(&mut c, with(texts(&[&short.text]), "normalize", st("NORMALIZE_L2"))).await;
    matches("NORMALIZE_L2", &v[0], &short.vector);
    let v = embed(&mut c, with(texts(&[&short.text]), "normalize", st("NORMALIZE_MODEL"))).await;
    matches("NORMALIZE_MODEL", &v[0], &short.vector);

    // pooling
    let v = embed(&mut c, with(texts(&[&short.text]), "pooling", st("POOLING_MEAN"))).await;
    matches("POOLING_MEAN", &v[0], &short.vector);
    for p in ["POOLING_CLS", "POOLING_LAST"] {
        let v = embed(&mut c, with(texts(&[&short.text]), "pooling", st(p))).await;
        assert!(cosine(&v[0], &short.vector) < 0.9999, "{p} is not the mean");
        assert!((norm(&v[0]) - 1.0).abs() < 1e-4, "{p} is still normalized");
    }
    let v = embed(&mut c, with(texts(&[&short.text]), "pooling", st("POOLING_MODEL"))).await;
    matches("POOLING_MODEL", &v[0], &short.vector);

    // output_dim
    let v = embed(&mut c, with(texts(&[&short.text]), "output_dim", int(32))).await;
    matches("output_dim 32", &v[0], &short.vector);
    let v = embed(&mut c, with(texts(&[&short.text]), "output_dim", int(0))).await;
    matches("output_dim 0", &v[0], &short.vector);
    let e = refused(&mut c, with(texts(&[&short.text]), "output_dim", int(16))).await;
    assert_eq!(refusal(&e), (Code::Unimplemented, 513, 6));
    assert!(e.message().starts_with("TURBO_E_UNSUPPORTED_OPTION field 6 (output_dim): "), "{}", e.message());
    let e = refused(&mut c, with(texts(&[&short.text]), "output_dim", int(33))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 6));
    s.stop().await;
}

#[tokio::test]
async fn parameter_refusals() {
    let (s, mut c) = serve(1).await;
    let r = || texts(&["a"]);

    // Not a field of turbo_embed_options, by name.
    for name in ["precision", "device", "priority", "Truncate", "TURBO_TRUNCATE"] {
        let e = refused(&mut c, with(r(), name, st("PRECISION_EXACT"))).await;
        assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 0), "{name}");
        assert!(e.message().starts_with("TURBO_E_INVALID_ARGUMENT: ") && e.message().contains(name), "{}", e.message());
    }

    // Another InferParameter type, or a number out of range: the field named.
    for (name, field, v) in [
        ("truncate", 1, int(2)),
        ("max_tokens", 2, st("8")),
        ("max_tokens", 2, ParameterChoice::Uint64Param(8)),
        ("max_tokens", 2, ParameterChoice::DoubleParam(8.0)),
        ("max_tokens", 2, int(-1)),
        ("max_tokens", 2, int(4294967296)),
        ("prompt_role", 3, ParameterChoice::BoolParam(true)),
        ("normalize", 4, int(0)),
        ("pooling", 5, ParameterChoice::DoubleParam(1.0)),
        ("output_dim", 6, st("32")),
        ("output_dim", 6, int(-32)),
    ] {
        let e = refused(&mut c, with(r(), name, v.clone())).await;
        assert_eq!(refusal(&e), (Code::InvalidArgument, 256, field), "{name} {v:?}");
        let prefix = format!("TURBO_E_INVALID_ARGUMENT field {field} ({name}): ");
        assert!(e.message().starts_with(&prefix), "{}", e.message());
    }

    // A string that is no constant of its field.
    for (name, v) in [
        ("truncate", "truncate_right"),
        ("truncate", "TURBO_TRUNCATE_RIGHT"),
        ("truncate", "PROMPT_QUERY"),
        ("prompt_role", "QUERY"),
        ("normalize", "NORMALIZE_l2"),
        ("pooling", "POOLING_MAX"),
    ] {
        let e = refused(&mut c, with(r(), name, st(v))).await;
        assert_eq!(refusal(&e), (Code::InvalidArgument, 262, 0), "{name} {v}");
        assert!(e.message().starts_with("TURBO_E_INVALID_ENUM: "), "{}", e.message());
        assert!(e.message().contains(name) && e.message().contains(v), "{}", e.message());
    }
    s.stop().await;
}
