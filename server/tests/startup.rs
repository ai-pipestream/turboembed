//! The Open Inference Protocol against the server in-process on the CPU
//! with testdata/tiny-bert-bundle: every RPC, against the bundle's
//! reference vectors and the statuses docs/kserve.md gives.

#[path = "kserve/mod.rs"]
mod kserve;

use kserve::*;

#[tokio::test]
async fn startup_refusals() {
    let at = "127.0.0.1:0".parse().unwrap();
    let t = tiny();
    // Paths that name no bundle, and two models of one name.
    for bundle in [format!("{}/", t.display()), format!("{}/.", t.display()), format!("{}/..", t.display())] {
        let m = ModelConfig { bundle, ..model(&t, 1) };
        assert!(Server::start(at, vec![m], MAX, Serve::default()).await.is_err());
    }
    assert!(Server::start(at, vec![model(&t, 1), model(&t, 1)], MAX, Serve::default()).await.is_err());

    // A load failure is the call, its status, the field and the message.
    let s = Server::start(at, vec![model(&t.join("../no-such-bundle"), 1)], MAX, Serve::default()).await.unwrap();
    let e = s.load().await.unwrap_err();
    assert!(e.contains("model no-such-bundle: turbo_model_load: TURBO_E_BUNDLE_NOT_FOUND: "), "{e}");
    s.stop().await;
    let s = Server::start(at, vec![ModelConfig { max_seq: 65, ..model(&t, 1) }], MAX, Serve::default()).await.unwrap();
    let e = s.load().await.unwrap_err();
    assert!(e.contains("turbo_session_create: TURBO_E_UNSUPPORTED_OPTION field 2 (max_seq): "), "{e}");
    let mut c = client(&s).await;
    assert!(!c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    s.stop().await;
    if version().split_whitespace().skip(1).all(|b| b == "cpu") {
        let s = Server::start(at, vec![ModelConfig { device: Device::Select, ..model(&t, 1) }], MAX, Serve::default())
            .await
            .unwrap();
        let e = s.load().await.unwrap_err();
        assert!(e.contains("turbo_runtime_select: TURBO_E_DEVICE_NOT_FOUND"), "{e}");
        s.stop().await;
    }

    // A configured max_batch and max_seq are the session's limits.
    let s = Server::start(at, vec![ModelConfig { max_batch: 2, max_seq: 8, ..model(&t, 1) }], MAX, Serve::default())
        .await
        .unwrap();
    s.load().await.unwrap();
    let mut c = client(&s).await;
    let m = c
        .model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        (m.properties["session_info.max_batch"].as_str(), m.properties["session_info.max_seq"].as_str()),
        ("2", "8")
    );
    assert_eq!(refusal(&refused(&mut c, texts(&["a", "b", "c"])).await).1, 771);
    assert_eq!(refusal(&refused(&mut c, raw_tokens(&[vec![101; 9]], false)).await).1, 771);
    assert_eq!(embed(&mut c, raw_tokens(&[vec![101; 8], vec![101; 8]], false)).await.len(), 2);
    s.stop().await;
}

/// One bundle served three times in one process, under its own name at
/// PRECISION_MODEL and under two more at PRECISION_FASTEST and
/// PRECISION_EXACT: each model says its tier, and each gives the reference
/// vectors (the cpu computes every tier in F32). The tier is the model's,
/// never a request's.
#[tokio::test]
async fn a_tier_per_model() {
    let t = tiny();
    let tier = |name: &str, precision: u32| ModelConfig { name: Some(name.into()), precision, ..model(&t, 1) };
    let models = vec![model(&t, 1), tier("tiny-fastest", 1), tier("tiny-exact", 2)];
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, Serve::default()).await.unwrap();
    s.load().await.unwrap();
    let mut c = client(&s).await;
    assert!(c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    let cases = reference();
    let group: Vec<&Case> = cases.iter().filter(|k| k.role == "PROMPT_NONE").collect();
    let texts_of: Vec<&str> = group.iter().map(|k| k.text.as_str()).collect();
    let mut manifests = Vec::new();
    for (name, want) in
        [(NAME, "PRECISION_MODEL"), ("tiny-fastest", "PRECISION_FASTEST"), ("tiny-exact", "PRECISION_EXACT")]
    {
        let ready = ModelReadyRequest { name: name.into(), version: String::new() };
        assert!(c.model_ready(ready).await.unwrap().into_inner().ready, "{name}");
        let m = c
            .model_metadata(ModelMetadataRequest { name: name.into(), version: String::new() })
            .await
            .unwrap()
            .into_inner();
        assert_eq!(m.name, name);
        assert_eq!(m.properties["session_info.precision"], want, "{name}");
        assert_eq!(m.properties["session_info.compute_dtype"], "DTYPE_F32", "{name}");
        manifests.push(m.properties["model_info.manifest_sha256"].clone());

        // The batch, and then each row alone, give the reference.
        let mut r = texts(&texts_of);
        r.model_name = name.into();
        let resp = c.model_infer(r).await.unwrap().into_inner();
        assert_eq!(resp.model_name, name);
        assert_eq!(resp.outputs[0].shape, [group.len() as i64, 32]);
        for (k, v) in group.iter().zip(vectors(&resp)) {
            matches(&format!("{name} batch `{}`", k.text), &v, &k.vector);
        }
        for k in &group {
            let mut r = texts(&[&k.text]);
            r.model_name = name.into();
            let v = embed(&mut c, r).await;
            matches(&format!("{name} alone `{}`", k.text), &v[0], &k.vector);
        }
    }
    assert!(manifests.iter().all(|m| m == &manifests[0]), "one bundle: {manifests:?}");
    let e = refused(&mut c, with(texts(&["a"]), "precision", st("PRECISION_FASTEST"))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 0));
    s.stop().await;
}

/// A text longer than the session's max_seq is refused with the limit
/// named, never cut on the server's own: the request cuts it by naming
/// `max_tokens` within the limit and `truncate`. A model whose device
/// takes fewer tokens than the bundle (a fixed-shape artifact) is the case
/// a CPU session with a configured max_seq stands in for here.
#[tokio::test]
async fn a_text_over_the_sessions_max_seq() {
    let at = "127.0.0.1:0".parse().unwrap();
    let s =
        Server::start(at, vec![ModelConfig { max_seq: 8, ..model(&tiny(), 1) }], MAX, Serve::default()).await.unwrap();
    s.load().await.unwrap();
    let mut c = client(&s).await;
    let cases = reference();
    let long = cases.iter().max_by_key(|k| k.ids.len()).unwrap();
    assert!(long.ids.len() > 8, "the longest reference case has {} tokens", long.ids.len());
    let m = c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap();
    assert_eq!(m.into_inner().properties["session_info.max_seq"], "8");

    // Default options: cut at the bundle's max_seq, then over the session's.
    let e = refused(&mut c, texts(&[&long.text])).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    assert!(e.message().contains("tokens is over the session's max_seq 8"), "{}", e.message());
    // TRUNCATE_RIGHT alone cuts at the bundle's max_seq (64), still too long.
    let e = refused(&mut c, with(texts(&[&long.text]), "truncate", st("TRUNCATE_RIGHT"))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    // The request asks for the cut: max_tokens within the limit.
    let r = with(with(texts(&[&long.text]), "truncate", st("TRUNCATE_RIGHT")), "max_tokens", int(8));
    let resp = c.model_infer(r).await.unwrap().into_inner();
    assert_eq!(resp.outputs[0].shape, [1, 32]);
    let v = vectors(&resp);
    assert_eq!(norm(&v[0]).round() as i64, 1, "a normalized vector");
    // max_tokens over the limit is refused too; TRUNCATE_NONE cuts nothing.
    let r = with(with(texts(&[&long.text]), "truncate", st("TRUNCATE_RIGHT")), "max_tokens", int(9));
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::OutOfRange, 771, 0));
    let r = with(with(texts(&[&long.text]), "truncate", st("TRUNCATE_NONE")), "max_tokens", int(8));
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::OutOfRange, 771, 0));
    // A short text needs no option.
    let short = cases.iter().min_by_key(|k| k.ids.len()).unwrap();
    assert!(short.ids.len() <= 8);
    assert_eq!(embed(&mut c, texts(&[&short.text])).await.len(), 1);
    s.stop().await;
}

/// The binary stops cleanly on SIGTERM, as on SIGINT, once ready: it says
/// which signal, stops answering, and exits 0.
#[cfg(unix)]
#[test]
fn the_binary_stops_on_sigterm() {
    use std::io::{BufRead, BufReader};
    let model = format!("bundle={},device={},sessions=1", tiny().display(), cpu());
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .args(["--listen", "127.0.0.1:0", "--model", &model])
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let mut err = BufReader::new(child.stderr.take().unwrap());
    let mut lines = Vec::new();
    let mut addr = None;
    loop {
        let mut line = String::new();
        assert!(err.read_line(&mut line).unwrap() > 0, "the server exited before ready: {lines:?}");
        if let Some(a) = line.trim().strip_prefix("turbo-kserve: listening on ") {
            addr = Some(a.to_string());
        }
        let ready = line.trim() == "turbo-kserve: ready";
        lines.push(line);
        if ready {
            break;
        }
    }
    let addr = addr.expect("the listening line");
    assert!(std::net::TcpStream::connect(&addr).is_ok(), "{addr} answers");
    let kill = std::process::Command::new("kill").args(["-TERM", &child.id().to_string()]).status().unwrap();
    assert!(kill.success());
    let mut rest = String::new();
    std::io::Read::read_to_string(&mut err, &mut rest).unwrap();
    let status = child.wait().unwrap();
    assert!(status.success(), "{status}: {rest}");
    assert!(rest.contains("turbo-kserve: stopping on SIGTERM"), "{rest}");
    assert!(rest.trim_end().ends_with("turbo-kserve: stopped"), "{rest}");
    assert!(std::net::TcpStream::connect(&addr).is_err(), "{addr} is closed");
}

/// The binary exits non-zero with the library's message when a bundle does
/// not load, configured by flags or by the environment.
#[test]
fn the_binary_exits_on_a_load_failure() {
    let bundle = tiny().join("../no-such-bundle");
    let model = format!("bundle={},device={},sessions=1", bundle.display(), cpu());
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .args(["--listen", "127.0.0.1:0", "--model", &model])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("turbo_model_load: TURBO_E_BUNDLE_NOT_FOUND: "), "{err}");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .env("TURBO_KSERVE_LISTEN", "127.0.0.1:0")
        .env("TURBO_KSERVE_MODELS", &model)
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("turbo_model_load: TURBO_E_BUNDLE_NOT_FOUND: "), "{err}");
    assert!(err.contains("listening on 127.0.0.1:"), "{err}");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .env_remove("TURBO_KSERVE_LISTEN")
        .env_remove("TURBO_KSERVE_MODELS")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
}
