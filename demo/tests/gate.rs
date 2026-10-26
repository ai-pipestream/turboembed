//! The gate in front of the allowlist: nothing off the list reaches the
//! server, a bearer token when one is set, the origin check and CORS, and
//! the rule that a write off loopback needs a token.

mod common;

use common::*;
use turbo_kserve::Serve;
use turbo_kserve::proto::ServerLiveRequest;
use turbo_kserve_demo::allow::METHODS;
use turbo_kserve_demo::{Allow, Demo, Options};

const LIVE: &str = "/inference.GRPCInferenceService/ServerLive";

/// The code a call is refused with: the unary error body's, or the end
/// envelope's for a streaming method.
async fn code(d: &Demo, path: &str, streaming: bool, headers: &[(&str, &str)]) -> String {
    let (kind, body) = if streaming { (STREAM, envelope(0, b"")) } else { (UNARY, Vec::new()) };
    let mut h = vec![("Content-Type", kind)];
    h.extend_from_slice(headers);
    let mut r = http(d, "POST", path, &h, &body).await;
    let text = if streaming {
        r.frames::<ServerLiveRequest>().await.1
    } else {
        while r.more().await {}
        String::from_utf8(r.body).unwrap()
    };
    let start = text.find("\"code\":\"").map(|i| i + 8).unwrap_or_else(|| panic!("{path}: {text}"));
    text[start..].split('"').next().unwrap().to_string()
}

#[tokio::test]
async fn by_default_no_write_and_no_unknown_method_reaches_the_server() {
    let d = demo_with(nowhere().await, Options::default()).await;
    assert!(!Allow::default().writes());
    for m in METHODS {
        let want = if m.is_write() { "permission_denied" } else { "unavailable" };
        assert_eq!(code(&d, m.path, m.streaming, &[]).await, want, "{}", m.path);
    }
    for path in [
        "/turbo.sink.v1.TurboSinkService/StreamDocuments",
        "/turbo.search.v1.TurboSearchService/IndexDocuments",
        "/turbo.vocabulary.v1.TurboVocabularyService/ImportDictionary",
        "/grpc.health.v1.Health/Check",
        "/grpc.reflection.v1.ServerReflection/ServerReflectionInfo",
    ] {
        assert_eq!(code(&d, path, false, &[]).await, "unimplemented", "{path}");
        assert_eq!(code(&d, path, true, &[]).await, "unimplemented", "{path}");
    }
}

#[tokio::test]
async fn with_a_token_set_a_call_without_it_is_unauthenticated() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let options = Options { token: Some("s3cret".into()), ..Options::default() };
    let d = demo_with(s.local_addr(), options).await;
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY)], b"").await;
    assert_eq!(r.status, 401);
    assert_eq!(code(&d, LIVE, false, &[("Authorization", "Bearer wrong")]).await, "unauthenticated");
    assert_eq!(code(&d, "/turbo.search.v1.TurboSearchService/WatchCollection", true, &[]).await, "unauthenticated");
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Authorization", "Bearer s3cret")], b"").await;
    assert_eq!(r.status, 200);
    // The page itself needs no token.
    assert_eq!(http(&d, "GET", "/", &[], b"").await.status, 200);
}

#[tokio::test]
async fn a_write_off_loopback_needs_a_token() {
    let to = format!("http://{}", nowhere().await);
    let any = "0.0.0.0:0".parse().unwrap();
    let allow = Allow::with(&["/turbo.search.v1.TurboSearchService/DeleteIndex".to_string()]).unwrap();
    let writes = Options { allow: allow.clone(), ..Options::default() };
    let e = Demo::start_with(any, &to, writes).await.err().unwrap();
    assert!(e.contains("a token is required off loopback"), "{e}");
    let with_token = Options { allow: allow.clone(), token: Some("t".into()), ..Options::default() };
    Demo::start_with(any, &to, with_token).await.unwrap().stop().await;
    Demo::start_with(any, &to, Options::default()).await.unwrap().stop().await;
    let local = Options { allow, ..Options::default() };
    Demo::start_with("127.0.0.1:0".parse().unwrap(), &to, local).await.unwrap().stop().await;
}

#[tokio::test]
async fn another_origin_gets_no_cors_grant_unless_listed() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo_with(s.local_addr(), Options::default()).await;
    let evil = [("Origin", "https://evil.example")];
    assert_eq!(code(&d, LIVE, false, &evil).await, "permission_denied");
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), evil[0]], b"").await;
    assert_eq!(r.header("access-control-allow-origin"), None);
    let pre = [evil[0], ("Access-Control-Request-Method", "POST")];
    let r = http(&d, "OPTIONS", LIVE, &pre, b"").await;
    assert_eq!(r.status, 403);
    assert_eq!(r.header("access-control-allow-origin"), None);
    // The page's own origin: the request's Host.
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Origin", "http://demo")], b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), None);

    let listed = Options { origins: vec!["https://app.example".into()], ..Options::default() };
    let d = demo_with(s.local_addr(), listed).await;
    let app = ("Origin", "https://app.example");
    let r = http(&d, "OPTIONS", LIVE, &[app, ("Access-Control-Request-Method", "POST")], b"").await;
    assert_eq!(r.status, 204);
    assert_eq!(r.header("access-control-allow-origin"), Some("https://app.example"));
    assert_eq!(r.header("access-control-allow-methods"), Some("POST"));
    let r = http(&d, "OPTIONS", LIVE, &[app, ("Access-Control-Request-Method", "GET")], b"").await;
    assert_eq!(r.status, 403);
    assert_eq!(r.header("access-control-allow-origin"), None);
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), app], b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), Some("https://app.example"));
    assert_eq!(r.header("access-control-expose-headers"), Some("turbo-code, turbo-field"));
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), evil[0]], b"").await;
    assert_eq!(r.status, 403);
    assert_eq!(r.header("access-control-allow-origin"), None);

    for bad in ["*", "https://app.example/"] {
        let o = Options { origins: vec![bad.into()], ..Options::default() };
        assert!(Demo::start_with("127.0.0.1:0".parse().unwrap(), "http://127.0.0.1:1", o).await.is_err(), "{bad}");
    }
}

#[tokio::test]
async fn origin_identity_includes_scheme_and_canonical_default_port() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let d = demo_with(s.local_addr(), Options::default()).await;
    let mut got = Vec::new();
    for origin in ["http://demo", "https://demo", "file://demo", "http://demo:80"] {
        let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Origin", origin)], b"").await;
        got.push((origin, r.status));
    }
    assert_eq!(got, vec![("http://demo", 200), ("https://demo", 403), ("file://demo", 403), ("http://demo:80", 200)]);
}

#[tokio::test]
async fn configured_origin_default_ports_are_canonical_and_invalid_authorities_fail_startup() {
    let s = server(&["tiny-bert-bundle"], Serve::default()).await;
    let options = Options { origins: vec!["https://app.example:443".into()], ..Options::default() };
    let d = demo_with(s.local_addr(), options).await;
    let r = http(&d, "POST", LIVE, &[("Content-Type", UNARY), ("Origin", "https://app.example")], b"").await;
    assert_eq!(r.status, 200);
    assert_eq!(r.header("access-control-allow-origin"), Some("https://app.example"));
    for bad in ["https://app.example:abc", "https://app.example:65536", "http://[not-ip]", "http://:"] {
        let options = Options { origins: vec![bad.into()], ..Options::default() };
        assert!(
            Demo::start_with("127.0.0.1:0".parse().unwrap(), "http://127.0.0.1:1", options).await.is_err(),
            "{bad}"
        );
    }
}

#[tokio::test]
async fn enabled_write_requires_auth_before_mutating_real_server() {
    use prost::Message;
    use turbo_kserve::proto::ModelReadyRequest;
    use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
    use turbo_kserve::proto::result::UnloadBundleRequest;
    let s = server(&["tiny-bert-bundle"], Serve { admin: true, ..Serve::default() }).await;
    let path = "/turbo.result.v1.AnalysisService/UnloadBundle";
    let options =
        Options { allow: Allow::with(&[path.into()]).unwrap(), token: Some("s3cret".into()), ..Options::default() };
    let d = demo_with(s.local_addr(), options).await;
    let body = UnloadBundleRequest { name: "tiny-bert-bundle".into() }.encode_to_vec();
    let mut c = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    for auth in [None, Some("Bearer wrong"), Some("Basic s3cret")] {
        let mut headers = vec![("Content-Type", UNARY)];
        if let Some(value) = auth {
            headers.push(("Authorization", value));
        }
        assert_eq!(http(&d, "POST", path, &headers, &body).await.status, 401);
        let r =
            c.model_ready(ModelReadyRequest { name: "tiny-bert-bundle".into(), version: String::new() }).await.unwrap();
        assert!(r.into_inner().ready);
    }
    let r = http(&d, "POST", path, &[("Content-Type", UNARY), ("Authorization", "Bearer s3cret")], &body).await;
    assert_eq!(r.status, 200);
    let r = c.model_ready(ModelReadyRequest { name: "tiny-bert-bundle".into(), version: String::new() }).await;
    assert!(r.is_err() || !r.unwrap().into_inner().ready);
}
