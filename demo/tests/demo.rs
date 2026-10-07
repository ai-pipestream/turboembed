//! The demo in front of the real server: turbo-kserve in-process on the CPU
//! with testdata/tiny-bert-bundle served at three tiers, the demo calling it
//! over gRPC, and these tests calling the demo over HTTP as the page does.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use turbo_kserve::Server;
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, ModelConfig};
use turbo_kserve_demo::Demo;
use turbo_kserve_demo::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve_demo::proto::model_infer_request::InferInputTensor;
use turbo_kserve_demo::proto::{InferTensorContents, ModelInferRequest};

const NAME: &str = "tiny-bert-bundle";
const TIERS: [(&str, &str); 3] =
    [(NAME, "PRECISION_MODEL"), ("tiny-fastest", "PRECISION_FASTEST"), ("tiny-exact", "PRECISION_EXACT")];

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// The runtime index of the CPU.
fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

/// The tiny bundle at each tier, the other two under links named for them,
/// since a model's name is its bundle directory's.
fn models() -> Vec<ModelConfig> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let n = N.fetch_add(1, Ordering::Relaxed);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("demo-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bundle = tiny().canonicalize().unwrap();
    TIERS
        .iter()
        .map(|(name, precision)| {
            let path = if *name == NAME {
                bundle.clone()
            } else {
                let link = dir.join(name);
                std::os::unix::fs::symlink(&bundle, &link).unwrap();
                link
            };
            let s = format!("bundle={},device={},sessions=1,precision={precision}", path.display(), cpu());
            ModelConfig::parse(&s).unwrap()
        })
        .collect()
}

/// The server loaded, and the demo in front of it offering every tier.
async fn start() -> (Server, Demo) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models(), DEFAULT_MAX_MESSAGE_BYTES).await.unwrap();
    s.load().await.unwrap();
    let names = TIERS.iter().map(|(n, _)| n.to_string()).collect();
    let d = Demo::start("127.0.0.1:0".parse().unwrap(), &format!("http://{}", s.local_addr()), names).await.unwrap();
    (s, d)
}

/// One HTTP/1.1 request; the status, the content type and the body.
async fn http(d: &Demo, method: &str, path: &str, body: Option<&str>) -> (u16, String, String) {
    let mut c = TcpStream::connect(d.local_addr()).await.unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: demo\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    c.write_all(req.as_bytes()).await.unwrap();
    let mut out = Vec::new();
    c.read_to_end(&mut out).await.unwrap();
    let out = String::from_utf8(out).unwrap();
    let (head, body) = out.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    let kind = head
        .lines()
        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-type: ").map(str::to_string))
        .unwrap_or_default();
    assert!(!head.to_ascii_lowercase().contains("transfer-encoding: chunked"), "{head}");
    (status, kind, body.to_string())
}

async fn get_json(d: &Demo, path: &str) -> (u16, Value) {
    let (status, kind, body) = http(d, "GET", path, None).await;
    assert_eq!(kind, "application/json", "{path}");
    (status, serde_json::from_str(&body).unwrap())
}

async fn post_json(d: &Demo, path: &str, body: Value) -> (u16, Value) {
    let (status, kind, body) = http(d, "POST", path, Some(&body.to_string())).await;
    assert_eq!(kind, "application/json", "{path}");
    (status, serde_json::from_str(&body).unwrap())
}

fn rows(v: &Value) -> Vec<Vec<f32>> {
    v["vectors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect())
        .collect()
}

/// The reference cases without a prompt role: texts from the manifest,
/// vectors from the reference safetensors.
fn reference() -> Vec<(String, Vec<f32>)> {
    let dir = tiny();
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir.join(m["reference"]["file"].as_str().unwrap())).unwrap();
    let h = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + h]).unwrap();
    let o = &header["embeddings"]["data_offsets"];
    let dim = header["embeddings"]["shape"][1].as_u64().unwrap() as usize;
    let emb: Vec<f32> = bytes[8 + h..][o[0].as_u64().unwrap() as usize..o[1].as_u64().unwrap() as usize]
        .chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    m["reference"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, c)| c["prompt_role"] == "PROMPT_NONE")
        .map(|(i, c)| (c["text"].as_str().unwrap().to_string(), emb[i * dim..(i + 1) * dim].to_vec()))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    d / (na * nb).sqrt()
}

#[tokio::test]
async fn the_page_is_served_whole() {
    let (_s, d) = start().await;
    for (path, kind, has) in [
        ("/", "text/html; charset=utf-8", "<script src=\"app.js\"></script>"),
        ("/app.js", "text/javascript; charset=utf-8", "api/infer"),
        ("/style.css", "text/css; charset=utf-8", "prefers-color-scheme"),
    ] {
        let (status, k, body) = http(&d, "GET", path, None).await;
        assert_eq!((status, k.as_str()), (200, kind), "{path}");
        assert!(body.contains(has), "{path}");
        // Nothing is loaded from anywhere but the demo.
        assert!(!body.contains("http://") && !body.contains("https://"), "{path} names another host");
    }
    assert_eq!(http(&d, "GET", "/nothing", None).await.0, 404);
}

#[tokio::test]
async fn status_and_metadata_come_from_the_server() {
    let (s, d) = start().await;
    let (status, v) = get_json(&d, "/api/status").await;
    assert_eq!(status, 200);
    assert_eq!(v["reachable"], true);
    assert_eq!(v["live"], true);
    assert_eq!(v["ready"], true);
    assert_eq!(v["name"], "turboembed");
    assert_eq!(v["server"], format!("http://{}", s.local_addr()));
    let names: Vec<Value> = TIERS.iter().map(|(n, _)| json!({ "name": n, "ready": true })).collect();
    assert_eq!(v["models"], Value::Array(names));

    for (name, precision) in TIERS {
        let (status, m) = get_json(&d, &format!("/api/models/{name}")).await;
        assert_eq!(status, 200, "{m}");
        assert_eq!(m["name"], name);
        assert_eq!(m["versions"], json!(["1"]));
        assert_eq!(m["properties"]["session_info.precision"], precision);
        assert_eq!(m["properties"]["model_info.dim"], "32");
        assert_eq!(m["properties"]["device_info.kind"], "DEVICE_CPU");
        assert_eq!(m["outputs"], json!([{ "name": "vectors", "datatype": "FP32", "shape": [-1, -1] }]));
    }

    // A name the server does not serve is the server's NOT_FOUND, passed on.
    let (status, e) = get_json(&d, "/api/models/no-such-model").await;
    assert_eq!(status, 502);
    assert_eq!(e["error"]["grpc"], "NotFound");
    assert_eq!(e["error"]["turbo_code"], "1280");
}

#[tokio::test]
async fn texts_through_the_demo_are_the_servers_vectors_at_every_tier() {
    let (s, d) = start().await;
    let cases = reference();
    assert!(cases.len() >= 2);
    let texts: Vec<&str> = cases.iter().map(|(t, _)| t.as_str()).collect();

    // The server's own answer over gRPC, for the bits to compare.
    let mut grpc = GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    for (name, _) in TIERS {
        let (status, v) = post_json(&d, "/api/infer", json!({ "model": name, "texts": texts })).await;
        assert_eq!(status, 200, "{v}");
        assert_eq!(v["model"], name);
        assert_eq!(v["version"], "1");
        assert_eq!(v["shape"], json!([texts.len(), 32]));
        assert_eq!(v["summary"]["backend"], "cpu");
        assert_eq!(v["summary"]["batch"], texts.len());
        let got = rows(&v);

        let r = ModelInferRequest {
            model_name: name.into(),
            inputs: vec![InferInputTensor {
                name: "texts".into(),
                datatype: "BYTES".into(),
                shape: vec![texts.len() as i64],
                contents: Some(InferTensorContents {
                    bytes_contents: texts.iter().map(|t| t.as_bytes().to_vec()).collect(),
                    ..Default::default()
                }),
                ..Default::default()
            }],
            ..Default::default()
        };
        let raw = grpc.model_infer(r).await.unwrap().into_inner().raw_output_contents.remove(0);
        let want: Vec<f32> = raw.chunks(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        // Exactly the server's floats: JSON carries each f32 so it reads back the same.
        assert_eq!(got.concat(), want, "{name}");

        for ((text, reference), row) in cases.iter().zip(&got) {
            let c = cosine(row, reference);
            assert!(c >= 0.9999, "{name} `{text}`: cosine {c} to the reference");
        }
    }
}

#[tokio::test]
async fn parameters_reach_the_server_and_its_refusals_come_back() {
    let (_s, d) = start().await;
    let body = |p: Value| json!({ "model": NAME, "texts": ["a cat", "a dog"], "parameters": p });

    let (_, plain) = post_json(&d, "/api/infer", body(json!({}))).await;
    let (status, query) = post_json(&d, "/api/infer", body(json!({ "prompt_role": "PROMPT_QUERY" }))).await;
    assert_eq!(status, 200, "{query}");
    assert_ne!(rows(&plain), rows(&query), "the bundle's query prefix changes the vectors");

    let (status, cut) =
        post_json(&d, "/api/infer", body(json!({ "max_tokens": 3, "truncate": "TRUNCATE_RIGHT" }))).await;
    assert_eq!(status, 200, "{cut}");

    // Over the session's max_seq: the library's TURBO_E_CAPACITY, as the server maps it.
    let (status, e) = post_json(&d, "/api/infer", body(json!({ "max_tokens": 100000 }))).await;
    assert_eq!(status, 502);
    assert_eq!(e["error"]["grpc"], "OutOfRange");
    assert_eq!(e["error"]["turbo_code"], "771");
    assert!(e["error"]["message"].as_str().unwrap().starts_with("TURBO_E_CAPACITY"), "{e}");

    // A value of no protocol parameter type is refused by the demo.
    let (status, e) = post_json(&d, "/api/infer", body(json!({ "normalize": true }))).await;
    assert_eq!(status, 400);
    assert!(e["error"]["message"].as_str().unwrap().contains("normalize"), "{e}");

    // Not the demo's JSON.
    let (status, kind, _) = http(&d, "POST", "/api/infer", Some("{\"texts\": 1}")).await;
    assert_eq!((status, kind.as_str()), (400, "application/json"));
}

#[tokio::test]
async fn a_server_that_is_not_there_is_reported() {
    // A port nothing listens on: bind one, then let it go.
    let addr = std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
    let d = Demo::start("127.0.0.1:0".parse().unwrap(), &format!("http://{addr}"), vec![NAME.into()]).await.unwrap();
    let (status, v) = get_json(&d, "/api/status").await;
    assert_eq!(status, 200);
    assert_eq!(v["reachable"], false);
    assert_eq!(v["models"], json!([{ "name": NAME }]));
    let (status, e) = post_json(&d, "/api/infer", json!({ "model": NAME, "texts": ["a"] })).await;
    assert_eq!(status, 502);
    assert_eq!(e["error"]["grpc"], "Unavailable");
    d.stop().await;
}
