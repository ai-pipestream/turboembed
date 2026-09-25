//! The server in-process on the CPU with testdata/tiny-bert-bundle, called
//! over gRPC with a tonic client: every RPC, against the bundle's reference
//! vectors and the statuses docs/kserve.md gives.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde_json::Value;
use tonic::transport::Channel;
use tonic::{Code, Status};
use turbo_kserve::Server;
use turbo_kserve::api::{Runtime, field, version};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
use turbo_kserve::proto::*;

type Client = GrpcInferenceServiceClient<Channel>;

const NAME: &str = "tiny-bert-bundle";

fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// The runtime index of the CPU.
fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

fn model(bundle: &Path, sessions: u32) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

async fn client(s: &Server) -> Client {
    GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap()
}

/// A server with the tiny bundle loaded, and a client.
async fn serve(sessions: u32) -> (Server, Client) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), sessions)], MAX).await.unwrap();
    s.load().await.unwrap();
    let c = client(&s).await;
    (s, c)
}

// ---- Requests -------------------------------------------------------------

fn input(name: &str, datatype: &str, shape: &[i64], contents: Option<InferTensorContents>) -> InferInputTensor {
    InferInputTensor {
        name: name.into(),
        datatype: datatype.into(),
        shape: shape.to_vec(),
        parameters: HashMap::new(),
        contents,
    }
}

fn request(inputs: Vec<InferInputTensor>, raw: Vec<Vec<u8>>) -> ModelInferRequest {
    ModelInferRequest {
        model_name: NAME.into(),
        model_version: String::new(),
        id: String::new(),
        parameters: HashMap::new(),
        inputs,
        outputs: vec![],
        raw_input_contents: raw,
    }
}

fn texts(t: &[&str]) -> ModelInferRequest {
    let c =
        InferTensorContents { bytes_contents: t.iter().map(|s| s.as_bytes().to_vec()).collect(), ..Default::default() };
    request(vec![input("texts", "BYTES", &[t.len() as i64], Some(c))], vec![])
}

fn raw_texts(t: &[&[u8]]) -> ModelInferRequest {
    let mut b = Vec::new();
    for s in t {
        b.extend((s.len() as u32).to_le_bytes());
        b.extend(*s);
    }
    request(vec![input("texts", "BYTES", &[t.len() as i64], None)], vec![b])
}

/// Rows padded with 0 and mask 0 to the longest: ids, mask, types.
fn pad(rows: &[Vec<i32>]) -> (i64, i64, [Vec<i32>; 3]) {
    let seq = rows.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = [vec![], vec![], vec![]];
    for r in rows {
        for k in 0..seq {
            out[0].push(r.get(k).copied().unwrap_or(0));
            out[1].push((k < r.len()) as i32);
            out[2].push(0);
        }
    }
    (rows.len() as i64, seq as i64, out)
}

fn ints(v: &[i32]) -> InferTensorContents {
    InferTensorContents { int_contents: v.to_vec(), ..Default::default() }
}

fn le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn typed_tokens(rows: &[Vec<i32>], with_types: bool) -> ModelInferRequest {
    let (b, s, [ids, mask, types]) = pad(rows);
    let mut i =
        vec![input("ids", "INT32", &[b, s], Some(ints(&ids))), input("mask", "INT32", &[b, s], Some(ints(&mask)))];
    if with_types {
        i.push(input("types", "INT32", &[b, s], Some(ints(&types))));
    }
    request(i, vec![])
}

fn raw_tokens(rows: &[Vec<i32>], with_types: bool) -> ModelInferRequest {
    let (b, s, [ids, mask, types]) = pad(rows);
    let mut i = vec![input("ids", "INT32", &[b, s], None), input("mask", "INT32", &[b, s], None)];
    let mut raw = vec![le(&ids), le(&mask)];
    if with_types {
        // Given first, to show the order is the inputs' order, not a fixed one.
        i.insert(0, input("types", "INT32", &[b, s], None));
        raw.insert(0, le(&types));
    }
    request(i, raw)
}

fn with(mut r: ModelInferRequest, name: &str, v: ParameterChoice) -> ModelInferRequest {
    r.parameters.insert(name.into(), InferParameter { parameter_choice: Some(v) });
    r
}

fn st(v: &str) -> ParameterChoice {
    ParameterChoice::StringParam(v.into())
}

fn int(v: i64) -> ParameterChoice {
    ParameterChoice::Int64Param(v)
}

// ---- Answers --------------------------------------------------------------

fn vectors(r: &ModelInferResponse) -> Vec<Vec<f32>> {
    assert_eq!(r.outputs.len(), 1);
    let o = &r.outputs[0];
    assert_eq!((o.name.as_str(), o.datatype.as_str()), ("vectors", "FP32"));
    assert!(o.contents.is_none() && o.parameters.is_empty());
    assert_eq!(r.raw_output_contents.len(), 1);
    let dim = o.shape[1] as usize;
    let v: Vec<f32> = r.raw_output_contents[0].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    assert_eq!(v.len(), o.shape[0] as usize * dim);
    v.chunks(dim).map(<[f32]>::to_vec).collect()
}

async fn embed(c: &mut Client, r: ModelInferRequest) -> Vec<Vec<f32>> {
    vectors(&c.model_infer(r).await.unwrap().into_inner())
}

/// The gRPC code, turbo-code and turbo-field of a refusal.
fn refusal(s: &Status) -> (Code, i32, u32) {
    let md = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_string();
    (s.code(), md("turbo-code").parse().unwrap(), md("turbo-field").parse().unwrap())
}

async fn refused(c: &mut Client, r: ModelInferRequest) -> Status {
    c.model_infer(r).await.expect_err("refused")
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

fn norm(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// docs/conformance.md, F32: cosine at least 0.9999, max abs diff at most 1e-4.
fn matches(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    let c = cosine(got, want);
    let d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(c >= 0.9999, "{what}: cosine {c}");
    assert!(d <= 1e-4, "{what}: max abs diff {d:e}");
}

// ---- The bundle's reference -----------------------------------------------

struct Case {
    text: String,
    role: String,
    ids: Vec<i32>,
    vector: Vec<f32>,
}

/// The reference cases as core/tests/conformance.rs reads them: texts and
/// roles from the manifest, ids (cut to each case's length) and vectors
/// from the reference safetensors.
fn reference() -> Vec<Case> {
    let dir = tiny();
    let m: Value = serde_json::from_slice(&std::fs::read(dir.join("manifest.json")).unwrap()).unwrap();
    let bytes = std::fs::read(dir.join(m["reference"]["file"].as_str().unwrap())).unwrap();
    let h = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + h]).unwrap();
    let data = &bytes[8 + h..];
    let tensor = |name: &str| {
        let o = &header[name]["data_offsets"];
        let shape: Vec<usize> =
            header[name]["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        (&data[o[0].as_u64().unwrap() as usize..o[1].as_u64().unwrap() as usize], shape)
    };
    let i32s = |b: &[u8]| b.chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<_>>();
    let (ids, ids_shape) = tensor("ids");
    let (lengths, _) = tensor("lengths");
    let (emb, emb_shape) = tensor("embeddings");
    let (ids, lengths) = (i32s(ids), i32s(lengths));
    let emb: Vec<f32> = emb.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let (width, dim) = (ids_shape[1], emb_shape[1]);
    m["reference"]["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, c)| Case {
            text: c["text"].as_str().unwrap().to_string(),
            role: c["prompt_role"].as_str().unwrap().to_string(),
            ids: ids[i * width..i * width + lengths[i] as usize].to_vec(),
            vector: emb[i * dim..(i + 1) * dim].to_vec(),
        })
        .collect()
}

// ---- Tests ----------------------------------------------------------------

#[tokio::test]
async fn readiness_before_and_after_load() {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), 1)], MAX).await.unwrap();
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

#[tokio::test]
async fn capacity() {
    let (s, mut c) = serve(1).await;
    let many: Vec<&str> = vec!["a"; 65];
    let e = refused(&mut c, texts(&many)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    assert!(e.message().starts_with("TURBO_E_CAPACITY: "), "{}", e.message());
    assert_eq!(embed(&mut c, texts(&many[..64])).await.len(), 64);

    let row = vec![101, 7592, 102];
    let rows = vec![row.clone(); 65];
    // Typed rows go to the library, which refuses them.
    let e = refused(&mut c, typed_tokens(&rows, false)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    // Raw rows larger than the session's buffers are the server's refusal.
    let e = refused(&mut c, raw_tokens(&rows, true)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    assert!(e.message().contains("max_batch 64"), "{}", e.message());
    let mut long = row.clone();
    long.resize(65, 1000);
    for r in [raw_tokens(std::slice::from_ref(&long), false), typed_tokens(std::slice::from_ref(&long), false)] {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    }
    // An oversized raw request with an option the library would refuse is
    // told CAPACITY.
    let e = refused(&mut c, with(raw_tokens(&rows, false), "output_dim", int(16))).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    s.stop().await;
}

/// sessions=1, and the one session held through Server::hold, which takes
/// it from the pool exactly as a ModelInfer does and keeps it until dropped.
/// While it is held every ModelInfer is BUSY at once, the calls that take
/// no session are answered, and once it is dropped the next request runs.
#[tokio::test]
async fn busy_when_every_session_is_held() {
    let (s, mut c) = serve(1).await;
    assert_eq!(embed(&mut c, texts(&["a"])).await.len(), 1);
    let held = s.hold(NAME).unwrap();
    for r in [texts(&["a"]), raw_tokens(&[vec![101, 102]], false)] {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), (Code::Unavailable, 769, 0));
        assert!(e.message().starts_with("TURBO_E_BUSY: "), "{}", e.message());
    }
    assert_eq!(s.hold(NAME).err().map(|f| f.code), Some(769));
    assert!(
        c.model_ready(ModelReadyRequest { name: NAME.into(), version: String::new() })
            .await
            .unwrap()
            .into_inner()
            .ready
    );
    assert!(c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: String::new() }).await.unwrap();
    c.server_metadata(ServerMetadataRequest {}).await.unwrap();
    // A request is checked before a session is taken: only one the server
    // would send to the library is told BUSY.
    let e = refused(&mut c, with(texts(&["a"]), "precision", st("PRECISION_EXACT"))).await;
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256, 0));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![2];
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::InvalidArgument, 260, 0));
    let e = refused(&mut c, raw_tokens(&vec![vec![101, 102]; 65], false)).await;
    assert_eq!(refusal(&e), (Code::OutOfRange, 771, 0));
    drop(held);
    assert_eq!(embed(&mut c, texts(&["a"])).await.len(), 1);
    s.stop().await;
}

/// A request message larger than --max-message-bytes is refused by gRPC with
/// RESOURCE_EXHAUSTED before it is read, and carries no turbo-code.
#[tokio::test]
async fn a_message_over_the_limit() {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), 1)], 4096).await.unwrap();
    s.load().await.unwrap();
    let mut c = client(&s).await;
    let big = "a ".repeat(4096);
    let e = refused(&mut c, texts(&[&big])).await;
    assert_eq!(e.code(), Code::ResourceExhausted, "{e:?}");
    assert!(e.metadata().get("turbo-code").is_none() && e.metadata().get("turbo-field").is_none(), "{e:?}");
    assert_eq!(embed(&mut c, texts(&[&big[..1000]])).await.len(), 1);
    s.stop().await;
}

/// Two sessions serve two requests at once.
#[tokio::test]
async fn one_request_per_session() {
    let (s, c) = serve(2).await;
    let held = s.hold(NAME).unwrap();
    let mut c1 = c.clone();
    assert_eq!(embed(&mut c1, texts(&["a", "b"])).await.len(), 2);
    let _second = s.hold(NAME).unwrap();
    assert_eq!(refusal(&refused(&mut c1, texts(&["a"])).await).1, 769);
    drop(held);
    assert_eq!(embed(&mut c1, texts(&["a"])).await.len(), 1);
    s.stop().await;
}

#[tokio::test]
async fn unknown_model_or_version() {
    let (s, mut c) = serve(1).await;
    let e = c.model_ready(ModelReadyRequest { name: "tiny".into(), version: String::new() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    assert!(e.message().starts_with("TURBO_E_BUNDLE_NOT_FOUND: "), "{}", e.message());
    let e = c.model_metadata(ModelMetadataRequest { name: "tiny".into(), version: String::new() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    let e = c.model_metadata(ModelMetadataRequest { name: NAME.into(), version: "0".into() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_name = "Tiny-Bert-Bundle".into();
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_version = "2".into();
    assert_eq!(refusal(&refused(&mut c, r).await), (Code::NotFound, 1280, 0));
    let mut r = texts(&["a"]);
    r.model_version = "1".into();
    assert_eq!(embed(&mut c, r).await.len(), 1);
    s.stop().await;
}

#[tokio::test]
async fn input_refusals() {
    let (s, mut c) = serve(1).await;
    let arg = (Code::InvalidArgument, 256, 0);
    let shape = (Code::InvalidArgument, 260, 0);
    let rows = vec![vec![101, 7592, 102]];

    let mut cases: Vec<(&str, ModelInferRequest, (Code, i32, u32))> = Vec::new();
    cases.push(("no inputs", request(vec![], vec![]), arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].datatype = "INT64".into();
    r.inputs[0].contents = Some(InferTensorContents { int64_contents: vec![101, 7592, 102], ..Default::default() });
    cases.push(("INT64 ids", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].datatype = "STRING".into();
    cases.push(("texts not BYTES", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs.remove(1);
    cases.push(("ids without mask", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs.push(texts(&["a"]).inputs.remove(0));
    cases.push(("texts and ids", r, arg));
    let mut r = typed_tokens(&rows, true);
    r.inputs[2].name = "mask".into();
    cases.push(("a name given twice", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].name = "text".into();
    cases.push(("an unknown input", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].parameters.insert("x".into(), InferParameter { parameter_choice: Some(int(1)) });
    cases.push(("input parameters", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor { name: "embeddings".into(), parameters: HashMap::new() }];
    cases.push(("an unknown output", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor { name: "vectors".into(), parameters: HashMap::new() }; 2];
    cases.push(("vectors twice", r, arg));
    let mut r = texts(&["a"]);
    r.outputs = vec![InferRequestedOutputTensor {
        name: "vectors".into(),
        parameters: HashMap::from([("x".into(), InferParameter { parameter_choice: Some(int(1)) })]),
    }];
    cases.push(("output parameters", r, arg));
    let mut r = texts(&["a"]);
    r.raw_input_contents = vec![vec![1, 0, 0, 0, b'a']];
    cases.push(("raw and typed", r, arg));
    let mut r = raw_tokens(&rows, false);
    r.raw_input_contents.pop();
    cases.push(("a raw entry short", r, arg));
    let mut r = raw_texts(&[b"a"]);
    r.raw_input_contents[0].push(9);
    cases.push(("malformed raw BYTES", r, arg));
    let mut r = texts(&["a"]);
    r.inputs[0].contents.as_mut().unwrap().int_contents = vec![1];
    cases.push(("texts in int_contents too", r, arg));
    let mut r = typed_tokens(&rows, false);
    r.inputs[1].contents = Some(InferTensorContents { uint_contents: vec![1, 1, 1], ..Default::default() });
    cases.push(("mask in uint_contents", r, arg));

    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![1, 1];
    cases.push(("texts of rank 2", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].shape = vec![3];
    cases.push(("ids of rank 1", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![-1];
    cases.push(("a negative extent", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![1 << 32];
    cases.push(("an extent over uint32_t", r, shape));
    let mut r = texts(&["a"]);
    r.inputs[0].shape = vec![2];
    cases.push(("fewer texts than the shape", r, shape));
    let mut r = raw_texts(&[b"a", b"b"]);
    r.inputs[0].shape = vec![1];
    cases.push(("more raw texts than the shape", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[1].shape = vec![3, 1];
    cases.push(("mask shaped unlike ids", r, shape));
    let mut r = typed_tokens(&rows, true);
    r.inputs[2].shape = vec![1, 2];
    cases.push(("types shaped unlike ids", r, shape));
    let mut r = typed_tokens(&rows, false);
    r.inputs[0].contents.as_mut().unwrap().int_contents.pop();
    cases.push(("typed ids short", r, shape));
    let mut r = raw_tokens(&rows, false);
    r.raw_input_contents[1].extend([0, 0, 0, 0]);
    cases.push(("raw mask long", r, shape));
    // 4 x batch x seq is 2^64 here, which wraps to 0 in a uint64_t: the
    // empty contents must not pass for it.
    for e in [1i64 << 31, u32::MAX as i64] {
        let mut r = raw_tokens(&rows, false);
        for i in &mut r.inputs {
            i.shape = vec![e, e];
        }
        r.raw_input_contents = vec![vec![], vec![]];
        cases.push(("raw rows past a uint64_t of bytes", r, shape));
    }

    cases.push(("not UTF-8", raw_texts(&[b"\xff\xfe"]), (Code::InvalidArgument, 258, 0)));

    for (what, r, want) in cases {
        let e = refused(&mut c, r).await;
        assert_eq!(refusal(&e), want, "{what}: {}", e.message());
    }
    s.stop().await;
}

#[tokio::test]
async fn startup_refusals() {
    let at = "127.0.0.1:0".parse().unwrap();
    let t = tiny();
    // Paths that name no bundle, and two models of one name.
    for bundle in [format!("{}/", t.display()), format!("{}/.", t.display()), format!("{}/..", t.display())] {
        let m = ModelConfig { bundle, ..model(&t, 1) };
        assert!(Server::start(at, vec![m], MAX).await.is_err());
    }
    assert!(Server::start(at, vec![model(&t, 1), model(&t, 1)], MAX).await.is_err());

    // A load failure is the call, its status, the field and the message.
    let s = Server::start(at, vec![model(&t.join("../no-such-bundle"), 1)], MAX).await.unwrap();
    let e = s.load().await.unwrap_err();
    assert!(e.contains("model no-such-bundle: turbo_model_load: TURBO_E_BUNDLE_NOT_FOUND: "), "{e}");
    s.stop().await;
    let s = Server::start(at, vec![ModelConfig { max_seq: 65, ..model(&t, 1) }], MAX).await.unwrap();
    let e = s.load().await.unwrap_err();
    assert!(e.contains("turbo_session_create: TURBO_E_UNSUPPORTED_OPTION field 2 (max_seq): "), "{e}");
    let mut c = client(&s).await;
    assert!(!c.server_ready(ServerReadyRequest {}).await.unwrap().into_inner().ready);
    s.stop().await;
    if version().split_whitespace().skip(1).all(|b| b == "cpu") {
        let s = Server::start(at, vec![ModelConfig { device: Device::Select, ..model(&t, 1) }], MAX).await.unwrap();
        let e = s.load().await.unwrap_err();
        assert!(e.contains("turbo_runtime_select: TURBO_E_DEVICE_NOT_FOUND"), "{e}");
        s.stop().await;
    }

    // A configured max_batch and max_seq are the session's limits.
    let s = Server::start(at, vec![ModelConfig { max_batch: 2, max_seq: 8, ..model(&t, 1) }], MAX).await.unwrap();
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

/// The binary exits non-zero with the library's message when a bundle does
/// not load.
#[test]
fn the_binary_exits_on_a_load_failure() {
    let bundle = tiny().join("../no-such-bundle");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve"))
        .args(["--listen", "127.0.0.1:0", "--model"])
        .arg(format!("bundle={},device={},sessions=1", bundle.display(), cpu()))
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("turbo_model_load: TURBO_E_BUNDLE_NOT_FOUND: "), "{err}");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_turbo-kserve")).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}
