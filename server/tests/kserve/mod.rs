//! What the Open Inference Protocol tests share: the served tiny bundle,
//! a tonic client, the request builders, and the bundle's reference.
//! Each test binary uses its own subset.
#![allow(dead_code, unused_imports)]

pub use std::collections::HashMap;
pub use std::path::{Path, PathBuf};
pub use tonic::{Code, Status};
pub use turbo_kserve::config::DEFAULT_MAX_MESSAGE_BYTES as MAX;
pub use turbo_kserve::proto::model_infer_request::{InferInputTensor, InferRequestedOutputTensor};
pub use turbo_kserve::proto::*;
pub use turbo_kserve::{Serve, Server};

use serde_json::Value;
use tonic::transport::Channel;
pub use turbo_kserve::api::Runtime;
pub use turbo_kserve::api::field;
pub use turbo_kserve::api::version;
pub use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::grpc_inference_service_client::GrpcInferenceServiceClient;
pub use turbo_kserve::proto::infer_parameter::ParameterChoice;

pub type Client = GrpcInferenceServiceClient<Channel>;

pub const NAME: &str = "tiny-bert-bundle";

pub fn tiny() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

/// The runtime index of the CPU.
pub fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

pub fn model(bundle: &Path, sessions: u32) -> ModelConfig {
    ModelConfig {
        bundle: bundle.to_str().unwrap().to_string(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions,
    }
}

pub async fn client(s: &Server) -> Client {
    GrpcInferenceServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap()
}

/// A server with the tiny bundle loaded, and a client.
pub async fn serve(sessions: u32) -> (Server, Client) {
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model(&tiny(), sessions)], MAX, Serve::default())
        .await
        .unwrap();
    s.load().await.unwrap();
    let c = client(&s).await;
    (s, c)
}

// ---- Requests -------------------------------------------------------------

pub fn input(name: &str, datatype: &str, shape: &[i64], contents: Option<InferTensorContents>) -> InferInputTensor {
    InferInputTensor {
        name: name.into(),
        datatype: datatype.into(),
        shape: shape.to_vec(),
        parameters: HashMap::new(),
        contents,
    }
}

pub fn request(inputs: Vec<InferInputTensor>, raw: Vec<Vec<u8>>) -> ModelInferRequest {
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

pub fn texts(t: &[&str]) -> ModelInferRequest {
    let c =
        InferTensorContents { bytes_contents: t.iter().map(|s| s.as_bytes().to_vec()).collect(), ..Default::default() };
    request(vec![input("texts", "BYTES", &[t.len() as i64], Some(c))], vec![])
}

pub fn raw_texts(t: &[&[u8]]) -> ModelInferRequest {
    let mut b = Vec::new();
    for s in t {
        b.extend((s.len() as u32).to_le_bytes());
        b.extend(*s);
    }
    request(vec![input("texts", "BYTES", &[t.len() as i64], None)], vec![b])
}

/// Rows padded with 0 and mask 0 to the longest: ids, mask, types.
pub fn pad(rows: &[Vec<i32>]) -> (i64, i64, [Vec<i32>; 3]) {
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

pub fn ints(v: &[i32]) -> InferTensorContents {
    InferTensorContents { int_contents: v.to_vec(), ..Default::default() }
}

pub fn le(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn typed_tokens(rows: &[Vec<i32>], with_types: bool) -> ModelInferRequest {
    let (b, s, [ids, mask, types]) = pad(rows);
    let mut i =
        vec![input("ids", "INT32", &[b, s], Some(ints(&ids))), input("mask", "INT32", &[b, s], Some(ints(&mask)))];
    if with_types {
        i.push(input("types", "INT32", &[b, s], Some(ints(&types))));
    }
    request(i, vec![])
}

pub fn raw_tokens(rows: &[Vec<i32>], with_types: bool) -> ModelInferRequest {
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

pub fn with(mut r: ModelInferRequest, name: &str, v: ParameterChoice) -> ModelInferRequest {
    r.parameters.insert(name.into(), InferParameter { parameter_choice: Some(v) });
    r
}

pub fn st(v: &str) -> ParameterChoice {
    ParameterChoice::StringParam(v.into())
}

pub fn int(v: i64) -> ParameterChoice {
    ParameterChoice::Int64Param(v)
}

// ---- Answers --------------------------------------------------------------

pub fn vectors(r: &ModelInferResponse) -> Vec<Vec<f32>> {
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

pub async fn embed(c: &mut Client, r: ModelInferRequest) -> Vec<Vec<f32>> {
    vectors(&c.model_infer(r).await.unwrap().into_inner())
}

/// The gRPC code, turbo-code and turbo-field of a refusal.
pub fn refusal(s: &Status) -> (Code, i32, u32) {
    let md = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_string();
    (s.code(), md("turbo-code").parse().unwrap(), md("turbo-field").parse().unwrap())
}

pub async fn refused(c: &mut Client, r: ModelInferRequest) -> Status {
    c.model_infer(r).await.expect_err("refused")
}

pub fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}

pub fn norm(a: &[f32]) -> f64 {
    a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()
}

/// docs/conformance.md, F32: cosine at least 0.9999, max abs diff at most 1e-4.
pub fn matches(what: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{what}: width");
    let c = cosine(got, want);
    let d = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(c >= 0.9999, "{what}: cosine {c}");
    assert!(d <= 1e-4, "{what}: max abs diff {d:e}");
}

// ---- The bundle's reference -----------------------------------------------

pub struct Case {
    pub text: String,
    pub role: String,
    pub ids: Vec<i32>,
    pub vector: Vec<f32>,
}

/// The reference cases as core/tests/conformance.rs reads them: texts and
/// roles from the manifest, ids (cut to each case's length) and vectors
/// from the reference safetensors.
pub fn reference() -> Vec<Case> {
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
