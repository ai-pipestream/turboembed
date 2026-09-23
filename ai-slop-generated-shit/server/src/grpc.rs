//! The gRPC binding of the Open Inference Protocol v2: `GRPCInferenceService`
//! over the generated protobuf types, converting to and from the shared
//! `oip` request and response.

use std::collections::BTreeMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};

use crate::engine::Engine;
use crate::error::{Result, ServeError};
use crate::oip::{self, Data, InferRequest, Param, Tensor};

/// The protocol's own types, generated from `proto/open_inference_grpc.proto`.
#[allow(missing_docs)]
pub mod inference {
    tonic::include_proto!("inference");
}

use inference::grpc_inference_service_server::GrpcInferenceService;
use inference::{
    infer_parameter, model_infer_request, model_infer_response, model_metadata_response, InferParameter,
    InferTensorContents, ModelInferRequest, ModelInferResponse, ModelMetadataRequest, ModelMetadataResponse,
    ModelReadyRequest, ModelReadyResponse, ServerLiveRequest, ServerLiveResponse, ServerMetadataRequest,
    ServerMetadataResponse, ServerReadyRequest, ServerReadyResponse,
};

pub use inference::grpc_inference_service_server::GrpcInferenceServiceServer;

/// Inferstream's extension service, generated from `proto/turbo_inferstream.proto`.
#[allow(missing_docs)]
pub mod ext {
    include!(concat!(env!("OUT_DIR"), "/ext/turbo.inferstream.rs"));
}

pub use ext::inferstream_extension_server::InferstreamExtensionServer;

/// The file descriptor set of both protos, for the reflection service.
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("inferstream_descriptor");

/// The service over the engine.
pub struct Service {
    /// Every model this service answers for.
    pub engine: Arc<Engine>,
}

fn params_from(map: &std::collections::HashMap<String, InferParameter>) -> BTreeMap<String, Param> {
    let mut out = BTreeMap::new();
    for (k, v) in map {
        let p = match &v.parameter_choice {
            Some(infer_parameter::ParameterChoice::BoolParam(b)) => Param::Bool(*b),
            Some(infer_parameter::ParameterChoice::Int64Param(i)) => Param::Int(*i),
            Some(infer_parameter::ParameterChoice::StringParam(s)) => Param::Str(s.clone()),
            Some(infer_parameter::ParameterChoice::DoubleParam(d)) => Param::Double(*d),
            Some(infer_parameter::ParameterChoice::Uint64Param(u)) => Param::Uint(*u),
            None => continue,
        };
        out.insert(k.clone(), p);
    }
    out
}

fn params_to(map: &BTreeMap<String, Param>) -> std::collections::HashMap<String, InferParameter> {
    map.iter()
        .map(|(k, v)| {
            let choice = match v {
                Param::Bool(b) => infer_parameter::ParameterChoice::BoolParam(*b),
                Param::Int(i) => infer_parameter::ParameterChoice::Int64Param(*i),
                Param::Str(s) => infer_parameter::ParameterChoice::StringParam(s.clone()),
                Param::Double(d) => infer_parameter::ParameterChoice::DoubleParam(*d),
                Param::Uint(u) => infer_parameter::ParameterChoice::Uint64Param(*u),
            };
            (k.clone(), InferParameter { parameter_choice: Some(choice) })
        })
        .collect()
}

/// Typed contents of an input tensor from either its `contents` or the
/// matching `raw_input_contents` entry.
fn data_from(t: &model_infer_request::InferInputTensor, raw: Option<&Vec<u8>>) -> Result<Data> {
    let elements: i64 = t.shape.iter().product();
    if let Some(raw) = raw {
        if t.contents.is_some() {
            return Err(ServeError::bad_request(format!(
                "input `{}` has both contents and raw_input_contents; the protocol allows one",
                t.name
            )));
        }
        return raw_to_data(&t.name, &t.datatype, raw, elements);
    }
    let c =
        t.contents.as_ref().ok_or_else(|| ServeError::bad_request(format!("input `{}` has no contents", t.name)))?;
    Ok(match t.datatype.as_str() {
        "BYTES" => Data::Bytes(c.bytes_contents.clone()),
        "FP32" => Data::Fp32(c.fp32_contents.clone()),
        "FP64" => Data::Fp64(c.fp64_contents.clone()),
        "INT32" | "INT16" | "INT8" => Data::Int32(c.int_contents.clone()),
        "INT64" => Data::Int64(c.int64_contents.clone()),
        "UINT32" | "UINT16" | "UINT8" => Data::Uint32(c.uint_contents.clone()),
        "UINT64" => Data::Uint64(c.uint64_contents.clone()),
        "BOOL" => Data::Bool(c.bool_contents.clone()),
        other => {
            return Err(ServeError::bad_request(format!(
                "input `{}`: datatype `{other}` is not carried in contents",
                t.name
            )))
        }
    })
}

/// Raw little-endian tensor bytes as typed contents. BYTES raw form is a
/// sequence of 4-byte little-endian length prefixes and payloads.
fn raw_to_data(name: &str, datatype: &str, raw: &[u8], elements: i64) -> Result<Data> {
    fn fixed<T, const N: usize>(name: &str, raw: &[u8], elements: i64, f: impl Fn([u8; N]) -> T) -> Result<Vec<T>> {
        if raw.len() % N != 0 || (elements >= 0 && raw.len() != elements as usize * N) {
            return Err(ServeError::bad_request(format!(
                "input `{name}`: {} raw bytes do not hold {elements} elements of {N} bytes",
                raw.len()
            )));
        }
        Ok(raw.chunks_exact(N).map(|c| f(c.try_into().expect("N bytes"))).collect())
    }
    Ok(match datatype {
        "FP32" => Data::Fp32(fixed(name, raw, elements, f32::from_le_bytes)?),
        "FP64" => Data::Fp64(fixed(name, raw, elements, f64::from_le_bytes)?),
        "INT32" => Data::Int32(fixed(name, raw, elements, i32::from_le_bytes)?),
        "INT64" => Data::Int64(fixed(name, raw, elements, i64::from_le_bytes)?),
        "UINT32" => Data::Uint32(fixed(name, raw, elements, u32::from_le_bytes)?),
        "UINT64" => Data::Uint64(fixed(name, raw, elements, u64::from_le_bytes)?),
        "BOOL" => Data::Bool(raw.iter().map(|&b| b != 0).collect()),
        "BYTES" => {
            let mut items = Vec::new();
            let mut at = 0usize;
            while at < raw.len() {
                if at + 4 > raw.len() {
                    return Err(ServeError::bad_request(format!(
                        "input `{name}`: raw BYTES content ends inside a length prefix"
                    )));
                }
                let len = u32::from_le_bytes(raw[at..at + 4].try_into().expect("4 bytes")) as usize;
                at += 4;
                if at + len > raw.len() {
                    return Err(ServeError::bad_request(format!(
                        "input `{name}`: raw BYTES element of {len} bytes runs past the content"
                    )));
                }
                items.push(raw[at..at + len].to_vec());
                at += len;
            }
            if elements >= 0 && items.len() != elements as usize {
                return Err(ServeError::bad_request(format!(
                    "input `{name}`: raw BYTES content holds {} strings but the shape declares {elements}",
                    items.len()
                )));
            }
            Data::Bytes(items)
        }
        other => {
            return Err(ServeError::bad_request(format!(
                "input `{name}`: datatype `{other}` is not carried in raw form by this server"
            )))
        }
    })
}

fn contents_of(data: &Data) -> InferTensorContents {
    let mut c = InferTensorContents::default();
    match data {
        Data::Bytes(v) => c.bytes_contents = v.clone(),
        Data::Fp32(v) => c.fp32_contents = v.clone(),
        Data::Fp64(v) => c.fp64_contents = v.clone(),
        Data::Int32(v) => c.int_contents = v.clone(),
        Data::Int64(v) => c.int64_contents = v.clone(),
        Data::Uint32(v) => c.uint_contents = v.clone(),
        Data::Uint64(v) => c.uint64_contents = v.clone(),
        Data::Bool(v) => c.bool_contents = v.clone(),
    }
    c
}

fn request_from(req: &ModelInferRequest) -> Result<InferRequest> {
    if !req.raw_input_contents.is_empty() && req.raw_input_contents.len() != req.inputs.len() {
        return Err(ServeError::bad_request(format!(
            "raw_input_contents has {} entries for {} inputs",
            req.raw_input_contents.len(),
            req.inputs.len()
        )));
    }
    let mut inputs = Vec::new();
    for (i, t) in req.inputs.iter().enumerate() {
        let raw = req.raw_input_contents.get(i);
        inputs.push(Tensor {
            name: t.name.clone(),
            datatype: t.datatype.clone(),
            shape: t.shape.clone(),
            data: data_from(t, raw)?,
            parameters: params_from(&t.parameters),
        });
    }
    Ok(InferRequest {
        id: req.id.clone(),
        parameters: params_from(&req.parameters),
        inputs,
        outputs: req.outputs.iter().map(|o| o.name.clone()).collect(),
    })
}

#[tonic::async_trait]
impl GrpcInferenceService for Service {
    async fn server_live(
        &self,
        _: Request<ServerLiveRequest>,
    ) -> std::result::Result<Response<ServerLiveResponse>, Status> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(
        &self,
        _: Request<ServerReadyRequest>,
    ) -> std::result::Result<Response<ServerReadyResponse>, Status> {
        Ok(Response::new(ServerReadyResponse { ready: !self.engine.is_empty() }))
    }

    async fn model_ready(
        &self,
        req: Request<ModelReadyRequest>,
    ) -> std::result::Result<Response<ModelReadyResponse>, Status> {
        let r = req.into_inner();
        check_version(&r.version)?;
        // An unknown model is NotFound on both bindings; `ready: false` is
        // for a model the server knows and has not finished loading.
        self.engine.model(&r.name).map_err(|e| e.grpc_status())?;
        Ok(Response::new(ModelReadyResponse { ready: true }))
    }

    async fn server_metadata(
        &self,
        _: Request<ServerMetadataRequest>,
    ) -> std::result::Result<Response<ServerMetadataResponse>, Status> {
        Ok(Response::new(ServerMetadataResponse {
            name: oip::SERVER_NAME.to_string(),
            version: oip::SERVER_VERSION.to_string(),
            extensions: oip::extensions(),
        }))
    }

    async fn model_metadata(
        &self,
        req: Request<ModelMetadataRequest>,
    ) -> std::result::Result<Response<ModelMetadataResponse>, Status> {
        let r = req.into_inner();
        check_version(&r.version)?;
        let served = self.engine.model(&r.name).map_err(|e| e.grpc_status())?;
        let m = oip::model_meta(&served);
        let conv = |t: &oip::TensorMeta| model_metadata_response::TensorMetadata {
            name: t.name.clone(),
            datatype: t.datatype.clone(),
            shape: t.shape.clone(),
        };
        Ok(Response::new(ModelMetadataResponse {
            name: m.name,
            versions: m.versions,
            platform: m.platform,
            inputs: m.inputs.iter().map(conv).collect(),
            outputs: m.outputs.iter().map(conv).collect(),
            properties: m.properties.into_iter().collect(),
        }))
    }

    async fn model_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> std::result::Result<Response<ModelInferResponse>, Status> {
        let r = req.into_inner();
        check_version(&r.model_version)?;
        let served = self.engine.model(&r.model_name).map_err(|e| e.grpc_status())?;
        let request = request_from(&r).map_err(|e| e.grpc_status())?;
        let resp = oip::infer(served, request).await.map_err(|e| e.grpc_status())?;
        Ok(Response::new(ModelInferResponse {
            model_name: resp.model_name,
            model_version: resp.model_version,
            id: resp.id,
            parameters: params_to(&resp.parameters),
            outputs: resp
                .outputs
                .iter()
                .map(|t| model_infer_response::InferOutputTensor {
                    name: t.name.clone(),
                    datatype: t.datatype.clone(),
                    shape: t.shape.clone(),
                    parameters: params_to(&t.parameters),
                    contents: Some(contents_of(&t.data)),
                })
                .collect(),
            raw_output_contents: Vec::new(),
        }))
    }
}

/// The one version this server has; another is a not-found, not a silent
/// substitution.
fn check_version(v: &str) -> std::result::Result<(), Status> {
    if v.is_empty() || v == oip::MODEL_VERSION {
        Ok(())
    } else {
        Err(Status::not_found(format!(
            "model version `{v}` does not exist; this server serves version {}",
            oip::MODEL_VERSION
        )))
    }
}

// ---------------------------------------------------------------------------
// Extension service: streamed inference and the model repository
// ---------------------------------------------------------------------------

use ext::inferstream_extension_server::InferstreamExtension;
use ext::{
    repository_index_response, ModelStreamInferResponse, RepositoryIndexRequest, RepositoryIndexResponse,
    RepositoryModelLoadRequest, RepositoryModelLoadResponse, RepositoryModelUnloadRequest,
    RepositoryModelUnloadResponse,
};

/// The extension service over the engine.
pub struct ExtService {
    /// Every model this service answers for.
    pub engine: Arc<Engine>,
}

fn response_to_proto(resp: &oip::InferResponse) -> ModelInferResponse {
    ModelInferResponse {
        model_name: resp.model_name.clone(),
        model_version: resp.model_version.clone(),
        id: resp.id.clone(),
        parameters: params_to(&resp.parameters),
        outputs: resp
            .outputs
            .iter()
            .map(|t| model_infer_response::InferOutputTensor {
                name: t.name.clone(),
                datatype: t.datatype.clone(),
                shape: t.shape.clone(),
                parameters: params_to(&t.parameters),
                contents: Some(contents_of(&t.data)),
            })
            .collect(),
        raw_output_contents: Vec::new(),
    }
}

#[tonic::async_trait]
impl InferstreamExtension for ExtService {
    type ModelStreamInferStream = std::pin::Pin<
        Box<dyn futures_core::Stream<Item = std::result::Result<ModelStreamInferResponse, Status>> + Send + 'static>,
    >;

    async fn model_stream_infer(
        &self,
        req: Request<ModelInferRequest>,
    ) -> std::result::Result<Response<Self::ModelStreamInferStream>, Status> {
        let r = req.into_inner();
        check_version(&r.model_version)?;
        let served = self.engine.model(&r.model_name).map_err(|e| e.grpc_status())?;
        let request = request_from(&r).map_err(|e| e.grpc_status())?;
        let (tx, rx) = tokio::sync::mpsc::channel::<std::result::Result<ModelStreamInferResponse, Status>>(16);
        if served.info().kind != turbo::types::ModelKind::Generative {
            // One response with the whole result.
            let resp = oip::infer(served, request).await.map_err(|e| e.grpc_status())?;
            let _ = tx
                .send(Ok(ModelStreamInferResponse {
                    error_message: String::new(),
                    infer_response: Some(response_to_proto(&resp)),
                }))
                .await;
        } else {
            let (messages, desc) = oip::generation_inputs(&request).map_err(|e| e.grpc_status())?;
            let mut pieces =
                crate::engine::generate(served.clone(), messages, desc).await.map_err(|e| e.grpc_status())?;
            let name = served.name.clone();
            let id = request.id.clone();
            tokio::spawn(async move {
                while let Some(piece) = pieces.recv().await {
                    let msg = match piece {
                        Ok(p) => {
                            let mut parameters = BTreeMap::new();
                            if p.done {
                                parameters.insert(
                                    "finish_reason".to_string(),
                                    Param::Str(format!("{:?}", p.finish_reason).to_lowercase()),
                                );
                                parameters.insert("prompt_tokens".to_string(), Param::Int(p.prompt_tokens as i64));
                                parameters
                                    .insert("generated_tokens".to_string(), Param::Int(p.generated_tokens as i64));
                            }
                            let resp = oip::InferResponse {
                                model_name: name.clone(),
                                model_version: oip::MODEL_VERSION.to_string(),
                                id: id.clone(),
                                parameters,
                                outputs: vec![Tensor {
                                    name: "text".into(),
                                    datatype: "BYTES".into(),
                                    shape: vec![1],
                                    data: Data::Bytes(vec![p.text.into_bytes()]),
                                    parameters: BTreeMap::new(),
                                }],
                            };
                            Ok(ModelStreamInferResponse {
                                error_message: String::new(),
                                infer_response: Some(response_to_proto(&resp)),
                            })
                        }
                        Err(e) => Ok(ModelStreamInferResponse { error_message: e.to_string(), infer_response: None }),
                    };
                    if tx.send(msg).await.is_err() {
                        // The client went away; dropping `pieces` cancels the generation.
                        return;
                    }
                }
            });
        }
        Ok(Response::new(Box::pin(ReceiverStream { rx })))
    }

    async fn repository_index(
        &self,
        _: Request<RepositoryIndexRequest>,
    ) -> std::result::Result<Response<RepositoryIndexResponse>, Status> {
        let models = self
            .engine
            .snapshot()
            .iter()
            .map(|s| repository_index_response::ModelIndex {
                name: s.name.clone(),
                version: oip::MODEL_VERSION.to_string(),
                state: "READY".to_string(),
                reason: String::new(),
                bundle: s.spec.bundle.display().to_string(),
                provider: s.spec.provider.clone(),
                device: s.device.name.clone(),
                kind: format!("{:?}", s.info().kind),
            })
            .collect();
        Ok(Response::new(RepositoryIndexResponse { models }))
    }

    async fn repository_model_load(
        &self,
        req: Request<RepositoryModelLoadRequest>,
    ) -> std::result::Result<Response<RepositoryModelLoadResponse>, Status> {
        let r = req.into_inner();
        let spec = crate::config::ModelSpec::from_request(
            if r.model_name.is_empty() { None } else { Some(r.model_name.clone()) },
            &r.bundle,
            &r.provider,
            r.ordinal,
            &r.buckets,
            r.sessions,
            r.generations,
        )
        .map_err(|e| e.grpc_status())?;
        let engine = self.engine.clone();
        let name = tokio::task::spawn_blocking(move || engine.load_model(&spec))
            .await
            .map_err(|e| Status::internal(format!("load task failed: {e}")))?
            .map_err(|e| e.grpc_status())?;
        Ok(Response::new(RepositoryModelLoadResponse { model_name: name }))
    }

    async fn repository_model_unload(
        &self,
        req: Request<RepositoryModelUnloadRequest>,
    ) -> std::result::Result<Response<RepositoryModelUnloadResponse>, Status> {
        let r = req.into_inner();
        self.engine.unload(&r.model_name).map_err(|e| e.grpc_status())?;
        Ok(Response::new(RepositoryModelUnloadResponse {}))
    }
}

/// A receiver as a `Stream`, without another crate.
struct ReceiverStream<T> {
    rx: tokio::sync::mpsc::Receiver<T>,
}

impl<T> futures_core::Stream for ReceiverStream<T> {
    type Item = T;
    fn poll_next(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<Option<T>> {
        self.rx.poll_recv(cx)
    }
}
