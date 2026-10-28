//! The Open Inference Protocol's six RPCs (docs/kserve.md): ModelInfer
//! as one write, one run and one read on one held session, and the
//! metadata and readiness calls around it.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use tonic::{Request, Response, Status};
use turbo::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I32, TURBO_EMBED_STAGE_COUNT, TURBO_PLACE_HOST,
    TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, TURBO_RERANK_STAGE_COUNT, TURBO_TASK_EMBED, TURBO_TASK_EXTRACTIVE_QA,
    TURBO_TASK_RERANK, turbo_embed_options, turbo_result_info, turbo_session_info, turbo_token_batch,
};

use crate::api::{self, *};
use crate::deadline::{Deadline, deadline_status};
use crate::infer::{self, Rows};
use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::*;
use crate::state::{Held, Loaded, Served, State};
use crate::status::{EMBED_OPTIONS, QA_OPTIONS, RERANK_OPTIONS, to_status};

/// The vectors of one result, owned until the response that carries them is
/// encoded; dropping them releases the buffer, then the result, then the
/// session.
pub(crate) struct Vectors {
    pub(crate) data: Data,
    pub(crate) _result: Output,
    pub(crate) _held: Held,
}

pub(crate) enum Data {
    /// The result's own host memory, from turbo_result_buffer.
    Mapped { ptr: *const u8, len: usize, _buffer: Buffer },
    /// What turbo_result_read downloaded.
    Read(Vec<u8>),
}

// The mapped memory belongs to the buffer held alongside it.
unsafe impl Send for Vectors {}

impl AsRef<[u8]> for Vectors {
    fn as_ref(&self) -> &[u8] {
        match &self.data {
            Data::Mapped { ptr, len, .. } if *len > 0 => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            Data::Mapped { .. } => &[],
            Data::Read(v) => v,
        }
    }
}

/// Copies little-endian int32 rows into a session's pinned buffer.
fn copy_rows(src: &[u8], dst: *mut u8) {
    if cfg!(target_endian = "little") {
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
    } else {
        for (k, c) in src.as_chunks::<4>().0.iter().enumerate() {
            let v = i32::from_le_bytes(*c);
            unsafe { dst.cast::<i32>().add(k).write_unaligned(v) };
        }
    }
}

fn token_batch(batch: u32, seq: u32, ids: *const i32, mask: *const i32, types: *const i32) -> turbo_token_batch {
    turbo_token_batch {
        struct_size: size_of::<turbo_token_batch>() as u32,
        batch,
        seq,
        row_stride: 0,
        ids,
        mask,
        types,
    }
}

/// The protocol's name for a result dtype.
pub(crate) fn datatype(dtype: u32) -> Option<&'static str> {
    match dtype {
        TURBO_DTYPE_F32 => Some("FP32"),
        TURBO_DTYPE_F16 => Some("FP16"),
        TURBO_DTYPE_BF16 => Some("BF16"),
        TURBO_DTYPE_I32 => Some("INT32"),
        _ => None,
    }
}

pub(crate) fn int(v: impl Into<i128>) -> InferParameter {
    let v: i128 = v.into();
    InferParameter {
        parameter_choice: Some(ParameterChoice::Int64Param(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64)),
    }
}

pub(crate) fn string(v: impl Into<String>) -> InferParameter {
    InferParameter { parameter_choice: Some(ParameterChoice::StringParam(v.into())) }
}

/// The scalar fields of turbo_result_info as response parameters.
pub(crate) fn result_parameters(i: &turbo_result_info) -> HashMap<String, InferParameter> {
    let mut p: HashMap<String, InferParameter> = [
        ("task", string(names::task(i.task))),
        ("batch", int(i.batch)),
        ("dim", int(i.dim)),
        ("dtype", string(names::dtype(i.dtype))),
        ("compute_dtype", string(names::dtype(i.compute_dtype))),
        ("placement", string(names::place(i.placement))),
        ("device", int(i.device)),
        ("bytes", int(i.bytes)),
        ("h2d_bytes", int(i.h2d_bytes)),
        ("d2h_bytes", int(i.d2h_bytes)),
        ("host_allocs", int(i.host_allocs)),
        ("device_allocs", int(i.device_allocs)),
        ("backend", string(api::field(&i.backend))),
        ("arch", string(api::field(&i.arch))),
        ("runtime_version", string(api::field(&i.runtime_version))),
        ("manifest_sha256", string(api::field(&i.manifest_sha256))),
        ("artifact_sha256", string(api::field(&i.artifact_sha256))),
        ("tokenizer_sha256", string(api::field(&i.tokenizer_sha256))),
        ("stage_count", int(i.stage_count)),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    if i.task == TURBO_TASK_EMBED || i.task == TURBO_TASK_EXTRACTIVE_QA {
        for (s, name) in
            names::EMBED_STAGES.iter().enumerate().take((i.stage_count as usize).min(TURBO_EMBED_STAGE_COUNT))
        {
            p.insert(format!("stage.{name}"), string(names::stage(i.stage[s])));
        }
    } else if i.task == TURBO_TASK_RERANK {
        for (s, name) in
            names::RERANK_STAGES.iter().enumerate().take((i.stage_count as usize).min(TURBO_RERANK_STAGE_COUNT))
        {
            p.insert(format!("stage.{name}"), string(names::stage(i.stage[s])));
        }
    }
    p
}

/// Raw token rows fit a session's pinned buffers, or TURBO_E_CAPACITY: what
/// turbo_embed_write_tokens gives that shape, refused before the copy.
fn fits(si: &turbo_session_info, batch: u32, seq: u32) -> Result<()> {
    if batch > si.max_batch || seq > si.max_seq {
        return Err(Failure::new(
            TURBO_E_CAPACITY,
            0,
            format!(
                "token rows [{batch}, {seq}] are over the session's max_batch {} or max_seq {}",
                si.max_batch, si.max_seq
            ),
        ));
    }
    Ok(())
}

/// One ModelInfer: one write, one run and one read on one held session.
fn infer(
    state: &State,
    req: ModelInferRequest,
    deadline: &Deadline,
) -> std::result::Result<ModelInferResponse, (Failure, &'static [&'static str])> {
    deadline.check_alive().map_err(|f| (f, &[][..]))?;
    let (served, loaded) = state.ready(&req.model_name, &req.model_version).map_err(|f| (f, &[][..]))?;
    match loaded.task {
        TURBO_TASK_EMBED => infer_embed(&served, &loaded, req, deadline).map_err(|f| (f, &EMBED_OPTIONS[..])),
        TURBO_TASK_RERANK => {
            crate::tasks::infer_rerank(&served, &loaded, req, deadline).map_err(|f| (f, &RERANK_OPTIONS[..]))
        }
        TURBO_TASK_EXTRACTIVE_QA => {
            crate::tasks::infer_qa(&served, &loaded, req, deadline).map_err(|f| (f, &QA_OPTIONS[..]))
        }
        other => Err((Failure::new(TURBO_E_UNSUPPORTED_TASK, 0, format!("unsupported task {other}")), &[][..])),
    }
}

fn infer_embed(
    served: &Arc<Served>,
    loaded: &Arc<Loaded>,
    req: ModelInferRequest,
    deadline: &Deadline,
) -> Result<ModelInferResponse> {
    let plan = infer::plan(&req)?;
    // Every session of a model is made alike, so the first one's limits are
    // the held one's: an oversized raw request is refused before BUSY.
    if let Rows::Raw { batch, seq, .. } = plan.rows {
        fits(&loaded.slots[0].info, batch, seq)?;
    }
    let batch_len = match &plan.rows {
        Rows::Texts(texts, _) => texts.len(),
        Rows::Typed { batch, .. } | Rows::Raw { batch, .. } => *batch as usize,
    };
    let _docs = loaded.metrics.track_docs(batch_len.max(1), Some(loaded.device));
    deadline.check_alive()?;
    let held = loaded.take()?;
    deadline.check_alive()?;
    let slot = held.slot();
    let opts: &turbo_embed_options = &plan.opts;
    match &plan.rows {
        Rows::Texts(texts, _) => slot.session.write_text(texts, opts)?,
        Rows::Typed { batch, seq, ids, mask, types } => {
            let t = types.map_or(std::ptr::null(), |t| t.as_ptr());
            slot.session.write_tokens(&token_batch(*batch, *seq, ids.as_ptr(), mask.as_ptr(), t), opts)?
        }
        Rows::Raw { batch, seq, ids, mask, types } => {
            // The copy below stays inside the held session's own buffers.
            fits(&slot.info, *batch, *seq)?;
            copy_rows(ids, slot.ptrs[0]);
            copy_rows(mask, slot.ptrs[1]);
            let t = match types {
                Some(t) => {
                    copy_rows(t, slot.ptrs[2]);
                    slot.ptrs[2] as *const i32
                }
                None => std::ptr::null(),
            };
            let b = token_batch(*batch, *seq, slot.ptrs[0] as *const i32, slot.ptrs[1] as *const i32, t);
            slot.session.write_tokens(&b, opts)?
        }
    }
    deadline.check_alive()?;
    loaded.metrics.record_batch_size(batch_len);
    let result = slot.session.run()?;
    deadline.check_alive()?;
    let first = result.info()?;
    let data = match first.placement {
        TURBO_PLACE_HOST | TURBO_PLACE_PINNED | TURBO_PLACE_SHARED => {
            let buffer = result.buffer()?;
            let ptr = buffer.host_ptr()?;
            Data::Mapped { ptr, len: first.bytes as usize, _buffer: buffer }
        }
        _ => Data::Read(result.read(first.bytes)?),
    };
    let info = result.info()?;
    let datatype = datatype(info.dtype).ok_or_else(|| {
        Failure::new(TURBO_E_INTERNAL, 0, format!("the result's dtype {} has no datatype in the protocol", info.dtype))
    })?;
    let vectors = Vectors { data, _result: result, _held: held };
    Ok(ModelInferResponse {
        model_name: served.name.clone(),
        model_version: loaded.revision.clone(),
        id: req.id.clone(),
        parameters: result_parameters(&info),
        outputs: vec![InferOutputTensor {
            name: "vectors".into(),
            datatype: datatype.into(),
            shape: vec![info.batch as i64, info.dim as i64],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_output_contents: vec![Bytes::from_owner(vectors)],
    })
}

pub(crate) fn tensor(name: &str, datatype: &str, shape: &[i64]) -> TensorMetadata {
    TensorMetadata { name: name.into(), datatype: datatype.into(), shape: shape.to_vec() }
}

pub(crate) struct Inference {
    pub(crate) state: Arc<State>,
}

pub(crate) type Reply<T> = std::result::Result<Response<T>, Status>;

fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

#[tonic::async_trait]
impl GrpcInferenceService for Inference {
    async fn server_live(&self, _: Request<ServerLiveRequest>) -> Reply<ServerLiveResponse> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(&self, _: Request<ServerReadyRequest>) -> Reply<ServerReadyResponse> {
        Ok(Response::new(ServerReadyResponse { ready: self.state.is_ready() }))
    }

    async fn model_ready(&self, req: Request<ModelReadyRequest>) -> Reply<ModelReadyResponse> {
        let req = req.into_inner();
        let (_, loaded) = self.state.peek(&req.name).map_err(plain)?;
        let ready = match loaded {
            None => false,
            Some(l) => {
                if !req.version.is_empty() && req.version != l.revision {
                    return Err(plain(Failure::new(
                        TURBO_E_BUNDLE_NOT_FOUND,
                        0,
                        format!("model {} has version {}, not {}", req.name, l.revision, req.version),
                    )));
                }
                !self.state.is_draining()
            }
        };
        Ok(Response::new(ModelReadyResponse { ready }))
    }

    async fn server_metadata(&self, _: Request<ServerMetadataRequest>) -> Reply<ServerMetadataResponse> {
        Ok(Response::new(ServerMetadataResponse { name: "turboembed".into(), version: version(), extensions: vec![] }))
    }

    async fn model_metadata(&self, req: Request<ModelMetadataRequest>) -> Reply<ModelMetadataResponse> {
        let req = req.into_inner();
        let (served, loaded) = self.state.ready(&req.name, &req.version).map_err(plain)?;
        let properties = properties(&self.state, &loaded).map_err(plain)?;
        let (inputs, outputs) = match loaded.task {
            TURBO_TASK_RERANK => (
                vec![tensor("query", "BYTES", &[-1]), tensor("passages", "BYTES", &[-1])],
                vec![tensor("scores", "FP32", &[-1, 1])],
            ),
            TURBO_TASK_EXTRACTIVE_QA => (
                vec![tensor("question", "BYTES", &[1]), tensor("context", "BYTES", &[1])],
                vec![
                    tensor("spans", "INT32", &[-1, 2]),
                    tensor("scores", "FP64", &[-1]),
                    tensor("probabilities", "FP64", &[-1]),
                ],
            ),
            _ => (
                vec![
                    tensor("texts", "BYTES", &[-1]),
                    tensor("ids", "INT32", &[-1, -1]),
                    tensor("mask", "INT32", &[-1, -1]),
                    tensor("types", "INT32", &[-1, -1]),
                ],
                vec![tensor("vectors", "FP32", &[-1, -1])],
            ),
        };
        Ok(Response::new(ModelMetadataResponse {
            name: served.name.clone(),
            versions: vec![loaded.revision.clone()],
            platform: String::new(),
            inputs,
            outputs,
            properties,
        }))
    }

    async fn model_infer(&self, req: Request<ModelInferRequest>) -> Reply<ModelInferResponse> {
        let deadline = Deadline::from_request(&req)?;
        deadline.check()?;
        let state = self.state.clone();
        let req = req.into_inner();
        let worker_deadline = deadline.clone();
        // The run blocks; a request whose client goes away keeps its
        // session until the run ends and the result is released.
        let out = tokio::task::spawn_blocking(move || infer(&state, req, &worker_deadline)).await.map_err(|e| {
            to_status(&Failure::new(TURBO_E_INTERNAL, 0, format!("the request's task failed: {e}")), &[])
        })?;
        if deadline.is_expired() {
            return Err(deadline_status());
        }
        out.map(Response::new).map_err(|(f, fields)| to_status(&f, fields))
    }
}

/// The model's properties, read from the library on each call.
pub(crate) fn properties(state: &State, loaded: &Loaded) -> api::Result<HashMap<String, String>> {
    let rt = state.runtime.get().ok_or_else(|| Failure::new(TURBO_E_INVALID_STATE, 0, "no runtime"))?;
    let mi = loaded.model.info()?;
    let si = loaded.slots[0].info;
    let di = rt.device_info(loaded.device)?;
    let cap = rt.capability(loaded.device, mi.task, si.precision)?;
    let dims: Vec<String> = mi
        .output_dims
        .iter()
        .take((mi.output_dims_count as usize).min(mi.output_dims.len()))
        .map(u32::to_string)
        .collect();
    let mut p: Vec<(&str, String)> = vec![
        ("model_info.task", names::task(mi.task)),
        ("model_info.dim", mi.dim.to_string()),
        ("model_info.pooling", names::pooling(mi.pooling)),
        ("model_info.normalize", names::normalize(mi.normalize)),
        ("model_info.max_seq", mi.max_seq.to_string()),
        ("model_info.max_batch", mi.max_batch.to_string()),
        ("model_info.dtype", names::dtype(mi.dtype)),
        ("model_info.model_id", api::field(&mi.model_id)),
        ("model_info.revision", api::field(&mi.revision)),
        ("model_info.manifest_sha256", api::field(&mi.manifest_sha256)),
        ("model_info.artifact_sha256", api::field(&mi.artifact_sha256)),
        ("model_info.tokenizer_sha256", api::field(&mi.tokenizer_sha256)),
        ("model_info.prefix_query", api::field(&mi.prefix_query)),
        ("model_info.prefix_document", api::field(&mi.prefix_document)),
        ("model_info.output_dims_count", mi.output_dims_count.to_string()),
        ("model_info.output_dims", dims.join(",")),
        ("session_info.max_batch", si.max_batch.to_string()),
        ("session_info.max_seq", si.max_seq.to_string()),
        ("session_info.precision", names::precision(si.precision)),
        ("session_info.compute_dtype", names::dtype(si.compute_dtype)),
        ("device_info.kind", names::device_kind(di.kind)),
        ("device_info.ordinal", di.ordinal.to_string()),
        ("device_info.unified_memory", di.unified_memory.to_string()),
        ("device_info.memory_total", di.memory_total.to_string()),
        ("device_info.arch", api::field(&di.arch)),
        ("device_info.name", api::field(&di.name)),
        ("device_info.vendor", api::field(&di.vendor)),
        ("device_info.backend", api::field(&di.backend)),
        ("device_info.runtime_version", api::field(&di.runtime_version)),
        ("device_info.driver_version", api::field(&di.driver_version)),
        ("capability.status", names::cap(cap.status)),
        ("capability.dtype", names::dtype(cap.dtype)),
        ("capability.options_honored", cap.options_honored.to_string()),
        ("capability.cosine_floor", cap.cosine_floor.to_string()),
        ("capability.speed_ratio", cap.speed_ratio.to_string()),
        ("capability.benchmark", api::field(&cap.benchmark)),
        ("capability.reason", api::field(&cap.reason)),
    ];
    if mi.task == TURBO_TASK_RERANK {
        p.push(("model_info.rerank_query_tokens", mi.rerank_query_tokens.to_string()));
        p.push(("model_info.rerank_pair_template", names::pair_template(mi.rerank_pair_template)));
    }
    Ok(p.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}
