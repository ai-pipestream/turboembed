//! The Open Inference Protocol's six RPCs (docs/kserve.md): ModelInfer
//! as one write, one run and one read on one held session, and the
//! metadata and readiness calls around it.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use tonic::{Request, Response, Status};
use turbo::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I32, TURBO_EMBED_STAGE_COUNT, TURBO_PLACE_HOST,
    TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, TURBO_TASK_EMBED, turbo_embed_options, turbo_result_info,
    turbo_session_info, turbo_token_batch,
};

use crate::api::{self, *};
use crate::infer::{self, Rows};
use crate::proto::grpc_inference_service_server::GrpcInferenceService;
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::*;
use crate::state::{Held, State, properties};
use crate::status::{EMBED_OPTIONS, to_status};

/// The vectors of one result, owned until the response that carries them is
/// encoded; dropping them releases the buffer, then the result, then the
/// session.
struct Vectors {
    data: Data,
    _result: Output,
    _held: Held,
}

enum Data {
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
fn datatype(dtype: u32) -> Option<&'static str> {
    match dtype {
        TURBO_DTYPE_F32 => Some("FP32"),
        TURBO_DTYPE_F16 => Some("FP16"),
        TURBO_DTYPE_BF16 => Some("BF16"),
        TURBO_DTYPE_I32 => Some("INT32"),
        _ => None,
    }
}

fn int(v: impl Into<i128>) -> InferParameter {
    let v: i128 = v.into();
    InferParameter {
        parameter_choice: Some(ParameterChoice::Int64Param(v.clamp(i64::MIN as i128, i64::MAX as i128) as i64)),
    }
}

fn string(v: impl Into<String>) -> InferParameter {
    InferParameter { parameter_choice: Some(ParameterChoice::StringParam(v.into())) }
}

/// The scalar fields of turbo_result_info as response parameters.
fn result_parameters(i: &turbo_result_info) -> HashMap<String, InferParameter> {
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
    if i.task == TURBO_TASK_EMBED {
        for (s, name) in
            names::EMBED_STAGES.iter().enumerate().take((i.stage_count as usize).min(TURBO_EMBED_STAGE_COUNT))
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
fn infer(state: &State, req: ModelInferRequest) -> Result<ModelInferResponse> {
    let (served, loaded) = state.ready(&req.model_name, &req.model_version)?;
    let plan = infer::plan(&req)?;
    // Every session of a model is made alike, so the first one's limits are
    // the held one's: an oversized raw request is refused before BUSY.
    if let Rows::Raw { batch, seq, .. } = plan.rows {
        fits(&loaded.slots[0].info, batch, seq)?;
    }
    let held = loaded.take()?;
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
    let result = slot.session.run()?;
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

fn tensor(name: &str, datatype: &str, shape: &[i64]) -> TensorMetadata {
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
        let ready =
            self.state.models.read().unwrap_or_else(|e| e.into_inner()).iter().all(|m| m.loaded.get().is_some());
        Ok(Response::new(ServerReadyResponse { ready }))
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
                true
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
        Ok(Response::new(ModelMetadataResponse {
            name: served.name.clone(),
            versions: vec![loaded.revision.clone()],
            platform: String::new(),
            inputs: vec![
                tensor("texts", "BYTES", &[-1]),
                tensor("ids", "INT32", &[-1, -1]),
                tensor("mask", "INT32", &[-1, -1]),
                tensor("types", "INT32", &[-1, -1]),
            ],
            outputs: vec![tensor("vectors", "FP32", &[-1, -1])],
            properties,
        }))
    }

    async fn model_infer(&self, req: Request<ModelInferRequest>) -> Reply<ModelInferResponse> {
        let state = self.state.clone();
        let req = req.into_inner();
        // The run blocks; a request whose client goes away keeps its
        // session until the run ends and the result is released.
        let out = tokio::task::spawn_blocking(move || infer(&state, req)).await.map_err(|e| {
            to_status(&Failure::new(TURBO_E_INTERNAL, 0, format!("the request's task failed: {e}")), &[])
        })?;
        out.map(Response::new).map_err(|f| to_status(&f, &EMBED_OPTIONS))
    }
}
