//! The Open Inference Protocol (KServe v2) over gRPC, served from the
//! library's C interface as docs/kserve.md maps it.

pub mod api;
pub mod config;
mod infer;
pub mod status;

/// The messages and service of open_inference_grpc.proto.
pub mod proto {
    #![allow(clippy::all)]
    tonic::include_proto!("inference");
}

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex, OnceLock};

use bytes::Bytes;
use tokio::net::TcpListener;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::codegen::http;
use tonic::transport::server::TcpIncoming;
use tonic::{Code, Request, Response, Status};
use turbo::{
    TURBO_DTYPE_BF16, TURBO_DTYPE_F16, TURBO_DTYPE_F32, TURBO_DTYPE_I32, TURBO_EMBED_STAGE_COUNT, TURBO_PLACE_HOST,
    TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, TURBO_TASK_EMBED, turbo_embed_options, turbo_result_info,
    turbo_session_desc, turbo_session_info, turbo_token_batch,
};

use crate::api::*;
use crate::config::{Device, ModelConfig};
use crate::infer::Rows;
use crate::proto::grpc_inference_service_server::{GrpcInferenceService, GrpcInferenceServiceServer};
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::model_metadata_response::TensorMetadata;
use crate::proto::*;
use crate::status::{EMBED_OPTIONS, SESSION_DESC, describe, to_status};

/// One session of a model, with its three pinned I32 buffers for raw token
/// rows, each [max_batch, max_seq]. Fields are dropped in order, children
/// first: the buffers, then the session.
struct Slot {
    /// ids, mask and types, and their host pointers.
    _buffers: [Buffer; 3],
    ptrs: [*mut u8; 3],
    /// turbo_session_get_info, read once at load: it is fixed at
    /// turbo_session_create.
    info: turbo_session_info,
    session: Session,
}

// The pointers are into the slot's own buffers, written only by the request
// that holds the slot.
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

/// A loaded model and its sessions, dropped sessions first, then the model.
struct Loaded {
    slots: Vec<Slot>,
    model: Model,
    device: u32,
    revision: String,
    idle: Mutex<Vec<usize>>,
}

/// A session taken from its model's pool, returned when this is dropped.
pub struct Held {
    loaded: Arc<Loaded>,
    index: usize,
}

impl Held {
    fn slot(&self) -> &Slot {
        &self.loaded.slots[self.index]
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.loaded.idle.lock().unwrap().push(self.index);
    }
}

impl Loaded {
    /// An idle session, or TURBO_E_BUSY at once when every one is held.
    fn take(self: &Arc<Loaded>) -> Result<Held> {
        let index = self.idle.lock().unwrap().pop().ok_or_else(|| {
            Failure::new(TURBO_E_BUSY, 0, format!("all {} sessions of the model are held", self.slots.len()))
        })?;
        Ok(Held { loaded: self.clone(), index })
    }
}

struct Served {
    name: String,
    config: ModelConfig,
    loaded: OnceLock<Arc<Loaded>>,
}

/// Dropped models first, then contexts, then the runtime.
struct State {
    models: Vec<Served>,
    contexts: Mutex<Vec<(u32, Arc<Context>)>>,
    runtime: OnceLock<Runtime>,
}

fn not_found(name: &str) -> Failure {
    Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, format!("no model {name} is served here"))
}

impl State {
    fn served(&self, name: &str) -> Result<&Served> {
        self.models.iter().find(|m| m.name == name).ok_or_else(|| not_found(name))
    }

    /// A ready model, the version empty or its revision.
    fn ready(&self, name: &str, version: &str) -> Result<(&Served, Arc<Loaded>)> {
        let m = self.served(name)?;
        let l = m
            .loaded
            .get()
            .ok_or_else(|| Failure::new(TURBO_E_INVALID_STATE, 0, format!("model {name} is not loaded yet")))?;
        if !version.is_empty() && version != l.revision {
            return Err(Failure::new(
                TURBO_E_BUNDLE_NOT_FOUND,
                0,
                format!("model {name} has version {}, not {version}", l.revision),
            ));
        }
        Ok((m, l.clone()))
    }

    /// Loads every model in order, stopping at the first failure with the
    /// call, its status, the field it names and the library's message.
    fn load(&self) -> std::result::Result<(), String> {
        let fail = |name: &str, call: &str, fields: &[&str]| {
            let (name, call) = (name.to_string(), call.to_string());
            let fields: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
            move |f: Failure| {
                let fields: Vec<&str> = fields.iter().map(String::as_str).collect();
                format!("model {name}: {call}: {}", describe(&f, &fields))
            }
        };
        let rt = match self.runtime.get() {
            Some(rt) => rt,
            None => {
                let rt = Runtime::create().map_err(fail("-", "turbo_runtime_create", &[]))?;
                self.runtime.get_or_init(|| rt)
            }
        };
        for m in &self.models {
            if m.loaded.get().is_some() {
                continue;
            }
            let (name, c) = (m.name.as_str(), &m.config);
            let device = match c.device {
                Device::Index(i) => i,
                Device::Select => rt.select(TURBO_TASK_EMBED).map_err(fail(name, "turbo_runtime_select", &[]))?,
            };
            let ctx = {
                let mut contexts = self.contexts.lock().unwrap();
                if !contexts.iter().any(|(d, _)| *d == device) {
                    let ctx = rt.context(device).map_err(fail(name, "turbo_context_create", &[]))?;
                    contexts.push((device, Arc::new(ctx)));
                }
                contexts.iter().find(|(d, _)| *d == device).unwrap().1.clone()
            };
            let model = ctx.load(&c.bundle).map_err(fail(name, "turbo_model_load", &[]))?;
            let info = model.info().map_err(fail(name, "turbo_model_get_info", &[]))?;
            let desc = turbo_session_desc {
                struct_size: size_of::<turbo_session_desc>() as u32,
                max_batch: c.max_batch,
                max_seq: c.max_seq,
                precision: c.precision,
                tuning: 0,
                tuning_budget_ms: 0,
            };
            let mut slots = Vec::new();
            for _ in 0..c.sessions {
                let session = model.session(&desc).map_err(fail(name, "turbo_session_create", &SESSION_DESC))?;
                let si = session.info().map_err(fail(name, "turbo_session_get_info", &[]))?;
                let alloc =
                    || {
                        let b = ctx
                            .alloc(TURBO_PLACE_PINNED, TURBO_DTYPE_I32, si.max_batch, si.max_seq)
                            .map_err(fail(name, "turbo_buffer_alloc", &[]))?;
                        let p = b.host_ptr().map_err(fail(name, "turbo_buffer_host_ptr", &[]))?;
                        Ok::<_, String>((b, p))
                    };
                let ((ids, a), (mask, b), (types, t)) = (alloc()?, alloc()?, alloc()?);
                slots.push(Slot { _buffers: [ids, mask, types], ptrs: [a, b, t], info: si, session });
            }
            let idle = Mutex::new((0..slots.len()).rev().collect());
            let loaded = Loaded { slots, model, device, revision: api::field(&info.revision), idle };
            let _ = m.loaded.set(Arc::new(loaded));
        }
        Ok(())
    }
}

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

/// The model's properties, read from the library on each call.
fn properties(state: &State, loaded: &Loaded) -> Result<HashMap<String, String>> {
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
    let p: Vec<(&str, String)> = vec![
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
        // Display gives the shortest decimal that reads back as the same f32.
        ("capability.cosine_floor", cap.cosine_floor.to_string()),
        ("capability.speed_ratio", cap.speed_ratio.to_string()),
        ("capability.benchmark", api::field(&cap.benchmark)),
        ("capability.reason", api::field(&cap.reason)),
    ];
    Ok(p.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

fn tensor(name: &str, datatype: &str, shape: &[i64]) -> TensorMetadata {
    TensorMetadata { name: name.into(), datatype: datatype.into(), shape: shape.to_vec() }
}

struct Inference {
    state: Arc<State>,
}

type Reply<T> = std::result::Result<Response<T>, Status>;

fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

#[tonic::async_trait]
impl GrpcInferenceService for Inference {
    async fn server_live(&self, _: Request<ServerLiveRequest>) -> Reply<ServerLiveResponse> {
        Ok(Response::new(ServerLiveResponse { live: true }))
    }

    async fn server_ready(&self, _: Request<ServerReadyRequest>) -> Reply<ServerReadyResponse> {
        let ready = self.state.models.iter().all(|m| m.loaded.get().is_some());
        Ok(Response::new(ServerReadyResponse { ready }))
    }

    async fn model_ready(&self, req: Request<ModelReadyRequest>) -> Reply<ModelReadyResponse> {
        let req = req.into_inner();
        let m = self.state.served(&req.name).map_err(plain)?;
        let ready = match m.loaded.get() {
            None => false,
            Some(_) => {
                self.state.ready(&req.name, &req.version).map_err(plain)?;
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
        let (m, loaded) = self.state.ready(&req.name, &req.version).map_err(plain)?;
        let properties = properties(&self.state, &loaded).map_err(plain)?;
        Ok(Response::new(ModelMetadataResponse {
            name: m.name.clone(),
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

/// tonic refuses a request message over the decoding limit with
/// OUT_OF_RANGE, read from the frame's length prefix before the message.
/// The service's own refusals all carry turbo-code, so an OUT_OF_RANGE
/// without it is that one, and is answered RESOURCE_EXHAUSTED as other gRPC
/// servers answer it: OUT_OF_RANGE is TURBO_E_CAPACITY's (docs/kserve.md).
fn too_large<B>(mut r: http::Response<B>) -> http::Response<B> {
    let h = r.headers_mut();
    let out_of_range = (Code::OutOfRange as i32).to_string();
    if h.get("grpc-status").is_some_and(|v| v.as_bytes() == out_of_range.as_bytes()) && !h.contains_key("turbo-code") {
        h.insert("grpc-status", (Code::ResourceExhausted as i32).into());
    }
    r
}

/// A server answering gRPC on its listener. Models are loaded by `load`,
/// until which ServerLive is true and ServerReady false.
pub struct Server {
    state: Arc<State>,
    addr: SocketAddr,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<std::result::Result<(), tonic::transport::Error>>,
}

impl Server {
    /// Binds `listen` and answers gRPC, with nothing loaded yet. A bundle
    /// path that names no bundle, or two models of one name, is refused. A
    /// request message over `max_message_bytes` is refused by gRPC with
    /// RESOURCE_EXHAUSTED before it is read.
    pub async fn start(
        listen: SocketAddr,
        models: Vec<ModelConfig>,
        max_message_bytes: usize,
    ) -> std::result::Result<Server, String> {
        let names = config::names(&models)?;
        let listener = TcpListener::bind(listen).await.map_err(|e| format!("listen on {listen}: {e}"))?;
        let addr = listener.local_addr().map_err(|e| format!("listen on {listen}: {e}"))?;
        let state = Arc::new(State {
            models: names
                .into_iter()
                .zip(models)
                .map(|(name, config)| Served { name, config, loaded: OnceLock::new() })
                .collect(),
            runtime: OnceLock::new(),
            contexts: Mutex::new(Vec::new()),
        });
        let svc = GrpcInferenceServiceServer::new(Inference { state: state.clone() })
            .max_decoding_message_size(max_message_bytes);
        let (stop, stopped) = oneshot::channel::<()>();
        let task = tokio::spawn(
            tonic::transport::Server::builder()
                .layer(tower::util::MapResponseLayer::new(too_large))
                .add_service(svc)
                .serve_with_incoming_shutdown(TcpIncoming::from(listener), async {
                    let _ = stopped.await;
                }),
        );
        Ok(Server { state, addr, stop: Some(stop), task })
    }

    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Loads every model and makes its sessions. The error names the model,
    /// the call, its status, the field and the library's message.
    pub async fn load(&self) -> std::result::Result<(), String> {
        let state = self.state.clone();
        tokio::task::spawn_blocking(move || state.load()).await.map_err(|e| format!("loading failed: {e}"))?
    }

    /// Takes an idle session of `model` from its pool as a ModelInfer does,
    /// holding it until the returned value is dropped: TURBO_E_BUSY when
    /// every session is held.
    pub fn hold(&self, model: &str) -> Result<Held> {
        let (_, loaded) = self.state.ready(model, "")?;
        loaded.take()
    }

    /// Stops answering and waits for the server to finish.
    pub async fn stop(mut self) {
        if let Some(s) = self.stop.take() {
            let _ = s.send(());
        }
        let _ = (&mut self.task).await;
    }
}
