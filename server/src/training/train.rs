//! TrainStaticModel: one training at a time, admitted after the request's
//! checks, then the bundle library's distillation (docs/static.md) into a
//! staging directory with each step streamed as progress, the seal that
//! verifies the bundle, and the move into the training directory under
//! the model's name, where the server loads and serves it. A training that
//! fails, or whose client goes away before the move, leaves nothing behind.

use std::fs;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use turbo_bundle::distill::from_base::{self, Options};
use turbo_bundle::distill::{self, Progress};

use super::{STAGING, blocking, check_name, describe::describe, invalid, plain, runtime, serving, teacher};
use crate::api::{
    Failure, Runtime, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_STATE,
};
use crate::config::{Device, ModelConfig};
use crate::proto::training::train_static_model_update::Update;
use crate::proto::training::{
    RefusedTerm, StaticDtype, StaticModelQuality, TrainStage, TrainStaticModelProgress, TrainStaticModelRequest,
    TrainStaticModelUpdate, TrainedStaticModel,
};
use crate::state::{MAX_SESSIONS_PER_MODEL, State};
use crate::vocabulary::{Store, Stored};

pub(crate) type UpdateStream = Pin<Box<dyn Stream<Item = Result<TrainStaticModelUpdate, Status>> + Send>>;

/// The most learned terms one training adds: each is a row through the
/// teacher.
pub(crate) const MAX_TERMS: usize = 1 << 18;

/// Model2Vec's defaults where the request leaves a field 0.
const DEFAULT_PCA_DIMS: u32 = 256;
const DEFAULT_SIF: f64 = 1e-4;
const DEFAULT_MAX_LENGTH: u32 = 512;
/// The made recipe's embed.max_seq (bundle/src/distill/from_base.rs).
const MAX_LENGTH: u32 = 8192;

/// Names each staging directory apart within one server.
static SEQ: AtomicU64 = AtomicU64::new(0);

/// What a training reaches beyond its request.
pub(crate) struct Job {
    pub(crate) state: Arc<State>,
    pub(crate) vocabulary: Option<Arc<Store>>,
    pub(crate) dir: PathBuf,
    pub(crate) publishing: Arc<Mutex<()>>,
}

/// The training slot, given back when dropped.
struct Slot(Arc<AtomicBool>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// A request with its defaults filled in and its checks passed.
struct Plan {
    name: String,
    base: PathBuf,
    options: Options,
    terms: Vec<String>,
    vocabulary: (String, String),
    sessions: u32,
}

type Tx = mpsc::Sender<Result<TrainStaticModelUpdate, Status>>;

fn progress(stage: TrainStage) -> TrainStaticModelProgress {
    TrainStaticModelProgress { stage: stage as i32, ..Default::default() }
}

fn update(p: TrainStaticModelProgress) -> TrainStaticModelUpdate {
    TrainStaticModelUpdate { update: Some(Update::Progress(p)) }
}

/// A distillation step as the progress it is streamed as.
fn step(p: Progress) -> TrainStaticModelProgress {
    match p {
        Progress::Kept { kept, vocab_size, added, refused } => TrainStaticModelProgress {
            completed_items: kept as u64,
            total_items: vocab_size as u64,
            added_terms: added as u64,
            refused_terms: refused as u64,
            ..progress(TrainStage::Kept)
        },
        Progress::Embedded { done, total } => TrainStaticModelProgress {
            completed_items: done as u64,
            total_items: total as u64,
            ..progress(TrainStage::Embedding)
        },
        Progress::Reduced { pca_dims, explained_variance } => TrainStaticModelProgress {
            completed_items: pca_dims as u64,
            explained_variance,
            ..progress(TrainStage::Reduced)
        },
        Progress::Measured { base_top1, static_top1, similarity_spearman } => TrainStaticModelProgress {
            quality: Some(StaticModelQuality { base_top1, static_top1, similarity_spearman }),
            ..progress(TrainStage::Measured)
        },
    }
}

/// The single terms of a learned vocabulary (vocabulary/learn.rs), in its
/// order: the rows whose kind is `term`, after the `#` lines and the
/// column header.
fn vocabulary_terms(tsv: &[u8]) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(tsv).map_err(|e| format!("the vocabulary is not UTF-8: {e}"))?;
    let mut rows = text.lines().filter(|l| !l.starts_with('#'));
    if !rows.next().is_some_and(|h| h.starts_with("term\tkind\t")) {
        return Err("the vocabulary has no term\\tkind column header".into());
    }
    Ok(rows
        .filter_map(|r| r.split_once('\t'))
        .filter(|(_, rest)| rest.starts_with("term\t"))
        .map(|(t, _)| t.into())
        .collect())
}

/// The request's checks, its defaults, and the vocabulary's terms.
fn plan(job: &Job, r: TrainStaticModelRequest) -> Result<Plan, Status> {
    check_name(1, &r.name)?;
    if job.dir.join(&r.name).exists() {
        return Err(plain(Failure::new(TURBO_E_INVALID_STATE, 1, format!("static model {} already exists", r.name))));
    }
    if job.state.peek(&r.name).is_ok() {
        return Err(invalid(1, format!("a model named {} is already served", r.name)));
    }
    let served = job.state.peek(&r.teacher).map_err(|f| plain(Failure { field: 2, ..f }))?.0;
    let (t, base) = teacher(&served).ok_or_else(|| {
        invalid(2, format!("model {} is not a loaded encoder a static model is distilled from", r.teacher))
    })?;
    let pca_dims = if r.pca_dims == 0 { DEFAULT_PCA_DIMS.min(t.hidden) } else { r.pca_dims };
    if pca_dims > t.hidden || pca_dims >= t.vocab_size {
        return Err(invalid(
            3,
            format!("pca_dims {pca_dims}: at most the teacher's width {} and under its vocabulary", t.hidden),
        ));
    }
    let max_length = if r.max_length == 0 { DEFAULT_MAX_LENGTH } else { r.max_length };
    if max_length > MAX_LENGTH {
        return Err(invalid(6, format!("max_length {max_length} is over {MAX_LENGTH}, a static model's max_seq")));
    }
    let sif_coefficient = if r.sif_coefficient == 0.0 { DEFAULT_SIF } else { r.sif_coefficient };
    if !(sif_coefficient > 0.0 && sif_coefficient < 1.0) {
        return Err(invalid(4, format!("sif_coefficient {sif_coefficient} is not in (0, 1)")));
    }
    let dtype = match StaticDtype::try_from(r.dtype) {
        Ok(StaticDtype::Unspecified | StaticDtype::F32) => "F32",
        Ok(StaticDtype::F16) => "F16",
        Err(_) => return Err(invalid(5, format!("dtype {}: not a StaticDtype", r.dtype))),
    };
    let sessions = r.sessions.max(1);
    if sessions > MAX_SESSIONS_PER_MODEL {
        return Err(plain(Failure::new(
            TURBO_E_CAPACITY,
            10,
            format!("sessions {sessions} is over max_sessions_per_model {MAX_SESSIONS_PER_MODEL}"),
        )));
    }
    let (mut terms, mut vocabulary) = (Vec::new(), (String::new(), String::new()));
    if !r.vocabulary_artifact_id.is_empty() {
        let id = &r.vocabulary_artifact_id;
        let store = job.vocabulary.as_ref().ok_or_else(|| {
            plain(Failure::new(TURBO_E_INVALID_STATE, 7, "no vocabulary store: start the server with --vocabulary-dir"))
        })?;
        let not_found =
            || plain(Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 7, format!("no learned vocabulary {id:?} in the store")));
        let Some(Stored::Vocabulary(d, bytes)) = store.read(id).map_err(runtime)? else { return Err(not_found()) };
        terms = vocabulary_terms(&bytes).map_err(|e| invalid(7, format!("vocabulary {id}: {e}")))?;
        vocabulary = (d.artifact_id, d.sha256);
    }
    terms.extend(r.terms);
    if terms.len() > MAX_TERMS {
        return Err(plain(Failure::new(
            TURBO_E_CAPACITY,
            8,
            format!("{} learned terms is over the most one training adds, {MAX_TERMS}", terms.len()),
        )));
    }
    let options = Options {
        id: r.name.clone(),
        revision: if r.revision.is_empty() { "1".into() } else { r.revision },
        pca_dims,
        sif_coefficient,
        dtype: dtype.into(),
        max_length,
    };
    Ok(Plan { name: r.name, base, options, terms, vocabulary, sessions })
}

/// Checks and admits the request, then streams the training. Every
/// refusal before the first message is the call's own status.
pub(super) async fn start(
    job: Job,
    req: TrainStaticModelRequest,
    training: Arc<AtomicBool>,
) -> Result<UpdateStream, Status> {
    // The checks first, so a bad request is refused for what it is even
    // while another training runs; the name is checked again when the
    // model is published.
    let job = Arc::new(job);
    let j = job.clone();
    let plan = blocking(move || plan(&j, req)).await?;
    if training.swap(true, Ordering::AcqRel) {
        return Err(plain(Failure::new(TURBO_E_BUSY, 0, "another training is running; try again when it ends")));
    }
    let slot = Slot(training);
    let (tx, rx) = mpsc::channel(16);
    let _ = tx.send(Ok(update(progress(TrainStage::Admitted)))).await;
    let staging = job.dir.join(format!("{STAGING}{}-{}", plan.name, SEQ.fetch_add(1, Ordering::Relaxed)));
    tokio::spawn(async move {
        let (t, s) = (tx.clone(), staging.clone());
        // A panic in the distillation is the stream's INTERNAL status, not
        // a stream that just ends.
        let out = match tokio::task::spawn_blocking(move || run(&job, &plan, &s, &t)).await {
            Ok(out) => out,
            Err(e) => Err(plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the training's task failed: {e}")))),
        };
        let _ = tokio::task::spawn_blocking(move || fs::remove_dir_all(staging)).await;
        // The slot is free once the staging is gone, before the last
        // message, so a client that has it can start the next training.
        drop(slot);
        let last = out.map(|trained| TrainStaticModelUpdate { update: Some(Update::Trained(trained)) });
        let _ = tx.send(last).await;
    });
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// The CPU's runtime index: a static model runs on the CPU backend.
fn cpu(state: &State) -> Result<u32, Status> {
    let rt = match state.runtime.get() {
        Some(rt) => rt,
        None => {
            let rt = Runtime::create().map_err(plain)?;
            state.runtime.get_or_init(|| rt)
        }
    };
    (0..)
        .map_while(|i| rt.device_info(i).ok().map(|d| (i, d)))
        .find(|(_, d)| d.kind == turbo::TURBO_DEVICE_CPU)
        .map(|(i, _)| i)
        .ok_or_else(|| runtime("the runtime lists no CPU"))
}

/// Distils, seals, moves and serves the model. The client going away is
/// seen between steps: before the move, the training stops there.
fn run(job: &Job, plan: &Plan, staging: &Path, tx: &Tx) -> Result<TrainedStaticModel, Status> {
    let gone = || if tx.is_closed() { Err(runtime("the client cancelled the training")) } else { Ok(()) };
    let fail = |what: &'static str| move |e: String| runtime(format!("{what}: {e}"));
    let mut recipe =
        from_base::recipe(&plan.base, &plan.options, &staging.join("work")).map_err(fail("the static recipe"))?;
    let bundle = staging.join("bundle");
    let staged = distill::stage_with(&recipe, &plan.base, &bundle, &plan.terms, &mut |p| {
        let _ = tx.blocking_send(Ok(update(step(p))));
    })
    .map_err(fail("distillation"))?;
    gone()?;
    distill::seal(&mut recipe, &bundle).map_err(fail("sealing"))?;
    let _ = tx.blocking_send(Ok(update(progress(TrainStage::Sealed))));
    let config = ModelConfig {
        bundle: job.dir.join(&plan.name).to_string_lossy().into_owned(),
        name: Some(plan.name.clone()),
        device: Device::Index(cpu(&job.state)?),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: plan.sessions,
    };
    let dest = PathBuf::from(&config.bundle);
    // Published whole under the lock: moved, registered, loaded and
    // described, or none of it. A delete waits for the lock, so it never
    // removes a bundle that is loading.
    let model = {
        let _one = job.publishing.lock().unwrap_or_else(|e| e.into_inner());
        gone()?;
        if dest.exists() {
            return Err(plain(Failure::new(
                TURBO_E_INVALID_STATE,
                1,
                format!("static model {} already exists", plan.name),
            )));
        }
        let registered = job.state.add(plan.name.clone(), config).map_err(|f| plain(Failure { field: 1, ..f }))?;
        let published = fs::rename(&bundle, &dest)
            .map_err(|e| runtime(format!("{}: {e}", dest.display())))
            .and_then(|()| job.state.load_one(&registered).map_err(|e| e.to_status()))
            .and_then(|()| {
                let served = serving(&job.state, &plan.name, &dest).is_some();
                describe(&dest, &plan.name, served).map_err(runtime)
            })
            .and_then(|m| if m.verified { Ok(m) } else { Err(runtime(format!("{}: {}", dest.display(), m.problem))) });
        if published.is_err() {
            let _ = job.state.remove_if(&registered);
            let _ = fs::remove_dir_all(&dest);
        }
        published?
    };
    let _ = tx.blocking_send(Ok(update(progress(TrainStage::Loaded))));
    Ok(TrainedStaticModel {
        model: Some(model),
        added_terms: staged.added,
        refused_terms: staged
            .refused
            .into_iter()
            .map(|(term, r)| RefusedTerm { term, reason: format!("{r:?}").to_ascii_uppercase() })
            .collect(),
        vocabulary_artifact_id: plan.vocabulary.0.clone(),
        vocabulary_sha256: plan.vocabulary.1.clone(),
    })
}
