//! The TurboTrainingService rpcs (proto/turbo_training.proto,
//! docs/grpc-training.md): the served encoders a static model can be
//! distilled from, the distillation itself, and the static models the
//! training directory holds. Training and deleting are admin calls, served
//! only with the admin flag. Without a training directory every call but
//! ListTeachers refuses UNIMPLEMENTED.

mod describe;
mod train;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use turbo::bundle::Bundle;

use crate::api::{
    Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT, TURBO_E_RUNTIME, TURBO_E_UNSUPPORTED,
};
use crate::kserve::Reply;
use crate::proto::training::turbo_training_service_server::TurboTrainingService;
use crate::proto::training::{
    DeleteStaticModelRequest, DeleteStaticModelResponse, ListStaticModelsRequest, ListStaticModelsResponse,
    ListTeachersRequest, ListTeachersResponse, Teacher, TrainStaticModelRequest,
};
use crate::state::{Served, State};
use crate::status::to_status;
use crate::vocabulary::Store;

/// What a staging directory's name starts with: training work in
/// progress, never a model, removed when the directory is opened.
const STAGING: &str = ".staging-";

/// The longest static model name, in bytes.
const MAX_NAME: usize = 128;

pub(crate) struct TrainingService {
    pub(crate) state: Arc<State>,
    pub(crate) vocabulary: Option<Arc<Store>>,
    /// The training directory: one sealed static model per child.
    pub(crate) dir: Option<PathBuf>,
    pub(crate) admin: bool,
    /// Held by the one training that may run.
    training: Arc<AtomicBool>,
    /// Held across a name's check, rename and registration, and across a
    /// delete, so a model is published or removed whole.
    publishing: Arc<Mutex<()>>,
}

/// The training directory in `dir`, made when it does not exist, with any
/// staging a stopped server left behind removed.
pub(crate) fn open(dir: &Path) -> Result<PathBuf, String> {
    fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    for entry in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
        if path.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(STAGING)) {
            fs::remove_dir_all(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        }
    }
    Ok(dir.to_path_buf())
}

impl TrainingService {
    pub(crate) fn new(
        state: Arc<State>,
        vocabulary: Option<Arc<Store>>,
        dir: Option<PathBuf>,
        admin: bool,
    ) -> TrainingService {
        TrainingService {
            state,
            vocabulary,
            dir,
            admin,
            training: Arc::new(AtomicBool::new(false)),
            publishing: Arc::new(Mutex::new(())),
        }
    }

    fn dir(&self) -> Result<PathBuf, Status> {
        self.dir.clone().ok_or_else(|| {
            plain(Failure::new(TURBO_E_UNSUPPORTED, 0, "no training directory: start the server with --training-dir"))
        })
    }

    fn admin(&self, what: &str) -> Result<(), Status> {
        if self.admin {
            return Ok(());
        }
        Err(plain(Failure::new(
            TURBO_E_UNSUPPORTED,
            0,
            format!("{what} is admin-only: start the server with --admin to serve it"),
        )))
    }
}

pub(crate) fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

pub(crate) fn invalid(field: u32, m: impl Into<String>) -> Status {
    plain(Failure::new(TURBO_E_INVALID_ARGUMENT, field, m.into()))
}

pub(crate) fn runtime(m: impl Into<String>) -> Status {
    plain(Failure::new(TURBO_E_RUNTIME, 0, m.into()))
}

/// A static model's name is its directory's and its served name: ASCII
/// letters, digits, '.', '_' and '-', not starting with '.'.
pub(crate) fn check_name(field: u32, name: &str) -> Result<(), Status> {
    if name.is_empty() || name.len() > MAX_NAME {
        return Err(invalid(field, format!("name: 1 to {MAX_NAME} bytes")));
    }
    if name.starts_with('.') || !name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-')) {
        return Err(invalid(field, format!("name {name:?}: ASCII letters, digits, '.', '_' and '-', not first '.'")));
    }
    Ok(())
}

/// Whether two bundle paths name one directory.
pub(crate) fn same_dir(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The registration serving `dir` under `name`, if any.
pub(crate) fn serving(state: &State, name: &str, dir: &Path) -> Option<Arc<Served>> {
    state.listing().into_iter().find(|m| m.name == name && same_dir(Path::new(&m.config.bundle), dir))
}

/// A served, loaded encoder a static model can be distilled from: an
/// embedding bundle with an architecture, not itself a static model.
pub(crate) fn teacher(m: &Served) -> Option<(Teacher, PathBuf)> {
    let loaded = m.loaded.get()?;
    if loaded.task != turbo::TURBO_TASK_EMBED {
        return None;
    }
    let path = PathBuf::from(&m.config.bundle);
    let b = Bundle::open(&path).ok()?;
    let mf = &b.manifest;
    let arch = mf.architecture.as_ref().filter(|_| mf.static_embedding.is_none())?;
    let t = Teacher {
        name: m.name.clone(),
        model_id: mf.model.id.clone(),
        revision: mf.model.revision.clone(),
        manifest_sha256: b.manifest_sha256.clone(),
        hidden: arch.hidden,
        vocab_size: arch.vocab_size,
    };
    Some((t, path))
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, Status> + Send + 'static) -> Result<T, Status> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the training's task failed: {e}"))))?
}

#[tonic::async_trait]
impl TurboTrainingService for TrainingService {
    type TrainStaticModelStream = train::UpdateStream;

    async fn list_teachers(&self, _: Request<ListTeachersRequest>) -> Reply<ListTeachersResponse> {
        let state = self.state.clone();
        let teachers =
            blocking(move || Ok(state.listing().iter().filter_map(|m| teacher(m)).map(|(t, _)| t).collect())).await?;
        Ok(Response::new(ListTeachersResponse { teachers }))
    }

    async fn train_static_model(&self, req: Request<TrainStaticModelRequest>) -> Reply<Self::TrainStaticModelStream> {
        let dir = self.dir()?;
        self.admin("TrainStaticModel")?;
        let job = train::Job {
            state: self.state.clone(),
            vocabulary: self.vocabulary.clone(),
            dir,
            publishing: self.publishing.clone(),
        };
        Ok(Response::new(train::start(job, req.into_inner(), self.training.clone()).await?))
    }

    async fn list_static_models(&self, _: Request<ListStaticModelsRequest>) -> Reply<ListStaticModelsResponse> {
        let dir = self.dir()?;
        let state = self.state.clone();
        let models = blocking(move || describe::list(&state, &dir).map_err(runtime)).await?;
        Ok(Response::new(ListStaticModelsResponse { models }))
    }

    async fn delete_static_model(&self, req: Request<DeleteStaticModelRequest>) -> Reply<DeleteStaticModelResponse> {
        let dir = self.dir()?;
        self.admin("DeleteStaticModel")?;
        let name = req.into_inner().name;
        check_name(1, &name)?;
        let (state, publishing) = (self.state.clone(), self.publishing.clone());
        let unloaded = blocking(move || {
            let path = dir.join(&name);
            let _one = publishing.lock().unwrap_or_else(|e| e.into_inner());
            if !path.join("manifest.json").is_file() {
                return Err(plain(Failure::new(
                    TURBO_E_BUNDLE_NOT_FOUND,
                    1,
                    format!("no static model {name} in the training directory"),
                )));
            }
            // Only the registration serving this directory goes: a model
            // served under the same name from elsewhere is not this one.
            let served = serving(&state, &name, &path);
            if let Some(m) = &served {
                state.remove_if(m).map_err(plain)?;
            }
            fs::remove_dir_all(&path).map_err(|e| runtime(format!("{}: {e}", path.display())))?;
            Ok(served.is_some())
        })
        .await?;
        Ok(Response::new(DeleteStaticModelResponse { unloaded }))
    }
}
