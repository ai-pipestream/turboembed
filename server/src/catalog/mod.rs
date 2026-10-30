//! The TurboCatalogService rpcs (proto/turbo_training.proto,
//! docs/grpc-catalog.md): the zoo index the server was started with, the
//! artifacts in its cache, licence acceptances, and installs. Recording
//! and revoking an acceptance and installing are served only with the
//! admin flag or in zoo mode. Listing acceptances and describing,
//! resolving and reading a model are not; in zoo mode a read of a
//! permitted artifact not yet cached fetches it first. Without a catalog
//! every call refuses UNIMPLEMENTED.

mod accept;
mod disk;
mod entry;
mod gate;
mod install;
mod read;
mod through;

pub(crate) use gate::check_load;
pub use read::READ_CHUNK_BYTES;

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tonic::{Request, Response, Status};
use turbo_zoo::{Acceptances, Index, Licence, LicenceClass, Requirement, Transport, UserId, ZooError};

use crate::api::{
    Failure, TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT,
    TURBO_E_INVALID_STATE, TURBO_E_RUNTIME, TURBO_E_UNSUPPORTED,
};
use crate::kserve::Reply;
use crate::proto::training::turbo_catalog_service_server::TurboCatalogService;
use crate::proto::training::{
    DescribeModelRequest, DescribeModelResponse, InstallModelRequest, LicenceTerms, ListInstalledModelsRequest,
    ListInstalledModelsResponse, ListLicenceAcceptancesRequest, ListLicenceAcceptancesResponse,
    ListModelCatalogRequest, ListModelCatalogResponse, ReadModelFileRequest, RecordLicenceAcceptanceRequest,
    RecordLicenceAcceptanceResponse, ResolveModelRequest, ResolveModelResponse, RevokeLicenceAcceptanceRequest,
    RevokeLicenceAcceptanceResponse,
};
use crate::status::to_status;

/// What the catalog serves: a zoo index, the zoo's home directory, and how
/// files are fetched. The home holds the cache under `artifacts/` and the
/// acceptance file `acceptances.json`, as the turbo-zoo command keeps them.
#[derive(Clone)]
pub struct Catalog {
    pub home: PathBuf,
    pub index: Arc<Index>,
    pub transport: Arc<dyn Transport + Send + Sync>,
}

impl fmt::Debug for Catalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Catalog").field("home", &self.home).field("artifacts", &self.index.list().len()).finish()
    }
}

impl Catalog {
    /// The catalog over HTTPS: the index at `index`, or the built-in one.
    /// With `store` (`file:///dir` or `s3://bucket/prefix`, docs/zoo.md),
    /// each file is read from the store first, and the store keeps every
    /// file fetched from its index address, so a store several servers
    /// share, or one seeded ahead, saves the fetch.
    #[cfg(feature = "catalog-http")]
    pub fn http(home: PathBuf, index: Option<&std::path::Path>, store: Option<&str>) -> Result<Catalog, String> {
        let index = match index {
            Some(p) => {
                let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Index::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))?
            }
            None => Index::builtin(),
        };
        let origin = turbo_zoo::Http::new();
        let transport: Arc<dyn Transport + Send + Sync> = match store {
            None => Arc::new(origin),
            Some(store) => Arc::new(store_first(store, origin)?),
        };
        Ok(Catalog { home, index: Arc::new(index), transport })
    }

    fn cache(&self) -> PathBuf {
        self.home.join("artifacts")
    }

    fn acceptances_path(&self) -> PathBuf {
        self.home.join("acceptances.json")
    }

    /// The acceptance file, empty when there is none yet.
    pub(crate) fn acceptances(&self) -> Result<Acceptances, ZooError> {
        let p = self.acceptances_path();
        if p.exists() { Acceptances::load(&p) } else { Ok(Acceptances::default()) }
    }
}

/// `store`, which keeps what it lacks, then the index's own addresses.
#[cfg(feature = "catalog-http")]
fn store_first(store: &str, origin: turbo_zoo::Http) -> Result<turbo_zoo::Sources, String> {
    let list = turbo_zoo::Sources::new();
    let list = if let Some(dir) = store.strip_prefix("file://") {
        if !dir.starts_with('/') {
            return Err(format!("{store}: a file:// store names an absolute directory"));
        }
        list.then_keep(store, turbo_zoo::Blobs::new(dir))
    } else if store.starts_with("s3://") {
        list.then_keep(store, turbo_zoo::S3Bucket::new(store)?)
    } else {
        return Err(format!("{store}: a catalog store is file:///dir or s3://bucket/prefix"));
    };
    Ok(list.then("origin", origin))
}

pub(crate) struct CatalogService {
    catalog: Option<Catalog>,
    admin: bool,
    /// The read-through gates in zoo mode; `None` refuses a read of a
    /// file not yet cached.
    through: Option<through::Fetching>,
    /// Held by the one install that may run.
    installing: Arc<AtomicBool>,
    /// Held across each read, change and write of the acceptance file.
    accepting: Arc<Mutex<()>>,
}

impl CatalogService {
    /// With `prefetch`, every permissive artifact is installed in the
    /// background from now on (through.rs).
    pub(crate) fn new(catalog: Option<Catalog>, admin: bool, zoo: bool, prefetch: bool) -> CatalogService {
        let through = zoo.then(through::Fetching::default);
        if prefetch && let Some(c) = &catalog {
            let (c, gates) = (c.clone(), through.clone().unwrap_or_default());
            tokio::task::spawn_blocking(move || through::prefetch(&c, &gates));
        }
        CatalogService {
            catalog,
            admin,
            through,
            installing: Arc::new(AtomicBool::new(false)),
            accepting: Arc::new(Mutex::new(())),
        }
    }

    fn catalog(&self) -> Result<Catalog, Status> {
        self.catalog.clone().ok_or_else(|| {
            plain(Failure::new(TURBO_E_UNSUPPORTED, 0, "no model catalog: start the server with --catalog-dir"))
        })
    }

    /// Admits an acceptance or an install: with the admin flag, or in zoo
    /// mode, where the user a request names is the caller's own word.
    fn admin(&self, what: &str) -> Result<(), Status> {
        if self.admin || self.through.is_some() {
            return Ok(());
        }
        Err(plain(Failure::new(
            TURBO_E_UNSUPPORTED,
            0,
            format!("{what} is admin-only: start the server with --admin or --zoo to serve it"),
        )))
    }
}

pub(crate) fn plain(f: Failure) -> Status {
    to_status(&f, &[])
}

pub(crate) fn invalid(field: u32, m: impl Into<String>) -> Status {
    plain(Failure::new(TURBO_E_INVALID_ARGUMENT, field, m.into()))
}

/// A zoo error as the status it is answered with.
pub(crate) fn zoo_status(e: ZooError) -> Status {
    let code = match &e {
        ZooError::NotFound(_) => TURBO_E_BUNDLE_NOT_FOUND,
        ZooError::Integrity { .. } => TURBO_E_BUNDLE_INTEGRITY,
        ZooError::Index(_) | ZooError::Acceptances(_) | ZooError::Refused { .. } | ZooError::NotAccepted { .. } => {
            TURBO_E_INVALID_STATE
        }
        ZooError::Transport { .. } | ZooError::Io { .. } => TURBO_E_RUNTIME,
    };
    plain(Failure::new(code, 0, e.to_string()))
}

pub(crate) fn user(field: u32, s: &str) -> Result<UserId, Status> {
    s.parse().map_err(|e: String| invalid(field, format!("user: {e}")))
}

pub(crate) fn terms(r: &Requirement) -> LicenceTerms {
    LicenceTerms { licence: r.licence.to_string(), text: r.text.clone() }
}

/// The terms a request names, as the turbo-zoo command's `accept` reads
/// them: a licence a user can accept, and a terms address only for a
/// licence without a text of its own.
fn requirement(licence: &str, text: &str) -> Result<Requirement, Status> {
    let licence: Licence = licence.parse().map_err(|e: String| invalid(2, e))?;
    if let Some(reason) = licence.refusal() {
        return Err(invalid(2, format!("{licence}: refused ({reason}); the zoo fetches nothing under it")));
    }
    if licence.class() == LicenceClass::Permissive {
        return Err(invalid(2, format!("{licence} is {}: there is nothing to accept", LicenceClass::Permissive)));
    }
    match (Requirement::of(licence), text) {
        (Some(r), "") => Ok(r),
        (Some(_), _) => Err(invalid(3, format!("{licence} has its own text; give no terms address"))),
        (None, "") => Err(invalid(3, format!("{licence} needs the terms address its catalog entry gives"))),
        (None, t) => Ok(Requirement { licence, text: t.to_owned() }),
    }
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, Status> + Send + 'static) -> Result<T, Status> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| plain(Failure::new(TURBO_E_INTERNAL, 0, format!("the catalog's task failed: {e}"))))?
}

#[tonic::async_trait]
impl TurboCatalogService for CatalogService {
    type InstallModelStream = install::UpdateStream;
    type ReadModelFileStream = read::ChunkStream;

    async fn list_model_catalog(&self, req: Request<ListModelCatalogRequest>) -> Reply<ListModelCatalogResponse> {
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = if req.user.is_empty() { None } else { Some(user(1, &req.user)?) };
        let entries = blocking(move || entry::list(&catalog, who.as_ref(), &req.kind, &req.language)).await?;
        Ok(Response::new(ListModelCatalogResponse { entries }))
    }

    async fn list_installed_models(
        &self,
        _: Request<ListInstalledModelsRequest>,
    ) -> Reply<ListInstalledModelsResponse> {
        let catalog = self.catalog()?;
        let models = blocking(move || {
            Ok(catalog.index.list().iter().filter_map(|a| entry::installed(&catalog.cache(), a)).collect())
        })
        .await?;
        Ok(Response::new(ListInstalledModelsResponse { models }))
    }

    async fn record_licence_acceptance(
        &self,
        req: Request<RecordLicenceAcceptanceRequest>,
    ) -> Reply<RecordLicenceAcceptanceResponse> {
        self.admin("RecordLicenceAcceptance")?;
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = user(1, &req.user)?;
        let r = requirement(&req.licence, &req.terms)?;
        let shown = accept::shown(&catalog, &r, &req.artifact_id, &req.text_sha256)?;
        let lock = self.accepting.clone();
        blocking(move || {
            let _held = lock.lock().unwrap_or_else(|p| p.into_inner());
            accept::record(&catalog, &who, &r, &shown).map(Response::new)
        })
        .await
    }

    async fn revoke_licence_acceptance(
        &self,
        req: Request<RevokeLicenceAcceptanceRequest>,
    ) -> Reply<RevokeLicenceAcceptanceResponse> {
        self.admin("RevokeLicenceAcceptance")?;
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = user(1, &req.user)?;
        let r = requirement(&req.licence, &req.terms)?;
        let lock = self.accepting.clone();
        blocking(move || {
            let _held = lock.lock().unwrap_or_else(|p| p.into_inner());
            accept::revoke(&catalog, &who, &r).map(Response::new)
        })
        .await
    }

    async fn list_licence_acceptances(
        &self,
        req: Request<ListLicenceAcceptancesRequest>,
    ) -> Reply<ListLicenceAcceptancesResponse> {
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = if req.user.is_empty() { None } else { Some(user(1, &req.user)?) };
        let admin = self.admin || self.through.is_some();
        blocking(move || accept::list(&catalog, who.as_ref(), admin).map(Response::new)).await
    }

    async fn install_model(&self, req: Request<InstallModelRequest>) -> Reply<Self::InstallModelStream> {
        self.admin("InstallModel")?;
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = user(1, &req.user)?;
        let stream = install::start(catalog, who, req.artifact_id, self.installing.clone()).await?;
        Ok(Response::new(stream))
    }

    async fn describe_model(&self, req: Request<DescribeModelRequest>) -> Reply<DescribeModelResponse> {
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = if req.user.is_empty() { None } else { Some(user(1, &req.user)?) };
        let entry = blocking(move || read::describe(&catalog, who.as_ref(), &req.artifact_id)).await?;
        Ok(Response::new(DescribeModelResponse { entry: Some(entry) }))
    }

    async fn resolve_model(&self, req: Request<ResolveModelRequest>) -> Reply<ResolveModelResponse> {
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = if req.user.is_empty() { None } else { Some(user(3, &req.user)?) };
        let (entry, path) =
            blocking(move || read::resolve(&catalog, who.as_ref(), &req.sha256, &req.artifact_id)).await?;
        Ok(Response::new(ResolveModelResponse { entry: Some(entry), path }))
    }

    async fn read_model_file(&self, req: Request<ReadModelFileRequest>) -> Reply<Self::ReadModelFileStream> {
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = user(1, &req.user)?;
        let through = self.through.clone();
        let stream = read::start(catalog, who, req.artifact_id, req.path, req.sha256, through).await?;
        Ok(Response::new(stream))
    }
}
