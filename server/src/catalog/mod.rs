//! The TurboCatalogService rpcs (proto/turbo_training.proto,
//! docs/grpc-catalog.md): the zoo index the server was started with, the
//! artifacts in its cache, licence acceptances, and installs. Recording an
//! acceptance and installing are admin calls, served only with the admin
//! flag. Without a catalog every call refuses UNIMPLEMENTED.

mod disk;
mod entry;
mod install;

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use tonic::{Request, Response, Status};
use turbo_zoo::{Acceptances, Index, Licence, LicenceClass, Requirement, Transport, UserId, ZooError};

use crate::api::{
    Failure, TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT,
    TURBO_E_INVALID_STATE, TURBO_E_RUNTIME, TURBO_E_UNSUPPORTED,
};
use crate::kserve::Reply;
use crate::proto::training::turbo_catalog_service_server::TurboCatalogService;
use crate::proto::training::{
    InstallModelRequest, LicenceTerms, ListInstalledModelsRequest, ListInstalledModelsResponse,
    ListModelCatalogRequest, ListModelCatalogResponse, RecordLicenceAcceptanceRequest, RecordLicenceAcceptanceResponse,
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
    #[cfg(feature = "catalog-http")]
    pub fn http(home: PathBuf, index: Option<&std::path::Path>) -> Result<Catalog, String> {
        let index = match index {
            Some(p) => {
                let bytes = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
                Index::parse(&bytes).map_err(|e| format!("{}: {e}", p.display()))?
            }
            None => Index::builtin(),
        };
        Ok(Catalog { home, index: Arc::new(index), transport: Arc::new(turbo_zoo::Http::new()) })
    }

    fn cache(&self) -> PathBuf {
        self.home.join("artifacts")
    }

    fn acceptances_path(&self) -> PathBuf {
        self.home.join("acceptances.json")
    }

    /// The acceptance file, empty when there is none yet.
    fn acceptances(&self) -> Result<Acceptances, ZooError> {
        let p = self.acceptances_path();
        if p.exists() { Acceptances::load(&p) } else { Ok(Acceptances::default()) }
    }
}

pub(crate) struct CatalogService {
    catalog: Option<Catalog>,
    admin: bool,
    /// Held by the one install that may run.
    installing: Arc<AtomicBool>,
    /// Held across each read, change and write of the acceptance file.
    accepting: Arc<Mutex<()>>,
}

impl CatalogService {
    pub(crate) fn new(catalog: Option<Catalog>, admin: bool) -> CatalogService {
        CatalogService {
            catalog,
            admin,
            installing: Arc::new(AtomicBool::new(false)),
            accepting: Arc::new(Mutex::new(())),
        }
    }

    fn catalog(&self) -> Result<Catalog, Status> {
        self.catalog.clone().ok_or_else(|| {
            plain(Failure::new(TURBO_E_UNSUPPORTED, 0, "no model catalog: start the server with --catalog-dir"))
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
        let lock = self.accepting.clone();
        blocking(move || {
            let _held = lock.lock().unwrap_or_else(|p| p.into_inner());
            let now = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let mut acceptances = catalog.acceptances().map_err(zoo_status)?;
            acceptances.accept_terms(&who, &r, now).map_err(zoo_status)?;
            std::fs::create_dir_all(&catalog.home)
                .map_err(|e| zoo_status(ZooError::Io { path: catalog.home.clone(), message: e.to_string() }))?;
            acceptances.save(&catalog.acceptances_path()).map_err(zoo_status)?;
            // Accepting again keeps the first time.
            let first =
                acceptances.iter().find(|a| a.user == who && a.license == r.licence && a.license_text == r.text);
            Ok(Response::new(RecordLicenceAcceptanceResponse {
                user: who.to_string(),
                accepted: Some(terms(&r)),
                accepted_at: first.map_or(now, |a| a.accepted_at),
            }))
        })
        .await
    }

    async fn install_model(&self, req: Request<InstallModelRequest>) -> Reply<Self::InstallModelStream> {
        self.admin("InstallModel")?;
        let catalog = self.catalog()?;
        let req = req.into_inner();
        let who = user(1, &req.user)?;
        let stream = install::start(catalog, who, req.artifact_id, self.installing.clone()).await?;
        Ok(Response::new(stream))
    }
}
