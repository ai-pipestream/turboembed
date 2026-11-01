//! Model sources a user registers (docs/grpc-catalog.md, "Model sources"):
//! the zoo's holders file, `holders.json` beside the acceptance file, read
//! and changed for one user at a time, and the Google sign-in that
//! registers a Drive folder with the user's own grant. An install for a
//! user tries their sources in order before the catalog's own transport.
//!
//! Under the admin flag every kind the server reads is taken: `file://`,
//! `s3://` (the server's `AWS_*` environment) and `gdrive://`. In zoo mode
//! alone, where the user a request names is the caller's own word, only a
//! `gdrive://` source signed in with the user's grant or an API key is: a
//! caller may not point the server at its own disk or reuse its
//! credentials.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
#[cfg(feature = "catalog-http")]
use std::time::Duration;
use std::time::Instant;

use tonic::Status;
use turbo_zoo::{Holder, Holders, SignIn, Transport, UserId, ZooError};

use super::{Catalog, invalid, plain, zoo_status};
use crate::api::{Failure, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED};
use crate::proto::training::{ListModelSourcesResponse, ModelSource, SourceSignIn};

/// How long a started sign-in may take.
pub const SIGN_IN_SECONDS: u64 = 600;

/// The most sign-ins waiting at once; the oldest is dropped past it.
#[cfg(feature = "catalog-http")]
const WAITING_MAX: usize = 256;

/// The OAuth client the server signs users in to Google Drive with, and
/// where the consent page sends them back: the page that calls
/// FinishDriveSignIn, registered with the client as its redirect.
#[derive(Clone)]
pub struct DriveSignIn {
    pub client_id: String,
    pub client_secret: Option<String>,
    pub redirect_uri: String,
    pub auth_endpoint: String,
    pub token_endpoint: String,
    /// The Drive API an install reads a signed-in source through.
    pub drive_endpoint: Option<String>,
}

impl std::fmt::Debug for DriveSignIn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriveSignIn")
            .field("client_id", &self.client_id)
            .field("redirect_uri", &self.redirect_uri)
            .finish_non_exhaustive()
    }
}

/// The variables [`DriveSignIn::from_vars`] reads.
pub const ENV_DRIVE_CLIENT_ID: &str = "TURBO_KSERVE_DRIVE_CLIENT_ID";
pub const ENV_DRIVE_CLIENT_SECRET: &str = "TURBO_KSERVE_DRIVE_CLIENT_SECRET";
pub const ENV_DRIVE_REDIRECT: &str = "TURBO_KSERVE_DRIVE_REDIRECT";

impl DriveSignIn {
    /// The client against Google's own endpoints.
    pub fn google(client_id: &str, client_secret: Option<&str>, redirect_uri: &str) -> DriveSignIn {
        DriveSignIn {
            client_id: client_id.to_owned(),
            client_secret: client_secret.map(str::to_owned),
            redirect_uri: redirect_uri.to_owned(),
            auth_endpoint: "https://accounts.google.com/o/oauth2/v2/auth".into(),
            token_endpoint: "https://oauth2.googleapis.com/token".into(),
            drive_endpoint: None,
        }
    }

    /// From the client id, secret and redirect variables: none when no
    /// client id is set, and an error when one is set without a redirect.
    pub fn from_vars(var: impl Fn(&str) -> Option<String>) -> Result<Option<DriveSignIn>, String> {
        let var = |k: &str| var(k).filter(|v| !v.is_empty());
        let Some(id) = var(ENV_DRIVE_CLIENT_ID) else { return Ok(None) };
        let redirect =
            var(ENV_DRIVE_REDIRECT).ok_or_else(|| format!("{ENV_DRIVE_CLIENT_ID} needs {ENV_DRIVE_REDIRECT}"))?;
        if !redirect.starts_with("https://") && !redirect.starts_with("http://127.0.0.1:") {
            return Err(format!("{ENV_DRIVE_REDIRECT}={redirect}: https://, or http://127.0.0.1:PORT on this machine"));
        }
        Ok(Some(DriveSignIn::google(&id, var(ENV_DRIVE_CLIENT_SECRET).as_deref(), &redirect)))
    }
}

/// A sign-in started and not yet finished.
#[cfg_attr(not(feature = "catalog-http"), allow(dead_code))]
struct Waiting {
    user: UserId,
    name: String,
    address: String,
    verifier: String,
    at: Instant,
}

/// The sources' state the service keeps: the sign-in client, the
/// sign-ins waiting, and the lock held across each change of the file.
#[derive(Default)]
#[cfg_attr(not(feature = "catalog-http"), allow(dead_code))]
pub(crate) struct Sources {
    pub(super) drive: Option<DriveSignIn>,
    waiting: Mutex<HashMap<String, Waiting>>,
    changing: Mutex<()>,
}

impl Sources {
    pub(crate) fn new(drive: Option<DriveSignIn>) -> Arc<Sources> {
        Arc::new(Sources { drive, ..Sources::default() })
    }
}

fn path(catalog: &Catalog) -> std::path::PathBuf {
    catalog.home.join("holders.json")
}

fn load(catalog: &Catalog) -> Result<Holders, Status> {
    Holders::load(&path(catalog)).map_err(zoo_status)
}

fn save(catalog: &Catalog, holders: &Holders) -> Result<(), Status> {
    std::fs::create_dir_all(&catalog.home)
        .map_err(|e| zoo_status(ZooError::Io { path: catalog.home.clone(), message: e.to_string() }))?;
    holders.save(&path(catalog)).map_err(zoo_status)
}

fn shown(h: &Holder) -> ModelSource {
    let sign_in = match h.sign_in {
        SignIn::Environment => SourceSignIn::Environment,
        SignIn::GoogleOauth { .. } => SourceSignIn::Google,
        SignIn::ApiKey { .. } => SourceSignIn::ApiKey,
    };
    ModelSource { name: h.name.clone(), address: h.address.clone(), sign_in: sign_in as i32 }
}

/// `who`'s sources in order; nothing without a user.
pub(super) fn list(
    catalog: &Catalog,
    s: &Sources,
    who: Option<&UserId>,
    admin: bool,
) -> Result<ListModelSourcesResponse, Status> {
    let drive_sign_in = cfg!(feature = "catalog-http") && s.drive.is_some();
    let sources = match who {
        Some(w) => load(catalog)?.list(w).iter().map(shown).collect(),
        None => Vec::new(),
    };
    let user = who.map(UserId::to_string).unwrap_or_default();
    Ok(ListModelSourcesResponse { user, sources, admin, drive_sign_in })
}

fn refused(m: String) -> Status {
    plain(Failure::new(TURBO_E_UNSUPPORTED, 3, m))
}

/// Whether the server takes `holder` from a caller, `operator` when it
/// runs with the admin flag. Field 3 is the request's address.
fn admissible(holder: &Holder, operator: bool) -> Result<(), Status> {
    let a = &holder.address;
    match holder.scheme() {
        "file" | "s3" if !operator => {
            Err(refused(format!("{a}: zoo mode takes gdrive:// sources only; file:// and s3:// need --admin")))
        }
        "file" => {
            let dir = a.strip_prefix("file://").unwrap_or_default();
            if !holder.sign_in.is_environment() || !(dir.starts_with('/') || Path::new(dir).is_absolute()) {
                return Err(invalid(3, format!("{a}: a file:// source names an absolute directory, with no sign-in")));
            }
            Ok(())
        }
        "s3" if !holder.sign_in.is_environment() => {
            Err(invalid(3, format!("{a}: an s3:// source signs in from the server's environment")))
        }
        "s3" if !cfg!(feature = "catalog-http") => {
            Err(refused(format!("{a}: this server was built without catalog-http")))
        }
        "s3" => Ok(()),
        "gdrive" if !cfg!(feature = "catalog-http") => {
            Err(refused(format!("{a}: this server was built without catalog-http")))
        }
        "gdrive" if holder.sign_in.is_environment() && !operator => {
            Err(refused(format!("{a}: in zoo mode a gdrive:// source signs in with Google or an API key")))
        }
        "gdrive" => drive_address(a),
        _ => Err(invalid(3, format!("{a}: a source is file://, s3:// or gdrive://"))),
    }
}

#[cfg(feature = "catalog-http")]
fn drive_address(a: &str) -> Result<(), Status> {
    turbo_zoo::GoogleDrive::check_address(a).map_err(|e| invalid(3, e))
}

#[cfg(not(feature = "catalog-http"))]
fn drive_address(a: &str) -> Result<(), Status> {
    Err(refused(format!("{a}: this server was built without catalog-http")))
}

fn holder(name: &str, address: &str, sign_in: SignIn) -> Result<Holder, Status> {
    Holder::new(name, address, sign_in).map_err(|e| invalid(2, e.to_string()))
}

/// Registers `holder` for `who`, in place of theirs of the same name.
fn put(catalog: &Catalog, s: &Sources, who: &UserId, h: Holder) -> Result<ModelSource, Status> {
    let _held = s.changing.lock().map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 0, e.to_string())))?;
    let mut holders = load(catalog)?;
    let out = shown(&h);
    holders.register(who, h).map_err(zoo_status)?;
    save(catalog, &holders)?;
    Ok(out)
}

pub(super) fn register(
    catalog: &Catalog,
    s: &Sources,
    who: &UserId,
    name: &str,
    address: &str,
    api_key: &str,
    operator: bool,
) -> Result<ModelSource, Status> {
    let sign_in = if api_key.is_empty() { SignIn::Environment } else { SignIn::ApiKey { key: api_key.to_owned() } };
    let h = holder(name, address, sign_in)?;
    if !api_key.is_empty() && h.scheme() != "gdrive" {
        return Err(invalid(4, "an API key is for a gdrive:// source"));
    }
    admissible(&h, operator)?;
    put(catalog, s, who, h)
}

/// Removes `who`'s source `name`; the file is written only when it was there.
pub(super) fn remove(catalog: &Catalog, s: &Sources, who: &UserId, name: &str) -> Result<bool, Status> {
    let _held = s.changing.lock().map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 0, e.to_string())))?;
    let mut holders = load(catalog)?;
    let was = holders.remove(who, name).is_some();
    if was {
        save(catalog, &holders)?;
    }
    Ok(was)
}

fn no_sign_in() -> Status {
    plain(Failure::new(
        TURBO_E_UNSUPPORTED,
        0,
        format!("Google sign-in is not configured: set {ENV_DRIVE_CLIENT_ID} and {ENV_DRIVE_REDIRECT}"),
    ))
}

#[cfg(feature = "catalog-http")]
fn oauth(d: &DriveSignIn) -> Result<turbo_zoo::GoogleOAuth, Status> {
    let secret = d.client_secret.as_deref();
    turbo_zoo::GoogleOAuth::with_endpoints(&d.client_id, secret, &d.auth_endpoint, &d.token_endpoint)
        .map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 0, e)))
}

/// Starts `who`'s sign-in for the source `name` at `address`: the
/// consent page, and the state it comes back with.
#[cfg(feature = "catalog-http")]
pub(super) fn start(s: &Sources, who: &UserId, name: &str, address: &str) -> Result<(String, String), Status> {
    let d = s.drive.as_ref().ok_or_else(no_sign_in)?;
    // Checked now with a stand-in grant, so a bad name or address fails
    // before the user is sent to Google.
    let probe = SignIn::GoogleOauth { client_id: d.client_id.clone(), client_secret: None, refresh_token: "-".into() };
    admissible(&holder(name, address, probe)?, false)?;
    let pkce = turbo_zoo::Pkce::new();
    let url = oauth(d)?.authorize_url(&d.redirect_uri, &pkce);
    let mut waiting = s.waiting.lock().map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 0, e.to_string())))?;
    let ttl = Duration::from_secs(SIGN_IN_SECONDS);
    waiting.retain(|_, w| w.at.elapsed() < ttl);
    if waiting.len() >= WAITING_MAX
        && let Some(oldest) = waiting.iter().min_by_key(|(_, w)| w.at).map(|(k, _)| k.clone())
    {
        waiting.remove(&oldest);
    }
    let w = Waiting {
        user: who.clone(),
        name: name.to_owned(),
        address: address.to_owned(),
        verifier: pkce.verifier,
        at: Instant::now(),
    };
    waiting.insert(pkce.state.clone(), w);
    Ok((url, pkce.state))
}

#[cfg(not(feature = "catalog-http"))]
pub(super) fn start(_s: &Sources, _who: &UserId, _name: &str, _address: &str) -> Result<(String, String), Status> {
    Err(no_sign_in())
}

/// Finishes the sign-in `state` names with `code`, for the user who
/// started it, and registers its source with the grant's refresh token.
/// The client secret stays the server's own and is not written with it.
#[cfg(feature = "catalog-http")]
pub(super) fn finish(
    catalog: &Catalog,
    s: &Sources,
    who: &UserId,
    state: &str,
    code: &str,
) -> Result<ModelSource, Status> {
    let d = s.drive.as_ref().ok_or_else(no_sign_in)?;
    let gone = || invalid(2, "no sign-in is waiting with this state: it finished, ran out or was never started");
    let w = {
        let mut waiting = s.waiting.lock().map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 0, e.to_string())))?;
        match waiting.get(state) {
            Some(w) if &w.user != who => return Err(gone()),
            Some(_) => waiting.remove(state).ok_or_else(gone)?,
            None => return Err(gone()),
        }
    };
    if w.at.elapsed() >= Duration::from_secs(SIGN_IN_SECONDS) {
        return Err(gone());
    }
    if code.is_empty() {
        return Err(invalid(3, "code is required"));
    }
    let pkce = turbo_zoo::Pkce { verifier: w.verifier, state: state.to_owned() };
    let grant = oauth(d)?
        .exchange(code, &d.redirect_uri, &pkce)
        .map_err(|e| plain(Failure::new(TURBO_E_INVALID_STATE, 3, e)))?;
    let refresh_token = grant.refresh_token.ok_or_else(|| {
        plain(Failure::new(
            TURBO_E_INVALID_STATE,
            3,
            "Google sent no refresh token; sign in again and allow offline access",
        ))
    })?;
    let sign_in = SignIn::GoogleOauth { client_id: d.client_id.clone(), client_secret: None, refresh_token };
    put(catalog, s, who, holder(&w.name, &w.address, sign_in)?)
}

#[cfg(not(feature = "catalog-http"))]
pub(super) fn finish(
    _c: &Catalog,
    _s: &Sources,
    _who: &UserId,
    _state: &str,
    _code: &str,
) -> Result<ModelSource, Status> {
    Err(no_sign_in())
}

/// What an install for `who` fetches through: their sources in order,
/// then the catalog's own transport. The first `file://` store keeps
/// what the sources after it serve. A source that cannot be opened is
/// passed over with the reason it gave in the list's errors.
pub(super) fn for_install(
    catalog: &Catalog,
    s: &Sources,
    who: &UserId,
) -> Result<Arc<dyn Transport + Send + Sync>, Status> {
    let holders = load(catalog)?;
    let mine = holders.list(who);
    if mine.is_empty() {
        return Ok(catalog.transport.clone());
    }
    let mut list = turbo_zoo::Sources::new();
    let mut kept = false;
    for h in mine {
        let name = format!("{} ({})", h.name, h.address);
        match h.scheme() {
            "file" => {
                let store = turbo_zoo::Blobs::new(h.address.trim_start_matches("file://"));
                list = if kept { list.then(name, store) } else { list.then_keep(name, store) };
                kept = true;
            }
            _ => match open(s, h) {
                Ok(t) => list = list.then(name, t),
                Err(e) => list = list.then(name, Unopened(e)),
            },
        }
    }
    Ok(Arc::new(list.then("catalog", catalog.transport.clone())))
}

/// A source that could not be opened, which answers every request with why.
struct Unopened(String);

impl Transport for Unopened {
    fn get(&self, _url: &turbo_zoo::Url, _size: u64) -> Result<Vec<u8>, String> {
        Err(self.0.clone())
    }
}

#[cfg(feature = "catalog-http")]
fn open(s: &Sources, h: &Holder) -> Result<Box<dyn Transport + Send + Sync>, String> {
    use turbo_zoo::{DriveAuth, GoogleDrive, Renewing, S3Bucket};
    match (h.scheme(), &h.sign_in) {
        ("s3", _) => Ok(Box::new(S3Bucket::new(&h.address)?)),
        ("gdrive", sign_in) => {
            let d = s.drive.as_ref();
            let auth = match sign_in {
                SignIn::Environment => DriveAuth::from_vars(&h.address, |k| std::env::var(k).ok())?,
                SignIn::ApiKey { key } => DriveAuth::ApiKey(key.clone()),
                SignIn::GoogleOauth { client_id, client_secret, refresh_token } => {
                    let d = d.filter(|d| &d.client_id == client_id).ok_or("signed in with another OAuth client")?;
                    let secret = client_secret.as_deref().or(d.client_secret.as_deref());
                    let o =
                        turbo_zoo::GoogleOAuth::with_endpoints(client_id, secret, &d.auth_endpoint, &d.token_endpoint)?;
                    DriveAuth::Refresh(Arc::new(Renewing::new(o, refresh_token)?))
                }
            };
            let endpoint = d.and_then(|d| d.drive_endpoint.as_deref());
            Ok(Box::new(GoogleDrive::with(&h.address, endpoint, auth)?))
        }
        (other, _) => Err(format!("{other}:// is not a source this server reads")),
    }
}

#[cfg(not(feature = "catalog-http"))]
fn open(_s: &Sources, h: &Holder) -> Result<Box<dyn Transport + Send + Sync>, String> {
    Err(format!("{}: this server was built without catalog-http", h.address))
}
