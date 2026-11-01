//! The kinds of source a holder can be (#1744): each claims one or more
//! schemes and opens a holder of its scheme into a transport. A
//! [`Registry`] holds them by scheme; [`Registry::builtin`] has `file`,
//! `grpc`, `grpcs`, `s3` and `gdrive`, and an application adds its own
//! with [`Registry::register`]. `TURBO_ZOO_SOURCE` and a user's
//! [`Holder`] list both open through it.

use std::sync::Arc;

use turbo_zoo::{Blobs, Holder, Keep, SignIn, Sources, Transport, UserId};

use crate::server::ZooServer;

/// A store that serves files and can keep those a later source serves.
pub trait Store: Transport + Keep + Send + Sync {}

impl<T: Transport + Keep + Send + Sync> Store for T {}

/// What a holder opens into.
pub enum Opened {
    /// A source, tried in its place in the list.
    Source(Box<dyn Transport + Send + Sync>),
    /// A store, tried in its place; the first one in a list also keeps
    /// what the sources after it serve.
    Store(Arc<dyn Store>),
}

/// A kind of source.
pub trait Kind: Send + Sync {
    /// The schemes it opens, as they come before `://`.
    fn schemes(&self) -> &[&str];

    /// The source `holder` names, read as `user`.
    fn open(&self, holder: &Holder, user: &UserId) -> Result<Opened, String>;
}

/// Kinds by scheme, in the order they were registered.
#[derive(Clone, Default)]
pub struct Registry {
    kinds: Vec<(String, Arc<dyn Kind>)>,
}

impl Registry {
    /// No kind at all.
    pub fn empty() -> Registry {
        Registry::default()
    }

    /// `file`, `grpc`, `grpcs`, `s3` and `gdrive`. A scheme whose feature
    /// this build leaves out is still known, and refused when opened.
    pub fn builtin() -> Registry {
        let mut r = Registry::empty();
        r.register(FileKind);
        r.register(ZooServerKind);
        r.register(S3Kind);
        r.register(DriveKind::default());
        r
    }

    /// Adds `kind`, which takes over any scheme a kind before it claimed.
    pub fn register(&mut self, kind: impl Kind + 'static) -> &mut Registry {
        let kind: Arc<dyn Kind> = Arc::new(kind);
        for scheme in kind.schemes() {
            match self.kinds.iter_mut().find(|(s, _)| s == scheme) {
                Some(slot) => slot.1 = kind.clone(),
                None => self.kinds.push(((*scheme).to_owned(), kind.clone())),
            }
        }
        self
    }

    /// The schemes it opens, in the order they were registered.
    pub fn schemes(&self) -> impl Iterator<Item = &str> {
        self.kinds.iter().map(|(s, _)| s.as_str())
    }

    /// The source `holder` names, through the kind of its scheme.
    pub fn open(&self, holder: &Holder, user: &UserId) -> Result<Opened, String> {
        let scheme = holder.scheme();
        match self.kinds.iter().find(|(s, _)| !scheme.is_empty() && s == scheme) {
            Some((_, kind)) => kind.open(holder, user),
            None if self.kinds.is_empty() => Err(format!("{}: no kind of source is registered", holder.address)),
            None => {
                let known: Vec<String> = self.schemes().map(|s| format!("{s}://")).collect();
                Err(format!("{}: a source is {} or origin", holder.address, known.join(", ")))
            }
        }
    }

    /// `holders` opened in order, read as `user`, then `origin` (the
    /// index's own addresses) when given. The first store keeps what the
    /// sources after it serve.
    pub fn sources<T>(&self, holders: &[Holder], user: &UserId, origin: Option<T>) -> Result<Sources, String>
    where
        T: Transport + Send + Sync + 'static,
    {
        let mut list = List::default();
        for h in holders {
            list.push(&h.address, self.open(h, user).map_err(|e| format!("{}: {e}", h.name))?);
        }
        let mut out = list.out;
        if let Some(t) = origin {
            out = out.then("origin", t);
        }
        Ok(out)
    }
}

/// A [`Sources`] list being built, and whether a store keeps yet.
#[derive(Default)]
pub(crate) struct List {
    pub(crate) out: Sources,
    kept: bool,
}

impl List {
    pub(crate) fn push(&mut self, name: &str, opened: Opened) {
        let out = std::mem::take(&mut self.out);
        self.out = match opened {
            Opened::Source(s) => out.then(name, s),
            Opened::Store(s) if !self.kept => {
                self.kept = true;
                out.then_keep(name, s)
            }
            Opened::Store(s) => out.then(name, s),
        };
    }
}

/// Refuses a sign-in a kind does not take.
fn environment_only(holder: &Holder) -> Result<(), String> {
    if holder.sign_in.is_environment() {
        Ok(())
    } else {
        Err(format!("{}: this kind of source signs in from the environment only", holder.address))
    }
}

/// `file:///dir`: the content-addressed store under `/dir`; on Windows a
/// native absolute path (drive-qualified or UNC) after `file://` too.
pub struct FileKind;

impl Kind for FileKind {
    fn schemes(&self) -> &[&str] {
        &["file"]
    }

    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        environment_only(holder)?;
        let dir = holder.address.strip_prefix("file://").unwrap_or_default();
        if !dir.starts_with('/') && !std::path::Path::new(dir).is_absolute() {
            return Err(format!("{}: a file:// source names an absolute directory", holder.address));
        }
        Ok(Opened::Store(Arc::new(Blobs::new(dir))))
    }
}

/// `grpc://host:port`, `grpcs://host:port`: a zoo server.
pub struct ZooServerKind;

impl Kind for ZooServerKind {
    fn schemes(&self) -> &[&str] {
        &["grpc", "grpcs"]
    }

    fn open(&self, holder: &Holder, user: &UserId) -> Result<Opened, String> {
        environment_only(holder)?;
        Ok(Opened::Source(Box::new(ZooServer::new(&holder.address, user.clone())?)))
    }
}

/// `s3://bucket/prefix`, configured from the `AWS_*` environment.
pub struct S3Kind;

impl Kind for S3Kind {
    fn schemes(&self) -> &[&str] {
        &["s3"]
    }

    #[cfg(feature = "s3")]
    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        environment_only(holder)?;
        Ok(Opened::Source(Box::new(crate::S3Bucket::new(&holder.address)?)))
    }

    #[cfg(not(feature = "s3"))]
    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        Err(format!("{}: s3:// needs the s3 feature, which this build leaves out", holder.address))
    }
}

/// `gdrive://<folder-id>[/name/...]`: a Google Drive folder, signed in
/// from the environment, an API key, or a stored Google grant.
#[derive(Clone, Default)]
pub struct DriveKind {
    drive: Option<String>,
    oauth: Option<(String, String)>,
}

impl DriveKind {
    /// Against other endpoints than Google's: the Drive API at `drive`,
    /// and the consent page and token endpoint at `auth` and `token`.
    pub fn with_endpoints(drive: &str, auth: &str, token: &str) -> DriveKind {
        DriveKind { drive: Some(drive.to_owned()), oauth: Some((auth.to_owned(), token.to_owned())) }
    }
}

impl Kind for DriveKind {
    fn schemes(&self) -> &[&str] {
        &["gdrive"]
    }

    #[cfg(feature = "gdrive")]
    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        use turbo_zoo::{DriveAuth, GoogleDrive, GoogleOAuth, Renewing};
        let address = holder.address.as_str();
        GoogleDrive::check_address(address)?;
        let auth = match &holder.sign_in {
            SignIn::Environment => DriveAuth::from_vars(address, |k| std::env::var(k).ok())?,
            SignIn::ApiKey { key } => DriveAuth::ApiKey(key.clone()),
            SignIn::GoogleOauth { client_id, client_secret, refresh_token } => {
                let oauth = match &self.oauth {
                    Some((a, t)) => GoogleOAuth::with_endpoints(client_id, client_secret.as_deref(), a, t)?,
                    None => GoogleOAuth::new(client_id, client_secret.as_deref())?,
                };
                DriveAuth::Refresh(Arc::new(Renewing::new(oauth, refresh_token)?))
            }
        };
        Ok(Opened::Source(Box::new(GoogleDrive::with(address, self.drive.as_deref(), auth)?)))
    }

    #[cfg(not(feature = "gdrive"))]
    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        let _ = (&self.drive, &self.oauth, SignIn::Environment);
        Err(format!("{}: gdrive:// needs the gdrive feature, which this build leaves out", holder.address))
    }
}
