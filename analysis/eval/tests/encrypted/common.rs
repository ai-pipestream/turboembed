use std::{
    fs,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};
use turbo_eval::artifact::encrypted::*;
use turbo_eval::artifact::{ArtifactKey, TemporaryFileStore};

static NEXT: AtomicU64 = AtomicU64::new(0);
pub struct Fixture {
    pub root: PathBuf,
    pub policy: Arc<Policy>,
    pub keys: Arc<Keys>,
}
impl Fixture {
    pub fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "encrypted-artifact-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("objects")).unwrap();
        fs::create_dir(root.join("scratch")).unwrap();
        Self { root, policy: Arc::new(Policy::default()), keys: Arc::new(Keys::default()) }
    }
    pub fn context(&self, principal: u8) -> SecurityContext {
        SecurityContext {
            principal: PrincipalId([principal; 16]),
            encoding: EncodingId([3; 16]),
            keys: self.keys.clone(),
            authorization: self.policy.clone(),
            retention: self.policy.clone(),
            audit: self.policy.clone(),
        }
    }
    pub fn store(&self, principal: u8) -> EncryptedStore<TemporaryFileStore> {
        EncryptedStore::new(
            TemporaryFileStore::new(self.root.join("objects")),
            self.context(principal),
            self.root.join("scratch"),
            256 * 1024 * 1024,
        )
    }
    pub fn physical(&self, principal: u8, key: &ArtifactKey) -> PathBuf {
        self.root.join("objects").join(format!("{}.{}", format!("{principal:02x}").repeat(16), key.as_str()))
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.root).unwrap();
    }
}
#[derive(Default)]
pub struct Policy {
    pub denied: AtomicBool,
    pub held: AtomicBool,
    pub audit_down: AtomicBool,
    pub events: Mutex<Vec<(Operation, Outcome)>>,
}
impl Authorization for Policy {
    fn authorize(&self, request: &Request) -> Result<(), Error> {
        if self.denied.load(Ordering::SeqCst)
            || ![PrincipalId([1; 16]), PrincipalId([2; 16])].contains(&request.principal)
        {
            Err(Error::Denied)
        } else {
            Ok(())
        }
    }
}
impl Retention for Policy {
    fn allow_delete(&self, _: &Request) -> Result<(), Error> {
        if self.held.load(Ordering::SeqCst) { Err(Error::Retained) } else { Ok(()) }
    }
}
impl Audit for Policy {
    fn intent(&self, _: &Request) -> Result<(), Error> {
        if self.audit_down.load(Ordering::SeqCst) { Err(Error::AuditUnavailable) } else { Ok(()) }
    }
    fn outcome(&self, request: &Request, outcome: Outcome) {
        self.events.lock().unwrap().push((request.operation, outcome));
    }
}
#[derive(Default)]
pub struct Keys {
    pub rotated: AtomicBool,
    pub revoked: AtomicBool,
}
impl KeyProvider for Keys {
    fn active(&self) -> Result<(KeyId, KeyMaterial), Error> {
        let id = KeyId([if self.rotated.load(Ordering::SeqCst) { 2 } else { 1 }; 16]);
        Ok((id, self.resolve(id)?))
    }
    fn resolve(&self, id: KeyId) -> Result<KeyMaterial, Error> {
        if self.revoked.load(Ordering::SeqCst) {
            return Err(Error::RevokedKey);
        }
        match id.0[0] {
            1 if id == KeyId([1; 16]) => Ok(KeyMaterial::new([71; 32])),
            2 if id == KeyId([2; 16]) => Ok(KeyMaterial::new([72; 32])),
            _ => Err(Error::UnavailableKey),
        }
    }
}
pub fn key(n: u8) -> ArtifactKey {
    ArtifactKey::new(format!("{n:02x}").repeat(16)).unwrap()
}
