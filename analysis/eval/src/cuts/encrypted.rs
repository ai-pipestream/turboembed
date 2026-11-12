//! Opt-in `EncryptedStore` wiring for CLI and pipeline cut exports (#2018).

use std::collections::BTreeMap;
use std::fs::{self, DirBuilder};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo_analysis_embed::Precision;

use super::{Dump, DumpConfig, Header, SetCuts, Writer, compare_dumps};
use crate::Result;
use crate::artifact::encrypted::{
    Audit, Authorization, EncodingId, EncryptedStore, Error, KeyProvider, Outcome, PrincipalId, Request, Retention,
    SecurityContext, new_artifact_key,
};
use crate::artifact::{EvaluationArtifactStore, TemporaryFileStore};

/// Default CLI caller principal scope (`747572626f2d6576616c2d6375747300`).
pub const CUTS_CLI_PRINCIPAL: PrincipalId = PrincipalId(*b"turbo-eval-cuts\0");

/// Authenticated encoding identifier for version-2 cut dumps (`turbo-eval-cuts2`).
pub const CUTS_ENCODING: EncodingId = EncodingId(*b"turbo-eval-cuts2");

/// Maximum ciphertext bytes for a 1 GiB version-2 cut dump (56-byte header,
/// 21 bytes of framing/tag per 64 KiB plaintext frame, 37-byte terminal frame).
pub const MAX_CUTS_CIPHERTEXT_BYTES: u64 = (1024 * 1024 * 1024) + 56 + (16_384 * 21) + 37;

struct LocalCutsPolicy {
    principal: PrincipalId,
}

impl Authorization for LocalCutsPolicy {
    fn authorize(&self, request: &Request) -> std::result::Result<(), Error> {
        if request.principal == self.principal { Ok(()) } else { Err(Error::Denied) }
    }
}

impl Retention for LocalCutsPolicy {
    fn allow_delete(&self, _: &Request) -> std::result::Result<(), Error> {
        Err(Error::Retained)
    }
}

impl Audit for LocalCutsPolicy {
    fn intent(&self, request: &Request) -> std::result::Result<(), Error> {
        if request.principal == self.principal { Ok(()) } else { Err(Error::Denied) }
    }

    fn outcome(&self, _: &Request, _: Outcome) {}
}

fn ensure_private_dir(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    let mut builder = DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(dir).map_err(|e| format!("{}: {e}", dir.display()))
}

fn build_store(dir: PathBuf, principal: PrincipalId, keys: Arc<dyn KeyProvider>) -> EncryptedStore<TemporaryFileStore> {
    let policy = Arc::new(LocalCutsPolicy { principal });
    let context = SecurityContext {
        principal,
        encoding: CUTS_ENCODING,
        keys,
        authorization: policy.clone(),
        retention: policy.clone(),
        audit: policy,
    };
    EncryptedStore::new(TemporaryFileStore::new(&dir), context, dir, MAX_CUTS_CIPHERTEXT_BYTES)
}

/// Opens or creates a local [`EncryptedStore`] backed by [`TemporaryFileStore`]
/// under `dir`, scoped to [`CUTS_CLI_PRINCIPAL`] and [`CUTS_ENCODING`].
pub fn encrypted_store(
    dir: impl Into<PathBuf>,
    keys: Arc<dyn KeyProvider>,
) -> Result<EncryptedStore<TemporaryFileStore>> {
    encrypted_store_with_principal(dir, CUTS_CLI_PRINCIPAL, keys)
}

/// Opens or creates a local [`EncryptedStore`] backed by [`TemporaryFileStore`]
/// under `dir`, scoped to `principal` and [`CUTS_ENCODING`].
pub fn encrypted_store_with_principal(
    dir: impl Into<PathBuf>,
    principal: PrincipalId,
    keys: Arc<dyn KeyProvider>,
) -> Result<EncryptedStore<TemporaryFileStore>> {
    let dir = dir.into();
    ensure_private_dir(&dir)?;
    Ok(build_store(dir, principal, keys))
}

fn open_existing_store(
    dir: &Path,
    principal: PrincipalId,
    keys: Arc<dyn KeyProvider>,
) -> Result<EncryptedStore<TemporaryFileStore>> {
    let meta = fs::metadata(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if !meta.is_dir() {
        return Err(format!("{}: not a directory", dir.display()));
    }
    Ok(build_store(dir.to_path_buf(), principal, keys))
}

impl DumpConfig {
    /// Plaintext local cut exports in `dir` (`<dir>/<set>.cuts`).
    pub fn plain(dir: impl Into<PathBuf>, precision: Precision) -> Self {
        Self { store: Box::new(TemporaryFileStore::new(dir)), precision, opaque_keys: false }
    }

    /// Encrypted local cut exports in `dir` through [`EncryptedStore`] with
    /// [`CUTS_CLI_PRINCIPAL`] and the caller-supplied `keys`.
    pub fn encrypted(dir: impl Into<PathBuf>, keys: Arc<dyn KeyProvider>, precision: Precision) -> Result<Self> {
        Self::encrypted_with_principal(dir, CUTS_CLI_PRINCIPAL, keys, precision)
    }

    /// Encrypted local cut exports in `dir` through [`EncryptedStore`] with
    /// `principal` and the caller-supplied `keys`. Checks that an active key
    /// is available before returning.
    pub fn encrypted_with_principal(
        dir: impl Into<PathBuf>,
        principal: PrincipalId,
        keys: Arc<dyn KeyProvider>,
        precision: Precision,
    ) -> Result<Self> {
        keys.active().map_err(|e| e.to_string())?;
        let store = encrypted_store_with_principal(dir, principal, keys)?;
        Ok(Self::with_encrypted_store(store, precision))
    }

    /// Encrypted cut exports through an existing [`EncryptedStore`], minting a
    /// fresh 32-hex [`crate::artifact::ArtifactKey`] per set via [`new_artifact_key`].
    pub fn with_encrypted_store<S: EvaluationArtifactStore + 'static>(
        store: EncryptedStore<S>,
        precision: Precision,
    ) -> Self {
        Self { store: Box::new(store), precision, opaque_keys: true }
    }

    /// Opens a cut-dump writer for `header` in this configuration's store.
    pub fn writer(&self, header: &Header) -> Result<Writer> {
        if self.opaque_keys {
            let key = new_artifact_key().map_err(|e| e.to_string())?;
            Writer::create_at(self.store.as_ref(), &key, header)
        } else {
            Writer::create_in(self.store.as_ref(), header)
        }
    }
}

/// Compares two directories of encrypted cut dumps using [`CUTS_CLI_PRINCIPAL`]
/// and the caller-supplied `keys`.
pub fn compare_encrypted_dirs(a: &Path, b: &Path, keys: Arc<dyn KeyProvider>) -> Result<Vec<SetCuts>> {
    compare_encrypted_dirs_with_principal(a, b, CUTS_CLI_PRINCIPAL, keys)
}

/// Compares two directories of encrypted cut dumps using `principal` and the
/// caller-supplied `keys`.
pub fn compare_encrypted_dirs_with_principal(
    a: &Path,
    b: &Path,
    principal: PrincipalId,
    keys: Arc<dyn KeyProvider>,
) -> Result<Vec<SetCuts>> {
    let store_a = open_existing_store(a, principal, keys.clone())?;
    let store_b = open_existing_store(b, principal, keys)?;
    compare_encrypted_stores(&store_a, &store_b)
}

/// Compares two opaque-keyed stores of encrypted cut dumps, matching dumps by
/// their authenticated header set name.
pub fn compare_encrypted_stores(
    a: &dyn EvaluationArtifactStore,
    b: &dyn EvaluationArtifactStore,
) -> Result<Vec<SetCuts>> {
    let read_all = |store: &dyn EvaluationArtifactStore| -> Result<BTreeMap<String, Dump>> {
        let mut keys = store.list().map_err(|e| e.to_string())?;
        keys.sort();
        let mut by_set = BTreeMap::new();
        for key in &keys {
            let dump = Dump::read_opaque(store, key)?;
            let set = dump.header.set.clone();
            if by_set.insert(set.clone(), dump).is_some() {
                return Err(format!("duplicate encrypted cut dump for set {set}"));
            }
        }
        Ok(by_set)
    };
    let (da, db) = (read_all(a)?, read_all(b)?);
    let sa: Vec<&String> = da.keys().collect();
    let sb: Vec<&String> = db.keys().collect();
    if sa != sb {
        return Err(format!("stores holds sets {sa:?} against {sb:?}"));
    }
    if da.is_empty() {
        return Err("no encrypted cut dumps".into());
    }
    da.keys().map(|set| compare_dumps(&da[set], &db[set])).collect()
}
