use std::{
    io::{self, Read, Write},
    path::PathBuf,
    sync::Arc,
};

use super::{
    Audit, Authorization, Binding, Encoder, EncodingId, Error, KeyProvider, Operation, Outcome, PrincipalId, Request,
    Retention, VerifiedReader,
};
use crate::artifact::{ArtifactKey, EvaluationArtifactStore, PendingArtifact, Publication, StorageError};

/// Capabilities supplied by the authenticated application, not request payloads.
#[derive(Clone)]
pub struct SecurityContext {
    pub principal: PrincipalId,
    pub encoding: EncodingId,
    pub keys: Arc<dyn KeyProvider>,
    pub authorization: Arc<dyn Authorization>,
    pub retention: Arc<dyn Retention>,
    pub audit: Arc<dyn Audit>,
}

/// Wraps a provider with encryption and an immutable authenticated caller scope.
/// Logical keys must be random 128-bit identifiers encoded as lower-case hex.
/// The backing provider and scratch root must be inaccessible to untrusted code.
pub struct EncryptedStore<S> {
    provider: S,
    context: SecurityContext,
    temporary_root: PathBuf,
    max_ciphertext_bytes: u64,
}

impl<S: EvaluationArtifactStore> EncryptedStore<S> {
    pub fn new(provider: S, context: SecurityContext, temporary_root: PathBuf, max_ciphertext_bytes: u64) -> Self {
        Self { provider, context, temporary_root, max_ciphertext_bytes }
    }

    fn request(&self, key: Option<&ArtifactKey>, operation: Operation) -> Result<Request, StorageError> {
        Ok(Request { principal: self.context.principal, artifact: key.map(parse).transpose()?, operation })
    }

    fn physical(&self, key: &ArtifactKey) -> Result<ArtifactKey, StorageError> {
        parse(key)?;
        ArtifactKey::new(format!("{}.{}", hex(&self.context.principal.0), key.as_str()))
    }

    fn binding(&self, key: &ArtifactKey) -> Binding {
        let mut hash = turbo_sha256::Sha256::new();
        hash.update(b"turboembed evaluation artifact v1");
        hash.update(&self.context.principal.0);
        hash.update(&self.context.encoding.0);
        hash.update(key.as_str().as_bytes());
        Binding(hash.finish().0)
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn parse(key: &ArtifactKey) -> Result<[u8; 16], StorageError> {
    let bytes = key.as_str().as_bytes();
    if bytes.len() != 32 || !bytes.iter().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b)) {
        return Err(StorageError::InvalidKey);
    }
    let mut id = [0; 16];
    for (i, pair) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let digit = |b: u8| if b.is_ascii_digit() { b - b'0' } else { b - b'a' + 10 };
        id[i] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(id)
}

fn check(context: &SecurityContext, request: &Request) -> Result<(), StorageError> {
    context.audit.intent(request).map_err(StorageError::Security)?;
    if let Err(error) = context.authorization.authorize(request) {
        context.audit.outcome(request, Outcome::Failed);
        return Err(StorageError::Security(error));
    }
    Ok(())
}

fn result<T>(context: &SecurityContext, request: &Request, value: Result<T, StorageError>) -> Result<T, StorageError> {
    let outcome = match &value {
        Ok(_) => Outcome::Succeeded,
        Err(StorageError::Published { .. }) => Outcome::PublishedWithWarning,
        Err(_) => Outcome::Failed,
    };
    context.audit.outcome(request, outcome);
    value
}

// Never propagate a provider's error message, URL, path or credential material.
fn sanitize(error: StorageError, key: Option<&ArtifactKey>) -> StorageError {
    match (error, key) {
        (StorageError::AlreadyExists(_), Some(k)) => StorageError::AlreadyExists(k.clone()),
        (StorageError::NotFound(_), Some(k)) => StorageError::NotFound(k.clone()),
        (StorageError::Published { .. }, Some(k)) => {
            StorageError::Published { key: k.clone(), source: io::Error::other("encrypted artifact durability failed") }
        }
        _ => StorageError::Security(Error::Io),
    }
}

impl<S: EvaluationArtifactStore> EvaluationArtifactStore for EncryptedStore<S> {
    fn begin(&self, key: &ArtifactKey) -> Result<Box<dyn PendingArtifact>, StorageError> {
        if self.max_ciphertext_bytes < 93 {
            return Err(StorageError::Security(Error::InvalidArtifact));
        }
        let request = self.request(Some(key), Operation::Begin)?;
        check(&self.context, &request)?;
        let value = (|| {
            let pending = self.provider.begin(&self.physical(key)?).map_err(|e| sanitize(e, Some(key)))?;
            let encoder =
                Encoder::new(pending, self.context.keys.as_ref(), self.binding(key)).map_err(StorageError::Security)?;
            Ok(Box::new(Pending {
                encoder: Some(encoder),
                context: self.context.clone(),
                request,
                key: key.clone(),
                limit: self.max_ciphertext_bytes,
            }) as Box<dyn PendingArtifact>)
        })();
        let request = self.request(Some(key), Operation::Begin)?;
        result(&self.context, &request, value)
    }

    fn open(&self, key: &ArtifactKey) -> Result<Box<dyn Read>, StorageError> {
        let request = self.request(Some(key), Operation::Read)?;
        check(&self.context, &request)?;
        let value = (|| {
            let source = self.provider.open(&self.physical(key)?).map_err(|e| sanitize(e, Some(key)))?;
            let reader = VerifiedReader::open(
                source,
                self.context.keys.as_ref(),
                self.binding(key),
                &self.temporary_root,
                self.max_ciphertext_bytes,
            )
            .map_err(StorageError::Security)?;
            Ok(Box::new(reader) as Box<dyn Read>)
        })();
        result(&self.context, &request, value)
    }

    fn list(&self) -> Result<Vec<ArtifactKey>, StorageError> {
        let request = self.request(None, Operation::List)?;
        check(&self.context, &request)?;
        let value = (|| {
            let prefix = format!("{}.", hex(&self.context.principal.0));
            let mut keys = Vec::new();
            for physical in self.provider.list().map_err(|e| sanitize(e, None))? {
                let Some(name) = physical.as_str().strip_prefix(&prefix) else { continue };
                let key = ArtifactKey::new(name)?;
                if parse(&key).is_ok() {
                    keys.push(key);
                }
            }
            Ok(keys)
        })();
        result(&self.context, &request, value)
    }

    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError> {
        let request = self.request(Some(key), Operation::Delete)?;
        check(&self.context, &request)?;
        let value = (|| {
            self.context.retention.allow_delete(&request).map_err(StorageError::Security)?;
            self.provider.delete(&self.physical(key)?).map_err(|e| sanitize(e, Some(key)))
        })();
        result(&self.context, &request, value)
    }
}

struct Pending {
    encoder: Option<Encoder<Box<dyn PendingArtifact>>>,
    context: SecurityContext,
    request: Request,
    key: ArtifactKey,
    limit: u64,
}

impl Write for Pending {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        // Header + terminal + bounded frame overhead. Reserve for all bytes,
        // before accepting any part of this call. Overflow permanently poisons.
        let encoder = self.encoder.as_mut().ok_or_else(|| io::Error::other(Error::Poisoned))?;
        let fits = encoder.projected_size(bytes.len()).is_some_and(|n| n <= self.limit);
        if !fits {
            self.encoder.take();
            return Err(io::Error::other(Error::InvalidArtifact));
        }
        self.encoder.as_mut().ok_or_else(|| io::Error::other(Error::Poisoned))?.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.encoder.as_mut().ok_or_else(|| io::Error::other(Error::Poisoned))?.flush()
    }
}

impl PendingArtifact for Pending {
    fn finalize(mut self: Box<Self>) -> Result<Publication, StorageError> {
        self.request.operation = Operation::Publish;
        check(&self.context, &self.request)?;
        let value = (|| {
            let encoder = self.encoder.take().ok_or(StorageError::Security(Error::Poisoned))?;
            self.context.keys.resolve(encoder.key_id()).map_err(StorageError::Security)?;
            let pending = encoder.finish().map_err(StorageError::Security)?;
            let publication = pending.finalize().map_err(|e| sanitize(e, Some(&self.key)))?;
            Ok(Publication {
                key: self.key.clone(),
                cleanup_error: publication
                    .cleanup_error
                    .map(|_| io::Error::other("encrypted artifact staging cleanup failed")),
            })
        })();
        if value.as_ref().is_ok_and(|p| p.cleanup_error.is_some()) {
            self.context.audit.outcome(&self.request, Outcome::PublishedWithWarning);
            value
        } else {
            result(&self.context, &self.request, value)
        }
    }
    fn abort(mut self: Box<Self>) -> Result<(), StorageError> {
        self.request.operation = Operation::Abort;
        // Cleanup must remain possible even after access is revoked.
        let audit = self.context.audit.intent(&self.request);
        let value = self
            .encoder
            .take()
            .map(|encoder| encoder.abort().abort().map_err(|e| sanitize(e, Some(&self.key))))
            .unwrap_or(Ok(()));
        let value = value.and(audit.map_err(StorageError::Security));
        result(&self.context, &self.request, value)
    }
}
