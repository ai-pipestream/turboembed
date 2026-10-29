//! DescribeModel, ResolveModel and ReadModelFile: one artifact's entry, the
//! artifact and file the index pins at a SHA-256, and an installed file's
//! bytes streamed to a user the licence check admits. A read is not an
//! admin call. The bytes are hashed as they are sent, and a file whose
//! size or SHA-256 no longer matches the index ends the stream with
//! DATA_LOSS instead of a clean end; the client verifies every digest too.

use std::fs;
use std::io::Read;
use std::pin::Pin;

use sha2::Digest as _;
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use turbo_zoo::{Offline, Policy, Sha256, UserId, Zoo, ZooError};

use super::{Catalog, entry, invalid, plain, zoo_status};
use crate::api::{Failure, TURBO_E_BUNDLE_INTEGRITY, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_STATE, TURBO_E_RUNTIME};
use crate::proto::training::model_file_chunk::Frame;
use crate::proto::training::{CatalogEntry, ModelFileChunk};

pub(crate) type ChunkStream = Pin<Box<dyn Stream<Item = Result<ModelFileChunk, Status>> + Send>>;

/// The most bytes one data frame carries.
pub const READ_CHUNK_BYTES: usize = 64 << 10;

fn not_found(m: String) -> Status {
    plain(Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 0, m))
}

fn sha256(field: u32, s: &str) -> Result<Sha256, Status> {
    s.parse().map_err(|e: String| invalid(field, format!("sha256: {e}")))
}

/// The artifact `id`'s entry; with a user, its missing terms are the ones
/// that user has not accepted.
pub(super) fn describe(catalog: &Catalog, user: Option<&UserId>, id: &str) -> Result<CatalogEntry, Status> {
    if id.is_empty() {
        return Err(invalid(2, "artifact_id is required"));
    }
    let a = catalog.index.get(id).map_err(zoo_status)?;
    let acceptances = match user {
        Some(_) => Some(catalog.acceptances().map_err(zoo_status)?),
        None => None,
    };
    Ok(entry::entry(a, user.zip(acceptances.as_ref()), &catalog.cache()))
}

/// The first artifact, in index order, with a file pinned at `sha256`, and
/// that file's path; only artifact `id`'s files when `id` is not empty.
pub(super) fn resolve(
    catalog: &Catalog,
    user: Option<&UserId>,
    sha: &str,
    id: &str,
) -> Result<(CatalogEntry, String), Status> {
    let want = sha256(1, sha)?;
    let candidates = match id {
        "" => catalog.index.list(),
        id => std::slice::from_ref(catalog.index.get(id).map_err(zoo_status)?),
    };
    let (a, f) = candidates
        .iter()
        .find_map(|a| a.files.iter().find(|f| f.sha256 == want).map(|f| (a, f)))
        .ok_or_else(|| not_found(format!("the index pins no file at SHA-256 {want}")))?;
    let path = f.path.as_str().to_owned();
    Ok((describe(catalog, user, a.id.as_str())?, path))
}

/// Admits the read, then streams the file: the catalog file first, then
/// the bytes. Every refusal before the first frame is the call's own
/// status.
pub(super) async fn start(
    catalog: Catalog,
    user: UserId,
    id: String,
    path: String,
    pin: String,
) -> Result<ChunkStream, Status> {
    if id.is_empty() {
        return Err(invalid(2, "artifact_id is required"));
    }
    if path.is_empty() {
        return Err(invalid(3, "path is required"));
    }
    let pin = if pin.is_empty() { None } else { Some(sha256(4, &pin)?) };
    let opened = super::blocking(move || open(&catalog, &user, &id, &path, pin)).await?;
    let (tx, rx) = mpsc::channel(4);
    tokio::task::spawn_blocking(move || send(opened, tx));
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// A file admitted for reading: its index record and its open handle.
struct Opened {
    id: String,
    file: turbo_zoo::File,
    handle: fs::File,
}

/// The licence check, then the file: listed in the artifact, pinned at
/// the caller's SHA-256 when one is given, and a plain file in the cache
/// at the size the index records.
fn open(catalog: &Catalog, user: &UserId, id: &str, path: &str, pin: Option<Sha256>) -> Result<Opened, Status> {
    let zoo = Zoo::new((*catalog.index).clone(), catalog.cache(), Policy::default(), &Offline);
    let a = zoo.permit_with(id, user, || catalog.acceptances()).map_err(zoo_status)?;
    let f = a
        .files
        .iter()
        .find(|f| f.path.as_str() == path)
        .ok_or_else(|| not_found(format!("{id} has no file {path}")))?;
    if let Some(pin) = pin
        && pin != f.sha256
    {
        return Err(plain(Failure::new(
            TURBO_E_INVALID_STATE,
            4,
            format!("{id}/{path}: the server's index pins SHA-256 {}, the request {pin}", f.sha256),
        )));
    }
    let local = catalog.cache().join(a.id.as_str()).join(f.path.as_str());
    let installed = fs::symlink_metadata(&local).is_ok_and(|m| m.is_file() && m.len() == f.size);
    if !installed {
        return Err(plain(Failure::new(
            TURBO_E_INVALID_STATE,
            0,
            format!("{id}/{path} is not installed on this server; install the artifact first"),
        )));
    }
    let handle =
        fs::File::open(&local).map_err(|e| zoo_status(ZooError::Io { path: local.clone(), message: e.to_string() }))?;
    Ok(Opened { id: id.to_owned(), file: f.clone(), handle })
}

/// Sends the catalog file, then the bytes, hashing them on the way; ends
/// with DATA_LOSS when they are not the size and SHA-256 the index
/// records. Stops when the client goes away.
fn send(o: Opened, tx: mpsc::Sender<Result<ModelFileChunk, Status>>) {
    let frame = |f| Ok(ModelFileChunk { frame: Some(f) });
    if tx.blocking_send(frame(Frame::File(entry::file(&o.file)))).is_err() {
        return;
    }
    let what = format!("{}/{}", o.id, o.file.path.as_str());
    let mut hash = sha2::Sha256::new();
    let mut sent = 0u64;
    // One byte past the recorded size is enough to know the file grew.
    let mut reader = o.handle.take(o.file.size.saturating_add(1));
    let mut buf = vec![0u8; READ_CHUNK_BYTES];
    loop {
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => {
                let _ = tx.blocking_send(Err(plain(Failure::new(TURBO_E_RUNTIME, 0, format!("{what}: {e}")))));
                return;
            }
        };
        sent += n as u64;
        if sent > o.file.size {
            break;
        }
        hash.update(&buf[..n]);
        if tx.blocking_send(frame(Frame::Data(buf[..n].to_vec()))).is_err() {
            return;
        }
    }
    let got: Sha256 = turbo_zoo::digest::Hex(hash.finalize().into());
    let fail = if sent > o.file.size {
        Some(format!("{what}: the cache holds more than {} bytes, the index {0}", o.file.size))
    } else if sent < o.file.size {
        Some(format!("{what}: the cache holds {sent} bytes, the index {}", o.file.size))
    } else if got != o.file.sha256 {
        Some(format!("{what}: the cached bytes have SHA-256 {got}, the index {}", o.file.sha256))
    } else {
        None
    };
    if let Some(m) = fail {
        let _ = tx.blocking_send(Err(plain(Failure::new(TURBO_E_BUNDLE_INTEGRITY, 0, m))));
    }
}
