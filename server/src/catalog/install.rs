//! InstallModel: one install at a time, admitted after the licence check
//! and a free disk check with a margin, then the zoo's fetch with each
//! file's request and arrival streamed as progress. The zoo verifies every
//! file before it writes it, and writes each under its name only whole
//! (docs/zoo.md), so an install that fails or is abandoned leaves no file
//! that does not verify.

use std::fs;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Status;
use turbo_zoo::{Artifact, Offline, Policy, Transport, Url, UserId, Zoo, ZooError, verify};

use super::{Catalog, blocking, disk, invalid, plain, zoo_status};
use crate::api::{Failure, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_RUNTIME};
use crate::proto::training::install_model_update::Update;
use crate::proto::training::{InstallModelUpdate, InstallProgress, InstallStage, InstalledModel};

pub(crate) type UpdateStream = Pin<Box<dyn Stream<Item = Result<InstallModelUpdate, Status>> + Send>>;

/// The free space an install leaves beyond the files it writes.
pub(crate) const DISK_MARGIN_BYTES: u64 = 64 << 20;

/// The install slot, given back when dropped.
struct Slot(Arc<AtomicBool>);

impl Drop for Slot {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

type Tx = mpsc::Sender<Result<InstallModelUpdate, Status>>;

fn progress(stage: InstallStage, file_index: u32, file_count: u32, path: &str, bytes: u64) -> InstallModelUpdate {
    InstallModelUpdate {
        update: Some(Update::Progress(InstallProgress {
            stage: stage as i32,
            file_index,
            file_count,
            path: path.to_owned(),
            bytes,
        })),
    }
}

/// Takes the slot, admits the install, and streams the fetch. Every
/// refusal before the first message is the call's own status.
pub(super) async fn start(
    catalog: Catalog,
    user: UserId,
    id: String,
    installing: Arc<AtomicBool>,
) -> Result<UpdateStream, Status> {
    if id.is_empty() {
        return Err(invalid(2, "artifact_id is required"));
    }
    if installing.swap(true, Ordering::AcqRel) {
        return Err(plain(Failure::new(TURBO_E_BUSY, 0, "another install is running; try again when it ends")));
    }
    let slot = Slot(installing);
    let (c, u, i) = (catalog.clone(), user.clone(), id.clone());
    let (needed, count) = blocking(move || admit(&c, &u, &i)).await?;
    let (tx, rx) = mpsc::channel(16);
    let _ = tx.send(Ok(progress(InstallStage::Admitted, 0, count, "", needed))).await;
    tokio::task::spawn_blocking(move || run(catalog, user, id, tx, slot));
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Whether the fetch will keep the cached copy of `f`: a plain file of its
/// size whose bytes verify, the test the zoo's fetch makes. Any other file
/// is fetched again and written whole, so its bytes count.
fn cached(cache: &std::path::Path, a: &Artifact, f: &turbo_zoo::File) -> bool {
    let p = cache.join(a.id.as_str()).join(f.path.as_str());
    match fs::symlink_metadata(&p) {
        Ok(m) if m.is_file() && m.len() == f.size => fs::read(&p).is_ok_and(|b| verify(a, f, &b).is_ok()),
        _ => false,
    }
}

/// The licence check, then the bytes the files still to fetch take, which
/// with the margin must fit in the cache's free space. Returns those bytes
/// and the artifact's file count.
fn admit(catalog: &Catalog, user: &UserId, id: &str) -> Result<(u64, u32), Status> {
    let zoo = Zoo::new((*catalog.index).clone(), catalog.cache(), Policy::default(), &Offline);
    let a = zoo.permit_with(id, user, || catalog.acceptances()).map_err(zoo_status)?;
    let cache = catalog.cache();
    fs::create_dir_all(&cache).map_err(|e| zoo_status(ZooError::Io { path: cache.clone(), message: e.to_string() }))?;
    let needed: u64 = a.files.iter().filter(|f| !cached(&cache, a, f)).map(|f| f.size).sum();
    if needed > 0 {
        let free = disk::free_bytes(&cache)
            .map_err(|e| plain(Failure::new(TURBO_E_RUNTIME, 0, format!("free space of {}: {e}", cache.display()))))?;
        if free < needed.saturating_add(DISK_MARGIN_BYTES) {
            return Err(plain(Failure::new(
                TURBO_E_CAPACITY,
                0,
                format!(
                    "{id} needs {needed} bytes and a margin of {DISK_MARGIN_BYTES} bytes; the cache's file system has {free} free"
                ),
            )));
        }
    }
    Ok((needed, a.files.len() as u32))
}

/// The catalog's transport, reporting each request and arrival. The fetch
/// asks for the files in index order and skips the cached ones, so a
/// request is for the first file at or after `next` with its address:
/// two files may share one.
struct Reporting<'a> {
    inner: &'a (dyn Transport + Send + Sync),
    artifact: &'a Artifact,
    tx: &'a Tx,
    next: AtomicUsize,
}

impl Transport for Reporting<'_> {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        let files = &self.artifact.files;
        let count = files.len() as u32;
        let from = self.next.load(Ordering::Relaxed);
        let (at, path) = match files.iter().enumerate().skip(from).find(|(_, f)| f.url == *url) {
            Some((i, f)) => {
                self.next.store(i + 1, Ordering::Relaxed);
                (i as u32, f.path.as_str().to_owned())
            }
            None => (0, String::new()),
        };
        let _ = self.tx.blocking_send(Ok(progress(InstallStage::Downloading, at, count, &path, size)));
        let got = self.inner.get(url, size)?;
        let _ = self.tx.blocking_send(Ok(progress(InstallStage::Downloaded, at, count, &path, got.len() as u64)));
        Ok(got)
    }
}

/// The fetch, on a blocking thread. The install carries on when the client
/// goes away: what it writes is verified, and the next install finds it.
fn run(catalog: Catalog, user: UserId, id: String, tx: Tx, _slot: Slot) {
    let outcome = (|| {
        let a = catalog.index.get(&id).map_err(zoo_status)?;
        let reporting = Reporting { inner: &*catalog.transport, artifact: a, tx: &tx, next: AtomicUsize::new(0) };
        let zoo = Zoo::new((*catalog.index).clone(), catalog.cache(), Policy::default(), &reporting);
        let fetched = zoo.fetch_with(&id, &user, || catalog.acceptances()).map_err(zoo_status)?;
        let total: u64 = a.files.iter().map(|f| f.size).sum();
        let count = a.files.len() as u32;
        let _ = tx.blocking_send(Ok(progress(InstallStage::Verified, 0, count, "", total)));
        Ok(InstalledModel {
            id: a.id.to_string(),
            version: a.version.clone(),
            dir: fetched.dir.to_string_lossy().into_owned(),
            files: fetched.files.iter().map(|p| p.to_string_lossy().into_owned()).collect(),
            total_bytes: total,
        })
    })();
    let _ = tx.blocking_send(outcome.map(|m| InstallModelUpdate { update: Some(Update::Installed(m)) }));
}
