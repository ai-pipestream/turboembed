//! Local artifact storage with temporary staging. Committed files persist
//! until explicitly deleted, so another run can compare them later.

use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::{ArtifactKey, EvaluationArtifactStore, PendingArtifact, Publication, StorageError};

static NEXT_STAGE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
pub struct TemporaryFileStore {
    directory: PathBuf,
}

impl TemporaryFileStore {
    pub fn new(directory: impl Into<PathBuf>) -> Self {
        Self { directory: directory.into() }
    }

    fn path(&self, key: &ArtifactKey) -> PathBuf {
        self.directory.join(key.as_str())
    }
}

fn io_error(operation: &'static str, source: io::Error) -> StorageError {
    StorageError::Io { operation, source }
}

impl EvaluationArtifactStore for TemporaryFileStore {
    fn begin(&self, key: &ArtifactKey) -> Result<Box<dyn PendingArtifact>, StorageError> {
        fs::create_dir_all(&self.directory).map_err(|e| io_error("create directory", e))?;
        let destination = self.path(key);
        match fs::symlink_metadata(&destination) {
            Ok(_) => return Err(StorageError::AlreadyExists(key.clone())),
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(io_error("inspect destination", e)),
        }
        for _ in 0..128 {
            let nonce = NEXT_STAGE.fetch_add(1, Ordering::Relaxed);
            let stage = self.directory.join(format!(".{key}.{}.{}.partial", std::process::id(), nonce));
            let mut options = OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            match options.open(&stage) {
                Ok(file) => {
                    return Ok(Box::new(Pending {
                        key: key.clone(),
                        stage,
                        destination,
                        directory: self.directory.clone(),
                        out: Some(BufWriter::new(file)),
                    }));
                }
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(io_error("create staging file", e)),
            }
        }
        Err(io_error(
            "create unique staging file",
            io::Error::new(io::ErrorKind::AlreadyExists, "staging names exhausted"),
        ))
    }

    fn open(&self, key: &ArtifactKey) -> Result<Box<dyn Read>, StorageError> {
        File::open(self.path(key)).map(|f| Box::new(f) as Box<dyn Read>).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound { StorageError::NotFound(key.clone()) } else { io_error("open", e) }
        })
    }

    fn list(&self) -> Result<Vec<ArtifactKey>, StorageError> {
        let mut keys = Vec::new();
        for entry in fs::read_dir(&self.directory).map_err(|e| io_error("list", e))? {
            let entry = entry.map_err(|e| io_error("list entry", e))?;
            let name = entry.file_name();
            // Do not list staging files, directories or symbolic links.
            if !entry.file_type().map_err(|e| io_error("inspect entry", e))?.is_file() {
                continue;
            }
            let Some(name) = name.to_str() else { continue };
            if name.starts_with('.') {
                continue;
            }
            keys.push(ArtifactKey::new(name)?);
        }
        keys.sort();
        Ok(keys)
    }

    fn delete(&self, key: &ArtifactKey) -> Result<(), StorageError> {
        fs::remove_file(self.path(key)).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                StorageError::NotFound(key.clone())
            } else {
                io_error("delete", e)
            }
        })?;
        sync_directory(&self.directory).map_err(|e| io_error("sync deletion", e))
    }
}

struct Pending {
    key: ArtifactKey,
    stage: PathBuf,
    destination: PathBuf,
    directory: PathBuf,
    out: Option<BufWriter<File>>,
}

impl Write for Pending {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.out.as_mut().expect("pending writer").write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.out.as_mut().expect("pending writer").flush()
    }
}

impl Pending {
    fn discard(&mut self) -> io::Result<()> {
        if let Some(out) = self.out.take() {
            // Close without flushing buffered bytes, before unlinking on Windows.
            let (file, _) = out.into_parts();
            drop(file);
        }
        match fs::remove_file(&self.stage) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
            result => result,
        }
    }
}

impl PendingArtifact for Pending {
    fn finalize(mut self: Box<Self>) -> Result<Publication, StorageError> {
        let out = self.out.as_mut().expect("pending writer");
        out.flush().map_err(|e| io_error("flush staging file", e))?;
        out.get_ref().sync_all().map_err(|e| io_error("sync staging file", e))?;
        // Same-directory hard linking atomically refuses every existing name,
        // including dangling symlinks. Never fall back to replacing rename.
        fs::hard_link(&self.stage, &self.destination).map_err(|e| {
            if e.kind() == io::ErrorKind::AlreadyExists {
                StorageError::AlreadyExists(self.key.clone())
            } else {
                io_error("publish without replacement", e)
            }
        })?;
        let cleanup_error = self.discard().err();
        sync_directory(&self.directory).map_err(|source| StorageError::Published { key: self.key.clone(), source })?;
        Ok(Publication { key: self.key.clone(), cleanup_error })
    }

    fn abort(mut self: Box<Self>) -> Result<(), StorageError> {
        self.discard().map_err(|e| io_error("abort staging file", e))
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        let _ = self.discard();
    }
}

fn sync_directory(path: &Path) -> io::Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()
    }
    #[cfg(not(unix))]
    {
        let _ = path;
        Ok(())
    }
}
