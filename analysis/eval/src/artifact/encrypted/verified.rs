//! Fully authenticate an exclusively owned ciphertext snapshot before release.
//! Plaintext buffers are bounded by the codec's frame limit and never spooled.

use super::{Binding, Decoder, Error, KeyProvider};
use ring::rand::{SecureRandom, SystemRandom};
use std::fs::{self, DirBuilder, File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

struct Snapshot {
    file: Option<File>,
    directory: PathBuf,
}

impl Snapshot {
    fn new(root: &Path) -> Result<Self, Error> {
        // The configured root must already exist, avoiding implicit permission changes.
        let mut random = [0; 16];
        SystemRandom::new().fill(&mut random).map_err(|_| Error::Entropy)?;
        let name: String = random.iter().map(|b| format!("{b:02x}")).collect();
        let directory = root.join(format!(".artifact-{name}"));
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory)?;
        let mut snapshot = Self { file: None, directory };
        let mut options = OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        snapshot.file = Some(options.open(snapshot.directory.join("ciphertext"))?);
        Ok(snapshot)
    }
}

impl Drop for Snapshot {
    fn drop(&mut self) {
        drop(self.file.take());
        let _ = fs::remove_file(self.directory.join("ciphertext"));
        let _ = fs::remove_dir(&self.directory);
    }
}

struct Tee<'a, R> {
    source: R,
    destination: &'a mut File,
}

impl<R: Read> Read for Tee<'_, R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let count = self.source.read(out)?;
        self.destination.write_all(&out[..count])?;
        Ok(count)
    }
}

/// Returns no plaintext until a bounded ciphertext snapshot has been authenticated
/// through its terminal record. Snapshot cleanup occurs on success and error paths.
/// On non-Unix platforms the caller must supply a root with equivalent private ACLs.
pub struct VerifiedReader {
    decoder: Option<Decoder<File>>,
    snapshot: Snapshot,
}

impl VerifiedReader {
    pub fn open<R: Read>(
        source: R,
        keys: &dyn KeyProvider,
        binding: Binding,
        temporary_root: &Path,
        max_ciphertext_bytes: u64,
    ) -> Result<Self, Error> {
        let limit = max_ciphertext_bytes.checked_add(1).ok_or(Error::InvalidArtifact)?;
        let mut snapshot = Snapshot::new(temporary_root)?;
        let file = snapshot.file.as_mut().ok_or(Error::Io)?;
        let tee = Tee { source: source.take(limit), destination: file };
        let mut verify = Decoder::new(tee, keys, binding)?;
        io::copy(&mut verify, &mut io::sink()).map_err(|_| Error::InvalidArtifact)?;
        drop(verify);
        if file.metadata()?.len() > max_ciphertext_bytes {
            return Err(Error::InvalidArtifact);
        }
        file.seek(SeekFrom::Start(0))?;
        // Keep this same private snapshot descriptor, never reopen the provider's object.
        let file = snapshot.file.take().ok_or(Error::Io)?;
        let decoder = Decoder::new(file, keys, binding)?;
        Ok(Self { decoder: Some(decoder), snapshot })
    }
}

impl Read for VerifiedReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.decoder.as_mut().ok_or_else(|| io::Error::other(Error::Io))?.read(out)
    }
}

impl Drop for VerifiedReader {
    fn drop(&mut self) {
        // Close the descriptor before Snapshot's cross-platform unlink.
        drop(self.decoder.take());
        let _ = &self.snapshot;
    }
}
