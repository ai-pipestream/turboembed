//! Versioned framed AES-256-GCM with per-artifact HKDF key separation.
//! Decoder callers must consume the terminal record before treating data as complete.

use super::{Binding, Error, KeyMaterial, KeyProvider};
use ring::{
    aead, hkdf,
    rand::{SecureRandom, SystemRandom},
};
use std::io::{self, Read, Write};

const MAGIC: &[u8; 8] = b"TEAEAD01";
const HEADER: usize = 56;
const BLOCK: usize = 65_536;
const TAG: usize = 16;

struct KeyLen;
impl hkdf::KeyType for KeyLen {
    fn len(&self) -> usize {
        32
    }
}

fn key(material: KeyMaterial, salt: &[u8]) -> Result<aead::LessSafeKey, Error> {
    let salt = hkdf::Salt::new(hkdf::HKDF_SHA256, salt);
    let extracted = salt.extract(&material.0);
    let mut bytes = [0; 32];
    extracted
        .expand(&[b"turboembed artifact AES-256-GCM v1"], KeyLen)
        .map_err(|_| Error::InvalidArtifact)?
        .fill(&mut bytes)
        .map_err(|_| Error::InvalidArtifact)?;
    let key = aead::UnboundKey::new(&aead::AES_256_GCM, &bytes).map_err(|_| Error::InvalidArtifact)?;
    Ok(aead::LessSafeKey::new(key))
}

fn nonce(index: u64) -> aead::Nonce {
    let mut nonce = [0; 12];
    nonce[4..].copy_from_slice(&index.to_be_bytes());
    aead::Nonce::assume_unique_for_key(nonce)
}

fn aad(header: &[u8; HEADER], binding: Binding, index: u64, kind: u8, length: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER + 45);
    out.extend_from_slice(header);
    out.extend_from_slice(&binding.0);
    out.extend_from_slice(&index.to_be_bytes());
    out.push(kind);
    out.extend_from_slice(&length.to_be_bytes());
    out
}

pub(super) struct Encoder<W: Write> {
    output: W,
    key: aead::LessSafeKey,
    header: [u8; HEADER],
    binding: Binding,
    index: u64,
    total: u64,
    buffer: Vec<u8>,
    poisoned: bool,
}

impl<W: Write> Encoder<W> {
    pub fn new(mut output: W, keys: &dyn KeyProvider, binding: Binding) -> Result<Self, Error> {
        let (id, material) = keys.active()?;
        let mut header = [0; HEADER];
        header[..8].copy_from_slice(MAGIC);
        header[8..24].copy_from_slice(&id.0);
        SystemRandom::new().fill(&mut header[24..]).map_err(|_| Error::Entropy)?;
        let key = key(material, &header[24..])?;
        output.write_all(&header)?;
        Ok(Self {
            output,
            key,
            header,
            binding,
            index: 0,
            total: 0,
            buffer: Vec::with_capacity(BLOCK + TAG),
            poisoned: false,
        })
    }

    pub(super) fn key_id(&self) -> super::KeyId {
        super::KeyId(self.header[8..24].try_into().expect("fixed key identifier"))
    }

    pub(super) fn abort(self) -> W {
        self.output
    }

    fn frame(&mut self, kind: u8, mut bytes: Vec<u8>) -> Result<(), Error> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        // Reserve the last possible index; never wrap a nonce counter.
        let next = self.index.checked_add(1).ok_or(Error::InvalidArtifact)?;
        let length = u32::try_from(bytes.len()).map_err(|_| Error::InvalidArtifact)?;
        self.poisoned = true;
        let aad = aad(&self.header, self.binding, self.index, kind, length);
        self.key
            .seal_in_place_append_tag(nonce(self.index), aead::Aad::from(aad), &mut bytes)
            .map_err(|_| Error::InvalidArtifact)?;
        self.output.write_all(&[kind])?;
        self.output.write_all(&length.to_be_bytes())?;
        self.output.write_all(&bytes)?;
        self.index = next;
        self.poisoned = false;
        Ok(())
    }

    fn drain(&mut self) -> Result<(), Error> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let bytes = std::mem::replace(&mut self.buffer, Vec::with_capacity(BLOCK + TAG));
        self.frame(0, bytes)
    }

    pub fn finish(mut self) -> Result<W, Error> {
        if self.poisoned {
            return Err(Error::Poisoned);
        }
        self.drain()?;
        let mut terminal = self.index.to_be_bytes().to_vec();
        terminal.extend_from_slice(&self.total.to_be_bytes());
        self.frame(1, terminal)?;
        self.output.flush()?;
        Ok(self.output)
    }
}

impl<W: Write> Write for Encoder<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.poisoned {
            return Err(io::Error::other(Error::Poisoned));
        }
        let count = bytes.len().min(BLOCK - self.buffer.len());
        let total = self.total.checked_add(count as u64).ok_or_else(|| io::Error::other(Error::InvalidArtifact))?;
        self.buffer.extend_from_slice(&bytes[..count]);
        self.total = total;
        if self.buffer.len() == BLOCK {
            self.drain().map_err(io::Error::other)?;
        }
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other(Error::Poisoned));
        }
        // A transport failure permanently poisons publication.
        self.poisoned = true;
        self.output.flush().map_err(|_| io::Error::other(Error::Io))?;
        self.poisoned = false;
        Ok(())
    }
}

pub(super) struct Decoder<R: Read> {
    input: R,
    key: aead::LessSafeKey,
    header: [u8; HEADER],
    binding: Binding,
    index: u64,
    total: u64,
    buffer: Vec<u8>,
    position: usize,
    complete: bool,
    failed: bool,
}

impl<R: Read> Decoder<R> {
    pub fn new(mut input: R, keys: &dyn KeyProvider, binding: Binding) -> Result<Self, Error> {
        let mut header = [0; HEADER];
        input.read_exact(&mut header).map_err(|_| Error::InvalidArtifact)?;
        if &header[..8] != MAGIC {
            return Err(Error::InvalidArtifact);
        }
        let id = super::KeyId(header[8..24].try_into().map_err(|_| Error::InvalidArtifact)?);
        let key = key(keys.resolve(id)?, &header[24..])?;
        Ok(Self {
            input,
            key,
            header,
            binding,
            index: 0,
            total: 0,
            buffer: Vec::new(),
            position: 0,
            complete: false,
            failed: false,
        })
    }

    fn frame(&mut self) -> Result<(), Error> {
        let mut frame = [0; 5];
        self.input.read_exact(&mut frame).map_err(|_| Error::InvalidArtifact)?;
        let kind = frame[0];
        let length = u32::from_be_bytes(frame[1..].try_into().map_err(|_| Error::InvalidArtifact)?) as usize;
        if !matches!((kind, length), (0, 1..=BLOCK) | (1, 16)) {
            return Err(Error::InvalidArtifact);
        }
        self.buffer.resize(length + TAG, 0);
        self.input.read_exact(&mut self.buffer).map_err(|_| Error::InvalidArtifact)?;
        let aad = aad(&self.header, self.binding, self.index, kind, length as u32);
        self.key
            .open_in_place(nonce(self.index), aead::Aad::from(aad), &mut self.buffer)
            .map_err(|_| Error::InvalidArtifact)?;
        self.buffer.truncate(length);
        self.position = 0;
        if kind == 1 {
            let count = u64::from_be_bytes(self.buffer[..8].try_into().map_err(|_| Error::InvalidArtifact)?);
            let total = u64::from_be_bytes(self.buffer[8..].try_into().map_err(|_| Error::InvalidArtifact)?);
            let mut suffix = [0];
            if count != self.index || total != self.total || self.input.read(&mut suffix)? != 0 {
                return Err(Error::InvalidArtifact);
            }
            self.buffer.clear();
            self.complete = true;
        } else {
            self.total = self.total.checked_add(length as u64).ok_or(Error::InvalidArtifact)?;
            self.index = self.index.checked_add(1).ok_or(Error::InvalidArtifact)?;
        }
        Ok(())
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::other(Error::InvalidArtifact));
        }
        if out.is_empty() {
            return Ok(0);
        }
        if self.position == self.buffer.len() && !self.complete {
            self.failed = true;
            self.frame().map_err(io::Error::other)?;
            self.failed = false;
        }
        let count = out.len().min(self.buffer.len() - self.position);
        out[..count].copy_from_slice(&self.buffer[self.position..self.position + count]);
        self.position += count;
        Ok(count)
    }
}
