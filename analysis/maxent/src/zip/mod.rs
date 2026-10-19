//! A read-only zip archive over bytes in memory, enough for OpenNLP model
//! files: stored and deflated entries, an archive comment, and data
//! descriptors (which are ignored, because sizes and CRCs are taken from
//! the central directory). Zip64, encryption and archives split over
//! several disks are refused with an error that says so.

use std::borrow::Cow;
use std::ops::Range;

use crate::crc32::crc32;
use crate::error::{Error, Result};
use crate::inflate::inflate;

const END_SIGNATURE: u32 = 0x0605_4b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
const CENTRAL_SIGNATURE: u32 = 0x0201_4b50;
const LOCAL_SIGNATURE: u32 = 0x0403_4b50;
const END_LEN: usize = 22;
const CENTRAL_LEN: usize = 46;
const LOCAL_LEN: usize = 30;
const ZIP64_LOCATOR_LEN: usize = 20;
const MAX_COMMENT: usize = 0xFFFF;

/// How an entry's data is stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    Stored,
    Deflated,
}

/// One central directory record. Names are kept as a range of the
/// archive's bytes rather than copied.
#[derive(Debug, Clone)]
struct Record {
    name: Range<usize>,
    method: Method,
    flags: u16,
    crc: u32,
    compressed: usize,
    size: usize,
    local_header: usize,
    data_limit: usize,
}

/// A zip archive. `B` is the archive's bytes, borrowed (`&[u8]`) or owned
/// (`Vec<u8>`).
#[derive(Debug, Clone)]
pub struct Zip<B: AsRef<[u8]>> {
    bytes: B,
    records: Vec<Record>,
}

impl<B: AsRef<[u8]>> Zip<B> {
    /// Reads the end of central directory record and the central directory.
    pub fn new(bytes: B) -> Result<Zip<B>> {
        let records = read_directory(bytes.as_ref())?;
        Ok(Zip { bytes, records })
    }

    /// The archive's bytes, exactly as given to [`Zip::new`], for hashing
    /// or storing the file as a whole.
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether the archive has no entries.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The entry names in central directory order, as raw bytes. OpenNLP
    /// writes UTF-8 names.
    pub fn names(&self) -> impl Iterator<Item = &[u8]> {
        let bytes = self.bytes.as_ref();
        self.records.iter().map(move |r| &bytes[r.name.clone()])
    }

    /// Whether the archive has an entry with this name.
    pub fn contains(&self, name: &str) -> bool {
        self.find(name).is_some()
    }

    /// The uncompressed size the directory states for an entry.
    pub fn size(&self, name: &str) -> Option<usize> {
        self.find(name).map(|r| r.size)
    }

    /// How an entry is stored.
    pub fn method(&self, name: &str) -> Option<Method> {
        self.find(name).map(|r| r.method)
    }

    /// An entry's data: borrowed from the archive when stored, inflated
    /// when deflated. The CRC-32 is checked either way.
    pub fn entry(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let record = self.find(name).ok_or_else(|| Error::MissingEntry(name.into()))?;
        let in_entry = |error| Error::Entry { entry: name.into(), error: Box::new(error) };
        let raw = self.raw_data(record).map_err(in_entry)?;
        let data = match record.method {
            Method::Stored => {
                if raw.len() != record.size {
                    let error = Error::Zip {
                        offset: record.local_header,
                        message: "a stored entry's compressed and uncompressed sizes differ",
                    };
                    return Err(in_entry(error));
                }
                Cow::Borrowed(raw)
            }
            Method::Deflated => Cow::Owned(inflate(raw, record.size).map_err(in_entry)?),
        };
        let actual = crc32(&data);
        if actual != record.crc {
            return Err(Error::Crc { entry: name.into(), expected: record.crc, actual });
        }
        Ok(data)
    }

    fn find(&self, name: &str) -> Option<&Record> {
        let bytes = self.bytes.as_ref();
        self.records.iter().find(|r| &bytes[r.name.clone()] == name.as_bytes())
    }

    /// The entry's stored bytes, located through its local header, whose
    /// name and extra field lengths can differ from the central record's.
    fn raw_data(&self, record: &Record) -> Result<&[u8]> {
        let bytes = self.bytes.as_ref();
        let at = record.local_header;
        // The offsets and sizes come from the archive and can be near
        // u32::MAX, so the sums are checked: on a 32-bit target they could
        // wrap.
        let header = at
            .checked_add(LOCAL_LEN)
            .and_then(|header_end| bytes.get(at..header_end))
            .ok_or(Error::Zip { offset: at, message: "a local header is cut off" })?;
        if u32_at(header, 0) != LOCAL_SIGNATURE {
            return Err(Error::Zip { offset: at, message: "no local header signature where the directory points" });
        }
        let past_end = |offset| Error::Zip { offset, message: "an entry's data runs past the end of the archive" };
        let start = (at + LOCAL_LEN)
            .checked_add(usize::from(u16_at(header, 26)) + usize::from(u16_at(header, 28)))
            .ok_or(past_end(at))?;
        let end = start.checked_add(record.compressed).ok_or(past_end(start))?;
        let data = bytes.get(start..end).ok_or(past_end(start))?;
        if end > record.data_limit {
            return Err(Error::Zip { offset: start, message: "an entry overlaps the central directory" });
        }
        if record.method == Method::Stored && record.compressed != record.size {
            return Err(Error::Zip {
                offset: at,
                message: "a stored entry's compressed and uncompressed sizes differ",
            });
        }
        let method = match record.method {
            Method::Stored => 0,
            Method::Deflated => 8,
        };
        if u16_at(header, 6) != record.flags || u16_at(header, 8) != method {
            return Err(Error::Zip { offset: at, message: "local header flags or method differ from the directory" });
        }
        if record.flags & 8 == 0
            && (u32_at(header, 14) != record.crc
                || u32_at(header, 18) as usize != record.compressed
                || u32_at(header, 22) as usize != record.size)
        {
            return Err(Error::Zip { offset: at, message: "local header CRC or sizes differ from the directory" });
        }
        let name_start = at + LOCAL_LEN;
        let name_end = name_start
            .checked_add(usize::from(u16_at(header, 26)))
            .ok_or(Error::Zip { offset: at, message: "a local name is cut off" })?;
        if bytes.get(name_start..name_end) != Some(&bytes[record.name.clone()]) {
            return Err(Error::Zip { offset: at, message: "local header name differs from the directory" });
        }
        Ok(data)
    }
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Finds the end of central directory record: the last position holding
/// its signature whose comment length reaches exactly to the end of the
/// archive. Searching backwards from the end finds it before any copy of
/// the signature that a comment might hold.
fn find_end(bytes: &[u8]) -> Result<usize> {
    if bytes.len() < END_LEN {
        return Err(Error::Zip { offset: 0, message: "too short to be a zip archive" });
    }
    let last = bytes.len() - END_LEN;
    let first = last.saturating_sub(MAX_COMMENT);
    for at in (first..=last).rev() {
        if u32_at(bytes, at) == END_SIGNATURE && at + END_LEN + usize::from(u16_at(bytes, at + 20)) == bytes.len() {
            return Ok(at);
        }
    }
    Err(Error::Zip { offset: bytes.len(), message: "no end of central directory record" })
}

fn read_directory(bytes: &[u8]) -> Result<Vec<Record>> {
    let end = find_end(bytes)?;
    let e = &bytes[end..end + END_LEN];
    if end >= ZIP64_LOCATOR_LEN && u32_at(bytes, end - ZIP64_LOCATOR_LEN) == ZIP64_LOCATOR_SIGNATURE {
        return Err(Error::Zip { offset: end - ZIP64_LOCATOR_LEN, message: "zip64 archives are not supported" });
    }
    if u16_at(e, 4) != 0 || u16_at(e, 6) != 0 || u16_at(e, 8) != u16_at(e, 10) {
        return Err(Error::Zip { offset: end, message: "archives split over several disks are not supported" });
    }
    let count = u16_at(e, 10);
    let dir_size = u32_at(e, 12);
    let dir_start = u32_at(e, 16);
    if count == 0xFFFF || dir_size == 0xFFFF_FFFF || dir_start == 0xFFFF_FFFF {
        return Err(Error::Zip { offset: end, message: "zip64 archives are not supported" });
    }
    let dir_start = dir_start as usize;
    if dir_start.checked_add(dir_size as usize).is_none_or(|dir_end| dir_end > end) {
        return Err(Error::Zip { offset: end, message: "the central directory lies outside the archive" });
    }

    let dir_end = dir_start + dir_size as usize;
    let mut records: Vec<Record> = Vec::new();
    let mut at = dir_start;
    for _ in 0..count {
        let c = bytes
            .get(at..at + CENTRAL_LEN)
            .filter(|_| at + CENTRAL_LEN <= end)
            .ok_or(Error::Zip { offset: at, message: "a central directory record is cut off" })?;
        if u32_at(c, 0) != CENTRAL_SIGNATURE {
            return Err(Error::Zip { offset: at, message: "no central directory signature where one is expected" });
        }
        if u16_at(c, 34) != 0 {
            return Err(Error::Zip { offset: at, message: "an entry starts on another disk" });
        }
        if u16_at(c, 8) & 1 != 0 {
            return Err(Error::Zip { offset: at, message: "encrypted entries are not supported" });
        }
        let method = match u16_at(c, 10) {
            0 => Method::Stored,
            8 => Method::Deflated,
            _ => {
                return Err(Error::Zip {
                    offset: at,
                    message: "compression methods other than stored and deflated are not supported",
                });
            }
        };
        let crc = u32_at(c, 16);
        let compressed = u32_at(c, 20);
        let size = u32_at(c, 24);
        let local_header = u32_at(c, 42);
        if compressed == 0xFFFF_FFFF || size == 0xFFFF_FFFF || local_header == 0xFFFF_FFFF {
            return Err(Error::Zip { offset: at, message: "zip64 entries are not supported" });
        }
        let name_len = usize::from(u16_at(c, 28));
        let rest = name_len + usize::from(u16_at(c, 30)) + usize::from(u16_at(c, 32));
        let name = at + CENTRAL_LEN..at + CENTRAL_LEN + name_len;
        if at + CENTRAL_LEN + rest > end {
            return Err(Error::Zip { offset: at, message: "a central directory record is cut off" });
        }
        records.push(Record {
            name,
            method,
            flags: u16_at(c, 8),
            crc,
            compressed: compressed as usize,
            size: size as usize,
            local_header: local_header as usize,
            data_limit: dir_start,
        });
        at += CENTRAL_LEN + rest;
    }
    if at != dir_end {
        return Err(Error::Zip { offset: at, message: "central directory records do not match its stated size" });
    }
    let mut names: Vec<&[u8]> = records.iter().map(|r| &bytes[r.name.clone()]).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::Zip { offset: dir_start, message: "two entries have the same name" });
    }
    Ok(records)
}

#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::stored_archive;

#[cfg(test)]
mod boundary_tests;
