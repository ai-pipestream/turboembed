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
    crc: u32,
    compressed: usize,
    size: usize,
    local_header: usize,
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

    /// The whole archive's bytes, as given to [`Zip::new`]: what a caller
    /// hashes to identify the file.
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
        let header =
            bytes.get(at..at + LOCAL_LEN).ok_or(Error::Zip { offset: at, message: "a local header is cut off" })?;
        if u32_at(header, 0) != LOCAL_SIGNATURE {
            return Err(Error::Zip { offset: at, message: "no local header signature where the directory points" });
        }
        let start = at + LOCAL_LEN + usize::from(u16_at(header, 26)) + usize::from(u16_at(header, 28));
        bytes
            .get(start..start + record.compressed)
            .ok_or(Error::Zip { offset: start, message: "an entry's data runs past the end of the archive" })
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

    let mut records: Vec<Record> = Vec::with_capacity(usize::from(count));
    let mut at = dir_start;
    for _ in 0..count {
        let c = bytes
            .get(at..at + CENTRAL_LEN)
            .filter(|_| at + CENTRAL_LEN <= end)
            .ok_or(Error::Zip { offset: at, message: "a central directory record is cut off" })?;
        if u32_at(c, 0) != CENTRAL_SIGNATURE {
            return Err(Error::Zip { offset: at, message: "no central directory signature where one is expected" });
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
            crc,
            compressed: compressed as usize,
            size: size as usize,
            local_header: local_header as usize,
        });
        at += CENTRAL_LEN + rest;
    }
    let mut names: Vec<&[u8]> = records.iter().map(|r| &bytes[r.name.clone()]).collect();
    names.sort_unstable();
    if names.windows(2).any(|w| w[0] == w[1]) {
        return Err(Error::Zip { offset: dir_start, message: "two entries have the same name" });
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a one-entry stored archive by hand.
    fn stored_archive(name: &str, data: &[u8], local_extra: &[u8], comment: &[u8]) -> Vec<u8> {
        let mut z = Vec::new();
        z.extend_from_slice(&LOCAL_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&crc32(data).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&(local_extra.len() as u16).to_le_bytes());
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(local_extra);
        z.extend_from_slice(data);
        let dir = z.len();
        z.extend_from_slice(&CENTRAL_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        z.extend_from_slice(&crc32(data).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(data.len() as u32).to_le_bytes());
        z.extend_from_slice(&(name.len() as u16).to_le_bytes());
        z.extend_from_slice(&[0; 12]);
        z.extend_from_slice(&0u32.to_le_bytes());
        z.extend_from_slice(name.as_bytes());
        let dir_len = z.len() - dir;
        z.extend_from_slice(&END_SIGNATURE.to_le_bytes());
        z.extend_from_slice(&[0, 0, 0, 0, 1, 0, 1, 0]);
        z.extend_from_slice(&(dir_len as u32).to_le_bytes());
        z.extend_from_slice(&(dir as u32).to_le_bytes());
        z.extend_from_slice(&(comment.len() as u16).to_le_bytes());
        z.extend_from_slice(comment);
        z
    }

    #[test]
    fn the_local_extra_field_length_is_honoured() {
        let z = stored_archive("a.txt", b"hello", b"\x99\x99\x02\x00xy", b"");
        let zip = Zip::new(&z[..]).unwrap();
        assert_eq!(&*zip.entry("a.txt").unwrap(), b"hello");
        assert!(matches!(zip.entry("a.txt").unwrap(), Cow::Borrowed(_)));
    }

    #[test]
    fn a_comment_holding_the_end_signature_is_skipped() {
        let mut comment = END_SIGNATURE.to_le_bytes().to_vec();
        comment.extend_from_slice(&[0; 30]);
        let z = stored_archive("a", b"x", b"", &comment);
        let zip = Zip::new(z).unwrap();
        assert_eq!(&*zip.entry("a").unwrap(), b"x");
    }

    #[test]
    fn a_missing_entry_is_named() {
        let z = stored_archive("a", b"x", b"", b"");
        let zip = Zip::new(z).unwrap();
        assert!(matches!(zip.entry("b"), Err(Error::MissingEntry(n)) if &*n == "b"));
    }

    #[test]
    fn a_wrong_crc_is_refused() {
        let mut z = stored_archive("a", b"x", b"", b"");
        z[30 + 1] = b'y'; // the data byte after the 30-byte header and the name
        let zip = Zip::new(z).unwrap();
        assert!(matches!(zip.entry("a"), Err(Error::Crc { .. })));
    }

    #[test]
    fn zip64_markers_are_refused() {
        let mut z = stored_archive("a", b"x", b"", b"");
        let end = z.len() - END_LEN;
        z[end + 16..end + 20].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 archives are not supported", .. })));

        let mut z = stored_archive("a", b"x", b"", b"");
        let end = z.len() - END_LEN;
        let central = u32_at(&z, end + 16) as usize;
        z[central + 24..central + 28].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 entries are not supported", .. })));

        let mut z = stored_archive("a", b"x", b"", b"");
        let end = z.len() - END_LEN;
        let mut locator = ZIP64_LOCATOR_SIGNATURE.to_le_bytes().to_vec();
        locator.extend_from_slice(&[0; 16]);
        z.splice(end..end, locator);
        assert!(matches!(Zip::new(z), Err(Error::Zip { message: "zip64 archives are not supported", .. })));
    }

    #[test]
    fn not_a_zip() {
        assert!(matches!(Zip::new(&b"short"[..]), Err(Error::Zip { message: "too short to be a zip archive", .. })));
        assert!(matches!(
            Zip::new(&[0u8; 100][..]),
            Err(Error::Zip { message: "no end of central directory record", .. })
        ));
    }
}
