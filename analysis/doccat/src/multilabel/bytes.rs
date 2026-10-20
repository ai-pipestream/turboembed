//! The model entry's byte codec: `DataOutputStream`-style big-endian
//! writes (`writeUTF` length-prefixed strings) and the bounds-checked
//! reader, which fails on truncated input instead of panicking.

use super::error::MultiLabelError;

/// `writeUTF`'s limit on a string's bytes, which the entry format keeps.
pub(crate) const MAX_UTF_BYTES: usize = u16::MAX as usize;

/// Writes `writeUTF`'s framing: a `u16` byte length, then the bytes.
pub(crate) fn put_utf(out: &mut Vec<u8>, s: &str) {
    debug_assert!(s.len() <= MAX_UTF_BYTES);
    out.extend_from_slice(&(s.len() as u16).to_be_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// A big-endian reader over the entry's bytes: every read is bounds
/// checked, so truncated input fails instead of panicking.
pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Reader<'a> {
        Reader { bytes, pos: 0 }
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// The bytes read so far; the entry's checksum covers everything
    /// before its own four bytes.
    pub(crate) fn position(&self) -> usize {
        self.pos
    }

    pub(crate) fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], MultiLabelError> {
        if self.remaining() < n {
            return Err(MultiLabelError::Truncated(what));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    pub(crate) fn i32(&mut self, what: &'static str) -> Result<i32, MultiLabelError> {
        Ok(i32::from_be_bytes(self.take(4, what)?.try_into().expect("4 bytes")))
    }

    pub(crate) fn f64(&mut self, what: &'static str) -> Result<f64, MultiLabelError> {
        Ok(f64::from_be_bytes(self.take(8, what)?.try_into().expect("8 bytes")))
    }

    pub(crate) fn utf8(&mut self, what: &'static str) -> Result<&'a str, MultiLabelError> {
        let len = usize::from(u16::from_be_bytes(self.take(2, what)?.try_into().expect("2 bytes")));
        let bytes = self.take(len, what)?;
        std::str::from_utf8(bytes).map_err(|_| MultiLabelError::Invalid(format!("{what} is not UTF-8")))
    }

    /// A count of records to read, refused past the remaining bytes so a
    /// corrupt count cannot drive a huge allocation.
    pub(crate) fn count(&mut self, what: &'static str) -> Result<usize, MultiLabelError> {
        let n = self.i32(what)?;
        if n < 0 {
            return Err(MultiLabelError::Invalid(format!("{what} of {n}")));
        }
        Ok(n as usize)
    }

    pub(crate) fn done(&self) -> Result<(), MultiLabelError> {
        if self.pos == self.bytes.len() {
            Ok(())
        } else {
            Err(MultiLabelError::Invalid(format!("{} trailing bytes after the checksum", self.remaining())))
        }
    }
}
