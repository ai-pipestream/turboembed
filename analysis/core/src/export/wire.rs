use super::*;

/// Writes the fields of the boundary encoding: little-endian counts and
/// length-prefixed bytes, with no padding. Codecs write their payloads
/// with the same writer, so a payload and its record share one set of
/// field rules.
#[derive(Default)]
pub struct Writer {
    pub(super) out: Vec<u8>,
}

impl Writer {
    pub fn new() -> Writer {
        Writer::default()
    }

    pub fn u8(&mut self, v: u8) {
        self.out.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f64(&mut self, v: f64) {
        self.u64(v.to_bits());
    }

    pub fn usize(&mut self, v: usize) -> Result<(), ExportError> {
        self.u32(u32::try_from(v).map_err(|_| malformed("a count longer than a u32 can hold"))?);
        Ok(())
    }

    pub fn bytes(&mut self, b: &[u8]) -> Result<(), ExportError> {
        self.usize(b.len())?;
        self.out.extend_from_slice(b);
        Ok(())
    }

    pub fn str(&mut self, s: &str) -> Result<(), ExportError> {
        self.bytes(s.as_bytes())
    }

    pub fn span(&mut self, s: Span) {
        self.u32(s.start);
        self.u32(s.end);
    }

    pub fn finish(self) -> Vec<u8> {
        self.out
    }
}

/// Reads what [`Writer`] wrote, refusing anything truncated or out of
/// range as [`ExportError::Malformed`] rather than panicking.
pub struct Reader<'a> {
    buf: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(buf: &'a [u8]) -> Reader<'a> {
        Reader { buf, at: 0 }
    }

    pub(super) fn take(&mut self, n: usize) -> Result<&'a [u8], ExportError> {
        let end = self.at.checked_add(n).ok_or_else(|| malformed("truncated"))?;
        let some = self.buf.get(self.at..end).ok_or_else(|| malformed("truncated"))?;
        self.at = end;
        Ok(some)
    }

    pub fn u8(&mut self) -> Result<u8, ExportError> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16, ExportError> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32, ExportError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64, ExportError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn f64(&mut self) -> Result<f64, ExportError> {
        Ok(f64::from_bits(self.u64()?))
    }

    pub fn usize(&mut self) -> Result<usize, ExportError> {
        usize::try_from(self.u32()?).map_err(|_| malformed("a count longer than a usize can hold"))
    }

    pub fn bytes(&mut self) -> Result<&'a [u8], ExportError> {
        let n = self.usize()?;
        self.take(n)
    }

    pub fn string(&mut self) -> Result<String, ExportError> {
        String::from_utf8(self.bytes()?.to_vec()).map_err(|_| malformed("a string that is not UTF-8"))
    }

    pub fn span(&mut self) -> Result<Span, ExportError> {
        let (start, end) = (self.u32()?, self.u32()?);
        if start > end {
            return Err(malformed("a span that ends before it starts"));
        }
        Ok(Span { start, end })
    }

    /// Whether the whole input has been read.
    pub fn is_done(&self) -> bool {
        self.at == self.buf.len()
    }
}
