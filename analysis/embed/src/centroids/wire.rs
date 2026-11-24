use super::*;

/// The bytes `write` writes into an empty buffer.
pub(crate) fn in_memory(write: impl FnOnce(&mut Vec<u8>) -> io::Result<()>) -> Vec<u8> {
    let mut buf = Vec::new();
    write(&mut buf).expect("writing to Vec<u8> cannot fail");
    buf
}

/// Everything `r` holds, or its read error as text.
pub(crate) fn read_all<R: Read>(mut r: R) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    r.read_to_end(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

pub(crate) fn write_str<W: Write>(w: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(bytes)
}

pub(crate) struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Cursor { bytes, pos: 0 }
    }

    pub(crate) fn read_array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        let end = self.pos.checked_add(N).ok_or_else(|| "unexpected EOF".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected EOF".into());
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..end]);
        self.pos = end;
        Ok(out)
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_array::<1>()?[0])
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.read_array::<4>()?))
    }

    pub(crate) fn read_f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.read_array::<8>()?))
    }

    pub(crate) fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u32()? as usize;
        let end = self.pos.checked_add(len).ok_or_else(|| "string length overflow".to_string())?;
        if end > self.bytes.len() {
            return Err("unexpected EOF while reading string".into());
        }
        let s = std::str::from_utf8(&self.bytes[self.pos..end]).map_err(|e| format!("invalid UTF-8: {e}"))?;
        self.pos = end;
        Ok(s.to_owned())
    }

    pub(crate) fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    pub(crate) fn finish(&self) -> Result<(), String> {
        if self.pos != self.bytes.len() {
            return Err(format!("{} trailing byte(s) after model data", self.bytes.len() - self.pos));
        }
        Ok(())
    }
}

#[cfg(test)]
mod buffer_tests {
    use super::*;

    #[test]
    fn in_memory_keeps_what_was_written_and_read_all_returns_it() {
        let bytes = in_memory(|w| write_str(w, "ab"));
        assert_eq!(bytes, [2, 0, 0, 0, b'a', b'b']);
        assert_eq!(read_all(&bytes[..]), Ok(bytes.clone()));
    }

    #[test]
    fn read_all_reports_a_failed_read_as_text() {
        struct Broken;
        impl Read for Broken {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("gone"))
            }
        }
        assert_eq!(read_all(Broken), Err("gone".to_string()));
    }
}
