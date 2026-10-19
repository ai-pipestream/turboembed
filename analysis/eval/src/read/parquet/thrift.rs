//! Thrift's compact protocol, read only, as far as Parquet's footer and
//! page headers use it: structs, lists, maps, booleans, integers, doubles
//! and binary strings.
//!
//! A struct is read field by field: the caller's function sees each
//! field's id and type and either reads the value or returns `false`,
//! and the field is then stepped over. Nesting is limited, so a damaged
//! or hostile footer cannot exhaust the stack.

use crate::Result;

/// The compact protocol's type codes, as a field header carries them.
pub const BOOL_TRUE: u8 = 1;
pub const BOOL_FALSE: u8 = 2;
pub const I8: u8 = 3;
pub const I16: u8 = 4;
pub const I32: u8 = 5;
pub const I64: u8 = 6;
pub const DOUBLE: u8 = 7;
pub const BINARY: u8 = 8;
pub const LIST: u8 = 9;
pub const SET: u8 = 10;
pub const MAP: u8 = 11;
pub const STRUCT: u8 = 12;

/// How deep structs and collections may nest.
const MAX_DEPTH: u32 = 64;

/// A cursor over compact protocol bytes.
pub struct Compact<'a> {
    buf: &'a [u8],
    pos: usize,
    depth: u32,
    short: bool,
}

impl<'a> Compact<'a> {
    pub fn new(buf: &'a [u8]) -> Compact<'a> {
        Compact { buf, pos: 0, depth: 0, short: false }
    }

    /// Bytes read so far.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Whether a read has needed more bytes than the buffer holds: a
    /// value ran past its end, or a list counted more elements than the
    /// bytes left. Reading the first part of a footer, this is where the
    /// part ends; reading a whole one, it is damage.
    pub fn ran_short(&self) -> bool {
        self.short
    }

    fn ends_inside(&mut self) -> String {
        self.short = true;
        "thrift: ends inside a value".into()
    }

    fn byte(&mut self) -> Result<u8> {
        let Some(&b) = self.buf.get(self.pos) else {
            return Err(self.ends_inside());
        };
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let Some(end) = self.pos.checked_add(n).filter(|&e| e <= self.buf.len()) else {
            return Err(self.ends_inside());
        };
        let b = &self.buf[self.pos..end];
        self.pos = end;
        Ok(b)
    }

    /// An unsigned LEB128 varint of at most 10 bytes.
    pub fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..70).step_by(7) {
            let b = self.byte()?;
            if shift == 63 && b > 1 {
                return Err("thrift: varint overflows u64".into());
            }
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err("thrift: varint longer than 10 bytes".into())
    }

    /// A zigzag varint, as i16, i32 and i64 are written.
    pub fn i64(&mut self) -> Result<i64> {
        let v = self.varint()?;
        Ok((v >> 1) as i64 ^ -((v & 1) as i64))
    }

    pub fn i32(&mut self) -> Result<i32> {
        let v = self.i64()?;
        i32::try_from(v).map_err(|_| format!("thrift: {v} does not fit an i32"))
    }

    pub fn binary(&mut self) -> Result<&'a [u8]> {
        let n = self.varint()?;
        let n = usize::try_from(n).map_err(|_| "thrift: binary length too large")?;
        self.take(n)
    }

    pub fn string(&mut self) -> Result<String> {
        let b = self.binary()?;
        String::from_utf8(b.to_vec()).map_err(|_| "thrift: string is not UTF-8".into())
    }

    /// A boolean field's value, which its type code carries.
    pub fn bool_field(ty: u8) -> Result<bool> {
        match ty {
            BOOL_TRUE => Ok(true),
            BOOL_FALSE => Ok(false),
            _ => Err(format!("thrift: type {ty} where a boolean was expected")),
        }
    }

    /// A list or set header: the element type and the element count.
    pub fn list_header(&mut self) -> Result<(u8, usize)> {
        self.list_header_in(self.buf.len())
    }

    /// A list or set header in the first bytes of a value `len` bytes
    /// long, the buffer being those bytes: the count is checked against
    /// the bytes left of `len`.
    pub fn list_header_in(&mut self, len: usize) -> Result<(u8, usize)> {
        let b = self.byte()?;
        let n = match b >> 4 {
            15 => usize::try_from(self.varint()?).map_err(|_| "thrift: list too long")?,
            n => usize::from(n),
        };
        // Each element takes at least one byte, booleans included, so a
        // count past the bytes left is damage, not a long list.
        let left = len.saturating_sub(self.pos);
        if n > left {
            self.short |= len == self.buf.len();
            return Err(format!("thrift: a list of {n} elements in {left} bytes"));
        }
        Ok((b & 0x0f, n))
    }

    /// The elements of a list, each read by `f`.
    pub fn list<T>(&mut self, mut f: impl FnMut(&mut Self, u8) -> Result<T>) -> Result<Vec<T>> {
        let (ty, n) = self.list_header()?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(f(self, ty)?);
        }
        Ok(out)
    }

    /// The fields of a struct, up to its stop byte. `f` is given each
    /// field's id and type; it reads the value and returns `true`, or
    /// returns `false` and the value is stepped over.
    pub fn read_struct(&mut self, mut f: impl FnMut(&mut Self, i16, u8) -> Result<bool>) -> Result<()> {
        self.enter()?;
        let mut last: i16 = 0;
        while let Some((id, ty)) = self.field_header(&mut last)? {
            if !f(self, id, ty)? {
                self.skip_field(ty)?;
            }
        }
        self.depth -= 1;
        Ok(())
    }

    /// The next field's id and type in a struct whose last field had id
    /// `last`, which is updated; `None` at the struct's stop byte.
    pub fn field_header(&mut self, last: &mut i16) -> Result<Option<(i16, u8)>> {
        let b = self.byte()?;
        if b == 0 {
            return Ok(None);
        }
        let id = match b >> 4 {
            0 => {
                let v = self.i64()?;
                i16::try_from(v).map_err(|_| format!("thrift: field id {v}"))?
            }
            d => last.checked_add(i16::from(d)).ok_or("thrift: field id overflows")?,
        };
        *last = id;
        Ok(Some((id, b & 0x0f)))
    }

    fn enter(&mut self) -> Result<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(format!("thrift: nested more than {MAX_DEPTH} deep"));
        }
        Ok(())
    }

    /// Steps over a field's value; a boolean field has none.
    pub fn skip_field(&mut self, ty: u8) -> Result<()> {
        match ty {
            BOOL_TRUE | BOOL_FALSE => Ok(()),
            _ => self.skip_value(ty),
        }
    }

    /// Steps over a value inside a list, set or map, where a boolean takes
    /// a byte.
    fn skip_value(&mut self, ty: u8) -> Result<()> {
        match ty {
            BOOL_TRUE | BOOL_FALSE | I8 => self.take(1).map(drop),
            I16 | I32 | I64 => self.varint().map(drop),
            DOUBLE => self.take(8).map(drop),
            BINARY => self.binary().map(drop),
            LIST | SET => {
                self.enter()?;
                let (et, n) = self.list_header()?;
                for _ in 0..n {
                    self.skip_value(et)?;
                }
                self.depth -= 1;
                Ok(())
            }
            MAP => {
                self.enter()?;
                let n = self.varint()?;
                if n > 0 {
                    let kv = self.byte()?;
                    for _ in 0..n {
                        self.skip_value(kv >> 4)?;
                        self.skip_value(kv & 0x0f)?;
                    }
                }
                self.depth -= 1;
                Ok(())
            }
            STRUCT => self.read_struct(|_, _, _| Ok(false)),
            _ => Err(format!("thrift: unknown type {ty}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_varints() {
        // 0, -1, 1, -2, 300 (600 = 0xd8 0x04), i64::MIN.
        let mut c = Compact::new(&[0, 1, 2, 3, 0xd8, 0x04]);
        let got: Vec<i64> = (0..5).map(|_| c.i64().unwrap()).collect();
        assert_eq!(got, [0, -1, 1, -2, 300]);
        let mut min = vec![0xff; 9];
        min.push(0x01);
        assert_eq!(Compact::new(&min).i64().unwrap(), i64::MIN);
        assert!(Compact::new(&[0x80; 11]).varint().is_err());
    }

    #[test]
    fn struct_fields_short_and_long_ids_and_skips() {
        // Field 1 i32 = 7; field 3 (delta 2) binary "ab"; field 20 by a
        // long header, bool true; field 21 a nested struct with a list of
        // two i32 and a map, skipped; stop.
        let bytes = [
            0x15, 14, // 1: i32 7
            0x28, 2, b'a', b'b', // 3: binary
            0x01, 40,   // 20: bool true, long form id zigzag(20) = 40
            0x1c, // 21: struct
            0x19, 0x25, 2, 4, // 1: list<i32> [1, 2]
            0x1b, 1, 0x58, 0, 0,    // 2: map<i32, binary> {0: ""}
            0x00, // the nested struct's stop
            0x00, // the stop
        ];
        let mut seen = Vec::new();
        let mut c = Compact::new(&bytes);
        c.read_struct(|c, id, ty| {
            seen.push(id);
            match id {
                1 => assert_eq!(c.i32()?, 7),
                3 => assert_eq!(c.binary()?, b"ab"),
                20 => assert!(Compact::bool_field(ty)?),
                _ => return Ok(false),
            }
            Ok(true)
        })
        .unwrap();
        assert_eq!(seen, [1, 3, 20, 21]);
        assert_eq!(c.pos(), bytes.len());
    }

    #[test]
    fn damage_is_an_error() {
        // A list claiming more elements than bytes left.
        assert!(Compact::new(&[0xf5, 0x7f]).list_header().is_err());
        // A struct with no stop byte.
        assert!(Compact::new(&[0x15, 2]).read_struct(|_, _, _| Ok(false)).is_err());
        // Structs nested past the limit.
        let deep = vec![0x1c; 200];
        let e = Compact::new(&deep).read_struct(|_, _, _| Ok(false)).unwrap_err();
        assert!(e.contains("nested"), "{e}");
        // A 10-byte varint whose 10th byte is 1 is u64::MAX; 2 overflows u64.
        let mut max = [0xffu8; 10];
        max[9] = 1;
        assert_eq!(Compact::new(&max).varint().unwrap(), u64::MAX);
        max[9] = 2;
        assert!(Compact::new(&max).varint().is_err());
    }

    #[test]
    fn running_short_is_told_apart() {
        let mut c = Compact::new(&[0x15, 2]);
        assert!(c.read_struct(|_, _, _| Ok(false)).is_err());
        assert!(c.ran_short());
        let mut c = Compact::new(&[0xf5, 0x7f]);
        assert!(c.list_header().is_err());
        assert!(c.ran_short());
        // Within a longer value, the same count fits.
        let mut c = Compact::new(&[0xf5, 0x7f]);
        assert_eq!(c.list_header_in(200).unwrap(), (5, 127));
        // Past that value's end it is damage.
        let mut c = Compact::new(&[0xf5, 0x7f]);
        assert!(c.list_header_in(100).is_err());
        assert!(!c.ran_short());
        // A field id that overflows is damage, with bytes to spare.
        let mut c = Compact::new(&[0x25, 2]);
        let mut last = i16::MAX - 1;
        assert!(c.field_header(&mut last).is_err());
        assert!(!c.ran_short());
    }
}
