//! Decoding one column chunk's pages into values: dictionary pages, data
//! pages of both versions, definition levels in the RLE and bit packed
//! hybrid, and values PLAIN or as dictionary indices.

use std::io::Read;

use flate2::read::MultiGzDecoder;

use super::meta::{Codec, ColumnMeta, Encoding, PageHeader, PageKind, Physical};
use super::snappy;
use crate::Result;

/// The values of one column, row by row, nulls included.
pub struct Values {
    pub kind: Kind,
    /// Whether each row is null.
    pub nulls: Vec<bool>,
}

pub enum Kind {
    /// BYTE_ARRAY values one after another in `data`; value `i` ends at
    /// `ends[i]` and starts where value `i - 1` ends. A null is empty.
    Bytes { data: Vec<u8>, ends: Vec<usize> },
    /// INT32 and INT64 values, as i64. A null is 0.
    Ints(Vec<i64>),
}

impl Values {
    pub fn new(physical: Physical) -> Result<Values> {
        let kind = match physical {
            Physical::ByteArray => Kind::Bytes { data: Vec::new(), ends: Vec::new() },
            Physical::Int32 | Physical::Int64 => Kind::Ints(Vec::new()),
            p => return Err(format!("{p} columns are not read")),
        };
        Ok(Values { kind, nulls: Vec::new() })
    }

    pub fn len(&self) -> usize {
        self.nulls.len()
    }

    fn push_null(&mut self) {
        match &mut self.kind {
            Kind::Bytes { data, ends } => ends.push(data.len()),
            Kind::Ints(v) => v.push(0),
        }
        self.nulls.push(true);
    }

    /// Value `i` of a dictionary, appended.
    fn push_from(&mut self, dict: &Values, i: usize) -> Result<()> {
        if i >= dict.len() {
            return Err(format!("dictionary index {i} past the dictionary's {} values", dict.len()));
        }
        match (&mut self.kind, &dict.kind) {
            (Kind::Bytes { data, ends }, Kind::Bytes { data: d, ends: e }) => {
                let start = if i == 0 { 0 } else { e[i - 1] };
                data.extend_from_slice(&d[start..e[i]]);
                ends.push(data.len());
            }
            (Kind::Ints(v), Kind::Ints(d)) => v.push(d[i]),
            _ => unreachable!("a dictionary has its column's type"),
        }
        self.nulls.push(false);
        Ok(())
    }
}

/// PLAIN values of one physical type read one at a time.
struct Plain<'a> {
    physical: Physical,
    buf: &'a [u8],
    pos: usize,
}

impl Plain<'_> {
    fn push_next(&mut self, out: &mut Values) -> Result<()> {
        let need = |n: usize, pos: usize, len: usize| {
            if pos + n > len { Err("PLAIN values end before the page's value count".to_owned()) } else { Ok(()) }
        };
        match (self.physical, &mut out.kind) {
            (Physical::ByteArray, Kind::Bytes { data, ends }) => {
                need(4, self.pos, self.buf.len())?;
                let b = &self.buf[self.pos..self.pos + 4];
                let n = u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize;
                self.pos += 4;
                need(n, self.pos, self.buf.len())?;
                data.extend_from_slice(&self.buf[self.pos..self.pos + n]);
                ends.push(data.len());
                self.pos += n;
            }
            (Physical::Int32, Kind::Ints(v)) => {
                need(4, self.pos, self.buf.len())?;
                let b = &self.buf[self.pos..self.pos + 4];
                v.push(i64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])));
                self.pos += 4;
            }
            (Physical::Int64, Kind::Ints(v)) => {
                need(8, self.pos, self.buf.len())?;
                let mut b = [0u8; 8];
                b.copy_from_slice(&self.buf[self.pos..self.pos + 8]);
                v.push(i64::from_le_bytes(b));
                self.pos += 8;
            }
            _ => unreachable!("values are made for their column's type"),
        }
        out.nulls.push(false);
        Ok(())
    }
}

/// `count` values of the RLE and bit packed hybrid, each `width` bits.
pub fn rle_hybrid(buf: &[u8], width: u32, count: usize) -> Result<Vec<u32>> {
    if width > 32 {
        return Err(format!("a bit width of {width}"));
    }
    let mut out = Vec::with_capacity(count);
    let mut pos = 0;
    let bytes = width.div_ceil(8) as usize;
    while out.len() < count {
        let mut header = 0u64;
        for shift in (0..35).step_by(7) {
            let b = *buf.get(pos).ok_or("levels or indices end early")?;
            pos += 1;
            header |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                break;
            }
        }
        if header & 1 == 0 {
            let run = (header >> 1) as usize;
            let b = buf.get(pos..pos + bytes).ok_or("levels or indices end early")?;
            pos += bytes;
            let v = b.iter().rev().fold(0u32, |a, &x| a << 8 | u32::from(x));
            out.extend(std::iter::repeat_n(v, run.min(count - out.len())));
        } else {
            let groups = (header >> 1) as usize;
            let len = groups * width as usize;
            let packed = buf.get(pos..pos + len).ok_or("levels or indices end early")?;
            pos += len;
            let mask = if width == 32 { u32::MAX } else { (1u32 << width) - 1 };
            let total = (groups * 8).min(count - out.len());
            for i in 0..total {
                let bit = i * width as usize;
                let mut v = 0u64;
                for (k, &b) in packed[bit / 8..packed.len().min(bit / 8 + 5)].iter().enumerate() {
                    v |= u64::from(b) << (8 * k);
                }
                out.push((v >> (bit % 8)) as u32 & mask);
            }
        }
    }
    Ok(out)
}

fn decompress(codec: Codec, src: &[u8], want: usize) -> Result<Vec<u8>> {
    let out = match codec {
        Codec::Uncompressed => src.to_vec(),
        Codec::Snappy => snappy::decompress(src, want)?,
        Codec::Gzip => {
            let mut out = Vec::with_capacity(want);
            MultiGzDecoder::new(src).read_to_end(&mut out).map_err(|e| format!("gzip: {e}"))?;
            out
        }
        c => return Err(format!("the {c} codec is not read; SNAPPY, GZIP and UNCOMPRESSED are")),
    };
    if out.len() != want {
        return Err(format!("a page expands to {} bytes, its header says {want}", out.len()));
    }
    Ok(out)
}

/// Appends the values of the column chunk `meta` in `file` to `out`.
/// `optional` says whether the column has definition levels (a top level
/// optional column, maximum level 1).
pub fn read_chunk(file: &[u8], meta: &ColumnMeta, optional: bool, out: &mut Values) -> Result<()> {
    let start = match meta.dictionary_page_offset {
        Some(d) if d > 0 && d < meta.data_page_offset => d,
        _ => meta.data_page_offset,
    };
    let end = start.checked_add(meta.total_compressed_size).filter(|&e| e <= file.len() as u64);
    let end = end.ok_or("the column chunk runs past the end of the file")? as usize;
    let mut pos = start as usize;
    let mut dict: Option<Values> = None;
    let mut seen: u64 = 0;
    while seen < meta.num_values {
        if pos >= end {
            return Err(format!("the column chunk ends after {seen} of {} values", meta.num_values));
        }
        let at = pos;
        let page = |e: String| format!("page at byte {at}: {e}");
        let h = PageHeader::read(&file[pos..end]).map_err(page)?;
        let body_start = pos + h.header_len;
        let body =
            file.get(body_start..body_start + h.compressed_size).filter(|_| body_start + h.compressed_size <= end);
        let body = body.ok_or_else(|| page("runs past its column chunk".into()))?;
        pos = body_start + h.compressed_size;
        match h.kind {
            PageKind::Dictionary { num_values, encoding } => {
                if !matches!(encoding, Encoding::Plain | Encoding::PlainDictionary) {
                    return Err(page(format!("a dictionary in {encoding}")));
                }
                let raw = decompress(meta.codec, body, h.uncompressed_size).map_err(page)?;
                let mut d = Values::new(meta.physical)?;
                let mut plain = Plain { physical: meta.physical, buf: &raw, pos: 0 };
                for _ in 0..num_values {
                    plain.push_next(&mut d).map_err(page)?;
                }
                dict = Some(d);
            }
            PageKind::Data { num_values, encoding, def_encoding } => {
                let raw = decompress(meta.codec, body, h.uncompressed_size).map_err(page)?;
                let (defs, values) = if optional {
                    if def_encoding != Encoding::Rle {
                        return Err(page(format!("definition levels in {def_encoding}")));
                    }
                    let n = raw.get(..4).ok_or_else(|| page("no definition levels".into()))?;
                    let n = u32::from_le_bytes([n[0], n[1], n[2], n[3]]) as usize;
                    let levels = raw.get(4..4 + n).ok_or_else(|| page("definition levels run past the page".into()))?;
                    (Some(rle_hybrid(levels, 1, num_values).map_err(page)?), &raw[4 + n..])
                } else {
                    (None, &raw[..])
                };
                page_values(meta.physical, encoding, defs, num_values, values, dict.as_ref(), out).map_err(page)?;
                seen += num_values as u64;
            }
            PageKind::DataV2 { num_values, encoding, def_len, rep_len, compressed } => {
                if rep_len != 0 {
                    return Err(page("repetition levels in a flat column".into()));
                }
                let levels = body.get(..def_len).ok_or_else(|| page("definition levels run past the page".into()))?;
                let defs = if optional { Some(rle_hybrid(levels, 1, num_values).map_err(page)?) } else { None };
                let rest = &body[def_len..];
                let want =
                    h.uncompressed_size.checked_sub(def_len).ok_or_else(|| page("levels past its size".into()))?;
                let raw = if compressed { decompress(meta.codec, rest, want).map_err(page)? } else { rest.to_vec() };
                page_values(meta.physical, encoding, defs, num_values, &raw, dict.as_ref(), out).map_err(page)?;
                seen += num_values as u64;
            }
            PageKind::Skipped => {}
        }
    }
    if seen != meta.num_values {
        return Err(format!("the column chunk holds {seen} values, its metadata says {}", meta.num_values));
    }
    Ok(())
}

/// Appends one data page's values, a null wherever `defs` has level 0.
fn page_values(
    physical: Physical,
    encoding: Encoding,
    defs: Option<Vec<u32>>,
    num_values: usize,
    buf: &[u8],
    dict: Option<&Values>,
    out: &mut Values,
) -> Result<()> {
    let present = defs.as_ref().map_or(num_values, |d| d.iter().filter(|&&l| l == 1).count());
    let is_present = |i: usize| defs.as_ref().is_none_or(|d| d[i] == 1);
    match encoding {
        Encoding::Plain => {
            let mut plain = Plain { physical, buf, pos: 0 };
            for i in 0..num_values {
                if is_present(i) { plain.push_next(out)? } else { out.push_null() }
            }
        }
        Encoding::RleDictionary | Encoding::PlainDictionary => {
            let dict = dict.ok_or("dictionary indices before a dictionary page")?;
            let width = u32::from(*buf.first().ok_or("no bit width")?);
            let idx = rle_hybrid(&buf[1..], width, present)?;
            let mut next = idx.iter();
            for i in 0..num_values {
                if is_present(i) {
                    let &k = next.next().ok_or("fewer indices than values")?;
                    out.push_from(dict, k as usize)?;
                } else {
                    out.push_null();
                }
            }
        }
        e => return Err(format!("values in {e}; PLAIN and dictionary pages are read")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_runs_and_packed_groups() {
        // A run of 3 ones (header 6), then one bit packed group of eight
        // 2-bit values 0..=3,0..=3 (header 3): 0b11_10_01_00 twice.
        let buf = [6, 1, 3, 0b1110_0100, 0b1110_0100];
        assert_eq!(rle_hybrid(&buf, 2, 11).unwrap(), [1, 1, 1, 0, 1, 2, 3, 0, 1, 2, 3]);
        // A packed group is cut at the count asked for.
        assert_eq!(rle_hybrid(&buf, 2, 5).unwrap(), [1, 1, 1, 0, 1]);
        // Width 0: every value is 0 and a run has no value bytes.
        assert_eq!(rle_hybrid(&[8], 0, 4).unwrap(), [0, 0, 0, 0]);
        assert!(rle_hybrid(&[6], 2, 3).is_err());
    }

    #[test]
    fn wide_packed_values() {
        // One group of eight 17-bit values, each 0x1_0001 + i.
        let vals: Vec<u64> = (0..8).map(|i| 0x1_0001 + i).collect();
        let mut packed = vec![0u8; 17];
        for (i, v) in vals.iter().enumerate() {
            for b in 0..17 {
                if v >> b & 1 == 1 {
                    let bit = i * 17 + b;
                    packed[bit / 8] |= 1 << (bit % 8);
                }
            }
        }
        let mut buf = vec![3];
        buf.extend(packed);
        let got = rle_hybrid(&buf, 17, 8).unwrap();
        assert_eq!(got.iter().map(|&v| u64::from(v)).collect::<Vec<_>>(), vals);
    }
}
