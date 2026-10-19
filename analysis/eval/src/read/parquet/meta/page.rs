//! Page headers (`PageHeader` and `PageKind`).

use super::{Encoding, usize_count};
use crate::Result;
use crate::read::parquet::thrift::Compact;

/// A page's size, which `parquet.thrift` declares an i32, so at most
/// 2^31 - 1 bytes.
fn page_size(what: &str, v: i64) -> Result<usize> {
    match i32::try_from(v) {
        Ok(n) if n >= 0 => Ok(n as usize),
        _ => Err(format!("{what} is {v}, outside 0 to {}", i32::MAX)),
    }
}

/// A page header's kind and the header of that kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageKind {
    Data {
        num_values: usize,
        encoding: Encoding,
        def_encoding: Encoding,
    },
    Dictionary {
        num_values: usize,
        encoding: Encoding,
    },
    DataV2 {
        num_values: usize,
        encoding: Encoding,
        def_len: usize,
        rep_len: usize,
        compressed: bool,
    },
    /// An index page, or a kind this reader does not know; stepped over.
    Skipped,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageHeader {
    pub kind: PageKind,
    pub uncompressed_size: usize,
    pub compressed_size: usize,
    /// The header's own length in bytes.
    pub header_len: usize,
}

impl PageHeader {
    /// The page header at the start of `buf`.
    pub fn read(buf: &[u8]) -> Result<PageHeader> {
        let mut c = Compact::new(buf);
        let (mut kind, mut unc, mut comp) = (None, None, None);
        let (mut data, mut dict, mut v2) = (None, None, None);
        c.read_struct(|c, id, _| {
            match id {
                1 => kind = Some(c.i32()?),
                2 => unc = Some(page_size("a page's uncompressed size", c.i64()?)?),
                3 => comp = Some(page_size("a page's compressed size", c.i64()?)?),
                5 => data = Some(data_page(c)?),
                7 => dict = Some(dict_page(c)?),
                8 => v2 = Some(data_page_v2(c)?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        let missing = |what: &str| format!("a page header has no {what}");
        let kind = match kind.ok_or_else(|| missing("type"))? {
            0 => data.ok_or_else(|| missing("data page header"))?,
            2 => dict.ok_or_else(|| missing("dictionary page header"))?,
            3 => v2.ok_or_else(|| missing("data page v2 header"))?,
            _ => PageKind::Skipped,
        };
        Ok(PageHeader {
            kind,
            uncompressed_size: unc.ok_or_else(|| missing("size"))?,
            compressed_size: comp.ok_or_else(|| missing("compressed size"))?,
            header_len: c.pos(),
        })
    }
}

fn data_page(c: &mut Compact) -> Result<PageKind> {
    let (mut n, mut enc, mut def) = (None, None, None);
    c.read_struct(|c, id, _| {
        match id {
            1 => n = Some(usize_count("a page's value count", c.i64()?)?),
            2 => enc = Some(Encoding::from(c.i32()?)),
            3 => def = Some(Encoding::from(c.i32()?)),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(PageKind::Data {
        num_values: n.ok_or("a data page has no value count")?,
        encoding: enc.ok_or("a data page has no encoding")?,
        def_encoding: def.ok_or("a data page has no level encoding")?,
    })
}

fn dict_page(c: &mut Compact) -> Result<PageKind> {
    let (mut n, mut enc) = (None, None);
    c.read_struct(|c, id, _| {
        match id {
            1 => n = Some(usize_count("a dictionary's value count", c.i64()?)?),
            2 => enc = Some(Encoding::from(c.i32()?)),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(PageKind::Dictionary {
        num_values: n.ok_or("a dictionary page has no value count")?,
        encoding: enc.ok_or("a dictionary page has no encoding")?,
    })
}

fn data_page_v2(c: &mut Compact) -> Result<PageKind> {
    let (mut n, mut enc, mut def_len, mut rep_len) = (None, None, None, None);
    // The format's default.
    let mut compressed = true;
    c.read_struct(|c, id, ty| {
        match id {
            1 => n = Some(usize_count("a page's value count", c.i64()?)?),
            4 => enc = Some(Encoding::from(c.i32()?)),
            5 => def_len = Some(usize_count("a page's level length", c.i64()?)?),
            6 => rep_len = Some(usize_count("a page's level length", c.i64()?)?),
            7 => compressed = Compact::bool_field(ty)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(PageKind::DataV2 {
        num_values: n.ok_or("a data page has no value count")?,
        encoding: enc.ok_or("a data page has no encoding")?,
        def_len: def_len.ok_or("a data page has no level length")?,
        rep_len: rep_len.ok_or("a data page has no level length")?,
        compressed,
    })
}
