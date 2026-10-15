//! The parts of Parquet's metadata the reader uses, as `parquet.thrift`
//! numbers their fields: the footer's schema and row groups, each column
//! chunk's metadata, and page headers. Every other field is stepped over.

use std::fmt;

use super::thrift::Compact;
use crate::Result;

/// A column's physical type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Physical {
    Boolean,
    Int32,
    Int64,
    Int96,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray,
    Other(i32),
}

impl Physical {
    fn from(v: i32) -> Physical {
        match v {
            0 => Physical::Boolean,
            1 => Physical::Int32,
            2 => Physical::Int64,
            3 => Physical::Int96,
            4 => Physical::Float,
            5 => Physical::Double,
            6 => Physical::ByteArray,
            7 => Physical::FixedLenByteArray,
            v => Physical::Other(v),
        }
    }
}

impl fmt::Display for Physical {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Physical::Boolean => f.write_str("BOOLEAN"),
            Physical::Int32 => f.write_str("INT32"),
            Physical::Int64 => f.write_str("INT64"),
            Physical::Int96 => f.write_str("INT96"),
            Physical::Float => f.write_str("FLOAT"),
            Physical::Double => f.write_str("DOUBLE"),
            Physical::ByteArray => f.write_str("BYTE_ARRAY"),
            Physical::FixedLenByteArray => f.write_str("FIXED_LEN_BYTE_ARRAY"),
            Physical::Other(v) => write!(f, "type {v}"),
        }
    }
}

/// A schema element's repetition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Repetition {
    Required,
    Optional,
    Repeated,
}

/// A column chunk's compression codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Uncompressed,
    Snappy,
    Gzip,
    Other(i32),
}

impl fmt::Display for Codec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Codec::Uncompressed => "UNCOMPRESSED",
            Codec::Snappy => "SNAPPY",
            Codec::Gzip => "GZIP",
            Codec::Other(3) => "LZO",
            Codec::Other(4) => "BROTLI",
            Codec::Other(5) => "LZ4",
            Codec::Other(6) => "ZSTD",
            Codec::Other(7) => "LZ4_RAW",
            Codec::Other(v) => return write!(f, "codec {v}"),
        };
        f.write_str(name)
    }
}

/// A page's value or level encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Plain,
    /// Written by older writers for dictionary pages and the pages that
    /// index into them; read as `Plain` and `RleDictionary` are.
    PlainDictionary,
    Rle,
    RleDictionary,
    Other(i32),
}

impl Encoding {
    fn from(v: i32) -> Encoding {
        match v {
            0 => Encoding::Plain,
            2 => Encoding::PlainDictionary,
            3 => Encoding::Rle,
            8 => Encoding::RleDictionary,
            v => Encoding::Other(v),
        }
    }
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Encoding::Plain => "PLAIN",
            Encoding::PlainDictionary => "PLAIN_DICTIONARY",
            Encoding::Rle => "RLE",
            Encoding::RleDictionary => "RLE_DICTIONARY",
            Encoding::Other(4) => "BIT_PACKED",
            Encoding::Other(5) => "DELTA_BINARY_PACKED",
            Encoding::Other(6) => "DELTA_LENGTH_BYTE_ARRAY",
            Encoding::Other(7) => "DELTA_BYTE_ARRAY",
            Encoding::Other(9) => "BYTE_STREAM_SPLIT",
            Encoding::Other(v) => return write!(f, "encoding {v}"),
        };
        f.write_str(name)
    }
}

/// One element of the flattened schema tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaElement {
    pub name: String,
    /// `None` for a group.
    pub physical: Option<Physical>,
    /// `None` for the root.
    pub repetition: Option<Repetition>,
    pub num_children: usize,
}

impl SchemaElement {
    /// A group has children, or no physical type; every other element is
    /// a leaf, which is a column.
    pub fn is_group(&self) -> bool {
        self.num_children > 0 || self.physical.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    pub physical: Physical,
    pub path: Vec<String>,
    pub codec: Codec,
    /// Values in the chunk, nulls included.
    pub num_values: u64,
    pub total_compressed_size: u64,
    pub data_page_offset: u64,
    pub dictionary_page_offset: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowGroup {
    pub num_rows: u64,
    pub columns: Vec<ColumnMeta>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMeta {
    pub num_rows: u64,
    pub schema: Vec<SchemaElement>,
    pub row_groups: Vec<RowGroup>,
}

fn count(what: &str, v: i64) -> Result<u64> {
    u64::try_from(v).map_err(|_| format!("{what} is {v}"))
}

fn usize_count(what: &str, v: i64) -> Result<usize> {
    usize::try_from(v).map_err(|_| format!("{what} is {v}"))
}

impl FileMeta {
    /// The footer's `FileMetaData`.
    pub fn read(buf: &[u8]) -> Result<FileMeta> {
        let mut c = Compact::new(buf);
        let (mut num_rows, mut schema, mut row_groups) = (None, None, None);
        c.read_struct(|c, id, _| {
            match id {
                2 => schema = Some(c.list(|c, _| schema_element(c))?),
                3 => num_rows = Some(count("the row count", c.i64()?)?),
                4 => row_groups = Some(c.list(|c, _| row_group(c))?),
                _ => return Ok(false),
            }
            Ok(true)
        })?;
        let schema: Vec<SchemaElement> = schema.ok_or("the footer has no schema")?;
        // Every child is an element of its own, so no element has as
        // many children as the schema has elements.
        if let Some(e) = schema.iter().find(|e| e.num_children >= schema.len()) {
            return Err(format!(
                "schema element {} has {} children in a schema of {}",
                e.name,
                e.num_children,
                schema.len()
            ));
        }
        Ok(FileMeta {
            num_rows: num_rows.ok_or("the footer has no row count")?,
            schema,
            row_groups: row_groups.ok_or("the footer has no row groups")?,
        })
    }
}

fn schema_element(c: &mut Compact) -> Result<SchemaElement> {
    let mut e = SchemaElement { name: String::new(), physical: None, repetition: None, num_children: 0 };
    let mut named = false;
    c.read_struct(|c, id, _| {
        match id {
            1 => e.physical = Some(Physical::from(c.i32()?)),
            3 => {
                e.repetition = Some(match c.i32()? {
                    0 => Repetition::Required,
                    1 => Repetition::Optional,
                    2 => Repetition::Repeated,
                    v => return Err(format!("repetition {v}")),
                })
            }
            4 => {
                e.name = c.string()?;
                named = true;
            }
            5 => e.num_children = usize_count("a child count", c.i64()?)?,
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    if !named {
        return Err("a schema element has no name".into());
    }
    Ok(e)
}

fn row_group(c: &mut Compact) -> Result<RowGroup> {
    let (mut columns, mut num_rows) = (None, None);
    c.read_struct(|c, id, _| {
        match id {
            1 => columns = Some(c.list(|c, _| column_chunk(c))?),
            3 => num_rows = Some(count("a row group's row count", c.i64()?)?),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(RowGroup {
        num_rows: num_rows.ok_or("a row group has no row count")?,
        columns: columns.ok_or("a row group has no columns")?,
    })
}

fn column_chunk(c: &mut Compact) -> Result<ColumnMeta> {
    let mut meta = None;
    let mut elsewhere = false;
    c.read_struct(|c, id, _| {
        match id {
            1 => elsewhere = !c.binary()?.is_empty(),
            3 => meta = Some(column_meta(c)?),
            // crypto_metadata and encrypted_column_metadata.
            8 | 9 => return Err("an encrypted column is not read".into()),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    if elsewhere {
        return Err("a column chunk is in another file".into());
    }
    meta.ok_or_else(|| "a column chunk has no metadata".into())
}

fn column_meta(c: &mut Compact) -> Result<ColumnMeta> {
    let (mut physical, mut path, mut codec, mut num_values) = (None, None, None, None);
    let (mut size, mut data, mut dict) = (None, None, None);
    c.read_struct(|c, id, _| {
        match id {
            1 => physical = Some(Physical::from(c.i32()?)),
            3 => path = Some(c.list(|c, _| c.string())?),
            4 => {
                codec = Some(match c.i32()? {
                    0 => Codec::Uncompressed,
                    1 => Codec::Snappy,
                    2 => Codec::Gzip,
                    v => Codec::Other(v),
                })
            }
            5 => num_values = Some(count("a column's value count", c.i64()?)?),
            7 => size = Some(count("a column's compressed size", c.i64()?)?),
            9 => data = Some(count("a data page offset", c.i64()?)?),
            11 => dict = Some(count("a dictionary page offset", c.i64()?)?),
            _ => return Ok(false),
        }
        Ok(true)
    })?;
    Ok(ColumnMeta {
        physical: physical.ok_or("a column has no type")?,
        path: path.ok_or("a column has no path")?,
        codec: codec.ok_or("a column has no codec")?,
        num_values: num_values.ok_or("a column has no value count")?,
        total_compressed_size: size.ok_or("a column has no size")?,
        data_page_offset: data.ok_or("a column has no data page offset")?,
        dictionary_page_offset: dict,
    })
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
                2 => unc = Some(usize_count("a page's size", c.i64()?)?),
                3 => comp = Some(usize_count("a page's compressed size", c.i64()?)?),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_child_count_past_the_schema_is_refused() {
        // Field 2, a list of one schema element named "schema" with 5
        // children; field 3, 0 rows; field 4, no row groups.
        let mut b = vec![0x29, 0x1c, 0x48, 6];
        b.extend(b"schema");
        b.extend([0x15, 10, 0x00, 0x16, 0, 0x19, 0x0c, 0x00]);
        assert_eq!(FileMeta::read(&b).unwrap_err(), "schema element schema has 5 children in a schema of 1");
        // With no children it reads.
        let at = b.iter().position(|&x| x == 10).unwrap();
        b[at] = 0;
        assert_eq!(FileMeta::read(&b).unwrap().schema[0].num_children, 0);
    }

    #[test]
    fn an_encrypted_column_is_refused() {
        // A column chunk with field 8 (crypto_metadata), an empty struct.
        let e = column_chunk(&mut Compact::new(&[0x8c, 0x00, 0x00])).unwrap_err();
        assert_eq!(e, "an encrypted column is not read");
    }
}
