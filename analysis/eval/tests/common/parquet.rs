//! A Parquet writer for tests: one row group, required columns, one
//! uncompressed PLAIN data page per column. It lets a test write exactly
//! the rows it needs, broken ones included; the files in
//! `tests/data/parquet` cover what pyarrow writes.

use std::fs;
use std::path::Path;

/// One column: its name and every row's value.
pub enum Col<'a> {
    Str(&'a str, Vec<&'a str>),
    Int(&'a str, Vec<i64>),
}

impl Col<'_> {
    fn name(&self) -> &str {
        match self {
            Col::Str(n, _) | Col::Int(n, _) => n,
        }
    }

    fn rows(&self) -> usize {
        match self {
            Col::Str(_, v) => v.len(),
            Col::Int(_, v) => v.len(),
        }
    }

    /// Parquet's physical type: BYTE_ARRAY or INT64.
    fn physical(&self) -> i64 {
        match self {
            Col::Str(..) => 6,
            Col::Int(..) => 2,
        }
    }

    fn plain(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Col::Str(_, v) => {
                for s in v {
                    out.extend((s.len() as u32).to_le_bytes());
                    out.extend(s.as_bytes());
                }
            }
            Col::Int(_, v) => v.iter().for_each(|i| out.extend(i.to_le_bytes())),
        }
        out
    }
}

/// Thrift's compact protocol, as far as the writer needs it.
#[derive(Default)]
struct Thrift {
    out: Vec<u8>,
    last: Vec<i16>,
}

impl Thrift {
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.out.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.out.push(v as u8);
    }

    fn zigzag(&mut self, v: i64) {
        self.varint(((v << 1) ^ (v >> 63)) as u64);
    }

    fn field(&mut self, id: i16, ty: u8) {
        let last = self.last.last_mut().unwrap();
        assert!(id > *last && id - *last <= 15);
        self.out.push(((id - *last) as u8) << 4 | ty);
        *last = id;
    }

    fn int(&mut self, id: i16, v: i64) {
        self.field(id, 6);
        self.zigzag(v);
    }

    fn i32(&mut self, id: i16, v: i64) {
        self.field(id, 5);
        self.zigzag(v);
    }

    fn bytes(&mut self, id: i16, b: &[u8]) {
        self.field(id, 8);
        self.varint(b.len() as u64);
        self.out.extend(b);
    }

    fn begin(&mut self) {
        self.last.push(0);
    }

    fn begin_field(&mut self, id: i16) {
        self.field(id, 12);
        self.begin();
    }

    fn end(&mut self) {
        self.out.push(0);
        self.last.pop();
    }

    fn list(&mut self, id: i16, ty: u8, n: usize) {
        assert!(n < 15);
        self.field(id, 9);
        self.out.push((n as u8) << 4 | ty);
    }
}

/// Writes `cols`, which must all have the same number of rows, to `path`.
pub fn write(path: &Path, cols: &[Col]) {
    write_chunks(path, cols, |offset, len| (offset as i64, len as i64));
}

/// As [`write`], with each column chunk's data page offset and compressed
/// size in the footer given by `chunk` from its true offset and length.
pub fn write_chunks(path: &Path, cols: &[Col], chunk: impl Fn(usize, usize) -> (i64, i64)) {
    let rows = cols[0].rows();
    assert!(cols.iter().all(|c| c.rows() == rows));
    let mut file = b"PAR1".to_vec();
    let mut chunks = Vec::new();
    for c in cols {
        let values = c.plain();
        let mut h = Thrift::default();
        h.begin();
        h.i32(1, 0); // DATA_PAGE
        h.i32(2, values.len() as i64);
        h.i32(3, values.len() as i64);
        h.begin_field(5);
        h.i32(1, rows as i64);
        h.i32(2, 0); // PLAIN
        h.i32(3, 3); // RLE
        h.i32(4, 3);
        h.end();
        h.end();
        let offset = file.len();
        file.extend(&h.out);
        file.extend(&values);
        chunks.push((offset, file.len() - offset));
    }
    let mut m = Thrift::default();
    m.begin();
    m.i32(1, 1);
    m.list(2, 12, cols.len() + 1);
    m.begin();
    m.bytes(4, b"schema");
    m.i32(5, cols.len() as i64);
    m.end();
    for c in cols {
        m.begin();
        m.i32(1, c.physical());
        m.i32(3, 0); // REQUIRED
        m.bytes(4, c.name().as_bytes());
        m.end();
    }
    m.int(3, rows as i64);
    m.list(4, 12, 1);
    m.begin();
    m.list(1, 12, cols.len());
    for (c, &(offset, len)) in cols.iter().zip(&chunks) {
        m.begin();
        m.int(2, offset as i64);
        m.begin_field(3);
        m.i32(1, c.physical());
        m.list(2, 5, 1);
        m.zigzag(0);
        m.list(3, 8, 1);
        m.varint(c.name().len() as u64);
        m.out.extend(c.name().as_bytes());
        m.i32(4, 0); // UNCOMPRESSED
        m.int(5, rows as i64);
        let (at, compressed) = chunk(offset, len);
        m.int(6, len as i64);
        m.int(7, compressed);
        m.int(9, at);
        m.end();
        m.end();
    }
    m.int(2, chunks.iter().map(|c| c.1 as i64).sum());
    m.int(3, rows as i64);
    m.end();
    m.end();
    file.extend(&m.out);
    file.extend((m.out.len() as u32).to_le_bytes());
    file.extend(b"PAR1");
    fs::write(path, file).unwrap();
}
