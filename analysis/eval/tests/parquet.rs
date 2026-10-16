//! The Parquet reader on files pyarrow wrote, and on files the test
//! writer in `common/parquet.rs` writes.
//!
//! The files under `tests/data/parquet` and the `.parquet` samples of
//! GutenQA and ConTEB were written with pyarrow 26.0.0 by this script,
//! run from `tests/data`:
//!
//! ```text
//! import pyarrow as pa, pyarrow.parquet as pq
//! names = ["alpha", "beta", "alpha", "gamma", "beta", "alpha", "delta", "alpha", "beta", "epsilon"]
//! texts = ["row %d: %s" % (i, "Zürich " * i) for i in range(10)]
//! texts[3] = None
//! table = pa.table({
//!     "name": pa.array(names, pa.string()),
//!     "id": pa.array([i * 1000003 - 5 for i in range(10)], pa.int64()),
//!     "text": pa.array(texts, pa.string()),
//!     "small": pa.array([-i for i in range(10)], pa.int32()),
//! })
//! small = dict(row_group_size=4, data_page_size=40, dictionary_pagesize_limit=24, write_batch_size=2)
//! pq.write_table(table, "parquet/snappy-dict-v1.parquet", compression="snappy", **small)
//! pq.write_table(table, "parquet/gzip-dict-v2.parquet", compression="gzip", data_page_version="2.0", **small)
//! required = pa.schema([pa.field(f.name, f.type, nullable=False) for f in table.schema])
//! req = table.set_column(2, "text", pa.array([t or "" for t in texts], pa.string())).cast(required)
//! pq.write_table(req, "parquet/plain-uncompressed.parquet", compression="none", use_dictionary=False)
//! pq.write_table(table.slice(0, 2), "parquet/zstd.parquet", compression="zstd")
//! # The samples: each table's rows, written with pyarrow's defaults
//! # (SNAPPY, dictionary pages, data page version 1), as the published
//! # sets are. Columns a row lacks are null.
//! book = "A_Made_Up_Tale_-_Nobody"
//! other = "Another_Tale_-_Nobody"
//! samples = {
//!     "gutenqa/gutenqa_chunks": [
//!         {"Book Name": book, "Book ID": 0, "Chunk ID": 1, "Chapter": "CHAPTER II.",
//!          "Chunk": "Señor Ruiz rowed across the lake at dawn."},
//!         {"Book Name": book, "Book ID": 0, "Chunk ID": 0, "Chapter": "CHAPTER I.",
//!          "Chunk": "It was a cold night in the village."},
//!         {"Book Name": other, "Book ID": 1, "Chunk ID": 0, "Chapter": "CHAPTER I.",
//!          "Chunk": "The lighthouse keeper trimmed his lamp."},
//!     ],
//!     "gutenqa/questions": [
//!         {"Book Name": book, "Book ID": 0, "Chunk ID": 1, "Question": "Who rowed across the lake?",
//!          "Answer": "Señor Ruiz did.", "Chunk Must Contain": "Señor Ruiz rowed across the lake"},
//!         {"Book Name": other, "Book ID": 1, "Chunk ID": 0, "Question": "What did the keeper trim?",
//!          "Answer": "His lamp.", "Chunk Must Contain": "a passage the chunk does not have"},
//!     ],
//!     "conteb/documents/train-00000-of-00001": [
//!         {"chunk_id": "paperA_1", "chunk": "The trial enrolled 120 patients in Zürich.", "og_chunk": "unused"},
//!         {"chunk_id": "paperA_0", "chunk": "Remdesivir was tested against a placebo."},
//!         {"chunk_id": "paperB_0", "chunk": "Masks reduced transmission in households."},
//!     ],
//!     "conteb/queries/train-00000-of-00001": [
//!         {"chunk_id": "paperA_1", "query": "How many patients were enrolled?", "answer": "120"},
//!         {"chunk_id": "paperB_0", "query": "Did masks reduce transmission?", "answer": "yes"},
//!     ],
//! }
//! for rel, rows in samples.items():
//!     pq.write_table(pa.Table.from_pylist(rows), rel + ".parquet")
//! ```
//!
//! The small page and dictionary limits give three row groups, several
//! pages per column chunk, and dictionaries that fill up, after which
//! pyarrow writes the rest of the chunk PLAIN.

mod common;

use std::fs;

use common::parquet::{Col, write};
use common::{data, scratch};
use turbo_eval::read::parquet::{Leading, Parquet, footer_len};

fn names() -> Vec<&'static str> {
    vec!["alpha", "beta", "alpha", "gamma", "beta", "alpha", "delta", "alpha", "beta", "epsilon"]
}

fn texts() -> Vec<Option<String>> {
    (0..10).map(|i| if i == 3 { None } else { Some(format!("row {i}: {}", "Zürich ".repeat(i))) }).collect()
}

fn check_table(file: &str) {
    let bytes = fs::read(data("parquet").join(file)).unwrap();
    let t = Parquet::open(file, &bytes).unwrap();
    assert_eq!(t.rows(), 10);
    let name = t.strings("name").unwrap();
    assert_eq!((0..10).map(|r| name.at(r).unwrap()).collect::<Vec<_>>(), names());
    let id = t.ints("id").unwrap();
    assert_eq!(
        (0..10).map(|r| id.at(r).unwrap()).collect::<Vec<_>>(),
        (0..10).map(|i| i * 1_000_003 - 5).collect::<Vec<_>>()
    );
    let small = t.ints("small").unwrap();
    assert_eq!((0..10).map(|r| small.at(r).unwrap()).collect::<Vec<_>>(), (0..10).map(|i| -i).collect::<Vec<_>>());
    let text = t.strings("text").unwrap();
    let want = texts();
    for (r, w) in want.iter().enumerate() {
        if file.starts_with("plain") {
            assert_eq!(text.at(r).unwrap(), w.as_deref().unwrap_or(""), "{file} row {r}");
        } else {
            assert_eq!(text.get(r), w.as_deref(), "{file} row {r}");
        }
    }
}

#[test]
fn reads_snappy_dictionary_pages_v1() {
    check_table("snappy-dict-v1.parquet");
}

#[test]
fn reads_gzip_dictionary_pages_v2() {
    check_table("gzip-dict-v2.parquet");
}

#[test]
fn reads_plain_uncompressed_required_columns() {
    check_table("plain-uncompressed.parquet");
}

#[test]
fn a_null_is_an_error_where_a_value_is_needed() {
    let bytes = fs::read(data("parquet").join("snappy-dict-v1.parquet")).unwrap();
    let t = Parquet::open("t.parquet", &bytes).unwrap();
    assert_eq!(t.strings("text").unwrap().at(3).unwrap_err(), "t.parquet: column text row 3 is null");
}

#[test]
fn refusals_name_what_is_not_read() {
    let bytes = fs::read(data("parquet").join("zstd.parquet")).unwrap();
    let t = Parquet::open("z.parquet", &bytes).unwrap();
    let e = t.strings("name").unwrap_err();
    assert!(e.starts_with("z.parquet: column name: row group 0: page at byte 4: "), "{e}");
    assert!(e.ends_with("the ZSTD codec is not read; SNAPPY, GZIP and UNCOMPRESSED are"), "{e}");

    let bytes = fs::read(data("parquet").join("plain-uncompressed.parquet")).unwrap();
    let t = Parquet::open("p.parquet", &bytes).unwrap();
    assert_eq!(t.strings("nope").unwrap_err(), "p.parquet: no column nope");
    assert_eq!(t.ints("name").unwrap_err(), "p.parquet: column name is not an INT32 or INT64 column");
    assert_eq!(t.strings("id").unwrap_err(), "p.parquet: column id is not a BYTE_ARRAY column");

    assert_eq!(Parquet::open("x", b"PAR1 not really PAR1").err().unwrap(), "x: footer length 544828524");
    assert_eq!(Parquet::open("x", b"{\"a\": 1}").err().unwrap(), "x: not a Parquet file");
}

#[test]
fn the_test_writer_round_trips() {
    let dir = scratch("parquet-writer");
    let path = dir.join("w.parquet");
    write(&path, &[Col::Str("s", vec!["", "señor", "x"]), Col::Int("n", vec![i64::MIN, 0, i64::MAX])]);
    let bytes = fs::read(&path).unwrap();
    let t = Parquet::open("w.parquet", &bytes).unwrap();
    let s = t.strings("s").unwrap();
    assert_eq!((0..3).map(|r| s.at(r).unwrap()).collect::<Vec<_>>(), ["", "señor", "x"]);
    let n = t.ints("n").unwrap();
    assert_eq!((0..3).map(|r| n.at(r).unwrap()).collect::<Vec<_>>(), [i64::MIN, 0, i64::MAX]);
    assert_eq!(n.at_u32(1), Ok(0));
    assert_eq!(n.at_u32(0).unwrap_err(), format!("w.parquet: column n row 0 is {}, not a u32", i64::MIN));
}

#[test]
fn damage_is_an_error() {
    let dir = scratch("parquet-damage");
    let path = dir.join("d.parquet");
    write(&path, &[Col::Str("s", vec!["abc", "def"])]);
    let good = fs::read(&path).unwrap();
    // A value length past its page: the first value's length is at byte
    // 4 + the page header.
    let mut bad = good.clone();
    let at = bad.windows(4).position(|w| w == 3u32.to_le_bytes()).unwrap();
    bad[at] = 200;
    let e = Parquet::open("d", &bad).unwrap().strings("s").err().unwrap();
    assert!(e.starts_with("d: column s: row group 0: page at byte 4: ") && e.contains("PLAIN values end"), "{e}");
    // A value that is not UTF-8.
    let mut bad = good.clone();
    let at = bad.windows(3).position(|w| w == b"def").unwrap();
    bad[at] = 0xff;
    assert_eq!(Parquet::open("d", &bad).unwrap().strings("s").err().unwrap(), "d: column s row 1 is not UTF-8");
    // A truncated file.
    assert!(Parquet::open("d", &good[..good.len() - 20]).is_err());
}

/// A file pyarrow wrote in three row groups, of 4, 4 and 2 rows, with its
/// size, its footer's length and where the footer starts.
fn three_groups(file: &str) -> (Vec<u8>, u64, u64, usize) {
    let bytes = fs::read(data("parquet").join(file)).unwrap();
    let size = bytes.len() as u64;
    let len = footer_len(file, size, &bytes[bytes.len() - 8..]).unwrap();
    let start = bytes.len() - 8 - len as usize;
    (bytes, size, len, start)
}

#[test]
fn leading_row_groups_from_every_first_part_of_a_footer() {
    for file in ["snappy-dict-v1.parquet", "gzip-dict-v2.parquet"] {
        let (bytes, size, len, start) = three_groups(file);
        let footer = &bytes[start..bytes.len() - 8];
        let whole = Leading::read(file, size, len, footer).unwrap();
        assert_eq!((whole.rows(), whole.row_groups()), (10, 3));
        let spans = whole.spans();
        assert_eq!(spans.iter().map(|s| s.rows).collect::<Vec<_>>(), [4, 4, 2]);
        // The row groups are back to back, from the magic number to the
        // footer.
        assert_eq!(spans[0].start, 4);
        assert!(spans.windows(2).all(|w| w[0].end == w[1].start), "{spans:?}");
        assert_eq!(spans[2].end, start as u64);
        // Every first part of the footer: an error until the row groups
        // start, then the row groups wholly inside it, never fewer than a
        // shorter part held.
        let mut read: Option<usize> = None;
        for cut in 0..=footer.len() {
            match Leading::read(file, size, len, &footer[..cut]) {
                Err(e) => {
                    assert_eq!(read, None, "{file}: {cut} bytes: {e}");
                    assert_eq!(e, format!("{file}: footer: the first {cut} bytes end before the row groups"));
                }
                Ok(l) => {
                    assert_eq!((l.rows(), l.row_groups()), (10, 3), "{file}: {cut} bytes");
                    let n = l.spans().len();
                    assert!(n >= read.unwrap_or(0), "{file}: {cut} bytes");
                    assert_eq!(l.spans(), &spans[..n], "{file}: {cut} bytes");
                    read = Some(n);
                }
            }
        }
        assert_eq!(read, Some(3));
    }
}

#[test]
fn the_leading_row_groups_are_read_from_the_head_of_the_file() {
    for file in ["snappy-dict-v1.parquet", "gzip-dict-v2.parquet"] {
        let (bytes, size, len, start) = three_groups(file);
        let leading = Leading::read(file, size, len, &bytes[start..bytes.len() - 8]).unwrap();
        for groups in 1..=3 {
            let end = leading.spans()[groups - 1].end as usize;
            let t = Parquet::open_head(file, &bytes[..end], &leading, groups).unwrap();
            let rows = [4, 8, 10][groups - 1];
            assert_eq!(t.rows(), rows);
            let name = t.strings("name").unwrap();
            assert_eq!(name.len(), rows);
            assert_eq!((0..rows).map(|r| name.at(r).unwrap()).collect::<Vec<_>>(), names()[..rows]);
            let id = t.ints("id").unwrap();
            assert_eq!(id.len(), rows);
            assert_eq!(id.at(rows - 1).unwrap(), (rows as i64 - 1) * 1_000_003 - 5);
            let text = t.strings("text").unwrap();
            for (r, w) in texts().iter().take(rows).enumerate() {
                assert_eq!(text.get(r), w.as_deref(), "{file} row {r}");
            }
            let e = Parquet::open_head(file, &bytes[..end - 1], &leading, groups).err().unwrap();
            assert_eq!(
                e,
                format!("{file}: the first {groups} row groups end at byte {end}, past the {} bytes read", end - 1)
            );
        }
        let e = Parquet::open_head(file, &bytes, &leading, 4).err().unwrap();
        assert_eq!(e, format!("{file}: 4 row groups wanted, the footer's first bytes hold 3"));
        assert_eq!(Parquet::open_head(file, b"PAR", &leading, 1).err().unwrap(), format!("{file}: not a Parquet file"));
        // A first part of the footer holding the first row group only.
        let one = (0..len as usize)
            .map(|cut| Leading::read(file, size, len, &bytes[start..start + cut]))
            .find_map(|l| l.ok().filter(|l| l.spans().len() == 1))
            .unwrap();
        let e = Parquet::open_head(file, &bytes, &one, 2).err().unwrap();
        assert_eq!(e, format!("{file}: 2 row groups wanted, the footer's first bytes hold 1"));
        assert_eq!(Parquet::open_head(file, &bytes, &one, 1).unwrap().rows(), 4);
    }
}

#[test]
fn footer_lengths_and_spans_are_checked() {
    assert_eq!(footer_len("x", 28, b"\x10\0\0\0PAR1"), Ok(16));
    assert_eq!(footer_len("x", 27, b"\x10\0\0\0PAR1").unwrap_err(), "x: footer length 16 in a file of 27 bytes");
    assert_eq!(footer_len("x", 100, b"\x10\0\0\0PARE").unwrap_err(), "x: an encrypted Parquet footer is not read");
    assert_eq!(footer_len("x", 100, b"\x10\0\0\0PAR2").unwrap_err(), "x: not a Parquet file");
    assert_eq!(footer_len("x", 100, b"PAR1").unwrap_err(), "x: 4 bytes where its last 8 were wanted");

    let file = "snappy-dict-v1.parquet";
    let (bytes, size, len, start) = three_groups(file);
    let footer = &bytes[start..bytes.len() - 8];
    // A file a byte shorter puts the footer's start inside the last row
    // group.
    let e = Leading::read(file, size - 1, len, footer).unwrap_err();
    assert_eq!(e, format!("{file}: row group 2 has a column chunk outside the file's data"));
    let e = Leading::read(file, size, len - 1, footer).unwrap_err();
    assert_eq!(e, format!("{file}: footer: {} bytes of a footer of {}", len, len - 1));
    let e = Leading::read(file, len + 11, len, footer).unwrap_err();
    assert_eq!(e, format!("{file}: footer length {len} in a file of {} bytes", len + 11));
}
