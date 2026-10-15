//! The Parquet reader on files pyarrow wrote, and on files the test
//! writer in `common/parquet.rs` writes.
//!
//! The files under `tests/data/parquet` and the `.parquet` samples of
//! GutenQA and ConTEB were written with pyarrow 26.0.0 by this script,
//! run from `tests/data`:
//!
//! ```text
//! import json, os
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
//! # Each sample's rows, one JSON object per line in the order below,
//! # written as a table with pyarrow's defaults (SNAPPY, dictionary pages,
//! # data page version 1), as the published sets are.
//! for rel, rows in samples:
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
use turbo_eval::read::parquet::Parquet;

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
