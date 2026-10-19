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
//!
//! `zstd-page.zst` is a PLAIN page of 1000 strings, each after its
//! length in 4 bytes, compressed as the pages of `zstd.parquet`, which
//! pyarrow wrote, are (one frame with its content size and no checksum)
//! by the `zstd` tool, version 1.5.5:
//!
//! ```text
//! for i in $(seq 0 999); do
//!   s="row $i: the quick brown fox jumps over the lazy dog $((i * i % 997))"
//!   printf "\\$(printf %03o ${#s})\\000\\000\\000%s" "$s"
//! done > page
//! zstd -1 --no-check page -o zstd-page.zst
//! ```

mod common;

use std::fs;

use common::parquet::{Codec, Col, write, write_chunks, write_compressed, write_with_page_sizes};
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

/// The ZSTD file pyarrow wrote: the table's first two rows. FineWeb-2's
/// text column is ZSTD-compressed, as the models-fineweb run showed
/// reading `data/afr_Latn/train/000_00000.parquet`.
#[test]
fn reads_zstd_pages() {
    let bytes = fs::read(data("parquet").join("zstd.parquet")).unwrap();
    let t = Parquet::open("zstd.parquet", &bytes).unwrap();
    assert_eq!(t.rows(), 2);
    let name = t.strings("name").unwrap();
    assert_eq!([name.at(0).unwrap(), name.at(1).unwrap()], names()[..2]);
    let id = t.ints("id").unwrap();
    assert_eq!([id.at(0).unwrap(), id.at(1).unwrap()], [-5, 999_998]);
    let text = t.strings("text").unwrap();
    assert_eq!([text.get(0), text.get(1)], [texts()[0].as_deref(), texts()[1].as_deref()]);
    let small = t.ints("small").unwrap();
    assert_eq!([small.at(0).unwrap(), small.at(1).unwrap()], [0, -1]);
}

fn zstd_rows() -> Vec<String> {
    (0..1000).map(|i| format!("row {i}: the quick brown fox jumps over the lazy dog {}", i * i % 997)).collect()
}

#[test]
fn reads_a_zstd_page() {
    let rows = zstd_rows();
    let col = || Col::Str("s", rows.iter().map(String::as_str).collect());
    let page = fs::read(data("parquet").join("zstd-page.zst")).unwrap();
    let path = scratch("parquet-zstd").join("z.parquet");
    write_compressed(&path, &[col()], Codec::Zstd, std::slice::from_ref(&page));
    let bytes = fs::read(&path).unwrap();
    let s = Parquet::open("z.parquet", &bytes).unwrap().strings("s").unwrap();
    assert_eq!((0..1000).map(|r| s.at(r).unwrap()).collect::<Vec<_>>(), rows);
    // A damaged frame is an error that gives the decoder's reason.
    let mut bad = page;
    bad[0] ^= 1;
    write_compressed(&path, &[col()], Codec::Zstd, &[bad]);
    let bytes = fs::read(&path).unwrap();
    let e = Parquet::open("z.parquet", &bytes).unwrap().strings("s").unwrap_err();
    assert!(e.starts_with("z.parquet: column s: row group 0: page at byte 4: "), "{e}");
    assert!(e.ends_with("zstd: bad magic number 0xfd2fb529"), "{e}");
}

#[test]
fn a_null_is_an_error_where_a_value_is_needed() {
    let bytes = fs::read(data("parquet").join("snappy-dict-v1.parquet")).unwrap();
    let t = Parquet::open("t.parquet", &bytes).unwrap();
    assert_eq!(t.strings("text").unwrap().at(3).unwrap_err(), "t.parquet: column text row 3 is null");
}

#[test]
fn refusals_name_what_is_not_read() {
    let path = scratch("parquet-refusals").join("l.parquet");
    write_compressed(&path, &[Col::Str("name", vec!["a", "b"])], Codec::Lz4Raw, &[vec![0; 10]]);
    let bytes = fs::read(&path).unwrap();
    let t = Parquet::open("l.parquet", &bytes).unwrap();
    let e = t.strings("name").unwrap_err();
    assert!(e.starts_with("l.parquet: column name: row group 0: page at byte 4: "), "{e}");
    assert!(e.ends_with("the LZ4_RAW codec is not read; SNAPPY, GZIP, ZSTD and UNCOMPRESSED are"), "{e}");

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

/// `parquet.thrift` declares a page's sizes i32, so a size below 0 or
/// past 2^31 - 1 is refused, before anything is allocated for it.
#[test]
fn page_sizes_outside_an_i32_are_refused() {
    let path = scratch("parquet-page-sizes").join("s.parquet");
    // Two values of 3 bytes, each after its length: 14 bytes.
    let col = || Col::Str("s", vec!["abc", "def"]);
    write_with_page_sizes(&path, &[col()], (14, 14));
    let bytes = fs::read(&path).unwrap();
    let s = Parquet::open("s.parquet", &bytes).unwrap().strings("s").unwrap();
    assert_eq!([s.at(0).unwrap(), s.at(1).unwrap()], ["abc", "def"]);
    let max = i64::from(i32::MAX);
    let cases = [
        ((-1, 14), "a page's uncompressed size is -1"),
        ((max + 1, 14), "a page's uncompressed size is 2147483648"),
        ((1 << 35, 14), "a page's uncompressed size is 34359738368"),
        ((14, -14), "a page's compressed size is -14"),
        ((14, max + 1), "a page's compressed size is 2147483648"),
    ];
    for (sizes, what) in cases {
        write_with_page_sizes(&path, &[col()], sizes);
        let bytes = fs::read(&path).unwrap();
        let e = Parquet::open("s.parquet", &bytes).unwrap().strings("s").unwrap_err();
        assert_eq!(e, format!("s.parquet: column s: row group 0: page at byte 4: {what}, outside 0 to 2147483647"));
    }
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
    // A character split between two rows: "ab\xC3" then "\xA9ef" is
    // valid UTF-8 as one buffer but in neither row.
    let mut bad = good.clone();
    let at = bad.windows(3).position(|w| w == b"abc").unwrap();
    bad[at + 2] = 0xC3;
    let at = bad.windows(3).position(|w| w == b"def").unwrap();
    bad[at] = 0xA9;
    assert_eq!(Parquet::open("d", &bad).unwrap().strings("s").err().unwrap(), "d: column s row 0 is not UTF-8");
    // A split character before a later invalid byte: the split is the
    // first bad row.
    write(&path, &[Col::Str("s", vec!["", "a", "x", "y", "z"])]);
    let mut bad = fs::read(&path).unwrap();
    let at = bad.windows(15).position(|w| w == b"\x01\0\0\0x\x01\0\0\0y\x01\0\0\0z").unwrap();
    (bad[at + 4], bad[at + 9], bad[at + 14]) = (0xC3, 0xA9, 0xFF);
    assert_eq!(Parquet::open("d", &bad).unwrap().strings("s").err().unwrap(), "d: column s row 2 is not UTF-8");
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

#[test]
fn a_whole_footer_that_ends_inside_a_row_group_is_damage() {
    let file = "snappy-dict-v1.parquet";
    let (bytes, _, len, start) = three_groups(file);
    let footer = &bytes[start..bytes.len() - 8];
    // The first part of the footer that holds two row groups, and five
    // bytes of the third, read as if it were the whole footer.
    let two = (0..len as usize)
        .find(|&cut| Leading::read(file, bytes.len() as u64, len, &footer[..cut]).is_ok_and(|l| l.spans().len() == 2))
        .unwrap();
    let cut = two + 5;
    let e = Leading::read(file, (start + cut + 8) as u64, cut as u64, &footer[..cut]).unwrap_err();
    assert!(e.starts_with(&format!("{file}: footer: ")) && !e.contains("end before"), "{e}");
}

#[test]
fn a_column_chunk_outside_the_file_s_data_is_refused() {
    let dir = scratch("parquet-chunks");
    let cols = [Col::Str("s", vec!["abc", "def"])];
    // Each case gives a column chunk's (offset, compressed size) from the
    // writer's (offset, size).
    type Chunk = fn(usize, usize) -> (i64, i64);
    let cases: [(&str, Chunk); 2] = [
        // Inside the magic number at the start.
        ("start.parquet", |_, len| (3, len as i64)),
        // Past the footer's start.
        ("size.parquet", |offset, _| (offset as i64, i64::MAX)),
    ];
    for (name, chunk) in cases {
        let path = dir.join(name);
        write_chunks(&path, &cols, chunk);
        let bytes = fs::read(&path).unwrap();
        let size = bytes.len() as u64;
        let len = footer_len(name, size, &bytes[bytes.len() - 8..]).unwrap();
        let footer = &bytes[bytes.len() - 8 - len as usize..bytes.len() - 8];
        let e = Leading::read(name, size, len, footer).unwrap_err();
        assert_eq!(e, format!("{name}: row group 0 has a column chunk outside the file's data"));
    }
}
