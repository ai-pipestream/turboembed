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

#[test]
fn hybrid_damage_is_an_error() {
    // A run value wider than the bit width: a definition level of 2
    // where the maximum is 1.
    assert_eq!(rle_hybrid(&[6, 2], 1, 3).unwrap_err(), "a run of 2, wider than 1 bits");
    // A run header longer than 5 bytes.
    assert_eq!(rle_hybrid(&[0x80; 6], 1, 1).unwrap_err(), "a run header longer than 5 bytes");
    // A huge count over a short buffer fails on the buffer, without
    // allocating for the count.
    assert_eq!(rle_hybrid(&[6, 1], 1, 1 << 40).unwrap_err(), "levels or indices end early");
}

#[test]
fn a_definition_level_past_the_maximum_is_an_error() {
    let mut out = Values::new(Physical::Int32).unwrap();
    let e = page_values(Physical::Int32, Encoding::Plain, Some(vec![1, 2]), 2, &[0; 8], None, &mut out);
    assert_eq!(e.unwrap_err(), "a definition level of 2 in a column whose maximum is 1");
}

#[test]
fn gzip_sizes_are_bounded() {
    use std::io::Write;
    // A size no deflate stream of this length can inflate to is
    // refused before the output is allocated.
    let e = decompress(Codec::Gzip, &[0; 10], 1 << 40).unwrap_err();
    assert_eq!(e, "a page of 10 bytes cannot inflate to the 1099511627776 its header says");
    // A stream longer than the header says is cut one byte past it.
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(&[7; 1000]).unwrap();
    let gz = gz.finish().unwrap();
    assert_eq!(decompress(Codec::Gzip, &gz, 10).unwrap_err(), "a page expands to 11 bytes, its header says 10");
    assert_eq!(decompress(Codec::Gzip, &gz, 1000).unwrap().as_ref(), &[7; 1000][..]);
}

#[test]
fn zstd_sizes_are_bounded() {
    // The frame and block headers are read before the output is
    // allocated: bytes that are no frame are refused whatever the size,
    // and so is a size past what a frame's blocks hold.
    let e = decompress(Codec::Zstd, &[0; 10], 1 << 40).unwrap_err();
    assert_eq!(e, "zstd: bad magic number 0x00000000");
    // A frame of 1000 sevens with no content size: the magic number, a
    // header with a 1 KiB window, and one RLE block.
    let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x00, 0x00, 0x43, 0x1f, 0x00, 7];
    assert_eq!(decompress(Codec::Zstd, &frame, 1000).unwrap().as_ref(), &[7; 1000][..]);
    assert_eq!(
        decompress(Codec::Zstd, &frame, 1 << 40).unwrap_err(),
        "zstd: the frames decode to at most 1000 bytes, fewer than the 1099511627776 asked for"
    );
    // The same with a single-segment header giving the content size in
    // 2 bytes (1000 - 256): any other size is refused.
    let frame = [0x28, 0xb5, 0x2f, 0xfd, 0x60, 0xe8, 0x02, 0x43, 0x1f, 0x00, 7];
    assert_eq!(decompress(Codec::Zstd, &frame, 1000).unwrap().as_ref(), &[7; 1000][..]);
    assert_eq!(
        decompress(Codec::Zstd, &frame, 10).unwrap_err(),
        "zstd: the content is longer than the 10 bytes asked for"
    );
    assert_eq!(
        decompress(Codec::Zstd, &frame, 1001).unwrap_err(),
        "zstd: the content is 1000 bytes, not the 1001 asked for"
    );
}

/// An empty page that expands to nothing is no bytes in every codec,
/// without decompressing; anything else is decompressed.
#[test]
fn an_empty_page_is_not_decompressed() {
    for codec in [Codec::Uncompressed, Codec::Snappy, Codec::Gzip, Codec::Zstd, Codec::Other(7)] {
        assert_eq!(decompress(codec, &[], 0).unwrap().as_ref(), &[] as &[u8], "{codec}");
    }
    assert_eq!(decompress(Codec::Zstd, &[], 1).unwrap_err(), "zstd: no frames: the input is empty");
    assert_eq!(decompress(Codec::Snappy, &[], 1).unwrap_err(), "snappy: no length");
}

/// Compact protocol bytes of a page header, every field in the long
/// form: the type, then the id as a zigzag varint.
struct Header(Vec<u8>);

impl Header {
    fn zigzag(&mut self, v: i64) {
        let mut v = ((v << 1) ^ (v >> 63)) as u64;
        while v >= 0x80 {
            self.0.push(v as u8 | 0x80);
            v >>= 7;
        }
        self.0.push(v as u8);
    }

    fn field(&mut self, id: i64, ty: u8) {
        self.0.push(ty);
        self.zigzag(id);
    }

    fn int(&mut self, id: i64, v: i64) {
        self.field(id, 5);
        self.zigzag(v);
    }

    /// A page header of `kind` with its sub-header `sub` (field id
    /// `sub_id`) of i32 fields, and a boolean field 7 when given.
    fn page(kind: i64, sizes: (usize, usize), sub_id: i64, sub: &[(i64, i64)], flag: Option<bool>) -> Vec<u8> {
        let mut h = Header(Vec::new());
        h.int(1, kind);
        h.int(2, sizes.0 as i64);
        h.int(3, sizes.1 as i64);
        h.field(sub_id, 12);
        for &(id, v) in sub {
            h.int(id, v);
        }
        if let Some(b) = flag {
            h.field(7, if b { 1 } else { 2 });
        }
        h.0.extend([0, 0]);
        h.0
    }
}

fn chunk_of(file: &[u8], num_values: u64) -> ColumnMeta {
    ColumnMeta {
        physical: Physical::Int32,
        path: vec!["n".into()],
        codec: Codec::Uncompressed,
        num_values,
        total_compressed_size: file.len() as u64,
        data_page_offset: 0,
        dictionary_page_offset: None,
    }
}

#[test]
fn a_page_with_more_values_than_its_chunk_is_refused() {
    // A version 1 page of three PLAIN INT32 values in a chunk of two.
    let mut file = Header::page(0, (12, 12), 5, &[(1, 3), (2, 0), (3, 3), (4, 3)], None);
    file.extend([0; 12]);
    let mut out = Values::new(Physical::Int32).unwrap();
    let e = read_chunk(&file, &chunk_of(&file, 2), false, &mut out).unwrap_err();
    assert_eq!(e, "page at byte 0: 3 values where the column chunk has 2 left");
    let mut out = Values::new(Physical::Int32).unwrap();
    read_chunk(&file, &chunk_of(&file, 3), false, &mut out).unwrap();
    assert_eq!(out.len(), 3);
}

/// parquet-java's version 2 page of an optional column whose values
/// are all null: definition levels, and a compressed values area of
/// no bytes.
#[test]
fn a_v2_page_of_nulls_has_no_values_to_decompress() {
    // Three values, all null: a run of three 0 levels in 2 bytes.
    let sub = [(1, 3), (2, 3), (3, 3), (4, 0), (5, 2), (6, 0)];
    let mut file = Header::page(3, (2, 2), 8, &sub, Some(true));
    file.extend([3 << 1, 0]);
    for codec in [Codec::Snappy, Codec::Gzip, Codec::Zstd] {
        let meta = ColumnMeta { codec, ..chunk_of(&file, 3) };
        let mut out = Values::new(Physical::Int32).unwrap();
        read_chunk(&file, &meta, true, &mut out).unwrap();
        assert_eq!(out.nulls, [true; 3], "{codec}");
    }
}

#[test]
fn an_all_null_dictionary_page_with_empty_values_buffer_is_accepted() {
    let dict = Values { kind: Kind::Ints(vec![42]), nulls: vec![false] };
    for encoding in [Encoding::RleDictionary, Encoding::PlainDictionary] {
        let mut out = Values::new(Physical::Int32).unwrap();
        page_values(Physical::Int32, encoding, Some(vec![0, 0, 0]), 3, &[], Some(&dict), &mut out).unwrap();
        assert_eq!(out.nulls, [true; 3], "{encoding}");
        let Kind::Ints(v) = &out.kind else { panic!("expected Ints") };
        assert_eq!(v, &[0, 0, 0]);
    }
}

#[test]
fn hybrid_zero_runs_and_overflowing_groups_are_refused_and_zero_bit_packed_groups_skip() {
    // An RLE header with run == 0 (0x00) is refused.
    assert_eq!(rle_hybrid(&[0, 0], 1, 1).unwrap_err(), "an empty run");
    // A bit-packed header with groups == 0 (0x01) is skipped as in arrow-rs,
    // reading the following RLE run of three 1s.
    assert_eq!(rle_hybrid(&[1, 6, 1], 1, 3).unwrap(), [1, 1, 1]);
    // A 5-byte varint header with huge groups and width == 32 must not overflow.
    let huge_groups = [0x81, 0x80, 0x80, 0x80, 0x7f];
    assert!(rle_hybrid(&huge_groups, 32, 1).is_err());
}

#[test]
fn an_uncompressed_v2_page_is_checked_against_its_size() {
    // A version 2 page, not compressed, whose header says 8 bytes of
    // values where it holds 4.
    let sub = [(1, 1), (2, 0), (3, 1), (4, 0), (5, 0), (6, 0)];
    let mut file = Header::page(3, (8, 4), 8, &sub, Some(false));
    file.extend([1, 0, 0, 0]);
    let mut out = Values::new(Physical::Int32).unwrap();
    let e = read_chunk(&file, &chunk_of(&file, 1), false, &mut out).unwrap_err();
    assert_eq!(e, "page at byte 0: an uncompressed page of 4 bytes, its header says 8");
}

#[test]
fn duplicate_dictionary_and_required_v2_def_levels_are_refused() {
    let mut file = Header::page(2, (4, 4), 7, &[(1, 1), (2, 0)], None);
    file.extend([7, 0, 0, 0]);
    let second = file.len();
    file.extend(Header::page(2, (4, 4), 7, &[(1, 1), (2, 0)], None));
    file.extend([8, 0, 0, 0]);
    let mut out = Values::new(Physical::Int32).unwrap();
    let e = read_chunk(&file, &chunk_of(&file, 1), false, &mut out).unwrap_err();
    assert_eq!(e, format!("page at byte {second}: a second dictionary page"));

    let sub = [(1, 1), (2, 0), (3, 1), (4, 0), (5, 2), (6, 0)];
    let mut v2 = Header::page(3, (6, 6), 8, &sub, Some(false));
    v2.extend([2, 1, 1, 0, 0, 0]);
    let mut out = Values::new(Physical::Int32).unwrap();
    let e = read_chunk(&v2, &chunk_of(&v2, 1), false, &mut out).unwrap_err();
    assert_eq!(e, "page at byte 0: definition levels in a required column");
}
