//! The magic table alone: OpenNLP's `KnownMagicsTest`.

use super::*;

// ---------------------------------------------------------------------------
// KnownMagicsTest. `magic_format` is `KnownMagics.formatOf`: the table
// alone, longest signature first. OpenNLP's precomputed base64 prefixes
// have no counterpart (the detector decodes and tests the first decoded
// byte instead), so their test becomes one that every signature's first
// byte passes that filter.

#[test]
fn every_signature_passes_the_first_byte_filter() {
    use turbo_normalize::assets::FIRST_BYTES;
    for &(magic, format) in MAGICS {
        assert!(FIRST_BYTES[usize::from(magic[0])], "{format:?}");
    }
    for b in [b'R', 0x30, b'B', b'b', b'<', b' ', b'\t', b'\r', b'\n', 0x00, 0xEF, 0xFE, 0xFF] {
        assert!(FIRST_BYTES[usize::from(b)], "{b:#x}");
    }
}

#[test]
fn entries_are_sorted_longest_first() {
    for pair in MAGICS.windows(2) {
        assert!(pair[0].0.len() >= pair[1].0.len(), "{:?} before {:?}", pair[0], pair[1]);
    }
}

#[test]
fn no_duplicate_magics() {
    let mut seen = std::collections::HashSet::new();
    for &(magic, format) in MAGICS {
        assert!(seen.insert(magic), "duplicate magic for {format:?}");
    }
}

#[test]
fn lookup_favors_the_longest_magic() {
    assert_eq!(magic_format(b"!<arch>\ndebian-binary junk").unwrap().name(), "deb");
    assert_eq!(magic_format(b"!<arch>\nsomething else").unwrap().name(), "ar");
}

#[test]
fn binary_dxf_sentinel() {
    let header = b"AutoCAD Binary DXF\r\n\x1a\0";
    assert_eq!(header.len(), 22);
    assert_eq!(magic_format(header).unwrap().name(), "dxf");
    let content = padded(header, 30);
    let encoded = std64(&content);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("dxf", "image/vnd.dxf"));
    assert_eq!(a.decode(&encoded).unwrap().len(), content.len());
}

const DXB_HEADER: &[u8] = b"AutoCAD DXB 1.0\r\n\x1a\0";

#[test]
fn dxb_point_drawing() {
    assert_eq!(DXB_HEADER.len(), 19);
    assert_eq!(magic_format(DXB_HEADER).unwrap().name(), "dxb");
    let drawing = cat(DXB_HEADER, &[2, 10, 0, 20, 0, 0]);
    let encoded = std64(&drawing);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("dxb", "image/vnd.dxb"));
    assert_eq!(a.decode(&encoded).unwrap(), drawing);
}

#[test]
fn truncated_dxb_sentinel_is_rejected() {
    for length in [0, 14, 15, 16, 17, 18] {
        assert_eq!(magic_format(&DXB_HEADER[..length]), None);
    }
}

#[test]
fn invalid_dxb_sentinel_is_rejected() {
    for sentinel in [
        &b"AutoCAD DXB 1.0\r\n0x1A00"[..],
        b"AutoCAD DXB 1.0\r\n^Z\0",
        b"AutoCAD DXB 1.0\n\r\x1a\0",
        b"AutoCAD DXB 1.0\r\n\x1a!",
    ] {
        let content = padded(sentinel, 30);
        assert_eq!(magic_format(&content), None);
        assert!(detect(&std64(&content)).is_empty());
    }
}

/// `KnownMagicsTest.openExrHeader`.
fn open_exr(flags: u32) -> Vec<u8> {
    let mut b = 20_000_630u32.to_le_bytes().to_vec();
    b.extend_from_slice(&(2 | flags).to_le_bytes());
    b.resize(32, 0);
    b
}

#[test]
fn open_exr_header_flags() {
    for flags in [0, 0x200, 0x400, 0x600, 0x800, 0xC00, 0x1000, 0x1400, 0x1800, 0x1C00] {
        let header = open_exr(flags);
        assert_eq!(magic_format(&header).unwrap().name(), "exr");
        let encoded = std64(&header);
        let a = one(&encoded);
        assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("exr", "image/x-exr"));
        assert_eq!(a.decode(&encoded).unwrap(), header);
    }
}

#[test]
fn declared_aces_type_is_preserved() {
    let header = open_exr(0);
    let uri = format!("data:image/aces;base64,{}", std64(&header));
    let a = one(&uri);
    assert_eq!((a.format_name(&uri), a.media_type(&uri)), ("exr", "image/aces"));
    assert_eq!(a.decode(&uri).unwrap(), header);
}

#[test]
fn truncated_open_exr_magic_is_rejected() {
    for length in 0..4 {
        assert_eq!(magic_format(&open_exr(0)[..length]), None);
    }
}

#[test]
fn big_endian_open_exr_magic_is_rejected() {
    let mut header = 20_000_630u32.to_be_bytes().to_vec();
    header.extend_from_slice(&2u32.to_be_bytes());
    header.resize(32, 0);
    assert_eq!(magic_format(&header), None);
    assert!(detect(&std64(&header)).is_empty());
}

#[test]
fn snappy_framed_stream() {
    let mut stream = 0x0000_06FFu32.to_le_bytes().to_vec();
    stream.extend_from_slice(b"sNaPpY");
    stream.extend_from_slice(&0x0000_12FEu32.to_le_bytes());
    stream.resize(32, 0);
    assert_eq!(magic_format(&stream).unwrap().name(), "sz");
    let encoded = std64(&stream);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("sz", "application/x-snappy-framed"));
    assert_eq!(a.decode(&encoded).unwrap(), stream);
}

#[test]
fn snappy_identifier_without_chunk_header_is_rejected() {
    let text = b"sNaPpY is the identifier, not the complete chunk";
    assert_eq!(magic_format(text), None);
    assert!(detect(&std64(text)).is_empty());
}

#[test]
fn tiff_byte_orders() {
    for little in [true, false] {
        for version in [42u16, 43] {
            let marker = if little { b'I' } else { b'M' };
            let mut header = vec![marker, marker];
            let mut put = |bytes: &[u8]| header.extend_from_slice(bytes);
            let order16 = |v: u16| if little { v.to_le_bytes() } else { v.to_be_bytes() };
            put(&order16(version));
            if version == 43 {
                put(&order16(8));
                put(&order16(0));
                put(&if little { 16u64.to_le_bytes() } else { 16u64.to_be_bytes() });
            } else {
                put(&if little { 8u32.to_le_bytes() } else { 8u32.to_be_bytes() });
            }
            header.resize(32, 0);
            assert_eq!(magic_format(&header), Some(AssetFormat::Tiff));
            let encoded = std64(&header);
            let a = one(&encoded);
            assert_eq!((a.format, a.media_type(&encoded)), (AssetFormat::Tiff, "image/tiff"));
            assert_eq!(a.decode(&encoded).unwrap(), header);
        }
    }
}

#[test]
fn xar_archive_media_type() {
    let mut header = [0u8; 28];
    header[..4].copy_from_slice(b"xar!");
    header[5] = 28;
    header[7] = 1;
    let encoded = std64(&header);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("xar", "application/x-xar"));
}

#[test]
fn xpm_format_and_media_type() {
    let xpm = "/* XPM */\nstatic char *image[] = {\n\"1 1 1 1\",\n\". c #000000\",\n\".\"};\n";
    let encoded = std64(xpm.as_bytes());
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("xpm", "image/x-xpixmap"));
}

#[test]
fn textual_hexadecimal_dxf_suffix_is_rejected() {
    assert_eq!(magic_format(b"AutoCAD Binary DXF\r\n0x1A00"), None);
}

#[test]
fn embedded_asset_constants_resolve() {
    use AssetFormat::*;
    let core = [
        Png, Jpeg, Gif, Pdf, Zip, Tiff, Gzip, SevenZip, Rar, Flac, Ogg, Midi, Sqlite, Elf, Pe, Class, Woff, Woff2, Mp3,
        Ole2, Zstd, Wasm,
    ];
    let names = [
        ("png", "image/png"),
        ("jpeg", "image/jpeg"),
        ("gif", "image/gif"),
        ("pdf", "application/pdf"),
        ("zip", "application/zip"),
        ("tiff", "image/tiff"),
        ("gzip", "application/gzip"),
        ("7z", "application/x-7z-compressed"),
        ("rar", "application/vnd.rar"),
        ("flac", "audio/flac"),
        ("ogg", "application/ogg"),
        ("midi", "audio/midi"),
        ("sqlite", "application/vnd.sqlite3"),
        ("elf", "application/x-elf"),
        ("pe", "application/vnd.microsoft.portable-executable"),
        ("class", "application/java-vm"),
        ("woff", "font/woff"),
        ("woff2", "font/woff2"),
        ("mp3", "audio/mpeg"),
        ("ole2", "application/x-ole-storage"),
        ("zstd", "application/zstd"),
        ("wasm", "application/wasm"),
    ];
    for (format, (name, media_type)) in core.into_iter().zip(names) {
        assert_eq!((format.name(), format.media_type()), (name, media_type));
        assert!(MAGICS.iter().any(|&(_, f)| f == format), "{name}");
    }
}

#[test]
fn riff_formats_are_not_table_entries() {
    for format in [AssetFormat::Webp, AssetFormat::Wav, AssetFormat::Avi] {
        assert!(MAGICS.iter().all(|&(_, f)| f != format));
    }
}

#[test]
fn matching_format_and_media_type_stay_together() {
    for &(magic, format) in MAGICS {
        assert_eq!(magic_format(magic), Some(format));
    }
}

#[test]
fn short_magics_are_limited_to_the_core_set() {
    for &(magic, format) in MAGICS {
        if magic.len() < 4 {
            assert!(matches!(format, AssetFormat::Jpeg | AssetFormat::Gzip | AssetFormat::Mp3), "{format:?}");
        }
    }
}

#[test]
fn format_names_round_trip() {
    for format in AssetFormat::SNIFFED {
        assert!(AssetFormat::from_names(format.name(), format.media_type()) == Some(format), "{format:?}");
        assert!(format.media_type().contains('/'));
    }
}
