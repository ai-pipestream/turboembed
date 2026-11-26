//! The Excel, pcapng, calendar and DEX headers.

use super::*;

// ---------------------------------------------------------------------------
// AssetExcelTypeTest

/// `AssetExcelTypeTest.header`: a BIFF2, BIFF3 or BIFF4 BOF record.
fn excel(version: u16, kind: u16, unused: u16) -> Vec<u8> {
    let record: u16 = match version {
        2 => 0x0009,
        3 => 0x0209,
        4 => 0x0409,
        _ => panic!("version must be 2, 3 or 4"),
    };
    let mut b = Vec::new();
    for field in [record, if version == 2 { 4 } else { 6 }, unused, kind, unused] {
        b.extend_from_slice(&field.to_le_bytes());
    }
    b.resize(32, 0);
    b
}

/// `AssetExcelTypeTest.assertFormat`: bare and as an untyped data URI.
fn assert_format(bytes: &[u8], format: &str, media_type: &str) {
    let encoded = std64(bytes);
    for text in [encoded.clone(), format!("{DATA_URI}{encoded}")] {
        let a = one(&text);
        assert_eq!((a.format_name(&text), a.media_type(&text)), (format, media_type));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn excel_worksheet_document_types() {
    for version in [2, 3, 4] {
        for kind in [16, 32, 64] {
            assert_format(&excel(version, kind, 0), "xls", &format!("application/vnd.ms-excel.sheet.{version}"));
        }
    }
}

#[test]
fn excel_workspace_document_types() {
    for version in [3, 4] {
        for unused in [0, 65535] {
            let media_type = format!("application/vnd.ms-excel.workspace.{version}");
            assert_format(&excel(version, 0x0100, unused), "xlw", &media_type);
        }
    }
}

#[test]
fn excel_unused_fields_are_ignored() {
    for version in [2, 3, 4] {
        assert_format(&excel(version, 0x0010, 0xBEEF), "xls", &format!("application/vnd.ms-excel.sheet.{version}"));
    }
}

#[test]
fn excel_unsupported_document_types_are_rejected() {
    let cases = [
        (2, 0),
        (2, 5),
        (2, 256),
        (2, 65535),
        (3, 0),
        (3, 5),
        (3, 6),
        (3, 272),
        (3, 65535),
        (4, 0),
        (4, 5),
        (4, 6),
        (4, 272),
        (4, 65535),
    ];
    for (version, kind) in cases {
        assert_unrecognized_default(&excel(version, kind, 0));
    }
}

#[test]
fn excel_truncated_bof_is_rejected() {
    let cases = [(2, 0), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9)];
    for (version, length) in cases.into_iter().chain([(4, 4), (4, 7), (4, 8), (4, 9)]) {
        let encoded = std64(&excel(version, 0x0010, 0)[..length]);
        assert!(detect(&format!("{DATA_URI}{encoded}")).is_empty(), "{version} {length}");
    }
}

#[test]
fn excel_complete_bof_without_file_body() {
    for version in [2u16, 3, 4] {
        let bytes = &excel(version, 0x0010, 0)[..if version == 2 { 8 } else { 10 }];
        let text = format!("{DATA_URI}{}", std64(bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), "xls");
        assert_eq!(a.media_type(&text), format!("application/vnd.ms-excel.sheet.{version}"));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn excel_supplied_type_remains_available_for_unknown_document_type() {
    let bytes = excel(3, 0xFFFF, 0);
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetPcapngSignatureTest

/// An empty pcapng section header block in either byte order.
fn section(little_endian: bool) -> Vec<u8> {
    let u32b = |v: u32| if little_endian { v.to_le_bytes() } else { v.to_be_bytes() };
    let u16b = |v: u16| if little_endian { v.to_le_bytes() } else { v.to_be_bytes() };
    let mut b = Vec::new();
    b.extend_from_slice(&u32b(0x0A0D_0D0A));
    b.extend_from_slice(&u32b(28));
    b.extend_from_slice(&u32b(0x1A2B_3C4D));
    b.extend_from_slice(&u16b(1));
    b.extend_from_slice(&u16b(0));
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&u32b(28));
    b
}

fn assert_pcapng(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "pcapng", "application/vnd.tcpdump.pcapng", transport);
}

#[test]
fn pcapng_empty_section_in_both_byte_orders() {
    for little in [false, true] {
        assert_pcapng(&section(little), "standard");
        assert_pcapng(&section(little), "uri");
    }
}

#[test]
fn pcapng_sections_with_different_byte_orders() {
    let bytes = cat(&section(true), &section(false));
    for transport in TRANSPORTS {
        assert_pcapng(&bytes, transport);
    }
}

#[test]
fn pcapng_incomplete_byte_order_magic_is_rejected() {
    for length in [0, 1, 3, 4, 5, 7, 8, 9, 10, 11] {
        assert_unrecognized_default(&section(false)[..length]);
        assert_unrecognized_default(&section(true)[..length]);
    }
}

#[test]
fn pcapng_signature_without_complete_block() {
    for little in [false, true] {
        assert_pcapng(&section(little)[..12], "uri");
    }
}

#[test]
fn pcapng_incorrect_byte_order_magic_is_rejected() {
    for magic in [
        0x0000_0000u32,
        0xFFFF_FFFF,
        0xA1B2_C3D4,
        0xD4C3_B2A1,
        0xA1B2_3C4D,
        0x4D3C_B2A1,
        0x1A2B_3C00,
        0x003C_2B1A,
        0x1A2B_3C4E,
        0x4E3C_2B1A,
    ] {
        let mut bytes = section(false);
        bytes[8..12].copy_from_slice(&magic.to_be_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn pcapng_incorrect_byte_order_offset_is_rejected() {
    for little in [false, true] {
        for offset in [4, 7, 9, 12, 24] {
            let mut bytes = section(little);
            bytes[8..12].fill(0);
            let magic = if little { 0x1A2B_3C4Du32.to_le_bytes() } else { 0x1A2B_3C4Du32.to_be_bytes() };
            bytes[offset..offset + 4].copy_from_slice(&magic);
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn pcapng_incorrect_section_type_is_rejected() {
    for kind in [0u32, 2, 0x0A0A_0A0A, 0x0D0A_0D0A, 0x0A0D_0D0B, 0x1A2B_3C4D] {
        let mut bytes = section(false);
        bytes[..4].copy_from_slice(&kind.to_be_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn pcapng_explicit_media_type_with_incorrect_byte_order_magic() {
    let mut bytes = section(false);
    bytes[8..12].fill(0);
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetCalendarHeaderTest

const CALENDAR_BODY: &str = "PRODID:-//Fieldwork Planner//EN\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n\
    UID:survey@example.org\r\nDTSTAMP:20260907T090000Z\r\nDTSTART:20260908T100000Z\r\n\
    SUMMARY:Survey schedule\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n";
const VCALENDAR: &str = "BEGIN:VCALENDAR";

fn assert_calendar(calendar: &str, transport: &str) {
    assert_identified(&AssetDetector::new(), calendar.as_bytes(), "ics", "text/calendar", transport);
}

#[test]
fn calendar_opening() {
    let mut lines = vec![VCALENDAR.to_string(), "begin:vcalendar".to_string(), "BeGiN:vCaLeNdAr".to_string()];
    for at in 1..VCALENDAR.len() {
        for space in [" ", "\t"] {
            lines.push(format!("{}\r\n{space}{}", &VCALENDAR[..at], &VCALENDAR[at..]));
        }
    }
    for line in lines {
        for transport in TRANSPORTS {
            assert_calendar(&format!("{line}\r\n{CALENDAR_BODY}"), transport);
        }
    }
}

#[test]
fn calendar_invalid_opening() {
    for opening in [
        "BEGIN:VCALENDAR-NOTES\r\n",
        "BEGIN:VCALENDARS\r\n",
        "BEGIN:VCALENDAR notes\r\n",
        "BEGIN:VCALENDAR\0\r\n",
        "BEGIN:VCALENDAR\r\n -NOTES\r\n",
        "BEGIN:VCALENDAR\r\n\tS\r\n",
        "BEGIN:VCALENDAR\n",
        "BEGIN:VCALENDAR\r",
        "BEGIN:VCALENDAR\rX\n",
        "BEGIN:VCAL\r\n  ENDAR\r\n",
        "BEGIN:VCAL\n ENDAR\r\n",
        "BEGIN:VCALENDAR;\r\n",
        "BEGIN;X=1:VCALENDAR\r\n",
        "BEGIN:VCALENDAR:\r\n",
    ] {
        assert_unrecognized_default(format!("{opening}{CALENDAR_BODY}").as_bytes());
    }
}

#[test]
fn calendar_changed_byte() {
    for at in 0..VCALENDAR.len() {
        for value in [0u8, 0x1A, 0x3F, 0x7F, 0x80, 0xFE, 0xFF] {
            let mut bytes = format!("{VCALENDAR}\r\n{CALENDAR_BODY}").into_bytes();
            bytes[at] = value;
            assert_unrecognized_default(&bytes);
            assert!(detect(&url64(&bytes)).is_empty());
        }
    }
}

#[test]
fn calendar_truncation() {
    let opening = format!("{VCALENDAR}\r\n");
    for length in 0..=16 {
        assert_unrecognized_default(&opening.as_bytes()[..length]);
    }
}

#[test]
fn calendar_opening_only() {
    assert_calendar(&format!("{VCALENDAR}\r\n"), "uri");
}

#[test]
fn calendar_later_version() {
    assert_calendar(&format!("{VCALENDAR}\r\nX-NOTE:{}\r\n{CALENDAR_BODY}", "survey ".repeat(100)), "mime76");
}

#[test]
fn calendar_repeated_folding() {
    assert_calendar(&format!("B{}EGIN:VCALENDAR\r\n{CALENDAR_BODY}", "\r\n ".repeat(1000)), "mime64");
}

#[test]
fn calendar_folded_suffix() {
    let calendar = format!("{VCALENDAR}{}-NOTES\r\n{CALENDAR_BODY}", "\r\n\t".repeat(1000));
    assert_unrecognized_default(calendar.as_bytes());
}

#[test]
fn calendar_unterminated_fold() {
    assert_unrecognized_default(format!("{VCALENDAR}\r\n ").as_bytes());
}

#[test]
fn calendar_declared_type() {
    let bytes = b"BEGIN:VCALENDAR-NOTES\r\nSurvey schedule";
    let text = format!("data:text/calendar;base64,{}", std64(bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("calendar", "text/calendar"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

#[test]
fn calendar_manual_example() {
    let calendar = format!("begin:vcalendar\r\n{CALENDAR_BODY}");
    let text = format!("{DATA_URI}{}", std64(calendar.as_bytes()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("ics", "text/calendar"));
    assert_eq!(a.decode(&text).unwrap(), calendar.as_bytes());
}

// ---------------------------------------------------------------------------
// AssetDexMagicTest

fn assert_dex(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "dex", "application/x-dex", transport);
}

#[test]
fn dex_encoded_note() {
    let note = b"dex\nThis is a note about Android bytecode, not an executable file.";
    for transport in TRANSPORTS {
        assert!(detect(&encode(note, transport)).is_empty(), "{transport}");
    }
}

#[test]
fn dex_version_and_transport() {
    for version in ["009", "013", "035", "037", "038", "039", "040", "041", "000", "999"] {
        let mut bytes = dex(version, 80);
        bytes[13] = 0xFB;
        bytes[14] = 0xFF;
        for transport in TRANSPORTS {
            assert_dex(&bytes, transport);
        }
    }
}

#[test]
fn dex_version_digit() {
    for offset in 4..7 {
        for value in [0u8, b' ', b'/', b'0', b'1', b'8', b'9', b':', b'A', b'a', 0x7F, 0x80, 0xB2, 0xD9, 0xFF] {
            let mut bytes = dex("035", 80);
            bytes[offset] = value;
            if value.is_ascii_digit() {
                assert_dex(&bytes, "standard");
                assert_dex(&bytes, "uri");
            } else {
                assert_unrecognized_default(&bytes);
            }
        }
    }
}

#[test]
fn dex_nonzero_terminator() {
    for terminator in [1u8, 9, 10, 13, 32, 48, 127, 128, 255] {
        let mut bytes = dex("035", 80);
        bytes[7] = terminator;
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn dex_misplaced_terminator() {
    for offset in [8, 9, 12, 16, 24] {
        let mut bytes = dex("035", 80);
        bytes[7..].fill(b'x');
        bytes[offset] = 0;
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn dex_truncated_magic() {
    for length in 0..8 {
        assert_unrecognized_default(&dex("035", length));
    }
}

#[test]
fn dex_magic_without_complete_file() {
    for length in [8, 9, 12, 16, 23, 24] {
        assert_dex(&dex("035", length), "uri");
    }
}

#[test]
fn dex_changed_prefix() {
    for offset in 0..4 {
        for value in [0u8, b' ', b'D', b'X', b'\r', 0xFF] {
            let mut bytes = dex("035", 80);
            bytes[offset] = value;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn dex_declared_type_with_invalid_magic() {
    for media_type in ["application/octet-stream", "application/x-dex"] {
        let bytes = dex("abc", 80);
        let text = format!("data:{media_type};base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), &media_type[media_type.find('/').unwrap() + 1..]);
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}
