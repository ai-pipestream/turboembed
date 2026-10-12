//! What a decoded header is: the signature table, then the checks some
//! formats need beyond their signature, then the formats recognized
//! without a table entry. A port of `CursorAssetDetector.formatOf` and the
//! checks it calls.

use turbo_normalize_layers::AssetFormat;

use super::magic;
use super::payload::Payload;
use super::svg;

/// Where the four-character form type sits in RIFF and FORM headers
/// (`FORM_TYPE_OFFSET`).
const FORM_TYPE_OFFSET: usize = 8;
/// Where the EMF signature ` EMF` sits in the EMR_HEADER record
/// (`EMF_SIGNATURE_OFFSET`).
const EMF_SIGNATURE_OFFSET: usize = 40;
/// DEX magic: `dex\n`, three version digits and a zero byte
/// (`DEX_VERSION_OFFSET`, `DEX_MAGIC_LENGTH`).
const DEX_VERSION_OFFSET: usize = 4;
const DEX_MAGIC_LENGTH: usize = 8;
/// DWF header `(DWF Vdd.dd)` (`DWF_VERSION_OFFSET`, `DWF_DECIMAL_OFFSET`,
/// `DWF_HEADER_LENGTH`).
const DWF_VERSION_OFFSET: usize = 6;
const DWF_DECIMAL_OFFSET: usize = 8;
const DWF_HEADER_LENGTH: usize = 12;
/// The WebVTT identifier and the UTF-8 byte-order mark that may precede it
/// (`WEBVTT_MAGIC_LENGTH`, `UTF8_BOM_LENGTH`).
const WEBVTT_MAGIC_LENGTH: usize = 6;
const UTF8_BOM_LENGTH: usize = 3;
/// The first line of an iCalendar stream, compared ignoring ASCII case
/// (`CALENDAR_OPENING`).
const CALENDAR_OPENING: &[u8] = b"begin:vcalendar";
/// The JPEG 2000 file-type box after the 12-byte signature box
/// (`JPEG2000_BOX_OFFSET`, `JPEG2000_BRAND_OFFSET`,
/// `JPEG2000_MIN_BOX_LENGTH`).
const JPEG2000_BOX_OFFSET: usize = 12;
const JPEG2000_BRAND_OFFSET: usize = 20;
const JPEG2000_MIN_BOX_LENGTH: u64 = 16;
/// The pcapng byte-order magic in the Section Header Block
/// (`PCAPNG_BYTE_ORDER_OFFSET`, `PCAPNG_BYTE_ORDER_MAGIC`).
const PCAPNG_BYTE_ORDER_OFFSET: usize = 8;
const PCAPNG_BYTE_ORDER_MAGIC: u32 = 0x1a2b_3c4d;
/// The BIFF BOF record: a 4-byte record header, the document type at byte
/// 6 (`BIFF_RECORD_HEADER_SIZE`, `BIFF_DOCUMENT_TYPE_OFFSET`).
const BIFF_RECORD_HEADER_SIZE: usize = 4;
const BIFF_DOCUMENT_TYPE_OFFSET: usize = 6;
/// BER tags and the indefinite length (`BER_SEQUENCE_TAG`,
/// `BER_OBJECT_IDENTIFIER_TAG`, `BER_INDEFINITE_LENGTH`).
const BER_SEQUENCE_TAG: u8 = 0x30;
const BER_OBJECT_IDENTIFIER_TAG: u8 = 0x06;
const BER_INDEFINITE_LENGTH: u8 = 0x80;
/// The content type 1.2.840.113549.1.9.16.1.31, id-ct-timestampedData
/// (RFC 5544), BER-encoded (`TIMESTAMPED_DATA_OID`).
const TIMESTAMPED_DATA_OID: [u8; 11] = [0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x09, 0x10, 0x01, 0x1f];

/// The format of a payload from its decoded header, or `None` when the
/// header is not one this detector recognizes. `text` and `payload` are
/// needed by the calendar and SVG checks, which read further than the
/// header.
pub(crate) fn format_of(header: &[u8], text: &[u8], payload: &Payload) -> Option<AssetFormat> {
    if let Some(known) = magic::known(header) {
        let confirmed = match known {
            AssetFormat::Aiff => {
                carries(header, FORM_TYPE_OFFSET, b"AIFF") || carries(header, FORM_TYPE_OFFSET, b"AIFC")
            }
            AssetFormat::Dex => has_dex_magic(header),
            AssetFormat::Dwf => has_dwf_header(header),
            AssetFormat::Emf => carries(header, EMF_SIGNATURE_OFFSET, b" EMF"),
            AssetFormat::Ics => has_calendar_opening(text, payload),
            AssetFormat::Jp2 => return jpeg2000(header),
            AssetFormat::Pcapng => has_pcapng_byte_order(header),
            AssetFormat::Svg => svg::is_svg(text, payload),
            AssetFormat::Vtt => has_webvtt_boundary(header),
            AssetFormat::XlsSheet2 | AssetFormat::XlsSheet3 | AssetFormat::XlsSheet4 => return excel(header, known),
            AssetFormat::Xml => return Some(if svg::is_svg(text, payload) { AssetFormat::Svg } else { known }),
            _ => true,
        };
        return confirmed.then_some(known);
    }
    if matches!(header.first(), Some(b'B' | b'b')) && has_calendar_opening(text, payload) {
        return Some(AssetFormat::Ics);
    }
    if svg::is_candidate(header) && svg::is_svg(text, payload) {
        return Some(AssetFormat::Svg);
    }
    if carries(header, 0, b"RIFF") {
        let form = header.get(FORM_TYPE_OFFSET..FORM_TYPE_OFFSET + 4);
        match form {
            Some(b"WEBP") => return Some(AssetFormat::Webp),
            Some(b"WAVE") => return Some(AssetFormat::Wav),
            Some(b"AVI ") => return Some(AssetFormat::Avi),
            _ => {}
        }
    }
    has_timestamped_data_type(header).then_some(AssetFormat::Tsd)
}

/// The width and height a PNG or GIF header states, when both are
/// positive: a PNG's IHDR chunk (it must be the first chunk, of length
/// 13), a GIF's logical screen size. OpenNLP reads no other format's size.
pub(crate) fn dimensions(format: AssetFormat, header: &[u8]) -> Option<(u32, u32)> {
    let (width, height) = match format {
        AssetFormat::Png if header.len() >= 24 && read_u32(header, 8) == 13 && carries(header, 12, b"IHDR") => {
            (read_u32(header, 16) as i32, read_u32(header, 20) as i32)
        }
        AssetFormat::Gif if header.len() >= 10 => (read_u16_le(header, 6) as i32, read_u16_le(header, 8) as i32),
        _ => return None,
    };
    (width > 0 && height > 0).then_some((width as u32, height as u32))
}

/// Whether `bytes` holds `tag` at `at`.
fn carries(bytes: &[u8], at: usize, tag: &[u8]) -> bool {
    bytes.get(at..at + tag.len()) == Some(tag)
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_u16_le(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

/// Whether `bytes[start..end]` are all ASCII digits.
fn decimal(bytes: &[u8], start: usize, end: usize) -> bool {
    bytes[start..end].iter().all(u8::is_ascii_digit)
}

/// DEX magic: three decimal version digits and a zero byte after `dex\n`.
/// The version is not limited to those a particular runtime reads.
fn has_dex_magic(header: &[u8]) -> bool {
    header.len() >= DEX_MAGIC_LENGTH
        && header[DEX_MAGIC_LENGTH - 1] == 0
        && decimal(header, DEX_VERSION_OFFSET, DEX_MAGIC_LENGTH - 1)
}

/// DWF: `(DWF V`, two digits, `.`, two digits, `)`.
fn has_dwf_header(header: &[u8]) -> bool {
    header.len() >= DWF_HEADER_LENGTH
        && header[DWF_DECIMAL_OFFSET] == b'.'
        && header[DWF_HEADER_LENGTH - 1] == b')'
        && decimal(header, DWF_VERSION_OFFSET, DWF_DECIMAL_OFFSET)
        && decimal(header, DWF_DECIMAL_OFFSET + 1, DWF_HEADER_LENGTH - 1)
}

/// WebVTT: the identifier ends the payload or is followed by a tab, LF, CR
/// or space.
fn has_webvtt_boundary(header: &[u8]) -> bool {
    let end = if header[0] == b'W' { WEBVTT_MAGIC_LENGTH } else { WEBVTT_MAGIC_LENGTH + UTF8_BOM_LENGTH };
    header.len() == end || matches!(header[end], b'\t' | b'\n' | b'\r' | b' ')
}

/// pcapng: the byte-order magic of the Section Header Block, in either
/// byte order.
fn has_pcapng_byte_order(header: &[u8]) -> bool {
    header.len() >= PCAPNG_BYTE_ORDER_OFFSET + 4 && {
        let magic = read_u32(header, PCAPNG_BYTE_ORDER_OFFSET);
        magic == PCAPNG_BYTE_ORDER_MAGIC || magic == PCAPNG_BYTE_ORDER_MAGIC.swap_bytes()
    }
}

/// JPEG 2000: the brand of the file-type box that follows the signature
/// box picks JP2, JPX, JPM or Motion JPEG 2000. The box length may be the
/// 64-bit extended form; a box too short for its fixed fields is rejected.
fn jpeg2000(header: &[u8]) -> Option<AssetFormat> {
    if header.len() < JPEG2000_BOX_OFFSET + JPEG2000_MIN_BOX_LENGTH as usize
        || !carries(header, JPEG2000_BOX_OFFSET + 4, b"ftyp")
    {
        return None;
    }
    let mut length = read_u32(header, JPEG2000_BOX_OFFSET) as u64;
    let mut brand = JPEG2000_BRAND_OFFSET;
    if length == 1 {
        if header.len() < JPEG2000_BOX_OFFSET + JPEG2000_MIN_BOX_LENGTH as usize + 8 {
            return None;
        }
        length = ((read_u32(header, brand) as u64) << 32) | read_u32(header, brand + 4) as u64;
        if length < JPEG2000_MIN_BOX_LENGTH + 8 {
            return None;
        }
        brand += 8;
    } else if length != 0 && length < JPEG2000_MIN_BOX_LENGTH {
        return None;
    }
    match header.get(brand..brand + 4)? {
        b"jp2 " => Some(AssetFormat::Jp2),
        b"jpx " => Some(AssetFormat::Jpx),
        b"jpm " => Some(AssetFormat::Jpm),
        b"mjp2" => Some(AssetFormat::Mj2),
        _ => None,
    }
}

/// BIFF2 to BIFF4: the BOF record must be complete within the header, and
/// its document type is a worksheet (0x10, 0x20, 0x40) of the version the
/// signature named, or a workspace (0x100) of BIFF3 or BIFF4.
fn excel(header: &[u8], worksheet: AssetFormat) -> Option<AssetFormat> {
    if header.len() < BIFF_DOCUMENT_TYPE_OFFSET + 2
        || header.len() < BIFF_RECORD_HEADER_SIZE + read_u16_le(header, 2) as usize
    {
        return None;
    }
    match read_u16_le(header, BIFF_DOCUMENT_TYPE_OFFSET) {
        0x0010 | 0x0020 | 0x0040 => Some(worksheet),
        0x0100 => match read_u16_le(header, 0) {
            0x0209 => Some(AssetFormat::XlwWorkspace3),
            0x0409 => Some(AssetFormat::XlwWorkspace4),
            _ => None,
        },
        _ => None,
    }
}

/// Whether the first unfolded line of the decoded payload is
/// `BEGIN:VCALENDAR` (ASCII case ignored) ended by CRLF. A CRLF followed
/// by a space or tab is a fold (RFC 5545 section 3.1) and is removed. The
/// payload is read a byte at a time as far as the decision needs; nothing
/// else of the calendar is checked.
fn has_calendar_opening(text: &[u8], payload: &Payload) -> bool {
    let mut matched = 0;
    let (mut after_cr, mut after_lf) = (false, false);
    for value in payload.decoder(text, usize::MAX) {
        if after_cr {
            if value != b'\n' {
                return false;
            }
            after_cr = false;
            after_lf = true;
            continue;
        }
        if after_lf {
            if value != b' ' && value != b'\t' {
                return matched == CALENDAR_OPENING.len();
            }
            after_lf = false;
            continue;
        }
        if value == b'\r' {
            after_cr = true;
        } else {
            if matched == CALENDAR_OPENING.len() {
                return false;
            }
            let expected = CALENDAR_OPENING[matched];
            if value != expected && value != expected.to_ascii_uppercase() {
                return false;
            }
            matched += 1;
        }
    }
    after_lf && matched == CALENDAR_OPENING.len()
}

/// Whether the header opens a CMS ContentInfo whose content type is
/// id-ct-timestampedData: a SEQUENCE (definite or indefinite length) whose
/// first element is that object identifier, all within the header. When
/// the sequence length is definite it must cover the identifier. Nothing
/// after the identifier is checked.
fn has_timestamped_data_type(header: &[u8]) -> bool {
    if header.len() < 2 || header[0] != BER_SEQUENCE_TAG {
        return false;
    }
    let Some(sequence_start) = ber_content_offset(header, 1) else { return false };
    if header.get(sequence_start) != Some(&BER_OBJECT_IDENTIFIER_TAG) {
        return false;
    }
    let Some(oid_start) = ber_content_offset(header, sequence_start + 1) else { return false };
    let oid_end = oid_start + TIMESTAMPED_DATA_OID.len();
    if ber_length(header, sequence_start + 1, oid_start) != Some(TIMESTAMPED_DATA_OID.len()) || oid_end > header.len() {
        return false;
    }
    let covers = ber_length(header, 1, sequence_start).is_none_or(|length| length >= oid_end - sequence_start);
    covers && header[oid_start..oid_end] == TIMESTAMPED_DATA_OID
}

/// Where the content starts after the BER length field at `at`, or `None`
/// when the field is incomplete or uses the reserved form 0xFF.
fn ber_content_offset(header: &[u8], at: usize) -> Option<usize> {
    let descriptor = *header.get(at)?;
    let length_bytes = if descriptor < BER_INDEFINITE_LENGTH { 0 } else { (descriptor & 0x7f) as usize };
    if descriptor == 0xff || length_bytes > header.len() - at - 1 {
        return None;
    }
    Some(at + 1 + length_bytes)
}

/// The BER length at `at`, capped at one more than the header length so a
/// long form never overflows (leading zero bytes are allowed, as BER
/// allows them), or `None` for the indefinite form.
fn ber_length(header: &[u8], at: usize, content: usize) -> Option<usize> {
    let descriptor = header[at];
    if descriptor == BER_INDEFINITE_LENGTH {
        return None;
    }
    let cap = header.len() + 1;
    if descriptor < BER_INDEFINITE_LENGTH {
        return Some(cap.min(descriptor as usize));
    }
    Some(header[at + 1..content].iter().fold(0usize, |length, &b| cap.min((length << 8) | b as usize)))
}
