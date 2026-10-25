//! WARC/1.1 archive (`warcinfo` + `resource` record) [`DocumentFormatter`] and parser.

use std::fmt::Write as _;
use std::io::Write;

use turbo_analysis::Document;

use super::jsonld_export::JsonLdFormatter;
use super::{DocumentFormatter, ExportOffsetMode, FormatExportError, OutputFormatDescriptor, sanitize_single_line};

const REQUIRED_LAYERS: &[&str] = &[];
const WARC_VERSION: &str = "WARC/1.1";
const FIXED_WARC_DATE: &str = "2026-01-01T00:00:00Z";

/// WARC/1.1 archive (`warc`) exporter.
///
/// Emits two deterministic WARC/1.1 records with exact byte `Content-Length`
/// headers and `\r\n\r\n` record trailers:
/// 1. `warcinfo` record describing the `turbo-formats` exporter and offset mode;
/// 2. `resource` record for `urn:turbo:document:<doc_id>` carrying the analyzed
///    document's JSON-LD payload.
#[derive(Copy, Clone, Debug, Default)]
pub struct WarcFormatter;

/// Parsed WARC/1.1 record used to verify exact `Content-Length` and CRLF framing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarcRecord {
    /// Header fields in `(name, value)` order.
    pub headers: Vec<(String, String)>,
    /// Exact payload block bytes of length `Content-Length`.
    pub block: Vec<u8>,
}

impl WarcRecord {
    /// Looks up a header value by case-insensitive ASCII name.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

impl DocumentFormatter for WarcFormatter {
    fn descriptor(&self) -> OutputFormatDescriptor {
        OutputFormatDescriptor {
            format_id: "warc",
            display_name: "WARC/1.1 Web Archive",
            media_type: "application/warc",
            file_extension: "warc",
            binary: true,
            lossless: false,
            required_layers: REQUIRED_LAYERS,
            description: "WARC/1.1 archive with deterministic per-doc_id WARC-Record-ID and WARC-Date headers, warcinfo and resource records, exact Content-Length, and CRLF framing.",
        }
    }

    fn format_document(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError> {
        let clean_id = sanitize_warc_token(doc_id);
        let target_uri = format!("urn:turbo:document:{clean_id}");
        let info_record_id = format!("<urn:turbo:warcinfo:{clean_id}>");
        let resource_record_id = format!("<urn:turbo:resource:{clean_id}>");

        let mut warcinfo_body = String::new();
        let _ = write!(warcinfo_body, "software: turbo-formats/doc_export\r\n");
        let _ = write!(warcinfo_body, "format: WARC File Format 1.1\r\n");
        let _ = write!(warcinfo_body, "offset-mode: {}\r\n", offsets.as_str());
        let _ = write!(warcinfo_body, "doc-id: {}\r\n", sanitize_single_line(doc_id));
        let info_bytes = warcinfo_body.as_bytes();

        let resource_bytes = JsonLdFormatter.format_to_vec(doc_id, doc, offsets)?;

        write_warc_record(
            out,
            &[
                ("WARC-Type", "warcinfo"),
                ("WARC-Date", FIXED_WARC_DATE),
                ("WARC-Record-ID", &info_record_id),
                ("Content-Type", "application/warc-fields"),
            ],
            info_bytes,
        )?;

        write_warc_record(
            out,
            &[
                ("WARC-Type", "resource"),
                ("WARC-Target-URI", &target_uri),
                ("WARC-Date", FIXED_WARC_DATE),
                ("WARC-Record-ID", &resource_record_id),
                ("WARC-Warcinfo-ID", &info_record_id),
                ("Content-Type", "application/ld+json; charset=utf-8"),
            ],
            &resource_bytes,
        )?;

        Ok(())
    }
}

fn write_warc_record(out: &mut dyn Write, headers: &[(&str, &str)], block: &[u8]) -> Result<(), FormatExportError> {
    let mut header_buf = String::new();
    let _ = write!(header_buf, "{WARC_VERSION}\r\n");
    for &(name, val) in headers {
        let _ = write!(header_buf, "{name}: {}\r\n", sanitize_single_line(val));
    }
    let _ = write!(header_buf, "Content-Length: {}\r\n\r\n", block.len());
    out.write_all(header_buf.as_bytes())?;
    out.write_all(block)?;
    out.write_all(b"\r\n\r\n")?;
    Ok(())
}

fn sanitize_warc_token(doc_id: &str) -> String {
    let mut out = String::with_capacity(doc_id.len());
    for b in doc_id.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b':') {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{:02X}", b);
        }
    }
    if out.is_empty() { "unnamed".to_owned() } else { out }
}

/// Parses a WARC/1.1 byte stream into [`WarcRecord`]s, validating the
/// `WARC/1.1\r\n` version line, CRLF-terminated headers, exact `Content-Length`,
/// and `\r\n\r\n` record trailer.
pub fn parse_warc_records(bytes: &[u8]) -> Result<Vec<WarcRecord>, FormatExportError> {
    let mut pos = 0usize;
    let mut records = Vec::new();
    while pos < bytes.len() {
        let Some(ver_end) = find_crlf(bytes, pos) else {
            return Err(FormatExportError::InvalidData("missing CRLF on WARC version line".into()));
        };
        let ver_line =
            std::str::from_utf8(&bytes[pos..ver_end]).map_err(|e| FormatExportError::InvalidData(e.to_string()))?;
        if ver_line != WARC_VERSION {
            return Err(FormatExportError::InvalidData(format!("expected '{WARC_VERSION}', got '{ver_line}'")));
        }
        pos = ver_end + 2;

        let mut headers = Vec::new();
        let mut content_length: Option<usize> = None;
        loop {
            let Some(line_end) = find_crlf(bytes, pos) else {
                return Err(FormatExportError::InvalidData("unterminated WARC header block".into()));
            };
            if line_end == pos {
                pos = line_end + 2;
                break;
            }
            let line = std::str::from_utf8(&bytes[pos..line_end])
                .map_err(|e| FormatExportError::InvalidData(e.to_string()))?;
            let Some((raw_name, raw_val)) = line.split_once(':') else {
                return Err(FormatExportError::InvalidData(format!("malformed WARC header '{line}'")));
            };
            let name = raw_name.trim().to_owned();
            let val = raw_val.trim().to_owned();
            if name.eq_ignore_ascii_case("Content-Length") {
                content_length = Some(
                    val.parse::<usize>()
                        .map_err(|_| FormatExportError::InvalidData(format!("invalid Content-Length '{val}'")))?,
                );
            }
            headers.push((name, val));
            pos = line_end + 2;
        }

        let len = content_length
            .ok_or_else(|| FormatExportError::InvalidData("missing Content-Length header in WARC record".into()))?;
        let Some(block_end) = pos.checked_add(len) else {
            return Err(FormatExportError::InvalidData(format!(
                "WARC record Content-Length {len} overflows buffer offset {pos}"
            )));
        };
        let Some(record_end) = block_end.checked_add(4) else {
            return Err(FormatExportError::InvalidData(format!(
                "WARC record block end {block_end} + trailer overflows usize"
            )));
        };
        if record_end > bytes.len() {
            return Err(FormatExportError::InvalidData(format!(
                "WARC record block of {len} bytes + trailer exceeds buffer length {}",
                bytes.len()
            )));
        }
        let block = bytes[pos..block_end].to_vec();
        if &bytes[block_end..record_end] != b"\r\n\r\n" {
            return Err(FormatExportError::InvalidData("missing '\\r\\n\\r\\n' WARC record trailer".into()));
        }
        pos = record_end;
        records.push(WarcRecord { headers, block });
    }
    Ok(records)
}

fn find_crlf(bytes: &[u8], start: usize) -> Option<usize> {
    bytes[start..].windows(2).position(|w| w == b"\r\n").map(|i| start + i)
}
