//! The forward scan that finds assets, OpenNLP's
//! `CursorAssetDetector.detect`: at each position it tries a `data:` URI,
//! a PEM envelope, a JSON Web Token and a bare base64 run, in that order,
//! and resumes after whatever it found. A run that is not an asset is
//! skipped whole. Every construct it looks for is ASCII, so the scan works
//! on bytes: a byte of a multi-byte UTF-8 sequence is never in any of its
//! sets, which is how OpenNLP's UTF-16 scan treats non-ASCII characters.

use turbo_analysis::Span;
use turbo_normalize_layers::{AssetEncoding, AssetFormat, subtype};
use turbo_unicode::props::white_space;

use super::payload::{Header, Payload, Wrapping, is_any_base64, is_base64, is_base64_url, line_break_end, scan_base64};
use super::{Asset, AssetDetector, hex, json, magic, sniff, unknown};

/// The shortest bare run considered, in base64 characters
/// (`MIN_BARE_PAYLOAD`). `data:` URIs have no minimum.
pub(crate) const MIN_BARE_PAYLOAD: usize = 32;

/// The longest JWT segment considered, in characters
/// (`MAX_JWT_SEGMENT_LENGTH`).
const MAX_JWT_SEGMENT_LENGTH: usize = 16 * 1024;

/// The longest declared media type of a `data:` URI, in UTF-16 code
/// units as OpenNLP counts them: a character at offset 128 or later ends
/// the candidate.
const MAX_MEDIA_TYPE_OFFSET: usize = 127;

const DATA_URI_SCHEME: &[u8] = b"data:";
const BASE64_MARKER: &[u8] = b";base64,";

/// One RFC 7468 label the detector accepts, with the format it reports.
struct PemType {
    label: &'static str,
    format: AssetFormat,
}

/// The labels in OpenNLP's order (`PEM_TYPES`). Every begin line ends in
/// `-----`, so at most one label matches a given line.
const PEM_TYPES: [PemType; 18] = [
    PemType { label: "CERTIFICATE", format: AssetFormat::PemCert },
    PemType { label: "X509 CERTIFICATE", format: AssetFormat::PemCert },
    PemType { label: "X509 CRL", format: AssetFormat::PemCrl },
    PemType { label: "CERTIFICATE REQUEST", format: AssetFormat::PemRequest },
    PemType { label: "NEW CERTIFICATE REQUEST", format: AssetFormat::PemRequest },
    PemType { label: "PKCS7", format: AssetFormat::PemPkcs7 },
    PemType { label: "CMS", format: AssetFormat::PemCms },
    PemType { label: "PRIVATE KEY", format: AssetFormat::PemKey },
    PemType { label: "ENCRYPTED PRIVATE KEY", format: AssetFormat::PemKey },
    PemType { label: "RSA PRIVATE KEY", format: AssetFormat::PemKey },
    PemType { label: "DSA PRIVATE KEY", format: AssetFormat::PemKey },
    PemType { label: "EC PRIVATE KEY", format: AssetFormat::PemKey },
    PemType { label: "PUBLIC KEY", format: AssetFormat::PemKey },
    PemType { label: "RSA PUBLIC KEY", format: AssetFormat::PemKey },
    PemType { label: "ATTRIBUTE CERTIFICATE", format: AssetFormat::PemAttributeCert },
    PemType { label: "DSA PARAMETERS", format: AssetFormat::PemParametersDsa },
    PemType { label: "EC PARAMETERS", format: AssetFormat::PemParametersEc },
    PemType { label: "DH PARAMETERS", format: AssetFormat::PemParametersDh },
];

/// The longest PEM body line (RFC 7468 section 2).
const PEM_LINE_WIDTH: usize = 64;

/// Scans `text` and reports each asset to `sink`, in text order.
pub(crate) fn scan(options: &AssetDetector, text: &str, sink: &mut dyn FnMut(Asset)) {
    let mut scanner = Scanner { options, text, bytes: text.as_bytes(), sink };
    scanner.run();
}

struct Scanner<'a, 't> {
    options: &'a AssetDetector,
    text: &'t str,
    bytes: &'t [u8],
    sink: &'a mut dyn FnMut(Asset),
}

impl Scanner<'_, '_> {
    fn run(&mut self) {
        let len = self.bytes.len();
        let mut i = 0;
        while i < len {
            if let Some(end) = self.data_uri(i).or_else(|| self.pem(i)).or_else(|| self.jwt(i)) {
                i = end;
                continue;
            }
            if is_any_base64(self.bytes[i]) {
                let wrapping = Wrapping { mime: true, soft: self.options.soft_line_breaks };
                let payload = scan_base64(self.bytes, i, wrapping);
                if self.bare_run(&payload) {
                    i = payload.end.max(i + 1);
                    continue;
                }
                if self.options.hex_dumps
                    && let Some(end) = self.hex_dump(i)
                {
                    i = end;
                    continue;
                }
                if self.options.report_unknown_base64 && unknown::looks_encoded(self.bytes, &payload) {
                    self.report(payload.start, payload.end, &payload, AssetFormat::Unknown, None, None);
                }
                i = payload.end.max(i + 1);
                continue;
            }
            i += 1;
        }
    }

    /// Builds the asset and hands it to the sink: the decoded length from
    /// the payload's counts and the size from the header where the format
    /// carries one.
    fn report(
        &mut self,
        start: usize,
        end: usize,
        payload: &Payload,
        format: AssetFormat,
        declared: Option<Span>,
        header: Option<&[u8]>,
    ) {
        let size = header.and_then(|h| sniff::dimensions(format, h));
        (self.sink)(Asset {
            span: span(start, end),
            payload: span(payload.start, payload.end),
            format,
            declared,
            encoding: payload.encoding,
            decoded_len: payload.decoded_len(),
            width: size.map(|s| s.0),
            height: size.map(|s| s.1),
        });
    }

    /// `data:<media type>;base64,<payload>` starting at `start`
    /// (`tryDataUri`). The media type may be empty or carry parameters,
    /// but holds no white space and no quote. The payload is not wrapped,
    /// may be of any length, and uses the standard alphabet. Its format is
    /// sniffed from the header; the media type is the declared one when it
    /// has a non-empty type and subtype around its slash, else the sniffed
    /// format's. A URI with neither is not reported.
    fn data_uri(&mut self, start: usize) -> Option<usize> {
        if !self.bytes[start..].starts_with(DATA_URI_SCHEME) {
            return None;
        }
        let media_start = start + DATA_URI_SCHEME.len();
        let mut i = media_start;
        let mut units = 0;
        while !self.bytes[i..].starts_with(BASE64_MARKER) {
            let c = self.text[i..].chars().next()?;
            if c <= ' ' || white_space(c) || c == '"' || c == '\'' || units + c.len_utf16() - 1 > MAX_MEDIA_TYPE_OFFSET
            {
                return None;
            }
            units += c.len_utf16();
            i += c.len_utf8();
        }
        let declared = &self.text[media_start..i];
        let payload_start = i + BASE64_MARKER.len();
        let wrapping = Wrapping { mime: false, soft: self.options.soft_line_breaks };
        let payload = scan_base64(self.bytes, payload_start, wrapping);
        if payload.end == payload_start || payload.encoding == AssetEncoding::Base64Url || !payload.valid {
            return None;
        }
        let header = Header::read(self.bytes, &payload);
        let sniffed = sniff::format_of(header.bytes(), self.bytes, &payload);
        let declares = declared.contains('/') && !subtype(declared).is_empty();
        let (format, declared) = match (sniffed, declares) {
            (_, true) => (sniffed.unwrap_or(AssetFormat::Declared), Some(span(media_start, i))),
            (Some(format), false) => (format, None),
            (None, false) => return None,
        };
        self.report(start, payload.end, &payload, format, declared, Some(header.bytes()));
        Some(payload.end)
    }

    /// A bare run of at least `MIN_BARE_PAYLOAD` characters whose decoded
    /// header a check recognizes (`tryBareRun`).
    fn bare_run(&mut self, payload: &Payload) -> bool {
        if payload.encoded_len < MIN_BARE_PAYLOAD || !payload.valid || !self.may_start_asset(payload) {
            return false;
        }
        let header = Header::read(self.bytes, payload);
        let Some(format) = sniff::format_of(header.bytes(), self.bytes, payload) else { return false };
        self.report(payload.start, payload.end, payload, format, None, Some(header.bytes()));
        true
    }

    /// Whether the first decoded byte of a payload is one an asset can
    /// start with (see `magic::FIRST_BYTES`).
    fn may_start_asset(&self, payload: &Payload) -> bool {
        payload.decoder(self.bytes, 1).next().is_some_and(|b| magic::FIRST_BYTES[b as usize])
    }

    /// A hex dump starting at `start` whose decoded header a check
    /// recognizes.
    fn hex_dump(&mut self, start: usize) -> Option<usize> {
        let payload = hex::scan(self.bytes, start)?;
        if !self.may_start_asset(&payload) {
            return None;
        }
        let header = Header::read(self.bytes, &payload);
        let format = sniff::format_of(header.bytes(), self.bytes, &payload)?;
        self.report(payload.start, payload.end, &payload, format, None, Some(header.bytes()));
        Some(payload.end)
    }

    /// An RFC 7468 envelope with a supported label starting at a line
    /// start (`tryPem`): the begin line, body lines of 1 to 64 standard
    /// base64 characters each ended by CRLF or LF, padding only at the end
    /// of the last line, a body length that is a multiple of four, and the
    /// end line of the same label followed by a line break or the end of
    /// the text.
    fn pem(&mut self, start: usize) -> Option<usize> {
        let b = self.bytes;
        if start > 0 && !matches!(b[start - 1], b'\r' | b'\n') {
            return None;
        }
        let rest = b[start..].strip_prefix(b"-----BEGIN ")?;
        let pem = PEM_TYPES
            .iter()
            .find(|t| rest.strip_prefix(t.label.as_bytes()).is_some_and(|after| after.starts_with(b"-----")))?;
        let begin_end = start + b"-----BEGIN ".len() + pem.label.len() + 5;
        let body_start = line_break_end(b, begin_end);
        if body_start == begin_end {
            return None;
        }
        let (mut i, mut encoded, mut padding, mut payload_end, mut saw_body) =
            (body_start, 0usize, 0, body_start, false);
        while i < b.len() {
            if let Some(end) = self.pem_end(i, pem.label) {
                if !saw_body || !(end == b.len() || matches!(b[end], b'\r' | b'\n')) || !encoded.is_multiple_of(4) {
                    return None;
                }
                let payload = Payload {
                    start: body_start,
                    end: payload_end,
                    encoding: AssetEncoding::Base64,
                    encoded_len: encoded,
                    padding,
                    valid: true,
                };
                let header = Header::read(b, &payload);
                self.report(start, end, &payload, pem.format, None, Some(header.bytes()));
                return Some(end);
            }
            if b[i..].starts_with(b"-----END ") {
                return None;
            }
            let line_end = i + b[i..].iter().position(|&c| c == b'\r' || c == b'\n').unwrap_or(b.len() - i);
            if line_end == i || line_end - i > PEM_LINE_WIDTH {
                return None;
            }
            let mut padded = false;
            for &c in &b[i..line_end] {
                if is_base64(c) && !padded {
                    encoded += 1;
                } else if c == b'=' && padding < 2 {
                    padding += 1;
                    encoded += 1;
                    padded = true;
                } else {
                    return None;
                }
            }
            saw_body = true;
            payload_end = line_end;
            let next = line_break_end(b, line_end);
            if next == line_end || (padded && self.pem_end(next, pem.label).is_none()) {
                return None;
            }
            i = next;
        }
        None
    }

    /// The end of the end line `-----END <label>-----` at `at`.
    fn pem_end(&self, at: usize, label: &str) -> Option<usize> {
        let rest = self.bytes[at..].strip_prefix(b"-----END ")?.strip_prefix(label.as_bytes())?;
        rest.starts_with(b"-----").then_some(at + b"-----END ".len() + label.len() + 5)
    }

    /// A compact JSON Web Token starting at `start` (`tryJwt`): three
    /// base64url segments joined by dots, standing apart from other
    /// base64url text. The header must decode to a UTF-8 JSON object with
    /// a non-empty string `alg`, the claims to a UTF-8 JSON object, and
    /// the signature must be unpadded and empty exactly when `alg` is
    /// `none`. The signature is not verified. The claims segment is the
    /// payload.
    fn jwt(&mut self, start: usize) -> Option<usize> {
        let b = self.bytes;
        if start > 0 && (is_base64_url(b[start - 1]) || b[start - 1] == b'.') {
            return None;
        }
        let header_end = self.segment_end(start);
        if header_end == start || b.get(header_end) != Some(&b'.') {
            return None;
        }
        let claims_start = header_end + 1;
        let claims_end = self.segment_end(claims_start);
        if claims_end == claims_start || b.get(claims_end) != Some(&b'.') {
            return None;
        }
        let signature_start = claims_end + 1;
        let end = self.segment_end(signature_start);
        let bounded = end == b.len() || !(is_base64_url(b[end]) || b[end] == b'.' || b[end] == b'=');
        if !bounded || (end - signature_start) % 4 == 1 {
            return None;
        }
        if [header_end - start, claims_end - claims_start, end - signature_start]
            .iter()
            .any(|&n| n > MAX_JWT_SEGMENT_LENGTH)
        {
            return None;
        }
        let header = self.segment_json(start, header_end, Some("alg"))?;
        self.segment_json(claims_start, claims_end, None)?;
        let algorithm = header.filter(|a| !a.is_empty())?;
        if (algorithm == "none") != (end == signature_start) {
            return None;
        }
        let payload = Payload {
            start: claims_start,
            end: claims_end,
            encoding: AssetEncoding::Base64Url,
            encoded_len: claims_end - claims_start,
            padding: 0,
            valid: true,
        };
        self.report(start, end, &payload, AssetFormat::Jwt, None, None);
        Some(end)
    }

    /// The end of the unpadded base64url segment at `start`, read no
    /// further than one character past the longest segment considered.
    fn segment_end(&self, start: usize) -> usize {
        let limit = self.bytes.len().min(start + MAX_JWT_SEGMENT_LENGTH + 1);
        start + self.bytes[start..limit].iter().take_while(|&&c| is_base64_url(c)).count()
    }

    /// Decodes a segment and checks that it is a UTF-8 JSON object.
    /// `Some(member)` when it is, with the captured member's value.
    fn segment_json(&self, start: usize, end: usize, member: Option<&str>) -> Option<Option<String>> {
        if (end - start) % 4 == 1 {
            return None;
        }
        let segment = Payload {
            start,
            end,
            encoding: AssetEncoding::Base64Url,
            encoded_len: end - start,
            padding: 0,
            valid: true,
        };
        let decoded: Vec<u8> = segment.decoder(self.bytes, usize::MAX).collect();
        let parsed = json::parse(std::str::from_utf8(&decoded).ok()?, member);
        parsed.valid.then_some(parsed.member)
    }
}

/// Byte offsets as a span. Texts are limited to `u32::MAX` bytes.
pub(crate) fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}
