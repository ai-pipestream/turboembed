//! Document format exporter SPI and registry (`conllu`, `brat`, `tsv`, `csv`,
//! `markdown`, `jsonld`, and `warc`).
//!
//! Every built-in [`DocumentFormatter`] renders an analyzed [`Document`] into a
//! standard NLP or document interchange format, projecting spans from any text
//! view back to source text (`ViewId::SOURCE`) via [`Document::to_source`] and
//! converting byte offsets through [`OffsetTable`] according to the caller's
//! [`ExportOffsetMode`].

pub mod brat_export;
pub mod conllu_export;
pub mod jsonld_export;
pub mod markdown_export;
pub mod tabular_export;
pub mod warc_export;

pub use brat_export::{BratFormatter, parse_brat_with_offsets};
pub use conllu_export::ConlluFormatter;
pub use jsonld_export::JsonLdFormatter;
pub use markdown_export::MarkdownFormatter;
pub use tabular_export::{CsvFormatter, TsvFormatter};
pub use warc_export::{WarcFormatter, WarcRecord, parse_warc_records};

use std::fmt;
use std::io::Write;
use std::sync::Arc;

use turbo_analysis::{Document, Key, KeyId, Layer, Span, ViewId};

use crate::error::FormatError;

/// Offset unit used when exporting character spans from [`ViewId::SOURCE`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExportOffsetMode {
    /// Zero-based UTF-8 byte offsets in the source text.
    #[default]
    Utf8,
    /// Zero-based UTF-16 code-unit offsets in the source text.
    Utf16,
    /// Zero-based Unicode scalar value (code point) offsets in the source text.
    CodePoint,
}

impl ExportOffsetMode {
    /// Short lowercase identifier (`"utf8"`, `"utf16"`, `"codepoint"`).
    pub const fn as_str(self) -> &'static str {
        match self {
            ExportOffsetMode::Utf8 => "utf8",
            ExportOffsetMode::Utf16 => "utf16",
            ExportOffsetMode::CodePoint => "codepoint",
        }
    }

    /// Converts a UTF-8 byte [`Span`] on `ViewId::SOURCE` into `(start, end)`
    /// in this offset unit using the document's cached [`OffsetTable`].
    pub fn convert_source_span(self, doc: &Document, source_span: Span) -> Result<(u32, u32), FormatExportError> {
        let source_text = doc.source();
        let len = source_text.len();
        let s = source_span.start as usize;
        let e = source_span.end as usize;
        if s > e || e > len || !source_text.is_char_boundary(s) || !source_text.is_char_boundary(e) {
            return Err(FormatExportError::InvalidData(format!(
                "source span {}..{} is out of bounds or not on a UTF-8 character boundary (source len {})",
                source_span.start, source_span.end, len
            )));
        }
        let table = doc.offsets(ViewId::SOURCE);
        Ok(match self {
            ExportOffsetMode::Utf8 => (source_span.start, source_span.end),
            ExportOffsetMode::Utf16 => {
                let (u_start, u_end) = table.span_to_utf16(source_text, source_span);
                (u_start.0, u_end.0)
            }
            ExportOffsetMode::CodePoint => {
                let (cp_start, cp_end) = table.span_to_code_point(source_text, source_span);
                (cp_start.0, cp_end.0)
            }
        })
    }

    /// Projects `span` from `view` to [`ViewId::SOURCE`] via
    /// [`Document::to_source`] and converts the resulting byte span into
    /// `(start, end)` in this offset unit.
    pub fn project_and_convert(
        self,
        doc: &Document,
        view: ViewId,
        span: Span,
    ) -> Result<(u32, u32), FormatExportError> {
        let src_span = doc.to_source(view, span);
        self.convert_source_span(doc, src_span)
    }
}

/// Metadata describing an export format registered in [`FormatterRegistry`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct OutputFormatDescriptor {
    /// Canonical lowercase identifier (for example `"conllu"`, `"brat"`).
    pub format_id: &'static str,
    /// Human-readable title for UIs and CLI help.
    pub display_name: &'static str,
    /// IANA media type (MIME type) of the exported payload.
    pub media_type: &'static str,
    /// Default file extension without a leading dot.
    pub file_extension: &'static str,
    /// Whether the output is binary framing rather than plain UTF-8 text.
    pub binary: bool,
    /// Whether the format preserves all exported layers losslessly.
    pub lossless: bool,
    /// Layer IDs required on the [`Document`] before formatting.
    pub required_layers: &'static [&'static str],
    /// Concise summary of the output format and its target consumers.
    pub description: &'static str,
}

/// Errors returned when registering or executing a [`DocumentFormatter`].
#[derive(Debug)]
pub enum FormatExportError {
    /// Format identifier was empty, contained whitespace, or used non-lowercase characters.
    InvalidFormatId(String),
    /// A formatter with the same `format_id` was already registered.
    DuplicateFormatId(String),
    /// Requested `format_id` is not registered in the [`FormatterRegistry`].
    UnknownFormatId(String),
    /// A required layer (such as `opennlp:tokens`) is missing from the [`Document`].
    MissingLayer(&'static str),
    /// Underlying format conversion error.
    Format(FormatError),
    /// Underlying I/O error while writing the formatted output.
    Io(std::io::Error),
    /// Document or layer data violated a format constraint.
    InvalidData(String),
}

impl fmt::Display for FormatExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatExportError::InvalidFormatId(id) => {
                write!(f, "invalid format_id '{id}': must be non-empty lowercase ASCII without whitespace")
            }
            FormatExportError::DuplicateFormatId(id) => {
                write!(f, "duplicate format_id '{id}' in FormatterRegistry")
            }
            FormatExportError::UnknownFormatId(id) => {
                write!(f, "unknown format_id '{id}' in FormatterRegistry")
            }
            FormatExportError::MissingLayer(layer) => {
                write!(f, "document is missing required layer '{layer}'")
            }
            FormatExportError::Format(err) => write!(f, "{err}"),
            FormatExportError::Io(err) => write!(f, "{err}"),
            FormatExportError::InvalidData(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for FormatExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            FormatExportError::Format(err) => Some(err),
            FormatExportError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<FormatError> for FormatExportError {
    fn from(value: FormatError) -> Self {
        FormatExportError::Format(value)
    }
}

impl From<std::io::Error> for FormatExportError {
    fn from(value: std::io::Error) -> Self {
        FormatExportError::Io(value)
    }
}

/// Service provider interface for exporting an analyzed [`Document`] into an
/// interchange format.
pub trait DocumentFormatter: Send + Sync {
    /// Static descriptor for this formatter.
    fn descriptor(&self) -> OutputFormatDescriptor;

    /// Formats `doc` (identified by `doc_id`) using `offsets` and writes the
    /// result to `out`.
    fn format_document(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<(), FormatExportError>;

    /// Convenience helper that formats `doc` into a byte vector.
    fn format_to_vec(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
    ) -> Result<Vec<u8>, FormatExportError> {
        let mut buf = Vec::new();
        self.format_document(doc_id, doc, offsets, &mut buf)?;
        Ok(buf)
    }

    /// Convenience helper that formats `doc` into a UTF-8 [`String`].
    fn format_to_string(
        &self,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
    ) -> Result<String, FormatExportError> {
        let bytes = self.format_to_vec(doc_id, doc, offsets)?;
        String::from_utf8(bytes).map_err(|e| FormatExportError::InvalidData(e.to_string()))
    }
}

/// Deterministic registry of [`DocumentFormatter`] implementations, kept sorted
/// in ascending `format_id` order.
#[derive(Clone, Default)]
pub struct FormatterRegistry {
    entries: Vec<Arc<dyn DocumentFormatter>>,
}

impl fmt::Debug for FormatterRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ids: Vec<&str> = self.entries.iter().map(|e| e.descriptor().format_id).collect();
        f.debug_struct("FormatterRegistry").field("formats", &ids).finish()
    }
}

impl FormatterRegistry {
    /// Creates an empty [`FormatterRegistry`].
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Creates a [`FormatterRegistry`] pre-populated with all 7 standard
    /// built-in formatters (`brat`, `conllu`, `csv`, `jsonld`, `markdown`,
    /// `tsv`, `warc`).
    pub fn standard() -> Self {
        let mut reg = Self::new();
        let builtins: [Arc<dyn DocumentFormatter>; 7] = [
            Arc::new(ConlluFormatter),
            Arc::new(BratFormatter),
            Arc::new(TsvFormatter),
            Arc::new(CsvFormatter),
            Arc::new(MarkdownFormatter),
            Arc::new(JsonLdFormatter),
            Arc::new(WarcFormatter),
        ];
        for fmt in builtins {
            reg.register(fmt).expect("standard built-in formatters have valid unique IDs");
        }
        reg
    }

    /// Validates and registers `formatter` in ascending `format_id` order.
    ///
    /// # Errors
    ///
    /// Returns [`FormatExportError::InvalidFormatId`] if `format_id` is empty,
    /// contains whitespace, or contains non-lowercase characters, or
    /// [`FormatExportError::DuplicateFormatId`] if `format_id` is already
    /// present.
    pub fn register(&mut self, formatter: Arc<dyn DocumentFormatter>) -> Result<(), FormatExportError> {
        let id = formatter.descriptor().format_id;
        validate_format_id(id)?;
        match self.entries.binary_search_by_key(&id, |e| e.descriptor().format_id) {
            Ok(_) => Err(FormatExportError::DuplicateFormatId(id.to_owned())),
            Err(pos) => {
                self.entries.insert(pos, formatter);
                Ok(())
            }
        }
    }

    /// Number of registered formatters.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all registered [`OutputFormatDescriptor`]s in ascending
    /// `format_id` order.
    pub fn list(&self) -> Vec<OutputFormatDescriptor> {
        self.entries.iter().map(|e| e.descriptor()).collect()
    }

    /// Looks up a formatter by `format_id`.
    ///
    /// Returns [`FormatExportError::InvalidFormatId`] if `format_id` is
    /// malformed or [`FormatExportError::UnknownFormatId`] if not found.
    pub fn get(&self, format_id: &str) -> Result<&dyn DocumentFormatter, FormatExportError> {
        validate_format_id(format_id)?;
        match self.entries.binary_search_by_key(&format_id, |e| e.descriptor().format_id) {
            Ok(idx) => Ok(self.entries[idx].as_ref()),
            Err(_) => Err(FormatExportError::UnknownFormatId(format_id.to_owned())),
        }
    }

    /// Formats `doc` using the formatter registered under `format_id`.
    pub fn format_document(
        &self,
        format_id: &str,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
        out: &mut dyn Write,
    ) -> Result<OutputFormatDescriptor, FormatExportError> {
        let formatter = self.get(format_id)?;
        formatter.format_document(doc_id, doc, offsets, out)?;
        Ok(formatter.descriptor())
    }

    /// Formats `doc` into a byte vector using the formatter registered under `format_id`.
    pub fn format_to_vec(
        &self,
        format_id: &str,
        doc_id: &str,
        doc: &Document,
        offsets: ExportOffsetMode,
    ) -> Result<(OutputFormatDescriptor, Vec<u8>), FormatExportError> {
        let mut buf = Vec::new();
        let desc = self.format_document(format_id, doc_id, doc, offsets, &mut buf)?;
        Ok((desc, buf))
    }
}

/// Validates that `id` is non-empty and consists solely of lowercase ASCII
/// letters, digits, hyphens, or underscores (no whitespace or uppercase).
pub fn validate_format_id(id: &str) -> Result<(), FormatExportError> {
    if id.is_empty() || !id.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_') {
        return Err(FormatExportError::InvalidFormatId(id.to_owned()));
    }
    Ok(())
}

/// Replaces ASCII and Unicode line breaks and tabs with spaces so single-line
/// fields cannot break line-oriented formats.
pub(crate) fn sanitize_single_line(text: &str) -> String {
    text.chars()
        .map(|c| {
            if matches!(c, '\t' | '\n' | '\r' | '\u{000B}' | '\u{000C}' | '\u{0085}' | '\u{2028}' | '\u{2029}') {
                ' '
            } else {
                c
            }
        })
        .collect()
}

/// Fetches the default instance of `L` and verifies that its recorded binding
/// (`Document::made_on`) matches `expected_anchor` (and that its item count
/// equals `expected_len` when provided).
pub(crate) fn get_bound_layer<L: Layer>(
    doc: &Document,
    expected_anchor: KeyId,
    expected_len: Option<usize>,
) -> Result<Option<&L>, FormatExportError> {
    let key = Key::<L>::DEFAULT;
    let Some(layer) = doc.get_key(&key) else {
        return Ok(None);
    };
    let on = doc.made_on(key.id());
    if let Some(&actual) = on.iter().find(|k| k.ty == expected_anchor.ty)
        && actual != expected_anchor
    {
        return Err(FormatExportError::InvalidData(format!(
            "layer {} is bound to {}, expected {}",
            key.id(),
            actual,
            expected_anchor
        )));
    }
    if let Some(len) = expected_len
        && layer.len() != len
    {
        return Err(FormatExportError::InvalidData(format!(
            "layer {} has {} items, expected {} for {}",
            key.id(),
            layer.len(),
            len,
            expected_anchor
        )));
    }
    Ok(Some(layer))
}
