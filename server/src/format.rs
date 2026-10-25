//! Document output format discovery and export handlers (`ListOutputFormats`
//! and `FormatDocument`, issue #1220), backed by
//! [`turbo_formats::doc_export::FormatterRegistry::standard`].

use std::sync::OnceLock;

use turbo_analysis::export::{DocumentGraph, import};
use turbo_formats::doc_export::{ExportOffsetMode, FormatExportError, FormatterRegistry, OutputFormatDescriptor};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::OffsetMode;
use crate::proto::result::{
    FormatDocumentRequest, FormatDocumentResponse, ListOutputFormatsRequest, ListOutputFormatsResponse,
    OutputFormatDescriptorProto,
};
use crate::results::document::codecs;
use crate::results::{MAX_GRAPH_BYTES, over};

const DEFAULT_DOC_ID: &str = "doc";

/// Returns the cached standard [`FormatterRegistry`] containing all 7 built-in
/// document formatters (`brat`, `conllu`, `csv`, `jsonld`, `markdown`, `tsv`,
/// `warc`).
pub fn registry() -> &'static FormatterRegistry {
    static REGISTRY: OnceLock<FormatterRegistry> = OnceLock::new();
    REGISTRY.get_or_init(FormatterRegistry::standard)
}

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn descriptor_to_proto(d: OutputFormatDescriptor) -> OutputFormatDescriptorProto {
    OutputFormatDescriptorProto {
        format_id: d.format_id.to_owned(),
        display_name: d.display_name.to_owned(),
        media_type: d.media_type.to_owned(),
        file_extension: d.file_extension.to_owned(),
        binary: d.binary,
        lossless: d.lossless,
        required_layers: d.required_layers.iter().map(|&s| s.to_owned()).collect(),
        description: d.description.to_owned(),
    }
}

fn to_export_offset_mode(mode: OffsetMode) -> ExportOffsetMode {
    match mode {
        OffsetMode::Utf8 => ExportOffsetMode::Utf8,
        OffsetMode::Utf16 => ExportOffsetMode::Utf16,
        OffsetMode::CodePoint => ExportOffsetMode::CodePoint,
    }
}

fn map_export_error(err: FormatExportError) -> Failure {
    bad(format!("format_document: {err}"))
}

/// Lists all registered output formats in deterministic `format_id` order.
pub fn list_output_formats(_req: &ListOutputFormatsRequest) -> ListOutputFormatsResponse {
    let formats = registry().list().into_iter().map(descriptor_to_proto).collect();
    ListOutputFormatsResponse { formats }
}

/// Imports a binary [`DocumentGraph`] and formats it using the requested
/// formatter and [`OffsetMode`].
pub fn format_document(req: &FormatDocumentRequest) -> Result<FormatDocumentResponse, Failure> {
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    let offset_mode = OffsetMode::from_proto(req.offsets)?;
    let format_id = req.format_id.trim();
    if format_id.is_empty() {
        return Err(bad("format_id: must be non-empty"));
    }
    // Validate the format id before decoding the graph.
    let _ = registry().get(format_id).map_err(map_export_error)?;
    if req.graph.is_empty() {
        return Err(bad("graph: document graph bytes are required"));
    }

    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;

    let trimmed_id = req.doc_id.trim();
    let doc_id = if trimmed_id.is_empty() { DEFAULT_DOC_ID } else { trimmed_id };
    let export_offsets = to_export_offset_mode(offset_mode);
    let (desc, content) =
        registry().format_to_vec(format_id, doc_id, &doc, export_offsets).map_err(map_export_error)?;

    Ok(FormatDocumentResponse {
        format_id: desc.format_id.to_owned(),
        media_type: desc.media_type.to_owned(),
        file_extension: desc.file_extension.to_owned(),
        content,
    })
}
