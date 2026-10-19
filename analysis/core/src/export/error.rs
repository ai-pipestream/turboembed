use super::*;

/// Why an export or an import was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExportError {
    /// The graph's format version is not this build's.
    UnsupportedVersion { found: u16, supported: u16 },
    /// A layer record's schema is not its type's current `SCHEMA`.
    IncompatibleSchema { layer: String, exported: u16, supported: u16 },
    /// No codec is registered for the layer type.
    NoCodec { layer: String },
    /// Two records name one layer instance (or two views).
    DuplicateLayer { layer: String, instance: String },
    /// A view instance the graph does not hold, named as a parent or
    /// by a span codec.
    UnknownView { instance: String },
    /// A reference to a layer instance the graph does not restore:
    /// an endpoint's layer, or one provenance names.
    DanglingReference { layer: String, reference: LayerRef },
    /// An edge or group endpoint past the end of its layer.
    DanglingEndpoint { layer: String, endpoint: NodeRef },
    /// The source record does not hold together: bytes that are not
    /// the text's, or an alignment that does not decode them to it.
    InvalidSource(String),
    /// A codec's layer was refused by the document, or its payload
    /// disagrees with its record.
    InvalidLayer { layer: String, detail: String },
    /// Bytes that are not a graph: truncated, trailing, or a field out
    /// of range.
    Malformed(String),
}

impl fmt::Display for ExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use ExportError::*;
        match self {
            UnsupportedVersion { found, supported } => {
                write!(f, "document graph format {found}, this build reads {supported}")
            }
            IncompatibleSchema { layer, exported, supported } => {
                write!(f, "layer {layer} was exported at schema {exported}, this build reads {supported}")
            }
            NoCodec { layer } => write!(f, "no codec registered for layer {layer}"),
            DuplicateLayer { layer, instance } => write!(f, "two records for layer instance {layer}@{instance}"),
            UnknownView { instance } => write!(f, "no view instance {instance:?}"),
            DanglingReference { layer, reference } => {
                write!(f, "layer {layer} refers to {reference}, which the graph does not restore")
            }
            DanglingEndpoint { layer, endpoint } => {
                write!(f, "layer {layer} has an endpoint {endpoint} past the end of its layer")
            }
            InvalidSource(detail) => f.write_str(detail),
            InvalidLayer { layer, detail } => write!(f, "layer {layer}: {detail}"),
            Malformed(detail) => write!(f, "malformed document graph: {detail}"),
        }
    }
}

impl std::error::Error for ExportError {}

pub(super) fn malformed(detail: impl Into<String>) -> ExportError {
    ExportError::Malformed(detail.into())
}
