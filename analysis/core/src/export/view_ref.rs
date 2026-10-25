//! A borrowed directory over [`DocumentGraph::to_bytes`] output:
//! listing the source, views and layers of a graph, and importing
//! only selected layers, without decoding or copying the codec
//! bodies that were not asked for.
//!
//! [`DocumentGraphRef::parse`] walks the same field order as
//! [`DocumentGraph::from_bytes`], borrowing strings and keeping one
//! byte range per record. Codec bodies are stepped over by their
//! length prefix, never read. Status and provenance records are
//! skipped field by field (they hold no borrowed span tables), so
//! the directory costs a few small strings per record and nothing
//! per body byte. [`DocumentGraphRef::import_selected`] then
//! decodes the source, every view (a document cannot exist without
//! its view texts, and layer spans name views by instance), and
//! exactly the selected layer records, and hands that graph to
//! [`import`].

use super::graph::{get_edits, get_provenance, get_status};
use super::*;
use std::ops::Range;

/// One view in a borrowed graph directory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewWireEntry<'a> {
    pub instance: &'a str,
    pub parent: Option<&'a str>,
    /// The view's text, when the view is present in the graph
    /// (absent for a skipped or failed view, and for the source
    /// view, whose text is the graph's source text).
    pub text: Option<&'a str>,
    /// The view record's whole byte range in the payload.
    pub record: Range<usize>,
}

/// One layer instance in a borrowed graph directory. The codec body
/// is located, not decoded: `body` is its byte range in the payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerWireEntry<'a> {
    pub layer: &'a str,
    pub instance: &'a str,
    pub schema: u16,
    /// The item count the record claims.
    pub len: u32,
    /// The producer and its configuration hash, from provenance.
    pub producer: &'a str,
    pub version: &'a str,
    pub config_hash: u64,
    /// The codec body's byte range, for a layer that has one.
    pub body: Option<Range<usize>>,
    /// The layer record's whole byte range in the payload.
    pub record: Range<usize>,
}

/// A borrowed directory over one graph payload; see the module docs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DocumentGraphRef<'a> {
    bytes: &'a [u8],
    source_bytes: &'a [u8],
    source_text: &'a str,
    source_head: Range<usize>,
    views: Vec<ViewWireEntry<'a>>,
    layers: Vec<LayerWireEntry<'a>>,
}

fn skip_strs(r: &mut Reader<'_>, n: usize) -> Result<(), ExportError> {
    for _ in 0..n {
        r.str()?;
    }
    Ok(())
}

fn skip_layer_ref(r: &mut Reader<'_>) -> Result<(), ExportError> {
    skip_strs(r, 2)
}

fn skip_error(r: &mut Reader<'_>) -> Result<(), ExportError> {
    match r.u8()? {
        0..=2 => skip_layer_ref(r)?,
        3 => {
            skip_layer_ref(r)?;
            skip_strs(r, 2)?;
        }
        4 => {
            skip_layer_ref(r)?;
            skip_layer_ref(r)?;
            r.u32()?;
            r.u32()?;
        }
        5 => {
            skip_layer_ref(r)?;
            skip_layer_ref(r)?;
            r.str()?;
        }
        6 => {
            skip_layer_ref(r)?;
            r.str()?;
        }
        7 | 9 => {
            r.str()?;
        }
        8 => skip_strs(r, 2)?,
        10 => {
            match r.u8()? {
                0 | 1 => {}
                tag => return Err(malformed(format!("an offset unit tag of {tag}"))),
            }
            r.u32()?;
            match r.u8()? {
                0 => {
                    r.u32()?;
                }
                1 => {}
                tag => return Err(malformed(format!("an offset reason tag of {tag}"))),
            }
        }
        11 => match r.u8()? {
            0 => {
                r.u64()?;
            }
            1 => {
                r.str()?;
            }
            tag => return Err(malformed(format!("a source error tag of {tag}"))),
        },
        tag => return Err(malformed(format!("an error tag of {tag}"))),
    }
    Ok(())
}

fn skip_status(r: &mut Reader<'_>) -> Result<(), ExportError> {
    match r.u8()? {
        0 => {}
        1 => {
            r.str()?;
        }
        2 => skip_error(r)?,
        tag => return Err(malformed(format!("a status tag of {tag}"))),
    }
    Ok(())
}

/// Provenance read for its producer fields only; the model,
/// execution and reference lists are stepped over borrowed.
fn read_producer<'a>(r: &mut Reader<'a>) -> Result<(&'a str, &'a str, u64), ExportError> {
    let annotator = r.str()?;
    let version = r.str()?;
    let config = r.u64()?;
    match r.u8()? {
        0 => {}
        1 => skip_strs(r, 3)?,
        tag => return Err(malformed(format!("a model tag of {tag}"))),
    }
    match r.u8()? {
        0 => {}
        1 => skip_strs(r, 3)?,
        tag => return Err(malformed(format!("an execution tag of {tag}"))),
    }
    for _ in 0..2 {
        for _ in 0..r.usize()? {
            skip_layer_ref(r)?;
        }
    }
    Ok((annotator, version, config))
}

fn skip_provenance(r: &mut Reader<'_>) -> Result<(), ExportError> {
    read_producer(r).map(|_| ())
}

fn skip_edits(r: &mut Reader<'_>) -> Result<(), ExportError> {
    for _ in 0..r.usize()? {
        r.span()?;
        r.span()?;
    }
    Ok(())
}

impl<'a> DocumentGraphRef<'a> {
    /// Indexes a graph payload without decoding any codec body.
    /// The same refusals as [`DocumentGraph::from_bytes`] for the
    /// parts read: bad magic, an unsupported format version,
    /// truncated or out-of-range fields, and trailing bytes are
    /// [`ExportError`]s, never panics.
    pub fn parse(bytes: &'a [u8]) -> Result<DocumentGraphRef<'a>, ExportError> {
        let mut r = Reader::new(bytes);
        if r.take(MAGIC.len())? != MAGIC {
            return Err(malformed("not a document graph (bad magic)"));
        }
        let version = r.u16()?;
        if version != FORMAT_VERSION {
            return Err(ExportError::UnsupportedVersion { found: version, supported: FORMAT_VERSION });
        }
        let source_bytes = r.bytes()?;
        let source_text = r.str()?;
        match r.u8()? {
            0 => {}
            1 => skip_edits(&mut r)?,
            tag => return Err(malformed(format!("an alignment tag of {tag}"))),
        }
        let source_head = 0..r.position();
        let mut views = Vec::new();
        for _ in 0..r.usize()? {
            let start = r.position();
            let instance = r.str()?;
            skip_status(&mut r)?;
            skip_provenance(&mut r)?;
            let parent = match r.u8()? {
                0 => None,
                1 => Some(r.str()?),
                tag => return Err(malformed(format!("a parent tag of {tag}"))),
            };
            let text = match r.u8()? {
                0 => None,
                1 => Some(r.str()?),
                tag => return Err(malformed(format!("a text tag of {tag}"))),
            };
            skip_edits(&mut r)?;
            views.push(ViewWireEntry { instance, parent, text, record: start..r.position() });
        }
        let mut layers = Vec::new();
        for _ in 0..r.usize()? {
            let start = r.position();
            let layer = r.str()?;
            let instance = r.str()?;
            let schema = r.u16()?;
            let len = r.u32()?;
            skip_status(&mut r)?;
            let (producer, version, config_hash) = read_producer(&mut r)?;
            let body = match r.u8()? {
                0 => None,
                1 => {
                    // The payload follows its u32 length prefix.
                    let payload_at = r.position() + 4;
                    let body = r.bytes()?;
                    Some(payload_at..payload_at + body.len())
                }
                tag => return Err(malformed(format!("a body tag of {tag}"))),
            };
            layers.push(LayerWireEntry {
                layer,
                instance,
                schema,
                len,
                producer,
                version,
                config_hash,
                body,
                record: start..r.position(),
            });
        }
        if !r.is_done() {
            return Err(malformed("trailing bytes"));
        }
        Ok(DocumentGraphRef { bytes, source_bytes, source_text, source_head, views, layers })
    }

    /// The source bytes the document was read from, borrowed.
    pub fn source_bytes(&self) -> &'a [u8] {
        self.source_bytes
    }

    /// The source text those bytes decode to, borrowed.
    pub fn source_text(&self) -> &'a str {
        self.source_text
    }

    /// Every view in the graph, in graph order.
    pub fn views(&self) -> &[ViewWireEntry<'a>] {
        &self.views
    }

    /// Every layer instance in the graph, in graph order.
    pub fn layers(&self) -> &[LayerWireEntry<'a>] {
        &self.layers
    }

    /// The directory entry of one layer instance, if held.
    pub fn layer(&self, layer: &str, instance: &str) -> Option<&LayerWireEntry<'a>> {
        self.layers.iter().find(|e| e.layer == layer && e.instance == instance)
    }

    /// The codec body bytes of one directory entry, borrowed, for a
    /// layer that has one. The bytes are the codec's payload only
    /// (no length prefix).
    pub fn body(&self, entry: &LayerWireEntry<'a>) -> Option<&'a [u8]> {
        entry.body.as_ref().map(|range| &self.bytes[range.clone()])
    }

    /// The graph holding only the selected `(layer, instance)`
    /// layers (all views and the source are kept whole, as layer
    /// spans name views by instance), with unselected codec bodies
    /// neither copied nor decoded. Records retain graph dependency order,
    /// regardless of selection order. A missing selection is an
    /// [`ExportError::DanglingReference`]; a repeated selection is an
    /// [`ExportError::DuplicateLayer`].
    pub fn selected_graph(&self, selected: &[(&str, &str)]) -> Result<DocumentGraph, ExportError> {
        let mut wanted = HashSet::with_capacity(selected.len());
        for &(layer, instance) in selected {
            if !wanted.insert((layer, instance)) {
                return Err(ExportError::DuplicateLayer { layer: layer.to_owned(), instance: instance.to_owned() });
            }
            if self.layer(layer, instance).is_none() {
                return Err(ExportError::DanglingReference {
                    layer: layer.to_owned(),
                    reference: LayerRef { layer: layer.to_owned(), instance: instance.to_owned() },
                });
            }
        }
        // Decode the head (source) and view records from their
        // ranges with the same readers `from_bytes` uses.
        let mut r = Reader::new(&self.bytes[self.source_head.clone()]);
        r.take(MAGIC.len())?;
        let version = r.u16()?;
        let source = SourceRecord {
            bytes: r.bytes()?.to_vec(),
            text: r.string()?,
            alignment: match r.u8()? {
                0 => None,
                1 => Some(get_edits(&mut r)?),
                tag => return Err(malformed(format!("an alignment tag of {tag}"))),
            },
        };
        let mut views = Vec::with_capacity(self.views.len());
        for entry in &self.views {
            let mut r = Reader::new(&self.bytes[entry.record.clone()]);
            views.push(ViewRecord {
                instance: r.string()?,
                status: get_status(&mut r)?,
                provenance: get_provenance(&mut r)?,
                parent: match r.u8()? {
                    0 => None,
                    1 => Some(r.string()?),
                    tag => return Err(malformed(format!("a parent tag of {tag}"))),
                },
                text: match r.u8()? {
                    0 => None,
                    1 => Some(r.string()?),
                    tag => return Err(malformed(format!("a text tag of {tag}"))),
                },
                edits: get_edits(&mut r)?,
            });
        }
        let mut layers = Vec::with_capacity(selected.len());
        for entry in self.layers.iter().filter(|entry| wanted.contains(&(entry.layer, entry.instance))) {
            let mut r = Reader::new(&self.bytes[entry.record.clone()]);
            layers.push(LayerRecord {
                layer: r.string()?,
                instance: r.string()?,
                schema: r.u16()?,
                len: r.u32()?,
                status: get_status(&mut r)?,
                provenance: get_provenance(&mut r)?,
                body: match r.u8()? {
                    0 => None,
                    1 => Some(r.bytes()?.to_vec()),
                    tag => return Err(malformed(format!("a body tag of {tag}"))),
                },
            });
        }
        Ok(DocumentGraph { version, source, views, layers })
    }

    /// Imports only the selected `(layer, instance)` layers (and
    /// every view) into a [`Document`], through [`selected_graph`]
    /// and [`import`]. Provenance references to unselected layers
    /// are dropped by that import exactly as its missing-input
    /// handling already permits for non-`Present` entries; a
    /// selected layer whose codec reads an unselected layer fails
    /// in its codec, as it would in a graph that never held it.
    ///
    /// [`selected_graph`]: DocumentGraphRef::selected_graph
    pub fn import_selected(&self, codecs: &Codecs, selected: &[(&str, &str)]) -> Result<Document, ExportError> {
        import(&self.selected_graph(selected)?, codecs)
    }
}
