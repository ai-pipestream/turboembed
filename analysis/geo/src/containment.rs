//! Place containment hierarchies, the `ContainmentSpine` TSV/CSV loader and
//! `HierarchyAnnotator` (`opennlp.tools.geo.PlaceHierarchy`,
//! `opennlp.geo.ContainmentSpine` and `HierarchyAnnotator`).

mod csv;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use turbo_analysis::{Annotator, Error, Refs, Span};
pub use turbo_analysis_layers::{Containment, ContainmentChain, ContainmentError, PlaceAncestor, PlaceAncestorRef};

use self::csv::{
    CsvRecordReader, WOF_ID_HEADER, WOF_NAME_HEADER, WOF_PARENT_HEADER, WOF_TYPE_HEADER, column_index,
    is_wof_root_parent,
};
use crate::entry::ATTRIBUTE_KEY_WHOSONFIRST;
use crate::geocode::Locations;
use crate::text::{Lines, is_blank};

/// The maximum number of ancestors returned for a single place
/// (`ContainmentSpine.MAX_DEPTH`).
pub const MAX_CONTAINMENT_DEPTH: usize = 64;

/// Why building or querying a [`ContainmentSpine`] or [`HierarchyAnnotator`]
/// failed.
#[derive(Debug)]
pub enum HierarchyError {
    /// A required argument (place id, name, type, or attribute key) was blank
    /// or a spine was built with zero places.
    InvalidArgument(String),
    /// A neutral TSV or Who's On First CSV table violated its format contract.
    Format(String),
    /// An underlying I/O or UTF-8 decoding error occurred.
    Io(std::io::Error),
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HierarchyError::InvalidArgument(msg) | HierarchyError::Format(msg) => f.write_str(msg),
            HierarchyError::Io(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl std::error::Error for HierarchyError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            HierarchyError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for HierarchyError {
    fn from(err: std::io::Error) -> Self {
        HierarchyError::Io(err)
    }
}

/// Resolves a place identifier to its outward administrative ancestors
/// (`opennlp.tools.geo.PlaceHierarchy`).
pub trait PlaceHierarchy {
    /// Returns the ancestors of `id`, immediate parent first and outermost
    /// ancestor last, or an empty vector when `id` is unknown or a root place.
    fn ancestors(&self, id: &str) -> Result<Vec<PlaceAncestor>, HierarchyError>;
}

impl<H: PlaceHierarchy + ?Sized> PlaceHierarchy for &H {
    fn ancestors(&self, id: &str) -> Result<Vec<PlaceAncestor>, HierarchyError> {
        (**self).ancestors(id)
    }
}

impl<H: PlaceHierarchy + ?Sized> PlaceHierarchy for Box<H> {
    fn ancestors(&self, id: &str) -> Result<Vec<PlaceAncestor>, HierarchyError> {
        (**self).ancestors(id)
    }
}

impl<H: PlaceHierarchy + ?Sized> PlaceHierarchy for std::sync::Arc<H> {
    fn ancestors(&self, id: &str) -> Result<Vec<PlaceAncestor>, HierarchyError> {
        (**self).ancestors(id)
    }
}

#[derive(Clone, Debug)]
struct Node {
    parent_id: Option<String>,
    name: String,
    kind: String,
}

/// An in-memory parent-pointer place hierarchy (`opennlp.geo.ContainmentSpine`).
#[derive(Clone, Debug)]
pub struct ContainmentSpine {
    nodes: HashMap<String, Node>,
}

impl ContainmentSpine {
    /// Creates a new builder for a [`ContainmentSpine`].
    pub fn builder() -> ContainmentSpineBuilder {
        ContainmentSpineBuilder::new()
    }

    /// Number of places in the spine.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the spine is empty (a built spine always has at least one place).
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl PlaceHierarchy for ContainmentSpine {
    fn ancestors(&self, id: &str) -> Result<Vec<PlaceAncestor>, HierarchyError> {
        if is_blank(id) {
            return Err(HierarchyError::InvalidArgument("id must not be null or blank".into()));
        }
        let Some(current) = self.nodes.get(id) else {
            return Ok(Vec::new());
        };
        let Some(mut parent_id) = current.parent_id.as_deref() else {
            return Ok(Vec::new());
        };

        let mut chain = Vec::new();
        let mut visited: HashSet<&str> = HashSet::new();
        visited.insert(id);

        while chain.len() < MAX_CONTAINMENT_DEPTH {
            if !visited.insert(parent_id) {
                break;
            }
            let Some(parent) = self.nodes.get(parent_id) else {
                break;
            };
            chain.push(
                PlaceAncestor::new(parent_id, &parent.name, &parent.kind)
                    .map_err(|e| HierarchyError::InvalidArgument(e.to_string()))?,
            );
            match parent.parent_id.as_deref() {
                Some(next) => parent_id = next,
                None => break,
            }
        }
        Ok(chain)
    }
}

/// Builds a [`ContainmentSpine`] from explicit entries, neutral TSV tables, or
/// Who's On First meta CSV files (`opennlp.geo.ContainmentSpine.Builder`).
#[derive(Clone, Debug, Default)]
pub struct ContainmentSpineBuilder {
    nodes: HashMap<String, Node>,
}

impl ContainmentSpineBuilder {
    /// Creates an empty builder.
    pub fn new() -> ContainmentSpineBuilder {
        ContainmentSpineBuilder::default()
    }

    /// Adds a place with an optional `parent_id`, canonical `name`, and place
    /// `kind`. Adding an existing `id` replaces the previous entry.
    pub fn add(
        &mut self,
        id: &str,
        parent_id: Option<&str>,
        name: &str,
        kind: &str,
    ) -> Result<&mut Self, HierarchyError> {
        if is_blank(id) {
            return Err(HierarchyError::InvalidArgument("id must not be null or blank".into()));
        }
        if is_blank(name) {
            return Err(HierarchyError::InvalidArgument("name must not be null or blank".into()));
        }
        if is_blank(kind) {
            return Err(HierarchyError::InvalidArgument("type must not be null or blank".into()));
        }
        if let Some(p) = parent_id
            && is_blank(p)
        {
            return Err(HierarchyError::InvalidArgument("parentId must not be blank".into()));
        }
        self.nodes.insert(
            id.to_string(),
            Node { parent_id: parent_id.map(Into::into), name: name.to_string(), kind: kind.to_string() },
        );
        Ok(self)
    }

    /// Reads a four-column tab-separated containment table (`id`, `parent_id`,
    /// `name`, `type`) from `path`. Blank lines and `#` comment lines are
    /// skipped.
    pub fn add_table(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, HierarchyError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        self.add_table_bytes(&bytes, &path.display().to_string())
    }

    /// Reads a four-column tab-separated containment table from `bytes`.
    pub fn add_table_bytes(&mut self, bytes: &[u8], source: &str) -> Result<&mut Self, HierarchyError> {
        let text = std::str::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        for (line_no, _, line) in Lines::new(text) {
            if is_blank(line) || line.starts_with('#') {
                continue;
            }
            let columns: Vec<&str> = line.split('\t').collect();
            if columns.len() != 4 {
                return Err(HierarchyError::Format(format!(
                    "Expected 4 tab-separated columns (id, parent_id, name, type) at {source} line {line_no}, got {}",
                    columns.len()
                )));
            }
            let parent = if columns[1].is_empty() { None } else { Some(columns[1]) };
            self.add(columns[0], parent, columns[2], columns[3]).map_err(|e| {
                HierarchyError::Format(format!("Invalid containment row at {source} line {line_no}: {e}"))
            })?;
        }
        Ok(self)
    }

    /// Reads a Who's On First meta CSV file (RFC 4180 with `id`, `parent_id`,
    /// `name`, and `placetype` headers) from `path`.
    pub fn add_wof_meta(&mut self, path: impl AsRef<Path>) -> Result<&mut Self, HierarchyError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        self.add_wof_meta_bytes(&bytes, &path.display().to_string())
    }

    /// Reads a Who's On First meta CSV file from `bytes`.
    pub fn add_wof_meta_bytes(&mut self, bytes: &[u8], source: &str) -> Result<&mut Self, HierarchyError> {
        let text = std::str::from_utf8(bytes).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut records = CsvRecordReader::new(text, source);
        let Some((header, _)) = records.next_record()? else {
            return Err(HierarchyError::Format(format!("Empty WOF meta CSV: {source}")));
        };
        let id_col = column_index(&header, WOF_ID_HEADER, source)?;
        let parent_col = column_index(&header, WOF_PARENT_HEADER, source)?;
        let name_col = column_index(&header, WOF_NAME_HEADER, source)?;
        let type_col = column_index(&header, WOF_TYPE_HEADER, source)?;
        let max_col = id_col.max(parent_col).max(name_col).max(type_col);

        while let Some((row, line_no)) = records.next_record()? {
            if row.len() == 1 && is_blank(&row[0]) {
                continue;
            }
            if row.len() <= max_col {
                return Err(HierarchyError::Format(format!(
                    "WOF meta row at {source} line {line_no} has {} columns, need at least {}",
                    row.len(),
                    max_col + 1
                )));
            }
            let id = &row[id_col];
            if is_blank(id) {
                return Err(HierarchyError::Format(format!(
                    "WOF meta row at {source} line {line_no} has an empty id column"
                )));
            }
            let raw_parent = &row[parent_col];
            let parent =
                if is_wof_root_parent(raw_parent, source, line_no)? { None } else { Some(raw_parent.as_str()) };
            let name = &row[name_col];
            if is_blank(name) {
                return Err(HierarchyError::Format(format!(
                    "WOF meta row for id '{id}' at {source} line {line_no} has an empty name"
                )));
            }
            let place_type = &row[type_col];
            if is_blank(place_type) {
                return Err(HierarchyError::Format(format!(
                    "WOF meta row for id '{id}' at {source} line {line_no} has an empty placetype"
                )));
            }
            self.add(id, parent, name, place_type)?;
        }
        Ok(self)
    }

    /// Finishes building the [`ContainmentSpine`]; requires at least one place.
    pub fn build(&self) -> Result<ContainmentSpine, HierarchyError> {
        if self.nodes.is_empty() {
            return Err(HierarchyError::InvalidArgument("ContainmentSpine requires at least one place".into()));
        }
        Ok(ContainmentSpine { nodes: self.nodes.clone() })
    }
}

/// Expands resolved mentions in `turbo:locations` into administrative
/// containment chains in `turbo:containment` (`opennlp.geo.HierarchyAnnotator`).
pub struct HierarchyAnnotator<H> {
    hierarchy: H,
    attribute_key: String,
}

impl<H: PlaceHierarchy> HierarchyAnnotator<H> {
    /// Creates an annotator joining on [`ATTRIBUTE_KEY_WHOSONFIRST`].
    pub fn new(hierarchy: H) -> HierarchyAnnotator<H> {
        HierarchyAnnotator { hierarchy, attribute_key: ATTRIBUTE_KEY_WHOSONFIRST.to_string() }
    }

    /// Creates an annotator joining on `attribute_key`.
    pub fn with_attribute_key(
        hierarchy: H,
        attribute_key: impl Into<String>,
    ) -> Result<HierarchyAnnotator<H>, HierarchyError> {
        let attribute_key = attribute_key.into();
        if is_blank(&attribute_key) {
            return Err(HierarchyError::InvalidArgument("attributeKey must not be null or blank".into()));
        }
        Ok(HierarchyAnnotator { hierarchy, attribute_key })
    }

    /// The attribute key used to join gazetteer entries to the hierarchy.
    pub fn attribute_key(&self) -> &str {
        &self.attribute_key
    }
}

impl<H: PlaceHierarchy + Send + Sync + 'static> Annotator for HierarchyAnnotator<H> {
    const ID: &'static str = "turbo-geo:hierarchy";
    type Reads = (Locations,);
    type Writes = (Containment,);

    fn annotate(&self, (locations,): Refs<'_, Self>) -> Result<(Containment,), Error> {
        let mut out = Containment::new(locations.spans.view());
        let mut seen_spans: HashSet<Span> = HashSet::new();

        for (ix, span) in locations.spans.iter() {
            if !seen_spans.insert(span) {
                continue;
            }
            let entry = &locations.entry[ix.get()];
            let Some(join_id) = entry.attribute(&self.attribute_key) else {
                continue;
            };
            let ancestors = self
                .hierarchy
                .ancestors(join_id.value())
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
            if ancestors.is_empty() {
                continue;
            }
            for ancestor in &ancestors {
                if ancestor.id() == join_id.value() {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!("place {} appears in its own ancestors", join_id.value()),
                    });
                }
            }
            out.push_ancestors(span, &ancestors)
                .map_err(|e| Error::Annotator { annotator: Self::ID, detail: e.to_string() })?;
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        turbo_analysis::Fnv1a::new().str(&self.attribute_key).finish()
    }
}
