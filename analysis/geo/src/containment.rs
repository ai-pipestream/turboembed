//! Place containment hierarchies, the `ContainmentSpine` TSV/CSV loader and
//! `HierarchyAnnotator` (`opennlp.tools.geo.PlaceHierarchy`,
//! `opennlp.geo.ContainmentSpine` and `HierarchyAnnotator`).

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use turbo_analysis::{Annotator, Error, Refs, Span};
pub use turbo_analysis_layers::{Containment, ContainmentChain, ContainmentError, PlaceAncestor, PlaceAncestorRef};

use crate::entry::ATTRIBUTE_KEY_WHOSONFIRST;
use crate::geocode::Locations;
use crate::text::{Lines, is_blank};

/// The maximum number of ancestors returned for a single place
/// (`ContainmentSpine.MAX_DEPTH`).
pub const MAX_CONTAINMENT_DEPTH: usize = 64;

const WOF_ID_HEADER: &str = "id";
const WOF_PARENT_HEADER: &str = "parent_id";
const WOF_NAME_HEADER: &str = "name";
const WOF_TYPE_HEADER: &str = "placetype";

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
        let text = std::str::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
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
        let text = std::str::from_utf8(bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
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
            let parent = if is_wof_root_parent(raw_parent, source, line_no)? {
                None
            } else {
                Some(raw_parent.as_str())
            };
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
            return Err(HierarchyError::InvalidArgument(
                "ContainmentSpine requires at least one place".into(),
            ));
        }
        Ok(ContainmentSpine { nodes: self.nodes.clone() })
    }
}

fn column_index(header: &[String], name: &str, source: &str) -> Result<usize, HierarchyError> {
    header.iter().position(|h| h == name).ok_or_else(|| {
        HierarchyError::Format(format!("WOF meta CSV at {source} is missing required column '{name}'"))
    })
}

fn is_wof_root_parent(value: &str, source: &str, line_no: usize) -> Result<bool, HierarchyError> {
    if is_blank(value) {
        return Ok(true);
    }
    let bytes = value.as_bytes();
    let start = usize::from(bytes[0] == b'-');
    if start == bytes.len() || !bytes[start..].iter().all(u8::is_ascii_digit) {
        return Err(HierarchyError::Format(format!(
            "WOF meta row at {source} line {line_no} has non-numeric parent_id '{value}'"
        )));
    }
    if start == 1 {
        return Ok(true);
    }
    Ok(bytes.iter().all(|&b| b == b'0'))
}

/// RFC 4180 CSV record reader matching `ContainmentSpine.CsvRecordReader`.
struct CsvRecordReader<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
    source: &'a str,
    line_number: usize,
}

impl<'a> CsvRecordReader<'a> {
    fn new(text: &'a str, source: &'a str) -> CsvRecordReader<'a> {
        CsvRecordReader { chars: text.chars().peekable(), source, line_number: 1 }
    }

    fn next_record(&mut self) -> Result<Option<(Vec<String>, usize)>, HierarchyError> {
        if self.chars.peek().is_none() {
            return Ok(None);
        }
        let record_line = self.line_number;
        let mut fields = Vec::new();
        let mut field = String::new();
        loop {
            match self.chars.peek().copied() {
                Some('"') => {
                    let _ = self.chars.next();
                    let finished = self.read_quoted_field(&mut field, record_line)?;
                    fields.push(std::mem::take(&mut field));
                    if finished {
                        return Ok(Some((fields, record_line)));
                    }
                }
                _ => {
                    let finished = self.read_unquoted_field(&mut field)?;
                    fields.push(std::mem::take(&mut field));
                    if finished {
                        return Ok(Some((fields, record_line)));
                    }
                }
            }
        }
    }

    fn read_unquoted_field(&mut self, field: &mut String) -> Result<bool, HierarchyError> {
        while let Some(c) = self.chars.next() {
            match c {
                ',' => return Ok(false),
                '\r' => {
                    if self.chars.peek() == Some(&'\n') {
                        let _ = self.chars.next();
                    }
                    self.line_number += 1;
                    return Ok(true);
                }
                '\n' => {
                    self.line_number += 1;
                    return Ok(true);
                }
                '"' => {
                    return Err(HierarchyError::Format(format!(
                        "Malformed CSV at {} line {}: stray quote in unquoted field",
                        self.source, self.line_number
                    )));
                }
                other => field.push(other),
            }
        }
        Ok(true)
    }

    fn read_quoted_field(&mut self, field: &mut String, record_line: usize) -> Result<bool, HierarchyError> {
        while let Some(c) = self.chars.next() {
            match c {
                '"' => {
                    if self.chars.peek() == Some(&'"') {
                        let _ = self.chars.next();
                        field.push('"');
                        continue;
                    }
                    return match self.chars.next() {
                        None => Ok(true),
                        Some(',') => Ok(false),
                        Some('\r') => {
                            if self.chars.peek() == Some(&'\n') {
                                let _ = self.chars.next();
                            }
                            self.line_number += 1;
                            Ok(true)
                        }
                        Some('\n') => {
                            self.line_number += 1;
                            Ok(true)
                        }
                        Some(_) => Err(HierarchyError::Format(format!(
                            "Malformed CSV at {} line {}: unexpected character after a closing quote",
                            self.source, self.line_number
                        ))),
                    };
                }
                '\r' => {
                    if self.chars.peek() == Some(&'\n') {
                        let _ = self.chars.next();
                    }
                    self.line_number += 1;
                    field.push('\n');
                }
                '\n' => {
                    self.line_number += 1;
                    field.push('\n');
                }
                other => field.push(other),
            }
        }
        Err(HierarchyError::Format(format!(
            "Malformed CSV at {} line {}: unterminated quoted field",
            self.source, record_line
        )))
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
            return Err(HierarchyError::InvalidArgument(
                "attributeKey must not be null or blank".into(),
            ));
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
            let ancestors = self.hierarchy.ancestors(join_id.value()).map_err(|e| Error::Annotator {
                annotator: Self::ID,
                detail: e.to_string(),
            })?;
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
            out.push_ancestors(span, &ancestors).map_err(|e| Error::Annotator {
                annotator: Self::ID,
                detail: e.to_string(),
            })?;
        }
        Ok((out,))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entry::{
        ATTRIBUTE_KEY_GEONAMES, AttributeValue, EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoPoint,
    };
    use crate::geocode::{GeoResolution, GeocodeAnnotator};
    use crate::region::DocumentRegionAnnotator;
    use turbo_analysis::{Analyzer, Document, Ix, IxRange, Key, Layer, Provenance, Spans, Tag, TagSet, ViewId};
    use turbo_analysis_layers::{Entities, Tokens};
    use turbo_numeric::{CONVERTED_MONEY, Currency, Date, EcbRates, Money, MoneyConverter, RegionAwareMoneyAnnotator};

    fn run_one(a: &Analyzer, mut doc: Document) -> Document {
        a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
        doc
    }

    fn park_slope_spine() -> ContainmentSpine {
        let mut b = ContainmentSpine::builder();
        b.add("85865587", Some("421205765"), "Park Slope", "neighbourhood")
            .unwrap()
            .add("421205765", Some("85977539"), "Brooklyn", "borough")
            .unwrap()
            .add("85977539", Some("85688543"), "New York", "locality")
            .unwrap()
            .add("85688543", Some("85633793"), "New York", "region")
            .unwrap()
            .add("85633793", None, "United States", "country")
            .unwrap();
        b.build().unwrap()
    }

    fn entry_with_wof(name: &str, wof_id: Option<&str>) -> GazetteerEntry {
        let attributes = match wof_id {
            Some(id) => vec![(
                ATTRIBUTE_KEY_WHOSONFIRST.to_string(),
                AttributeValue::new(id, "test", "fixture").unwrap(),
            )],
            None => Vec::new(),
        };
        GazetteerEntry::new(EntryParts {
            source: "test".into(),
            record_id: name.into(),
            name: name.into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(0.0, 0.0).unwrap(),
            bounding_box: None,
            country_code: Some("US".into()),
            containment: Vec::new(),
            population: 1000,
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes,
        })
        .unwrap()
    }

    #[test]
    fn ancestors_walk_outward_and_handle_roots_missing_parents_duplicates_cycles_and_depth_cap() {
        let spine = park_slope_spine();
        let chain = spine.ancestors("85865587").unwrap();
        assert_eq!(chain.len(), 4);
        assert_eq!(chain[0].name(), "Brooklyn");
        assert_eq!(chain[0].place_type(), "borough");
        assert_eq!(chain[1].place_type(), "locality");
        assert_eq!(chain[2].place_type(), "region");
        assert_eq!(chain[3].name(), "United States");
        assert_eq!(chain[3].place_type(), "country");

        assert!(spine.ancestors("999").unwrap().is_empty());
        assert!(spine.ancestors("85633793").unwrap().is_empty());

        // Missing parent reference ends chain.
        let mut b = ContainmentSpine::builder();
        b.add("x", Some("y"), "X", "t").unwrap().add("y", Some("zz"), "Y", "t").unwrap();
        let missing = b.build().unwrap();
        assert_eq!(missing.ancestors("x").unwrap(), vec![PlaceAncestor::new("y", "Y", "t").unwrap()]);
        assert!(missing.ancestors("y").unwrap().is_empty());

        // Duplicate IDs keep last addition.
        let mut b_dup = ContainmentSpine::builder();
        b_dup
            .add("child", Some("dup"), "Child", "t")
            .unwrap()
            .add("dup", None, "First", "t1")
            .unwrap()
            .add("dup", None, "Second", "t2")
            .unwrap();
        assert_eq!(
            b_dup.build().unwrap().ancestors("child").unwrap(),
            vec![PlaceAncestor::new("dup", "Second", "t2").unwrap()]
        );

        // Self-loop and triangle cycles terminate without repeating an identifier.
        let mut b_self = ContainmentSpine::builder();
        b_self.add("a", Some("a"), "A", "t").unwrap();
        assert!(b_self.build().unwrap().ancestors("a").unwrap().is_empty());

        let mut b_tri = ContainmentSpine::builder();
        b_tri
            .add("a", Some("b"), "A", "t")
            .unwrap()
            .add("b", Some("c"), "B", "t")
            .unwrap()
            .add("c", Some("a"), "C", "t")
            .unwrap();
        assert_eq!(
            b_tri.build().unwrap().ancestors("a").unwrap(),
            vec![PlaceAncestor::new("b", "B", "t").unwrap(), PlaceAncestor::new("c", "C", "t").unwrap()]
        );

        // 64-ancestor depth cap.
        let mut b_deep = ContainmentSpine::builder();
        for i in 0..70 {
            let id = format!("p{i}");
            let parent = (i < 69).then(|| format!("p{}", i + 1));
            let name = format!("P{i}");
            b_deep.add(&id, parent.as_deref(), &name, "t").unwrap();
        }
        let deep_chain = b_deep.build().unwrap().ancestors("p0").unwrap();
        assert_eq!(deep_chain.len(), 64);
        assert_eq!(deep_chain[0].id(), "p1");
        assert_eq!(deep_chain[63].id(), "p64");
    }

    #[test]
    fn neutral_table_and_wof_meta_csv_loaders_and_validation() {
        let tsv = "# id\tparent\tname\ttype\nQ123\tQ60\tSoHo\tneighbourhood\nQ60\tQ1384\tNew York City\tcity\nQ1384\t\tNew York\tstate\n";
        let mut b = ContainmentSpine::builder();
        b.add_table_bytes(tsv.as_bytes(), "containment.tsv").unwrap();
        let chain = b.build().unwrap().ancestors("Q123").unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain[0].name(), "New York City");
        assert_eq!(chain[1].name(), "New York");

        // WOF CSV with quoted commas, quoted newlines, CRLF, and doubled quotes.
        let csv = "bbox,id,name,parent_id,placetype,source\n\"1,2,3,4\",85977539,Brooklyn,85688543,borough,mz\n\"5,6,7,8\",85688543,\"New York, the Big Apple\",-1,locality,mz\n";
        let mut b_wof = ContainmentSpine::builder();
        b_wof.add_wof_meta_bytes(csv.as_bytes(), "wof.csv").unwrap();
        let wof_chain = b_wof.build().unwrap().ancestors("85977539").unwrap();
        assert_eq!(wof_chain.len(), 1);
        assert_eq!(wof_chain[0].name(), "New York, the Big Apple");

        let crlf_csv = "id,parent_id,name,placetype\r\n999,123,Child,neighbourhood\r\n123,456,\"Sao Paulo\r\nZona Norte\",locality\r\n456,-1,\"The \"\"Big Apple\"\"\",country\r\n";
        let mut b_crlf = ContainmentSpine::builder();
        b_crlf.add_wof_meta_bytes(crlf_csv.as_bytes(), "crlf.csv").unwrap();
        let crlf_chain = b_crlf.build().unwrap().ancestors("999").unwrap();
        assert_eq!(
            crlf_chain,
            vec![
                PlaceAncestor::new("123", "Sao Paulo\nZona Norte", "locality").unwrap(),
                PlaceAncestor::new("456", "The \"Big Apple\"", "country").unwrap(),
            ]
        );

        // Error reporting checks.
        let unterm = "id,parent_id,name,placetype\n123,456,\"Sao Paulo,locality\n456,-1,Brazil,country\n";
        let err = ContainmentSpine::builder().add_wof_meta_bytes(unterm.as_bytes(), "u.csv").unwrap_err().to_string();
        assert!(err.contains("unterminated quoted field") && err.contains("line 2"), "{err}");

        let empty_name = "id,parent_id,name,placetype\n123,456,\"Sao Paulo\nZona Norte\",locality\n456,-1,,country\n";
        let err2 = ContainmentSpine::builder()
            .add_wof_meta_bytes(empty_name.as_bytes(), "en.csv")
            .unwrap_err()
            .to_string();
        assert!(err2.contains("empty name") && err2.contains("line 4"), "{err2}");

        let stray = "id,parent_id,name,placetype\n123,456,Kings\"County,county\n";
        let err3 = ContainmentSpine::builder().add_wof_meta_bytes(stray.as_bytes(), "s.csv").unwrap_err().to_string();
        assert!(err3.contains("stray quote") && err3.contains("line 2"), "{err3}");

        let after_q = "id,parent_id,name,placetype\n123,456,\"Kings\"County,county\n";
        let err4 = ContainmentSpine::builder().add_wof_meta_bytes(after_q.as_bytes(), "a.csv").unwrap_err().to_string();
        assert!(err4.contains("after a closing quote") && err4.contains("line 2"), "{err4}");

        for bad_parent in ["abc", "+1", "-x"] {
            let bad_csv = format!("id,parent_id,name,placetype\n1,{bad_parent},Place,locality\n");
            assert!(ContainmentSpine::builder().add_wof_meta_bytes(bad_csv.as_bytes(), "p.csv").is_err());
        }

        // Malformed UTF-8 rejected as Io error.
        assert!(matches!(
            ContainmentSpine::builder().add_table_bytes(&[b'x', b'\t', b'\t', 0xc3, b'\t', b't', b'\n'], "bad.tsv"),
            Err(HierarchyError::Io(_))
        ));
    }

    #[test]
    fn hierarchy_annotator_expands_resolved_mentions_and_handles_multi_candidate_spans() {
        let mut b = ContainmentSpine::builder();
        b.add("85865587", Some("421205765"), "Park Slope", "neighbourhood")
            .unwrap()
            .add("421205765", Some("85977539"), "Brooklyn", "borough")
            .unwrap()
            .add("85977539", None, "New York", "locality")
            .unwrap();
        let spine = b.build().unwrap();

        let text = "A stroll through Park Slope.";
        let mention = Span::new(17, 27);
        let mut doc = Document::new(text);
        let mut locs = Locations::new(ViewId::SOURCE);
        locs.push(GeoResolution::new(mention, entry_with_wof("Park Slope", Some("85865587")), 0.9).unwrap());
        doc.add(&Key::DEFAULT, locs, Provenance::by("test")).unwrap();

        let analyzer = Analyzer::builder()
            .add(HierarchyAnnotator::new(spine.clone()))
            .given(Key::<Locations>::DEFAULT)
            .build()
            .unwrap();
        let out = run_one(&analyzer, doc);
        let cont = out.get::<Containment>().unwrap();
        assert_eq!(cont.len(), 1);
        assert_eq!(cont.spans.get(Ix::new(0)), mention);
        let names: Vec<&str> = cont.ancestors(Ix::new(0)).map(|a| a.name).collect();
        assert_eq!(names, ["Brooklyn", "New York"]);

        // Unexpandable best candidate suppresses second candidate on same span.
        let mut doc2 = Document::new("A week in Paris.");
        let m2 = Span::new(10, 15);
        let mut locs2 = Locations::new(ViewId::SOURCE);
        locs2.push(GeoResolution::new(m2, entry_with_wof("Paris", None), 0.7).unwrap());
        locs2.push(GeoResolution::new(m2, entry_with_wof("Paris", Some("85865587")), 0.3).unwrap());
        doc2.add(&Key::DEFAULT, locs2, Provenance::by("test")).unwrap();
        let out2 = run_one(&analyzer, doc2);
        assert_eq!(out2.get::<Containment>().unwrap().len(), 0);

        // Custom attribute key joins on that attribute.
        let mut doc3 = Document::new(text);
        let mut locs3 = Locations::new(ViewId::SOURCE);
        let entry_geo = GazetteerEntry::new(EntryParts {
            source: "test".into(),
            record_id: "Park Slope".into(),
            name: "Park Slope".into(),
            alternate_names: Vec::new(),
            location: GeoPoint::new(0.0, 0.0).unwrap(),
            bounding_box: None,
            country_code: Some("US".into()),
            containment: Vec::new(),
            population: 1000,
            feature_class: Some(FEATURE_CLASS_CITY.into()),
            attributes: vec![
                (
                    ATTRIBUTE_KEY_WHOSONFIRST.to_string(),
                    AttributeValue::new("77", "test", "fixture").unwrap(),
                ),
                (
                    ATTRIBUTE_KEY_GEONAMES.to_string(),
                    AttributeValue::new("85865587", "test", "fixture").unwrap(),
                ),
            ],
        })
        .unwrap();
        locs3.push(GeoResolution::new(mention, entry_geo, 0.9).unwrap());
        doc3.add(&Key::DEFAULT, locs3, Provenance::by("test")).unwrap();

        let custom_ann = HierarchyAnnotator::with_attribute_key(spine, ATTRIBUTE_KEY_GEONAMES).unwrap();
        let custom_analyzer = Analyzer::builder().add(custom_ann).given(Key::<Locations>::DEFAULT).build().unwrap();
        let out3 = run_one(&custom_analyzer, doc3);
        assert_eq!(out3.get::<Containment>().unwrap().len(), 1);
    }

    #[test]
    fn end_to_end_pipeline_from_entities_to_converted_money() {
        struct SydneyGeocoder;
        impl crate::geocode::Geocoder for SydneyGeocoder {
            fn resolve(
                &self,
                _text: &str,
                location_mentions: &[Span],
            ) -> Result<Vec<GeoResolution>, crate::geocode::GeocodeError> {
                let entry = GazetteerEntry::new(EntryParts {
                    source: "test".into(),
                    record_id: "sydney".into(),
                    name: "Sydney".into(),
                    alternate_names: Vec::new(),
                    location: GeoPoint::new(-33.87, 151.21).unwrap(),
                    bounding_box: None,
                    country_code: Some("AU".into()),
                    containment: Vec::new(),
                    population: 5_300_000,
                    feature_class: Some(FEATURE_CLASS_CITY.into()),
                    attributes: Vec::new(),
                })
                .unwrap();
                location_mentions
                    .iter()
                    .map(|&m| GeoResolution::new(m, entry.clone(), 0.9))
                    .collect()
            }
        }

        let rates = EcbRates::from_bytes(b"Date,USD,AUD,\n2026-07-10,1.2000,1.8000,\n").unwrap();
        let text = "a Sydney startup raised $3 million this week";
        let mut doc = Document::new(text);
        doc.add(
            &Key::DEFAULT,
            Tokens { spans: Spans::new(ViewId::SOURCE), sentence: None },
            Provenance::by("test"),
        )
        .unwrap();
        let kinds = TagSet::new("ner", &["location"]);
        let mut spans = Spans::new(ViewId::SOURCE);
        let start = text.find("Sydney").unwrap();
        spans.push(Span::new(start as u32, (start + 6) as u32));
        doc.add(
            &Key::DEFAULT,
            Entities {
                spans,
                tokens: vec![IxRange::new(0, 0)],
                kind: vec![Tag(0)],
                kinds,
                score: vec![1.0],
            },
            Provenance::by("test"),
        )
        .unwrap();

        let usd = Currency::parse("USD").unwrap();
        let as_of = Date::parse("2026-07-10").unwrap();
        let analyzer = Analyzer::builder()
            .add(GeocodeAnnotator::new(SydneyGeocoder))
            .add(DocumentRegionAnnotator::new())
            .add(RegionAwareMoneyAnnotator::new())
            .step(MoneyConverter::new(rates, usd, as_of).bound())
            .given(Key::<Entities>::DEFAULT)
            .build()
            .unwrap();

        let out = run_one(&analyzer, doc);
        let identified = out.get::<Money>().unwrap();
        assert_eq!(identified.len(), 1);
        assert_eq!(identified.get(Ix::new(0)).currency.as_str(), "AUD");
        let id_span = identified.get(Ix::new(0)).span;
        assert_eq!(&out.source()[id_span.start as usize..id_span.end as usize], "$3 million");

        let converted = out.get_key(&CONVERTED_MONEY).unwrap();
        assert_eq!(converted.len(), 1);
        assert_eq!(converted.get(Ix::new(0)).currency.as_str(), "USD");
        assert_eq!(converted.amount(Ix::new(0)).to_plain_string(), "2000000");
        assert_eq!(converted.get(Ix::new(0)).span, identified.get(Ix::new(0)).span);
    }
}
