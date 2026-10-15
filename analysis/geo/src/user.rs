//! A gazetteer over a hand written table (`opennlp.geo.UserGazetteer`) and
//! the suppression rules file it reads.

use std::path::Path;

use crate::entry::{GeoBoundingBox, GeoPoint};
use crate::error::{ColumnCount, LoadError, RowProblem, TableKind};
use crate::gazetteer::indexed_gazetteer;
use crate::geonames::{double, population};
use crate::index::{Alternates, GazetteerIndex};
use crate::load::{RowParser, Scratch, TableFormat, line_of, load_bytes};
use crate::store::{Row, RowAttribute};
use crate::suppression::Suppression;
use crate::text::{BYTE_ORDER_MARK, Lines, is_blank, trim};

/// A gazetteer over a user written tab separated table, for places a
/// public dataset does not know (customer sites, internal facility names),
/// usually laid over a base gazetteer with an
/// [`OverlayGazetteer`](crate::OverlayGazetteer).
///
/// One row per place, the columns in this order; trailing columns may be
/// left out, and an empty column (after trimming Unicode white space) is
/// absent:
///
/// 1. record id, unique within the file
/// 2. canonical name
/// 3. alternate names, separated by `|`
/// 4. latitude in decimal degrees
/// 5. longitude in decimal degrees
/// 6. ISO 3166-1 alpha-2 country code, two capital letters
/// 7. feature class, conventionally `CITY`, `ADMIN` or `POI`
/// 8. population
/// 9. bounding box as `west,south,east,north`
/// 10. administrative containment, outermost first, separated by `|`
/// 11. and after: one `key=value` attribute each
///
/// A row needs both coordinates or a bounding box; with only the box, the
/// location is the box's center. Lines starting with `#` and blank lines
/// are skipped, and a byte order mark before the first line is dropped. A
/// canonical or alternate name that folds to nothing fails the load with
/// its line, since the author can fix it.
pub struct UserGazetteer {
    source: String,
    index: GazetteerIndex,
}

const FORMAT: TableFormat = TableFormat {
    table: TableKind::User,
    skip_comments: true,
    strip_byte_order_mark: true,
    alternates: Alternates::Reject,
};

impl UserGazetteer {
    /// Loads the table in the file at `path`, scoping its records to the
    /// dataset `source`, which must not be blank.
    pub fn load(path: impl AsRef<Path>, source: &str) -> Result<UserGazetteer, LoadError> {
        check_source(source)?;
        let bytes = std::fs::read(path)?;
        UserGazetteer::from_bytes(&bytes, source)
    }

    /// Loads a table from its bytes.
    pub fn from_bytes(bytes: &[u8], source: &str) -> Result<UserGazetteer, LoadError> {
        check_source(source)?;
        let index = load_bytes(bytes, FORMAT, &UserRows { source })?;
        Ok(UserGazetteer { source: source.to_string(), index })
    }

    /// The dataset identifier the records are scoped to.
    pub fn source(&self) -> &str {
        &self.source
    }

    pub(crate) fn index(&self) -> &GazetteerIndex {
        &self.index
    }

    pub(crate) fn from_index(source: String, index: GazetteerIndex) -> UserGazetteer {
        UserGazetteer { source, index }
    }

    /// Reads suppression rules from the file at `path`: one rule per line
    /// with up to three tab separated columns, name, country code and
    /// feature class, where trailing columns may be left out and an empty
    /// column is no filter. Comment and blank lines are skipped as in the
    /// table format. An empty file holds no rules.
    pub fn load_suppressions(path: impl AsRef<Path>) -> Result<Vec<Suppression>, LoadError> {
        UserGazetteer::suppressions_from_bytes(&std::fs::read(path)?)
    }

    /// Reads suppression rules from their bytes.
    pub fn suppressions_from_bytes(bytes: &[u8]) -> Result<Vec<Suppression>, LoadError> {
        let text = std::str::from_utf8(bytes).map_err(|e| LoadError::Utf8 { line: line_of(bytes, e.valid_up_to()) })?;
        let text = text.strip_prefix(BYTE_ORDER_MARK).unwrap_or(text);
        let mut rules = Vec::new();
        for (line, _, content) in Lines::new(text) {
            if is_blank(content) || content.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = content.split('\t').collect();
            if fields.len() > 3 {
                let problem = RowProblem::Columns { found: fields.len(), expected: ColumnCount::AtMost(3) };
                return Err(LoadError::Row { line, table: TableKind::Suppressions, problem });
            }
            let filter = |i: usize| (!absent(&fields, i)).then(|| trim(fields[i]));
            let rule = Suppression::new(trim(fields[0]), filter(1), filter(2)).map_err(|e| LoadError::Row {
                line,
                table: TableKind::Suppressions,
                problem: RowProblem::Suppression(e),
            })?;
            rules.push(rule);
        }
        Ok(rules)
    }
}

indexed_gazetteer!(UserGazetteer, |g| vec![g.source.as_str()]);

fn check_source(source: &str) -> Result<(), LoadError> {
    if is_blank(source) { Err(LoadError::BlankSource) } else { Ok(()) }
}

/// Whether column `i` is missing or empty after trimming.
fn absent(fields: &[&str], i: usize) -> bool {
    fields.get(i).is_none_or(|f| trim(f).is_empty())
}

/// Adds the trimmed, non-empty elements of a `|` separated column.
fn push_list<'t>(cell: &'t str, out: &mut Vec<&'t str>) {
    out.extend(cell.split('|').map(trim).filter(|s| !s.is_empty()));
}

/// The bounding box column: four decimal edges, west, south, east, north.
fn parse_box(cell: &str) -> Result<GeoBoundingBox, RowProblem> {
    let edges: Vec<&str> = cell.split(',').collect();
    let &[w, s, e, n] = edges.as_slice() else {
        return Err(RowProblem::BoxEdges(cell.to_string()));
    };
    Ok(GeoBoundingBox::new(double(w)?, double(s)?, double(e)?, double(n)?)?)
}

struct UserRows<'a> {
    source: &'a str,
}

impl<'t, 'a: 't> RowParser<'t> for UserRows<'a> {
    fn parse<'s>(&'s self, line: &'t str, s: &'s mut Scratch<'t>) -> Result<Row<'s>, RowProblem> {
        s.fields.clear();
        s.fields.extend(line.split('\t'));
        let f = &s.fields;
        if f.len() < 2 {
            return Err(RowProblem::Columns { found: f.len(), expected: ColumnCount::AtLeast(2) });
        }
        let bounding_box = if absent(f, 8) { None } else { Some(parse_box(trim(f[8]))?) };
        s.attributes.clear();
        for cell in &f[f.len().min(10)..] {
            let cell = trim(cell);
            let separator = cell.find('=').filter(|&i| i >= 1).ok_or_else(|| RowProblem::AttributePair(cell.into()))?;
            let key = trim(&cell[..separator]);
            let value = trim(&cell[separator + 1..]);
            if is_blank(value) {
                return Err(RowProblem::Entry(crate::entry::EntryError::Attribute("value")));
            }
            if s.attributes.iter().any(|a| a.key == key) {
                return Err(RowProblem::DuplicateAttribute(key.into()));
            }
            s.attributes.push(RowAttribute { key, value, source: self.source, notes: "" });
        }
        s.names.clear();
        if !absent(f, 2) {
            push_list(f[2], &mut s.names);
        }
        let location = match (absent(f, 3), absent(f, 4)) {
            (false, false) => GeoPoint::new(double(f[3])?, double(f[4])?)?,
            (true, true) => bounding_box.ok_or(RowProblem::NoLocation)?.center(),
            _ => return Err(RowProblem::HalfLocation),
        };
        s.containment.clear();
        if !absent(f, 9) {
            push_list(f[9], &mut s.containment);
        }
        let population = if absent(f, 7) { 0 } else { population(f[7])? };
        Ok(Row {
            source: self.source,
            record_id: trim(f[0]),
            name: trim(f[1]),
            alternates: &s.names,
            location,
            bounding_box,
            country: (!absent(f, 5)).then(|| trim(f[5])),
            containment: &s.containment,
            population,
            feature_class: (!absent(f, 6)).then(|| trim(f[6])),
            attributes: &s.attributes,
        })
    }
}
