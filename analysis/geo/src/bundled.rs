//! The bundled gazetteer: Natural Earth's populated places
//! (`opennlp.geo.BundledGazetteer`).

use std::sync::OnceLock;

use turbo_jvm::text::parse_double;

use crate::entry::GeoPoint;
use crate::error::{LoadError, NumberFormat, RowProblem, TableKind};
use crate::gazetteer::indexed_gazetteer;
use crate::index::{Alternates, GazetteerIndex};
use crate::load::{RowParser, Scratch, TableFormat, load_bytes};
use crate::store::{Row, RowAttribute};
use crate::text::parse_long;

/// The bundled table, OpenNLP's `naturalearth-populated-places.txt` byte for
/// byte (`data/README.md`).
pub const BUNDLED_TABLE: &str = include_str!("../data/naturalearth-populated-places.txt");

/// The file name of the bundled table, as OpenNLP's resource is named.
pub const BUNDLED_TABLE_NAME: &str = "naturalearth-populated-places.txt";

/// The gazetteer over the bundled table: the 7342 populated places of
/// Natural Earth's 1:10m Populated Places, which is in the public domain,
/// so lookups work with nothing downloaded. The names are ASCII; a query
/// with accents finds them through the folding.
///
/// [`BundledGazetteer::get`] parses and indexes the table once, on first
/// use, and returns the same instance after that.
pub struct BundledGazetteer {
    index: GazetteerIndex,
}

/// Eleven fields separated by semicolons: source, record id, name,
/// alternate names, latitude, longitude, country code, containment,
/// population, feature class and attributes. The list fields are `|`
/// separated, and the attributes are `key=value` pairs whose provenance is
/// the row's source. Nothing is trimmed.
const FIELDS: usize = 11;

const FORMAT: TableFormat = TableFormat {
    table: TableKind::Bundled,
    skip_comments: true,
    strip_byte_order_mark: false,
    alternates: Alternates::Reject,
};

impl BundledGazetteer {
    /// The shared instance. Parsing the bundled table cannot fail; a test
    /// loads it.
    pub fn get() -> &'static BundledGazetteer {
        static INSTANCE: OnceLock<BundledGazetteer> = OnceLock::new();
        INSTANCE.get_or_init(|| BundledGazetteer::parse().expect("the bundled gazetteer table is well formed"))
    }

    /// Parses and indexes the bundled table into a new instance, for
    /// measuring the load.
    pub fn parse() -> Result<BundledGazetteer, LoadError> {
        Ok(BundledGazetteer { index: load_bytes(BUNDLED_TABLE.as_bytes(), FORMAT, &BundledRows)? })
    }
}

indexed_gazetteer!(BundledGazetteer, |g| g.index.sources());

/// A `|` separated list field: nothing for an empty field, otherwise every
/// element, empty ones included, which the record check then refuses.
fn push_list<'t>(field: &'t str, out: &mut Vec<&'t str>) {
    if !field.is_empty() {
        out.extend(field.split('|'));
    }
}

struct BundledRows;

impl<'t> RowParser<'t> for BundledRows {
    fn parse<'s>(&'s self, line: &'t str, s: &'s mut Scratch<'t>) -> Result<Row<'s>, RowProblem> {
        s.fields.clear();
        s.fields.extend(line.split(';'));
        if s.fields.len() != FIELDS {
            return Err(RowProblem::Bundled(line.to_string()));
        }
        let f = &s.fields;
        let number = |cell: &str| NumberFormat(cell.to_string());
        let latitude = parse_double(f[4]).map_err(|_| number(f[4]))?;
        let longitude = parse_double(f[5]).map_err(|_| number(f[5]))?;
        let population = parse_long(f[8]).map_err(|_| number(f[8]))?;
        s.names.clear();
        push_list(f[3], &mut s.names);
        s.containment.clear();
        push_list(f[7], &mut s.containment);
        s.attributes.clear();
        if !f[10].is_empty() {
            for pair in f[10].split('|') {
                let equals = pair.find('=').filter(|&i| i > 0 && i + 1 < pair.len());
                let Some(equals) = equals else { return Err(RowProblem::AttributePair(pair.into())) };
                let key = &pair[..equals];
                if s.attributes.iter().any(|a| a.key == key) {
                    return Err(RowProblem::DuplicateAttribute(key.into()));
                }
                s.attributes.push(RowAttribute { key, value: &pair[equals + 1..], source: f[0], notes: "" });
            }
        }
        Ok(Row {
            source: f[0],
            record_id: f[1],
            name: f[2],
            alternates: &s.names,
            location: GeoPoint::new(latitude, longitude)?,
            bounding_box: None,
            country: (!f[6].is_empty()).then_some(f[6]),
            containment: &s.containment,
            population,
            feature_class: (!f[9].is_empty()).then_some(f[9]),
            attributes: &s.attributes,
        })
    }
}
