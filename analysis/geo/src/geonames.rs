//! A gazetteer over a file in the GeoNames main table format
//! (`opennlp.geo.GeoNamesGazetteer`).

use std::path::Path;

use turbo_jvm::text::parse_double;

use crate::entry::{FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY, FEATURE_CLASS_POI, GeoPoint};
use crate::error::{ColumnCount, LoadError, NumberFormat, RowProblem, TableKind};
use crate::gazetteer::indexed_gazetteer;
use crate::index::{Alternates, GazetteerIndex};
use crate::load::{RowParser, Scratch, TableFormat, load_bytes, load_path};
use crate::store::Row;
use crate::text::{parse_long, trim};

/// A gazetteer over a caller supplied file in the
/// GeoNames main table format (the `readme.txt` of the GeoNames dump):
/// one tab separated row of 19 columns per place.
///
/// The file is the caller's: nothing is bundled, and the GeoNames license
/// terms (CC BY 4.0, which asks for attribution) travel with it.
///
/// A record's names are the `name` column, the `asciiname` column when it
/// differs, and every comma separated alternate name, each trimmed of
/// Unicode white space, without the empty ones, the ones equal to the name
/// and repeats. An alternate name that folds to nothing (punctuation only)
/// is kept on the record but left out of the index, because the file is not
/// the caller's to correct. The feature classes map coarsely: `P` rows are
/// `CITY`, `A` rows `ADMIN`, every other row `POI`. Lines starting with `#`
/// are rows like any other and fail as such; a byte order mark before the
/// first line is dropped.
pub struct GeoNamesGazetteer {
    index: GazetteerIndex,
}

/// The dataset identifier the records are scoped to.
pub const GEONAMES_SOURCE: &str = "geonames";

const COLUMNS: usize = 19;

const FORMAT: TableFormat = TableFormat {
    table: TableKind::GeoNames,
    skip_comments: false,
    strip_byte_order_mark: true,
    alternates: Alternates::Skip,
};

impl GeoNamesGazetteer {
    /// Loads the table in the file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<GeoNamesGazetteer, LoadError> {
        Ok(GeoNamesGazetteer { index: load_path(path.as_ref(), FORMAT, &GeoNamesRows)? })
    }

    /// Loads a table from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<GeoNamesGazetteer, LoadError> {
        Ok(GeoNamesGazetteer { index: load_bytes(bytes, FORMAT, &GeoNamesRows)? })
    }

    pub(crate) fn index(&self) -> &GazetteerIndex {
        &self.index
    }

    pub(crate) fn from_index(index: GazetteerIndex) -> GeoNamesGazetteer {
        GeoNamesGazetteer { index }
    }
}

indexed_gazetteer!(GeoNamesGazetteer, |_g| vec![GEONAMES_SOURCE]);

/// The coarse class of a GeoNames feature class letter.
fn feature_class(class: &str) -> &'static str {
    match class {
        "P" => FEATURE_CLASS_CITY,
        "A" => FEATURE_CLASS_ADMIN,
        _ => FEATURE_CLASS_POI,
    }
}

/// `Double.parseDouble` of a trimmed cell.
pub(crate) fn double(cell: &str) -> Result<f64, NumberFormat> {
    let cell = trim(cell);
    parse_double(cell).map_err(|_| NumberFormat(cell.to_string()))
}

/// `Long.parseLong` of a trimmed cell, 0 for an empty one.
pub(crate) fn population(cell: &str) -> Result<i64, NumberFormat> {
    let cell = trim(cell);
    if cell.is_empty() { Ok(0) } else { parse_long(cell).map_err(|_| NumberFormat(cell.to_string())) }
}

/// Adds the trimmed alternate names of a comma separated cell to `names`,
/// leaving out the empty ones, the ones equal to `name` and repeats, as
/// OpenNLP's `LinkedHashSet` does.
pub(crate) fn push_alternates<'t>(cell: &'t str, name: &str, names: &mut Vec<&'t str>) {
    for alternate in cell.split(',') {
        push_alternate(trim(alternate), name, names);
    }
}

pub(crate) fn push_alternate<'t>(alternate: &'t str, name: &str, names: &mut Vec<&'t str>) {
    if !alternate.is_empty() && alternate != name && !names.contains(&alternate) {
        names.push(alternate);
    }
}

struct GeoNamesRows;

impl<'t> RowParser<'t> for GeoNamesRows {
    fn parse<'s>(&'s self, line: &'t str, s: &'s mut Scratch<'t>) -> Result<Row<'s>, RowProblem> {
        s.fields.clear();
        s.fields.extend(line.split('\t'));
        if s.fields.len() != COLUMNS {
            return Err(RowProblem::Columns { found: s.fields.len(), expected: ColumnCount::Exactly(COLUMNS) });
        }
        let f = &s.fields;
        let name = trim(f[1]);
        s.names.clear();
        push_alternate(trim(f[2]), name, &mut s.names);
        push_alternates(f[3], name, &mut s.names);
        let location = GeoPoint::new(double(f[4])?, double(f[5])?)?;
        let country = trim(f[8]);
        let population = population(f[14])?;
        Ok(Row {
            source: GEONAMES_SOURCE,
            record_id: trim(f[0]),
            name,
            alternates: &s.names,
            location,
            bounding_box: None,
            country: (!country.is_empty()).then_some(country),
            containment: &[],
            population,
            feature_class: Some(feature_class(trim(f[6]))),
            attributes: &[],
        })
    }
}
