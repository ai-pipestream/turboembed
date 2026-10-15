//! A gazetteer over a division table derived from Overture Maps data
//! (`opennlp.geo.OvertureGazetteer`).

use std::path::Path;

use crate::entry::{FEATURE_CLASS_ADMIN, FEATURE_CLASS_CITY, FEATURE_CLASS_POI, GeoPoint};
use crate::error::{ColumnCount, LoadError, RowProblem, TableKind};
use crate::gazetteer::indexed_gazetteer;
use crate::geonames::{double, population, push_alternates};
use crate::index::{Alternates, GazetteerIndex};
use crate::load::{RowParser, Scratch, TableFormat, load_bytes, load_path};
use crate::store::Row;
use crate::text::trim;

/// A gazetteer over a caller supplied division table derived from the
/// Overture Maps divisions theme
/// by OpenNLP's `OvertureGenerator`: one tab separated row of 8 columns per
/// division (id, primary name, comma separated alternate names, latitude,
/// longitude, country code, Overture subtype, population).
///
/// The upstream data is under the Open Database License, whose attribution
/// and share-alike terms follow any table derived from it, so nothing is
/// bundled here: the caller supplies the table.
///
/// Names are read as the GeoNames loader reads them, and an alternate name
/// that folds to nothing is likewise kept but not indexed. Subtypes map
/// coarsely: `locality` is `CITY`; `borough`, `macrohood`, `neighborhood`
/// and `microhood` are `POI`; every other subtype is `ADMIN`. Lines
/// starting with `#` carry the derivation record and are skipped.
pub struct OvertureGazetteer {
    index: GazetteerIndex,
}

/// The dataset identifier the records are scoped to.
pub const OVERTURE_SOURCE: &str = "overture";

const COLUMNS: usize = 8;

const FORMAT: TableFormat = TableFormat {
    table: TableKind::Overture,
    skip_comments: true,
    strip_byte_order_mark: true,
    alternates: Alternates::Skip,
};

impl OvertureGazetteer {
    /// Loads the table in the file at `path`.
    pub fn load(path: impl AsRef<Path>) -> Result<OvertureGazetteer, LoadError> {
        Ok(OvertureGazetteer { index: load_path(path.as_ref(), FORMAT, &OvertureRows)? })
    }

    /// Loads a table from its bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<OvertureGazetteer, LoadError> {
        Ok(OvertureGazetteer { index: load_bytes(bytes, FORMAT, &OvertureRows)? })
    }

    pub(crate) fn index(&self) -> &GazetteerIndex {
        &self.index
    }

    pub(crate) fn from_index(index: GazetteerIndex) -> OvertureGazetteer {
        OvertureGazetteer { index }
    }
}

indexed_gazetteer!(OvertureGazetteer, |_g| vec![OVERTURE_SOURCE]);

/// The coarse class of an Overture division subtype.
fn feature_class(subtype: &str) -> &'static str {
    match subtype {
        "locality" => FEATURE_CLASS_CITY,
        "borough" | "macrohood" | "neighborhood" | "microhood" => FEATURE_CLASS_POI,
        _ => FEATURE_CLASS_ADMIN,
    }
}

struct OvertureRows;

impl<'t> RowParser<'t> for OvertureRows {
    fn parse<'s>(&'s self, line: &'t str, s: &'s mut Scratch<'t>) -> Result<Row<'s>, RowProblem> {
        s.fields.clear();
        s.fields.extend(line.split('\t'));
        if s.fields.len() != COLUMNS {
            return Err(RowProblem::Columns { found: s.fields.len(), expected: ColumnCount::Exactly(COLUMNS) });
        }
        let f = &s.fields;
        let name = trim(f[1]);
        s.names.clear();
        s.seen.clear();
        push_alternates(f[2], name, &mut s.names, &mut s.seen);
        let location = GeoPoint::new(double(f[3])?, double(f[4])?)?;
        let country = trim(f[5]);
        let population = population(f[7])?;
        Ok(Row {
            source: OVERTURE_SOURCE,
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
