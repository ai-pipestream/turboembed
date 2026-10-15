//! A gazetteer over records a caller builds (`opennlp.geo.InMemoryGazetteer`).

use crate::entry::GazetteerEntry;
use crate::error::IndexError;
use crate::fold::Folder;
use crate::gazetteer::indexed_gazetteer;
use crate::index::{Alternates, BuildError, Fragment, GazetteerIndex, IndexBuilder};
use crate::store::{Row, RowAttribute};

/// A gazetteer over caller supplied records, indexed and ranked like the
/// table loaders' records, for place records from a database, a service or
/// anywhere else without a file format in between. The bundled table is
/// not involved.
pub struct InMemoryGazetteer {
    index: GazetteerIndex,
}

impl InMemoryGazetteer {
    /// Indexes `entries`. Two records with the same source and record id,
    /// or a record with a canonical or alternate name that folds to
    /// nothing, are refused.
    pub fn from_entries(entries: &[GazetteerEntry]) -> Result<InMemoryGazetteer, IndexError> {
        let mut fragment = Fragment::new();
        let mut folder = Folder::new();
        let mut key = String::new();
        let mut alternates: Vec<&str> = Vec::new();
        let mut containment: Vec<&str> = Vec::new();
        let mut attributes: Vec<RowAttribute<'_>> = Vec::new();
        for entry in entries {
            let p = entry.parts();
            alternates.clear();
            alternates.extend(p.alternate_names.iter().map(String::as_str));
            containment.clear();
            containment.extend(p.containment.iter().map(String::as_str));
            attributes.clear();
            attributes.extend(p.attributes.iter().map(|(k, v)| RowAttribute {
                key: k,
                value: v.value(),
                source: v.source(),
                notes: v.notes(),
            }));
            let row = Row {
                source: &p.source,
                record_id: &p.record_id,
                name: &p.name,
                alternates: &alternates,
                location: p.location,
                bounding_box: p.bounding_box,
                country: p.country_code.as_deref(),
                containment: &containment,
                population: p.population,
                feature_class: p.feature_class.as_deref(),
                attributes: &attributes,
            };
            fragment.push(&mut folder, &mut key, &row, entry.country(), Alternates::Reject, 0)?;
        }
        let mut builder = IndexBuilder::new();
        builder.add(fragment).map_err(|e| match e {
            BuildError::Capacity(c) => IndexError::Capacity(c),
            BuildError::Duplicate { row, record_id } => {
                IndexError::Duplicate { source: entries[row].parts().source.clone(), record_id }
            }
            BuildError::Unmatchable { name, source, record_id, .. } => {
                IndexError::Unmatchable { name, source, record_id }
            }
        })?;
        Ok(InMemoryGazetteer { index: builder.build() })
    }

    pub(crate) fn index(&self) -> &GazetteerIndex {
        &self.index
    }

    pub(crate) fn from_index(index: GazetteerIndex) -> InMemoryGazetteer {
        InMemoryGazetteer { index }
    }
}

indexed_gazetteer!(InMemoryGazetteer, |g| g.index.sources());
