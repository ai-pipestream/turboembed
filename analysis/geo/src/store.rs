//! The records of a gazetteer, stored by column: every string in one text
//! buffer addressed by offsets, every number in a vector of its own, and
//! the few distinct sources and feature classes as small integers. A record
//! is an index into the columns, and [`Entry`] is the view of one.

use std::collections::HashMap;
use std::fmt;

use crate::entry::{EntryError, GeoBoundingBox, GeoPoint, RegionCode};
use crate::text::is_blank;

/// The marker of an absent value in a `u32` or `u16` column.
pub(crate) const NONE32: u32 = u32::MAX;
pub(crate) const NONE16: u16 = u16::MAX;

/// A store that would outgrow its 32-bit offsets or 16-bit codes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CapacityError {
    /// More than 4 GiB of text, or more than 2^32 - 1 strings, records or
    /// attributes.
    Text,
    Records,
    /// More than 65 535 distinct sources, feature classes or attribute
    /// sources together.
    Codes,
}

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            CapacityError::Text => "the gazetteer holds more than 4 GiB of names or 2^32 strings",
            CapacityError::Records => "the gazetteer holds more than 2^32 - 1 records or attributes",
            CapacityError::Codes => "the gazetteer holds more than 65535 distinct sources and feature classes",
        })
    }
}

impl std::error::Error for CapacityError {}

/// One attribute of a [`Row`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct RowAttribute<'r> {
    pub key: &'r str,
    pub value: &'r str,
    pub source: &'r str,
    pub notes: &'r str,
}

/// One record as a loader parsed it, borrowing its text, before it is
/// checked and stored.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Row<'r> {
    pub source: &'r str,
    pub record_id: &'r str,
    pub name: &'r str,
    pub alternates: &'r [&'r str],
    pub location: GeoPoint,
    pub bounding_box: Option<GeoBoundingBox>,
    pub country: Option<&'r str>,
    pub containment: &'r [&'r str],
    pub population: i64,
    pub feature_class: Option<&'r str>,
    pub attributes: &'r [RowAttribute<'r>],
}

impl Row<'_> {
    /// The checks of OpenNLP's `GazetteerEntry` constructor, in its order.
    /// Returns the country code.
    pub(crate) fn check(&self) -> Result<Option<RegionCode>, EntryError> {
        if is_blank(self.source) {
            return Err(EntryError::Blank("source"));
        }
        if is_blank(self.record_id) {
            return Err(EntryError::Blank("recordId"));
        }
        if is_blank(self.name) {
            return Err(EntryError::Blank("name"));
        }
        if self.alternates.iter().any(|n| is_blank(n)) {
            return Err(EntryError::BlankElement("alternateNames"));
        }
        let country = self.country.map(RegionCode::from_record).transpose()?;
        if self.containment.iter().any(|n| is_blank(n)) {
            return Err(EntryError::BlankElement("containment"));
        }
        if self.population < 0 {
            return Err(EntryError::Population(self.population));
        }
        if self.feature_class.is_some_and(is_blank) {
            return Err(EntryError::Blank("featureClass"));
        }
        for a in self.attributes {
            if is_blank(a.key) {
                return Err(EntryError::Attribute("key"));
            }
            if is_blank(a.value) {
                return Err(EntryError::Attribute("value"));
            }
            if is_blank(a.source) {
                return Err(EntryError::Attribute("source"));
            }
        }
        Ok(country)
    }
}

/// The small integer codes of the few distinct short strings: sources,
/// feature classes and attribute sources share one table.
#[derive(Clone, Debug, Default)]
pub(crate) struct Codes {
    names: Vec<Box<str>>,
    ids: HashMap<Box<str>, u16>,
}

impl Codes {
    pub(crate) fn code(&mut self, name: &str) -> Result<u16, CapacityError> {
        if let Some(&id) = self.ids.get(name) {
            return Ok(id);
        }
        let id = u16::try_from(self.names.len()).ok().filter(|&id| id != NONE16).ok_or(CapacityError::Codes)?;
        self.names.push(name.into());
        self.ids.insert(name.into(), id);
        Ok(id)
    }

    pub(crate) fn name(&self, code: u16) -> &str {
        &self.names[usize::from(code)]
    }

    pub(crate) fn len(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn from_names(names: Vec<Box<str>>) -> Codes {
        let ids = names.iter().enumerate().map(|(i, n)| (n.clone(), i as u16)).collect();
        Codes { names, ids }
    }

    pub(crate) fn names(&self) -> &[Box<str>] {
        &self.names
    }
}

/// The records, by column. See the module documentation.
#[derive(Clone, Debug, Default)]
pub(crate) struct EntryStore {
    /// Every string of every record, one after the other.
    pub(crate) text: String,
    /// String `i` is `text[bounds[i]..bounds[i + 1]]`; starts with 0.
    pub(crate) bounds: Vec<u32>,
    pub(crate) codes: Codes,
    pub(crate) latitude: Vec<f64>,
    pub(crate) longitude: Vec<f64>,
    pub(crate) population: Vec<u64>,
    /// The first string of each record, and the string count at the end:
    /// the record id, the name, the alternate names, the containment, then
    /// the key, value and notes of each attribute.
    pub(crate) strings: Vec<u32>,
    /// The number of alternate names of each record.
    pub(crate) alternates: Vec<u32>,
    /// The first attribute of each record, and the attribute count at the
    /// end.
    pub(crate) attributes: Vec<u32>,
    /// The index of each record's box in `boxes`, or `NONE32`.
    pub(crate) box_of: Vec<u32>,
    pub(crate) source: Vec<u16>,
    /// The feature class code, or `NONE16`.
    pub(crate) feature: Vec<u16>,
    /// Two ASCII capital letters, or two zero bytes for none.
    pub(crate) country: Vec<[u8; 2]>,
    /// West, south, east, north.
    pub(crate) boxes: Vec<[f64; 4]>,
    /// Each attribute's key, value and notes as string indices, and its
    /// source as a code.
    pub(crate) attribute_key: Vec<u32>,
    pub(crate) attribute_value: Vec<u32>,
    pub(crate) attribute_notes: Vec<u32>,
    pub(crate) attribute_source: Vec<u16>,
}

impl EntryStore {
    pub(crate) fn new() -> EntryStore {
        EntryStore { bounds: vec![0], strings: vec![0], attributes: vec![0], ..EntryStore::default() }
    }

    pub(crate) fn len(&self) -> usize {
        self.latitude.len()
    }

    pub(crate) fn string(&self, i: u32) -> &str {
        let i = i as usize;
        &self.text[self.bounds[i] as usize..self.bounds[i + 1] as usize]
    }

    fn push_string(&mut self, s: &str) -> Result<u32, CapacityError> {
        let index = u32::try_from(self.bounds.len() - 1).ok().filter(|&i| i != NONE32).ok_or(CapacityError::Text)?;
        self.text.push_str(s);
        self.bounds.push(u32::try_from(self.text.len()).map_err(|_| CapacityError::Text)?);
        Ok(index)
    }

    /// Stores a checked row and returns its record number.
    pub(crate) fn push(&mut self, row: &Row<'_>, country: Option<RegionCode>) -> Result<u32, CapacityError> {
        let id = u32::try_from(self.len()).ok().filter(|&i| i != NONE32).ok_or(CapacityError::Records)?;
        let source = self.codes.code(row.source)?;
        let feature = row.feature_class.map(|f| self.codes.code(f)).transpose()?.unwrap_or(NONE16);
        self.push_string(row.record_id)?;
        self.push_string(row.name)?;
        for s in row.alternates.iter().chain(row.containment) {
            self.push_string(s)?;
        }
        for a in row.attributes {
            let source = self.codes.code(a.source)?;
            let (key, value, notes) =
                (self.push_string(a.key)?, self.push_string(a.value)?, self.push_string(a.notes)?);
            self.attribute_key.push(key);
            self.attribute_value.push(value);
            self.attribute_notes.push(notes);
            self.attribute_source.push(source);
        }
        let attributes = u32::try_from(self.attribute_key.len()).map_err(|_| CapacityError::Records)?;
        self.latitude.push(row.location.latitude());
        self.longitude.push(row.location.longitude());
        self.population.push(row.population as u64);
        self.strings.push((self.bounds.len() - 1) as u32);
        self.alternates.push(row.alternates.len() as u32);
        self.attributes.push(attributes);
        match row.bounding_box {
            Some(b) => {
                self.box_of.push(u32::try_from(self.boxes.len()).map_err(|_| CapacityError::Records)?);
                self.boxes.push([b.west(), b.south(), b.east(), b.north()]);
            }
            None => self.box_of.push(NONE32),
        }
        self.source.push(source);
        self.feature.push(feature);
        self.country.push(country.map_or([0, 0], RegionCode::bytes));
        Ok(id)
    }

    /// Moves the records of `other` to the end of this store.
    pub(crate) fn append(&mut self, other: EntryStore) -> Result<(), CapacityError> {
        let mut remap = Vec::with_capacity(other.codes.len());
        for name in other.codes.names() {
            remap.push(self.codes.code(name)?);
        }
        let code = |c: u16| if c == NONE16 { NONE16 } else { remap[usize::from(c)] };
        let text_base = u32::try_from(self.text.len()).map_err(|_| CapacityError::Text)?;
        let string_base = (self.bounds.len() - 1) as u32;
        let attribute_base = self.attribute_key.len() as u32;
        let box_base = self.boxes.len() as u32;
        let total_text = self.text.len() + other.text.len();
        let total_strings = self.bounds.len() + other.bounds.len();
        if u32::try_from(total_text).is_err() || u32::try_from(total_strings).is_err() {
            return Err(CapacityError::Text);
        }
        if u32::try_from(self.len() + other.len() + 1).is_err()
            || u32::try_from(self.attribute_key.len() + other.attribute_key.len()).is_err()
        {
            return Err(CapacityError::Records);
        }
        self.text.push_str(&other.text);
        self.bounds.extend(other.bounds[1..].iter().map(|b| b + text_base));
        self.latitude.extend_from_slice(&other.latitude);
        self.longitude.extend_from_slice(&other.longitude);
        self.population.extend_from_slice(&other.population);
        self.strings.extend(other.strings[1..].iter().map(|s| s + string_base));
        self.alternates.extend_from_slice(&other.alternates);
        self.attributes.extend(other.attributes[1..].iter().map(|a| a + attribute_base));
        self.box_of.extend(other.box_of.iter().map(|&b| if b == NONE32 { NONE32 } else { b + box_base }));
        self.source.extend(other.source.iter().map(|&c| code(c)));
        self.feature.extend(other.feature.iter().map(|&c| code(c)));
        self.country.extend_from_slice(&other.country);
        self.boxes.extend_from_slice(&other.boxes);
        self.attribute_key.extend(other.attribute_key.iter().map(|s| s + string_base));
        self.attribute_value.extend(other.attribute_value.iter().map(|s| s + string_base));
        self.attribute_notes.extend(other.attribute_notes.iter().map(|s| s + string_base));
        self.attribute_source.extend(other.attribute_source.iter().map(|&c| code(c)));
        Ok(())
    }

    /// The bytes the store holds on the heap, for the memory report.
    pub(crate) fn heap_bytes(&self) -> usize {
        fn v<T>(x: &Vec<T>) -> usize {
            x.capacity() * size_of::<T>()
        }
        self.text.capacity()
            + v(&self.bounds)
            + v(&self.latitude)
            + v(&self.longitude)
            + v(&self.population)
            + v(&self.strings)
            + v(&self.alternates)
            + v(&self.attributes)
            + v(&self.box_of)
            + v(&self.source)
            + v(&self.feature)
            + v(&self.country)
            + v(&self.boxes)
            + v(&self.attribute_key)
            + v(&self.attribute_value)
            + v(&self.attribute_notes)
            + v(&self.attribute_source)
    }

    /// Releases the spare capacity the loaders' growth left.
    pub(crate) fn shrink_to_fit(&mut self) {
        self.text.shrink_to_fit();
        self.bounds.shrink_to_fit();
        self.latitude.shrink_to_fit();
        self.longitude.shrink_to_fit();
        self.population.shrink_to_fit();
        self.strings.shrink_to_fit();
        self.alternates.shrink_to_fit();
        self.attributes.shrink_to_fit();
        self.box_of.shrink_to_fit();
        self.source.shrink_to_fit();
        self.feature.shrink_to_fit();
        self.country.shrink_to_fit();
        self.boxes.shrink_to_fit();
        self.attribute_key.shrink_to_fit();
        self.attribute_value.shrink_to_fit();
        self.attribute_notes.shrink_to_fit();
        self.attribute_source.shrink_to_fit();
    }
}

/// One record of a gazetteer, a view into its store
/// (`opennlp.tools.geo.GazetteerEntry`). Two views are equal when they are
/// the same record of the same gazetteer.
#[derive(Clone, Copy)]
pub struct Entry<'a> {
    store: &'a EntryStore,
    id: u32,
}

impl PartialEq for Entry<'_> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.store, other.store) && self.id == other.id
    }
}

impl Eq for Entry<'_> {}

impl fmt::Debug for Entry<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Entry")
            .field("source", &self.source())
            .field("record_id", &self.record_id())
            .field("name", &self.name())
            .field("country_code", &self.country_code())
            .field("population", &self.population())
            .finish()
    }
}

/// One attribute of an [`Entry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attribute<'a> {
    pub key: &'a str,
    pub value: &'a str,
    pub source: &'a str,
    pub notes: &'a str,
}

impl<'a> Entry<'a> {
    pub(crate) fn new(store: &'a EntryStore, id: u32) -> Entry<'a> {
        Entry { store, id }
    }

    /// The record's place in its gazetteer's store.
    pub fn index(&self) -> u32 {
        self.id
    }

    fn i(&self) -> usize {
        self.id as usize
    }

    /// The dataset identifier, such as `naturalearth`.
    pub fn source(&self) -> &'a str {
        self.store.codes.name(self.store.source[self.i()])
    }

    /// The identifier within the source dataset.
    pub fn record_id(&self) -> &'a str {
        self.store.string(self.store.strings[self.i()])
    }

    /// The canonical name.
    pub fn name(&self) -> &'a str {
        self.store.string(self.store.strings[self.i()] + 1)
    }

    /// The alternate names, in the order the record gives them.
    pub fn alternate_names(&self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        let store = self.store;
        let first = store.strings[self.i()] + 2;
        (first..first + store.alternates[self.i()]).map(move |s| store.string(s))
    }

    pub fn location(&self) -> GeoPoint {
        GeoPoint::new(self.store.latitude[self.i()], self.store.longitude[self.i()]).expect("checked when stored")
    }

    pub fn bounding_box(&self) -> Option<GeoBoundingBox> {
        let b = self.store.box_of[self.i()];
        (b != NONE32).then(|| {
            let [w, s, e, n] = self.store.boxes[b as usize];
            GeoBoundingBox::new(w, s, e, n).expect("checked when stored")
        })
    }

    /// The ISO 3166-1 alpha-2 country code.
    pub fn country_code(&self) -> Option<&'a str> {
        let c = &self.store.country[self.i()];
        (c[0] != 0).then(|| std::str::from_utf8(c).expect("ASCII letters"))
    }

    pub(crate) fn region(&self) -> Option<RegionCode> {
        RegionCode::from_bytes(self.store.country[self.i()])
    }

    /// The administrative containment chain, outermost first.
    pub fn containment(&self) -> impl ExactSizeIterator<Item = &'a str> + 'a {
        let store = self.store;
        let i = self.i();
        let first = store.strings[i] + 2 + store.alternates[i];
        // The three strings of each attribute come after the containment.
        let end = store.strings[i + 1] - 3 * (store.attributes[i + 1] - store.attributes[i]);
        (first..end).map(move |s| store.string(s))
    }

    /// The population, 0 when unknown.
    pub fn population(&self) -> u64 {
        self.store.population[self.i()]
    }

    /// The feature class, conventionally one of the `FEATURE_CLASS_*`
    /// constants.
    pub fn feature_class(&self) -> Option<&'a str> {
        let f = self.store.feature[self.i()];
        (f != NONE16).then(|| self.store.codes.name(f))
    }

    /// The attributes, in the order the record gives them.
    pub fn attributes(&self) -> impl ExactSizeIterator<Item = Attribute<'a>> + 'a {
        let store = self.store;
        (store.attributes[self.i()]..store.attributes[self.i() + 1]).map(move |a| {
            let a = a as usize;
            Attribute {
                key: store.string(store.attribute_key[a]),
                value: store.string(store.attribute_value[a]),
                source: store.codes.name(store.attribute_source[a]),
                notes: store.string(store.attribute_notes[a]),
            }
        })
    }

    /// The attribute with `key`.
    pub fn attribute(&self, key: &str) -> Option<Attribute<'a>> {
        self.attributes().find(|a| a.key == key)
    }

    /// The record as an owned entry.
    pub fn to_entry(&self) -> crate::GazetteerEntry {
        use crate::entry::{AttributeValue, EntryParts};
        crate::GazetteerEntry::new(EntryParts {
            source: self.source().into(),
            record_id: self.record_id().into(),
            name: self.name().into(),
            alternate_names: self.alternate_names().map(Into::into).collect(),
            location: self.location(),
            bounding_box: self.bounding_box(),
            country_code: self.country_code().map(Into::into),
            containment: self.containment().map(Into::into).collect(),
            population: self.population() as i64,
            feature_class: self.feature_class().map(Into::into),
            attributes: self
                .attributes()
                .map(|a| (a.key.into(), AttributeValue::new(a.value, a.source, a.notes).expect("checked when stored")))
                .collect(),
        })
        .expect("checked when stored")
    }
}
