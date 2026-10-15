//! The parts of a gazetteer record: points, bounding boxes, region codes,
//! attribute values and the owned entry, each checked as OpenNLP's records
//! check themselves at construction.

use std::fmt;

use crate::store::{Row, RowAttribute};
use crate::text::is_blank;

/// The conventional feature class of a populated place (city, town,
/// village, settlement).
pub const FEATURE_CLASS_CITY: &str = "CITY";
/// The conventional feature class of an administrative area (region,
/// district, province).
pub const FEATURE_CLASS_ADMIN: &str = "ADMIN";
/// The conventional feature class of a point of interest that is not a
/// settlement.
pub const FEATURE_CLASS_POI: &str = "POI";

/// The conventional attribute key of a US FIPS code.
pub const ATTRIBUTE_KEY_FIPS: &str = "fips";
/// The conventional attribute key of a US Census GEOID.
pub const ATTRIBUTE_KEY_GEOID: &str = "geoid";
/// The conventional attribute key of a US Census ZIP Code Tabulation Area.
pub const ATTRIBUTE_KEY_ZCTA: &str = "zcta";
/// The conventional attribute key of a Wikidata item id.
pub const ATTRIBUTE_KEY_WIKIDATA: &str = "wikidata";
/// The conventional attribute key of a GeoNames id.
pub const ATTRIBUTE_KEY_GEONAMES: &str = "geonames";
/// The conventional attribute key of a Who's On First id.
pub const ATTRIBUTE_KEY_WHOSONFIRST: &str = "whosonfirst";

/// The provenance of an attribute value that is a project judgment with no
/// dataset behind it.
pub const SOURCE_UNSPECIFIED: &str = "UNSPECIFIED";

/// A value that breaks a constraint of a gazetteer record. The message is
/// the one OpenNLP's `IllegalArgumentException` carries.
#[derive(Clone, Debug, PartialEq)]
pub enum EntryError {
    Latitude(f64),
    Longitude(f64),
    BoxWest(f64),
    BoxEast(f64),
    BoxSouth(f64),
    BoxNorth(f64),
    /// South above north.
    BoxInverted {
        south: f64,
        north: f64,
    },
    /// A required text that is empty or Unicode white space only.
    Blank(&'static str),
    /// An alternate name or containment level that is blank: the field and
    /// the whole list as Java's `List.toString` writes it.
    BlankElement(&'static str, String),
    /// A country code that is not two ASCII capital letters.
    CountryCode(String),
    Population(i64),
    /// An attribute with a blank key or a blank value or source.
    Attribute(&'static str),
    /// An attribute key that occurs twice.
    DuplicateAttribute(String),
}

impl fmt::Display for EntryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use turbo_jvm::text::double_to_string as d;
        match self {
            EntryError::Latitude(v) => write!(f, "latitude must be in [-90, 90], got: {}", d(*v)),
            EntryError::Longitude(v) => write!(f, "longitude must be in [-180, 180], got: {}", d(*v)),
            EntryError::BoxWest(v) => write!(f, "west must be in [-180, 180], got: {}", d(*v)),
            EntryError::BoxEast(v) => write!(f, "east must be in [-180, 180], got: {}", d(*v)),
            EntryError::BoxSouth(v) => write!(f, "south must be in [-90, 90], got: {}", d(*v)),
            EntryError::BoxNorth(v) => write!(f, "north must be in [-90, 90], got: {}", d(*v)),
            EntryError::BoxInverted { south, north } => {
                write!(f, "south must not be greater than north, got: {} > {}", d(*south), d(*north))
            }
            EntryError::Blank(what) => write!(f, "{what} must not be null or blank"),
            EntryError::BlankElement(what, list) => {
                write!(f, "{what} must not contain a null or blank element, got: {list}")
            }
            EntryError::CountryCode(code) => write!(
                f,
                "countryCode must be an ISO 3166-1 alpha-2 code (two ASCII capital letters) or null, got: {code}"
            ),
            EntryError::Population(p) => write!(f, "population must not be negative, got: {p}"),
            EntryError::Attribute(what) => write!(f, "{what} must not be null or blank"),
            EntryError::DuplicateAttribute(key) => write!(f, "the attribute key repeats: {key}"),
        }
    }
}

impl std::error::Error for EntryError {}

/// A point in WGS84 decimal degrees (`opennlp.tools.geo.GeoPoint`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    latitude: f64,
    longitude: f64,
}

impl GeoPoint {
    /// A point; the latitude must be in [-90, 90] and the longitude in
    /// [-180, 180], neither NaN.
    pub fn new(latitude: f64, longitude: f64) -> Result<GeoPoint, EntryError> {
        if !(-90.0..=90.0).contains(&latitude) {
            return Err(EntryError::Latitude(latitude));
        }
        if !(-180.0..=180.0).contains(&longitude) {
            return Err(EntryError::Longitude(longitude));
        }
        Ok(GeoPoint { latitude, longitude })
    }

    pub fn latitude(&self) -> f64 {
        self.latitude
    }

    pub fn longitude(&self) -> f64 {
        self.longitude
    }
}

/// A bounding box in WGS84 decimal degrees, its edges in the order west,
/// south, east, north of RFC 7946 (`opennlp.tools.geo.GeoBoundingBox`). A
/// box whose west edge is greater than its east edge crosses the
/// antimeridian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoBoundingBox {
    west: f64,
    south: f64,
    east: f64,
    north: f64,
}

impl GeoBoundingBox {
    pub fn new(west: f64, south: f64, east: f64, north: f64) -> Result<GeoBoundingBox, EntryError> {
        let longitude = -180.0..=180.0;
        let latitude = -90.0..=90.0;
        if !longitude.contains(&west) {
            return Err(EntryError::BoxWest(west));
        }
        if !longitude.contains(&east) {
            return Err(EntryError::BoxEast(east));
        }
        if !latitude.contains(&south) {
            return Err(EntryError::BoxSouth(south));
        }
        if !latitude.contains(&north) {
            return Err(EntryError::BoxNorth(north));
        }
        if south > north {
            return Err(EntryError::BoxInverted { south, north });
        }
        Ok(GeoBoundingBox { west, south, east, north })
    }

    pub fn west(&self) -> f64 {
        self.west
    }

    pub fn south(&self) -> f64 {
        self.south
    }

    pub fn east(&self) -> f64 {
        self.east
    }

    pub fn north(&self) -> f64 {
        self.north
    }

    /// Whether `point` lies in the box, edges included.
    pub fn contains(&self, point: GeoPoint) -> bool {
        if point.latitude < self.south || point.latitude > self.north {
            return false;
        }
        let longitude = point.longitude;
        if self.west <= self.east {
            longitude >= self.west && longitude <= self.east
        } else {
            longitude >= self.west || longitude <= self.east
        }
    }

    /// The center of the box. For a box crossing the antimeridian it lies
    /// on the covered side: the center of a box from 170 to -170 is at
    /// longitude 180. The arithmetic is Java's, so the doubles are equal.
    pub fn center(&self) -> GeoPoint {
        let latitude = (self.south + self.north) / 2.0;
        if self.west <= self.east {
            return GeoPoint { latitude, longitude: (self.west + self.east) / 2.0 };
        }
        let span = 360.0 - (self.west - self.east);
        let longitude = self.west + span / 2.0;
        GeoPoint { latitude, longitude: if longitude > 180.0 { longitude - 360.0 } else { longitude } }
    }
}

/// An ISO 3166-1 alpha-2 region code, two ASCII capital letters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RegionCode([u8; 2]);

/// A region code argument that is not two ASCII letters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidRegionCode(pub String);

impl fmt::Display for InvalidRegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "isoCountryCode must be an ISO 3166-1 alpha-2 code (two ASCII letters), got: {}", self.0)
    }
}

impl std::error::Error for InvalidRegionCode {}

impl RegionCode {
    /// Two ASCII letters of either case, as `Gazetteer.byRegion` takes
    /// them; they are upper-cased.
    pub fn parse(code: &str) -> Result<RegionCode, InvalidRegionCode> {
        match code.as_bytes() {
            &[a, b] if a.is_ascii_alphabetic() && b.is_ascii_alphabetic() => {
                Ok(RegionCode([a.to_ascii_uppercase(), b.to_ascii_uppercase()]))
            }
            _ => Err(InvalidRegionCode(code.to_string())),
        }
    }

    /// A record's country code: exactly two ASCII capital letters.
    pub(crate) fn from_record(code: &str) -> Result<RegionCode, EntryError> {
        match code.as_bytes() {
            &[a, b] if a.is_ascii_uppercase() && b.is_ascii_uppercase() => Ok(RegionCode([a, b])),
            _ => Err(EntryError::CountryCode(code.to_string())),
        }
    }

    pub(crate) fn from_bytes(bytes: [u8; 2]) -> Option<RegionCode> {
        bytes.iter().all(u8::is_ascii_uppercase).then_some(RegionCode(bytes))
    }

    pub fn as_str(&self) -> &str {
        std::str::from_utf8(&self.0).expect("two ASCII letters")
    }

    pub(crate) fn bytes(self) -> [u8; 2] {
        self.0
    }

    /// The code's place in the 26 by 26 table of codes.
    pub(crate) fn index(self) -> usize {
        usize::from(self.0[0] - b'A') * 26 + usize::from(self.0[1] - b'A')
    }
}

impl fmt::Display for RegionCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An attribute value with its provenance (`opennlp.tools.geo.AttributeValue`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AttributeValue {
    value: String,
    source: String,
    notes: String,
}

impl AttributeValue {
    /// A value and the dataset it came from, neither blank, and free notes,
    /// possibly empty.
    pub fn new(
        value: impl Into<String>,
        source: impl Into<String>,
        notes: impl Into<String>,
    ) -> Result<AttributeValue, EntryError> {
        let (value, source, notes) = (value.into(), source.into(), notes.into());
        if is_blank(&value) {
            return Err(EntryError::Attribute("value"));
        }
        if is_blank(&source) {
            return Err(EntryError::Attribute("source"));
        }
        Ok(AttributeValue { value, source, notes })
    }

    pub fn value(&self) -> &str {
        &self.value
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn notes(&self) -> &str {
        &self.notes
    }
}

/// The parts of a [`GazetteerEntry`], checked by [`GazetteerEntry::new`].
#[derive(Clone, Debug, PartialEq)]
pub struct EntryParts {
    /// The dataset identifier, such as `naturalearth` or `geonames`.
    pub source: String,
    /// The identifier within that dataset.
    pub record_id: String,
    pub name: String,
    pub alternate_names: Vec<String>,
    pub location: GeoPoint,
    pub bounding_box: Option<GeoBoundingBox>,
    /// Two ASCII capital letters, or none.
    pub country_code: Option<String>,
    /// The administrative containment chain, outermost first.
    pub containment: Vec<String>,
    /// Zero when unknown.
    pub population: i64,
    /// Conventionally one of the `FEATURE_CLASS_*` constants.
    pub feature_class: Option<String>,
    /// The attributes in the order given; a key may occur once.
    pub attributes: Vec<(String, AttributeValue)>,
}

/// One gazetteer record, owned (`opennlp.tools.geo.GazetteerEntry`). The
/// gazetteers hold their records in a compact store and hand out
/// [`Entry`](crate::Entry) views; this type is for records built by a
/// caller, for [`InMemoryGazetteer`](crate::InMemoryGazetteer).
#[derive(Clone, Debug, PartialEq)]
pub struct GazetteerEntry {
    parts: EntryParts,
    country: Option<RegionCode>,
}

impl GazetteerEntry {
    /// Checks every part as OpenNLP's record constructor does.
    pub fn new(parts: EntryParts) -> Result<GazetteerEntry, EntryError> {
        let alternates: Vec<&str> = parts.alternate_names.iter().map(String::as_str).collect();
        let containment: Vec<&str> = parts.containment.iter().map(String::as_str).collect();
        let attributes: Vec<RowAttribute<'_>> = parts
            .attributes
            .iter()
            .map(|(k, v)| RowAttribute { key: k, value: v.value(), source: v.source(), notes: v.notes() })
            .collect();
        let country = Row {
            source: &parts.source,
            record_id: &parts.record_id,
            name: &parts.name,
            alternates: &alternates,
            location: parts.location,
            bounding_box: parts.bounding_box,
            country: parts.country_code.as_deref(),
            containment: &containment,
            population: parts.population,
            feature_class: parts.feature_class.as_deref(),
            attributes: &attributes,
        }
        .check()?;
        for (i, (key, _)) in parts.attributes.iter().enumerate() {
            if parts.attributes[..i].iter().any(|(k, _)| k == key) {
                return Err(EntryError::DuplicateAttribute(key.clone()));
            }
        }
        Ok(GazetteerEntry { parts, country })
    }

    pub fn parts(&self) -> &EntryParts {
        &self.parts
    }

    pub fn into_parts(self) -> EntryParts {
        self.parts
    }

    pub(crate) fn country(&self) -> Option<RegionCode> {
        self.country
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_and_boxes_check_their_ranges() {
        assert!(GeoPoint::new(90.0, -180.0).is_ok());
        assert!(matches!(GeoPoint::new(f64::NAN, 0.0), Err(EntryError::Latitude(v)) if v.is_nan()));
        assert!(GeoPoint::new(0.0, 180.5).is_err());
        assert!(GeoBoundingBox::new(10.0, 5.0, -10.0, 1.0).is_err());
        assert_eq!(GeoPoint::new(0.0, 200.0).unwrap_err().to_string(), "longitude must be in [-180, 180], got: 200.0");
    }

    #[test]
    fn a_box_across_the_antimeridian_is_centered_on_its_covered_side() {
        let b = GeoBoundingBox::new(170.0, -10.0, -170.0, 10.0).unwrap();
        assert_eq!(b.center(), GeoPoint::new(0.0, 180.0).unwrap());
        assert!(b.contains(GeoPoint::new(0.0, -175.0).unwrap()));
        assert!(!b.contains(GeoPoint::new(0.0, 0.0).unwrap()));
        let c = GeoBoundingBox::new(160.0, 0.0, -170.0, 0.0).unwrap();
        assert_eq!(c.center().longitude(), 175.0);
        let d = GeoBoundingBox::new(170.0, 0.0, -160.0, 0.0).unwrap();
        assert_eq!(d.center().longitude(), -175.0);
    }

    #[test]
    fn region_codes_take_either_case_and_records_take_capitals() {
        assert_eq!(RegionCode::parse("de").unwrap().as_str(), "DE");
        assert!(RegionCode::parse("D").is_err());
        assert!(RegionCode::parse("D1").is_err());
        assert!(RegionCode::parse("ÄB").is_err());
        assert!(RegionCode::from_record("de").is_err());
        assert_eq!(RegionCode::from_record("DE").unwrap().index(), 3 * 26 + 4);
    }
}
