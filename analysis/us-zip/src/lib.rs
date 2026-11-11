//! US ZIP Code Tabulation Areas (ZCTAs): where a five-digit ZIP Code is,
//! from public domain US Census Bureau data bundled with the crate.
//!
//! [`UsZipTable::bundled`] holds the 33,791 ZCTAs of the 2020 Census (as
//! the 2024 Gazetteer lists them), each with
//!
//! - the state that holds most of its land, as a USPS code (`IL`),
//! - its land area, a point inside it (the Census internal point) and the
//!   bounding box of its 1:500,000 shape,
//! - the counties and the Census places (cities, towns, villages, CDPs)
//!   it overlaps, each with its share of the ZCTA's land, largest first,
//!
//! and the 32,188 places and 3,235 counties those name, and for every
//! three-digit prefix the states that hold its ZCTAs. `data/README.md`
//! says which Census files they are built from and how.
//!
//! A ZCTA is the Census Bureau's area for the ZIP Code most addresses in
//! it use, so not every ZIP Code has one: a ZIP Code of post office boxes
//! only, one of a single organization and a military ZIP Code have none,
//! and [`UsZipTable::zcta`] returns `None` for them. A ZCTA's place is a
//! Census place, which is not always the city name the Postal Service
//! uses for its mail, and a ZCTA outside every place has none.
//!
//! ```
//! use turbo_us_zip::UsZipTable;
//!
//! let zips = UsZipTable::bundled();
//! let z = zips.zcta("62704-1234").unwrap();
//! assert_eq!(z.state(), "IL");
//! assert_eq!(z.counties().next().unwrap().0.name(), "Sangamon County");
//! assert_eq!(z.places().next().unwrap().0.name(), "Springfield");
//! ```

mod entry;
mod parse;

use std::fmt;
use std::ops::Range;
use std::sync::OnceLock;

use turbo_geo::{GeoBoundingBox, GeoPoint};

pub use entry::{SOURCE_CENSUS_ZCTA, zcta_entry};

/// The bundled ZCTA table (`data/us-zcta.txt`).
pub const ZCTA_TABLE: &str = include_str!("../data/us-zcta.txt");
/// The bundled place table (`data/us-places.txt`).
pub const PLACE_TABLE: &str = include_str!("../data/us-places.txt");
/// The bundled county table (`data/us-counties.txt`).
pub const COUNTY_TABLE: &str = include_str!("../data/us-counties.txt");
/// The bundled prefix table (`data/zip3-states.txt`).
pub const PREFIX_TABLE: &str = include_str!("../data/zip3-states.txt");

/// A table that does not parse: which one, its line and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableError {
    pub table: &'static str,
    pub line: usize,
    pub problem: String,
}

impl fmt::Display for TableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} line {}: {}", self.table, self.line, self.problem)
    }
}

impl std::error::Error for TableError {}

#[derive(Clone, Copy, Debug)]
struct Part {
    geoid: u32,
    share: u16,
}

#[derive(Debug)]
struct ZctaRow {
    /// The line of its table it was read from.
    line: usize,
    code: u32,
    state: [u8; 2],
    /// Hundredths of a square kilometre.
    land: u32,
    /// Ten-thousandths of a degree.
    lat: i32,
    lon: i32,
    /// West, south, east, north in thousandths of a degree.
    bbox: [i32; 4],
    counties: Range<u32>,
    places: Range<u32>,
}

#[derive(Debug)]
struct PlaceRow {
    geoid: u32,
    state: [u8; 2],
    name: Box<str>,
    kind: Box<str>,
    /// Latitude and longitude in ten-thousandths of a degree.
    point: Option<[i32; 2]>,
}

#[derive(Debug)]
struct CountyRow {
    geoid: u32,
    state: [u8; 2],
    name: Box<str>,
}

#[derive(Debug)]
struct PrefixRow {
    prefix: u16,
    state: [u8; 2],
    zctas: u32,
}

/// The ZCTAs, places, counties and prefixes, each sorted by its code and
/// found by binary search.
#[derive(Debug)]
pub struct UsZipTable {
    zctas: Vec<ZctaRow>,
    parts: Vec<Part>,
    places: Vec<PlaceRow>,
    counties: Vec<CountyRow>,
    prefixes: Vec<PrefixRow>,
}

impl UsZipTable {
    /// The bundled tables, parsed on first use. They cannot fail to parse;
    /// a test parses them.
    pub fn bundled() -> &'static UsZipTable {
        static TABLE: OnceLock<UsZipTable> = OnceLock::new();
        TABLE.get_or_init(|| {
            UsZipTable::parse(ZCTA_TABLE, PLACE_TABLE, COUNTY_TABLE, PREFIX_TABLE)
                .expect("the bundled ZIP tables are well formed")
        })
    }

    /// Tables in the bundled format (`data/README.md`), such as ones built
    /// from later Census files. Every county and place a ZCTA names must
    /// be in its table.
    pub fn parse(zctas: &str, places: &str, counties: &str, prefixes: &str) -> Result<UsZipTable, TableError> {
        let mut t = UsZipTable {
            zctas: Vec::new(),
            parts: Vec::new(),
            places: Vec::new(),
            counties: Vec::new(),
            prefixes: Vec::new(),
        };
        parse::zctas(zctas, &mut t.zctas, &mut t.parts)?;
        parse::places(places, &mut t.places)?;
        parse::counties(counties, &mut t.counties)?;
        parse::prefixes(prefixes, &mut t.prefixes)?;
        for z in &t.zctas {
            let missing = |kind: &str, g: u32| TableError {
                table: "us-zcta",
                line: z.line,
                problem: format!("ZCTA {:05} names {kind} {g} that its table lacks", z.code),
            };
            for p in &t.parts[range(&z.counties)] {
                t.county_ix(p.geoid).ok_or_else(|| missing("county", p.geoid))?;
            }
            for p in &t.parts[range(&z.places)] {
                t.place_ix(p.geoid).ok_or_else(|| missing("place", p.geoid))?;
            }
        }
        Ok(t)
    }

    /// The number of ZCTAs.
    pub fn len(&self) -> usize {
        self.zctas.len()
    }

    pub fn is_empty(&self) -> bool {
        self.zctas.is_empty()
    }

    /// The ZCTA of a ZIP Code written as five ASCII digits, or as ZIP+4
    /// (`62704-1234`), when it has one.
    pub fn zcta(&self, zip: &str) -> Option<Zcta<'_>> {
        let five = match zip.as_bytes() {
            [_, _, _, _, _] => zip,
            [_, _, _, _, _, b'-', rest @ ..] if rest.len() == 4 && rest.iter().all(u8::is_ascii_digit) => &zip[..5],
            _ => return None,
        };
        let code = parse::digits(five, 5)?;
        let i = self.zctas.binary_search_by_key(&code, |r| r.code).ok()?;
        Some(Zcta { table: self, row: &self.zctas[i] })
    }

    /// Every ZCTA, in code order.
    pub fn zctas(&self) -> impl Iterator<Item = Zcta<'_>> {
        self.zctas.iter().map(|row| Zcta { table: self, row })
    }

    /// The place of a seven-digit Census GEOID (`1772000`).
    pub fn place(&self, geoid: &str) -> Option<Place<'_>> {
        self.place_ix(parse::digits(geoid, 7)?).map(|i| Place(&self.places[i]))
    }

    /// The county of a five-digit Census GEOID (`17167`).
    pub fn county(&self, geoid: &str) -> Option<County<'_>> {
        self.county_ix(parse::digits(geoid, 5)?).map(|i| County(&self.counties[i]))
    }

    /// The states that hold ZCTAs of the three-digit prefix (`627`), with
    /// how many ZCTAs each, in the order of their codes. A prefix with no
    /// ZCTA has none.
    pub fn prefix_states(&self, prefix: &str) -> impl Iterator<Item = (&str, u32)> {
        let p = parse::digits(prefix, 3).map_or(u16::MAX, |p| p as u16);
        let start = self.prefixes.partition_point(|r| r.prefix < p);
        self.prefixes[start..].iter().take_while(move |r| r.prefix == p).map(|r| (code(&r.state), r.zctas))
    }

    fn place_ix(&self, geoid: u32) -> Option<usize> {
        self.places.binary_search_by_key(&geoid, |r| r.geoid).ok()
    }

    fn county_ix(&self, geoid: u32) -> Option<usize> {
        self.counties.binary_search_by_key(&geoid, |r| r.geoid).ok()
    }
}

fn range(r: &Range<u32>) -> Range<usize> {
    r.start as usize..r.end as usize
}

fn code(s: &[u8; 2]) -> &str {
    // The parser admits only ASCII capitals.
    std::str::from_utf8(s).unwrap_or("")
}

fn point(lat: i32, lon: i32) -> GeoPoint {
    GeoPoint::new(f64::from(lat) / 1e4, f64::from(lon) / 1e4).expect("points are checked when parsed")
}

/// One ZCTA.
#[derive(Clone, Copy)]
pub struct Zcta<'a> {
    table: &'a UsZipTable,
    row: &'a ZctaRow,
}

impl<'a> Zcta<'a> {
    /// The five-digit code.
    pub fn code(&self) -> String {
        format!("{:05}", self.row.code)
    }

    /// The USPS code of the state that holds most of its land.
    pub fn state(&self) -> &'a str {
        code(&self.row.state)
    }

    /// Its land area in square kilometres, to a hundredth.
    pub fn land_km2(&self) -> f64 {
        f64::from(self.row.land) / 100.0
    }

    /// The Census internal point: inside the ZCTA, near its centre, to a
    /// ten-thousandth of a degree.
    pub fn point(&self) -> GeoPoint {
        point(self.row.lat, self.row.lon)
    }

    /// The bounding box of its 1:500,000 shape, rounded outward to a
    /// thousandth of a degree and widened, where needed, to hold
    /// [`Zcta::point`].
    pub fn bounding_box(&self) -> GeoBoundingBox {
        let [w, s, e, n] = self.row.bbox.map(|v| f64::from(v) / 1e3);
        GeoBoundingBox::new(w, s, e, n).expect("boxes are checked when parsed")
    }

    /// The counties it overlaps with their thousandths of its land,
    /// largest first.
    pub fn counties(&self) -> impl Iterator<Item = (County<'a>, u16)> + use<'a> {
        let t = self.table;
        t.parts[range(&self.row.counties)]
            .iter()
            .filter_map(move |p| t.county_ix(p.geoid).map(|i| (County(&t.counties[i]), p.share)))
    }

    /// The Census places it overlaps with their thousandths of its land,
    /// largest first; none when all its land is outside every place.
    pub fn places(&self) -> impl Iterator<Item = (Place<'a>, u16)> + use<'a> {
        let t = self.table;
        t.parts[range(&self.row.places)]
            .iter()
            .filter_map(move |p| t.place_ix(p.geoid).map(|i| (Place(&t.places[i]), p.share)))
    }
}

impl fmt::Debug for Zcta<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Zcta").field("code", &self.code()).field("state", &self.state()).finish_non_exhaustive()
    }
}

/// One Census place.
#[derive(Clone, Copy, Debug)]
pub struct Place<'a>(&'a PlaceRow);

impl<'a> Place<'a> {
    /// The seven-digit GEOID: the state's FIPS code and the place's.
    pub fn geoid(&self) -> String {
        format!("{:07}", self.0.geoid)
    }

    /// The USPS code of its state.
    pub fn state(&self) -> &'a str {
        code(&self.0.state)
    }

    /// Its name without the Census description (`Springfield`).
    pub fn name(&self) -> &'a str {
        &self.0.name
    }

    /// The Census description of the place (`city`, `town`, `village`,
    /// `CDP`), empty when the name had none.
    pub fn kind(&self) -> &'a str {
        &self.0.kind
    }

    /// The Census internal point; none for the places of the island
    /// areas (American Samoa, Guam, the Northern Mariana Islands and the
    /// US Virgin Islands), which the Census Gazetteer leaves out.
    pub fn point(&self) -> Option<GeoPoint> {
        self.0.point.map(|[lat, lon]| point(lat, lon))
    }
}

/// One county or county equivalent.
#[derive(Clone, Copy, Debug)]
pub struct County<'a>(&'a CountyRow);

impl<'a> County<'a> {
    /// The five-digit GEOID: the state's FIPS code and the county's.
    pub fn geoid(&self) -> String {
        format!("{:05}", self.0.geoid)
    }

    /// The USPS code of its state.
    pub fn state(&self) -> &'a str {
        code(&self.0.state)
    }

    /// Its name as the Census list writes it (`Sangamon County`).
    pub fn name(&self) -> &'a str {
        &self.0.name
    }
}
