//! The four tab separated tables, read strictly: every row must have its
//! fields, every number its digits, every code its length.

use std::ops::Range;

use turbo_geo::{GeoBoundingBox, GeoPoint};

use crate::{CountyRow, Part, PlaceRow, PrefixRow, TableError, ZctaRow};

/// The data lines of `text` (not empty, not starting with `#`), with
/// their line numbers from one.
fn lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(i, l)| (i + 1, l.strip_suffix('\r').unwrap_or(l)))
        .filter(|(_, l)| !l.is_empty() && !l.starts_with('#'))
}

struct Fields<'t> {
    table: &'static str,
    line: usize,
    it: std::str::Split<'t, char>,
}

impl<'t> Fields<'t> {
    fn new(table: &'static str, line: usize, text: &'t str) -> Fields<'t> {
        Fields { table, line, it: text.split('\t') }
    }

    fn err(&self, what: &str) -> TableError {
        TableError { table: self.table, line: self.line, problem: what.to_string() }
    }

    fn next(&mut self, what: &str) -> Result<&'t str, TableError> {
        self.it.next().ok_or_else(|| self.err(&format!("no {what} field")))
    }

    fn end(&mut self) -> Result<(), TableError> {
        match self.it.next() {
            None => Ok(()),
            Some(_) => Err(self.err("too many fields")),
        }
    }

    fn digits(&mut self, what: &str, len: usize) -> Result<u32, TableError> {
        let f = self.next(what)?;
        digits(f, len).ok_or_else(|| self.err(&format!("{what} is not {len} ASCII digits: {f:?}")))
    }

    fn state(&mut self) -> Result<[u8; 2], TableError> {
        let f = self.next("state")?;
        match f.as_bytes() {
            &[a, b] if a.is_ascii_uppercase() && b.is_ascii_uppercase() => Ok([a, b]),
            _ => Err(self.err(&format!("state is not two capital letters: {f:?}"))),
        }
    }

    fn fixed(&mut self, what: &str, places: usize) -> Result<i64, TableError> {
        let f = self.next(what)?;
        fixed(f, places).ok_or_else(|| self.err(&format!("{what} is not a decimal with {places} places: {f:?}")))
    }

    /// A latitude and longitude field pair, both empty for no point.
    fn optional_point(&mut self) -> Result<Option<[i32; 2]>, TableError> {
        let (lat, lon) = (self.next("lat")?, self.next("lon")?);
        if lat.is_empty() && lon.is_empty() {
            return Ok(None);
        }
        let bad = |what: &str, v: &str| self.err(&format!("{what} is not a decimal with 4 places: {v:?}"));
        let la = degrees(fixed(lat, 4), 4).ok_or_else(|| bad("lat", lat))?;
        let lo = degrees(fixed(lon, 4), 4).ok_or_else(|| bad("lon", lon))?;
        GeoPoint::new(f64::from(la) / 1e4, f64::from(lo) / 1e4).map_err(|e| self.err(&e.to_string()))?;
        Ok(Some([la, lo]))
    }

    /// A coordinate in degrees with `places` decimals, as an integer of
    /// `10^-places` degrees, refused past 180 degrees either way before
    /// it is narrowed.
    fn degrees(&mut self, what: &str, places: usize) -> Result<i32, TableError> {
        let f = self.next(what)?;
        degrees(fixed(f, places), places)
            .ok_or_else(|| self.err(&format!("{what} is not a coordinate with {places} decimal places: {f:?}")))
    }

    fn text(&mut self, what: &str) -> Result<&'t str, TableError> {
        let f = self.next(what)?;
        if f.trim().is_empty() {
            return Err(self.err(&format!("{what} is blank")));
        }
        Ok(f)
    }
}

/// `s` as a number when it is exactly `len` ASCII digits.
pub(crate) fn digits(s: &str, len: usize) -> Option<u32> {
    let b = s.as_bytes();
    (b.len() == len && b.iter().all(u8::is_ascii_digit))
        .then(|| b.iter().fold(0u32, |n, &d| n * 10 + u32::from(d - b'0')))
}

/// `s` (`-89.6820`) as an integer of `10^-places` units when it has a sign
/// or none, at least one digit before the point and exactly `places`
/// after it.
fn fixed(s: &str, places: usize) -> Option<i64> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let (whole, frac) = s.split_once('.')?;
    if whole.is_empty() || whole.len() > 9 || frac.len() != places {
        return None;
    }
    let all = |p: &str| p.bytes().all(|b| b.is_ascii_digit());
    if !all(whole) || !all(frac) {
        return None;
    }
    let v = format!("{whole}{frac}").bytes().fold(0i64, |n, d| n * 10 + i64::from(d - b'0'));
    Some(if neg { -v } else { v })
}

/// `v` (in `10^-places` degrees) when it is within 180 degrees of zero.
fn degrees(v: Option<i64>, places: usize) -> Option<i32> {
    let limit = 180 * 10i64.pow(places as u32);
    v.filter(|v| v.abs() <= limit).and_then(|v| i32::try_from(v).ok())
}

/// A list of `geoid:share` parts (`17167:988,17001:12`), each GEOID
/// `len` digits and each share 1 to 1000, appended to `parts`.
fn parts(f: &mut Fields<'_>, what: &str, len: usize, parts: &mut Vec<Part>) -> Result<Range<u32>, TableError> {
    let s = f.next(what)?;
    let start = parts.len() as u32;
    if !s.is_empty() {
        for p in s.split(',') {
            let bad = || f.err(&format!("{what} part is not GEOID:share: {p:?}"));
            let (g, share) = p.split_once(':').ok_or_else(bad)?;
            let geoid = digits(g, len).ok_or_else(bad)?;
            let share = share.parse::<u16>().ok().filter(|s| (1..=1000).contains(s)).ok_or_else(bad)?;
            let list = &parts[start as usize..];
            if list.iter().any(|q| q.geoid == geoid) {
                return Err(f.err(&format!("{what} names {g} twice")));
            }
            if list.last().is_some_and(|q| q.share < share) {
                return Err(f.err(&format!("{what} shares are not largest first")));
            }
            parts.push(Part { geoid, share });
        }
    }
    Ok(start..parts.len() as u32)
}

pub(crate) fn zctas(text: &str, out: &mut Vec<ZctaRow>, all: &mut Vec<Part>) -> Result<(), TableError> {
    for (n, l) in lines(text) {
        let mut f = Fields::new("us-zcta", n, l);
        let code = f.digits("zcta", 5)?;
        let state = f.state()?;
        let land = f.fixed("land_km2", 2)?;
        let lat = f.degrees("lat", 4)?;
        let lon = f.degrees("lon", 4)?;
        let mut bbox = [0i32; 4];
        for (v, what) in bbox.iter_mut().zip(["west", "south", "east", "north"]) {
            *v = f.degrees(what, 3)?;
        }
        let counties = parts(&mut f, "counties", 5, all)?;
        let places = parts(&mut f, "places", 7, all)?;
        f.end()?;
        if counties.is_empty() {
            return Err(f.err("no county"));
        }
        if out.last().is_some_and(|r| r.code >= code) {
            return Err(f.err("ZCTAs out of order"));
        }
        let land = u32::try_from(land).map_err(|_| f.err("negative land area"))?;
        let [w, s, e, no] = bbox.map(|v| f64::from(v) / 1e3);
        let bx = GeoBoundingBox::new(w, s, e, no).map_err(|e| f.err(&e.to_string()))?;
        let pt = GeoPoint::new(f64::from(lat) / 1e4, f64::from(lon) / 1e4).map_err(|e| f.err(&e.to_string()))?;
        if !bx.contains(pt) {
            return Err(f.err("the point is outside the box"));
        }
        out.push(ZctaRow { line: n, code, state, land, lat, lon, bbox, counties, places });
    }
    Ok(())
}

pub(crate) fn places(text: &str, out: &mut Vec<PlaceRow>) -> Result<(), TableError> {
    for (n, l) in lines(text) {
        let mut f = Fields::new("us-places", n, l);
        let geoid = f.digits("geoid", 7)?;
        let state = f.state()?;
        let name = f.text("name")?.into();
        let kind = f.next("kind")?.into();
        let point = f.optional_point()?;
        f.end()?;
        if out.last().is_some_and(|r: &PlaceRow| r.geoid >= geoid) {
            return Err(f.err("places out of order"));
        }
        out.push(PlaceRow { geoid, state, name, kind, point });
    }
    Ok(())
}

pub(crate) fn counties(text: &str, out: &mut Vec<CountyRow>) -> Result<(), TableError> {
    for (n, l) in lines(text) {
        let mut f = Fields::new("us-counties", n, l);
        let geoid = f.digits("geoid", 5)?;
        let state = f.state()?;
        let name = f.text("name")?.into();
        f.end()?;
        if out.last().is_some_and(|r: &CountyRow| r.geoid >= geoid) {
            return Err(f.err("counties out of order"));
        }
        out.push(CountyRow { geoid, state, name });
    }
    Ok(())
}

pub(crate) fn prefixes(text: &str, out: &mut Vec<PrefixRow>) -> Result<(), TableError> {
    for (n, l) in lines(text) {
        let mut f = Fields::new("zip3-states", n, l);
        let prefix = f.digits("prefix", 3)? as u16;
        let states = f.next("states")?;
        f.end()?;
        for s in states.split(',') {
            let bad = || f.err(&format!("state is not CODE:count: {s:?}"));
            let (code, count) = s.split_once(':').ok_or_else(bad)?;
            let code: [u8; 2] = code.as_bytes().try_into().map_err(|_| bad())?;
            if !code.iter().all(u8::is_ascii_uppercase) {
                return Err(bad());
            }
            let count = count.parse::<u32>().ok().filter(|&c| c > 0).ok_or_else(bad)?;
            if out.last().is_some_and(|r: &PrefixRow| (r.prefix, r.state) >= (prefix, code)) {
                return Err(f.err("prefixes out of order"));
            }
            out.push(PrefixRow { prefix, state: code, zctas: count });
        }
    }
    Ok(())
}
