//! Readers for the Census Bureau files the tables are built from: the
//! Gazetteer files (tab separated), the 2020 relationship files and the
//! county list (pipe separated), and the parts of a shapefile the boxes
//! come from.

use std::collections::HashMap;
use std::path::Path;

/// The text of `path`: UTF-8 without its byte order mark, or, when the
/// bytes are not UTF-8, ISO 8859-1, each byte one character.
pub fn text(path: &Path) -> (String, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digest = turbo_sha256::sha256(&bytes).to_hex();
    let body = bytes.strip_prefix(b"\xef\xbb\xbf").unwrap_or(&bytes);
    let s = match std::str::from_utf8(body) {
        Ok(s) => s.to_string(),
        Err(_) => body.iter().map(|&b| char::from(b)).collect(),
    };
    (s, digest)
}

/// The rows of a delimited table with a header line, each a map from
/// column name to trimmed value; `need` names the columns that must exist.
pub struct Table {
    columns: HashMap<String, usize>,
    rows: Vec<Vec<String>>,
}

impl Table {
    pub fn parse(text: &str, sep: char, need: &[&str], what: &str) -> Table {
        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().unwrap_or_else(|| panic!("{what}: empty"));
        let columns: HashMap<String, usize> =
            header.split(sep).enumerate().map(|(i, c)| (c.trim().to_string(), i)).collect();
        for n in need {
            assert!(columns.contains_key(*n), "{what}: no column {n} in {header:?}");
        }
        let rows = lines.map(|l| l.split(sep).map(|f| f.trim().to_string()).collect()).collect();
        Table { columns, rows }
    }

    pub fn rows(&self) -> impl Iterator<Item = Row<'_>> {
        self.rows.iter().map(|r| Row { columns: &self.columns, values: r })
    }
}

pub struct Row<'a> {
    columns: &'a HashMap<String, usize>,
    values: &'a [String],
}

impl Row<'_> {
    pub fn get(&self, column: &str) -> &str {
        self.values.get(self.columns[column]).map_or("", String::as_str)
    }

    pub fn int(&self, column: &str) -> i64 {
        let v = self.get(column);
        v.parse().unwrap_or_else(|_| panic!("{column}: not an integer: {v:?}"))
    }

    /// A decimal in degrees as an integer of `10^-places` degrees,
    /// rounded half away from zero.
    pub fn fixed(&self, column: &str, places: u32) -> i64 {
        let v = self.get(column);
        let f: f64 = v.parse().unwrap_or_else(|_| panic!("{column}: not a number: {v:?}"));
        (f * 10f64.powi(places as i32)).round() as i64
    }
}

/// The bounding box of every record of a polygon shapefile, by the value
/// of the attribute `key` (`GEOID20`) of its record in the `.dbf` table,
/// as west, south, east, north in thousandths of a degree, rounded
/// outward so the box still holds the shape.
pub fn shape_boxes(shp: &Path, dbf: &Path, key: &str) -> (HashMap<String, [i64; 4]>, String, String) {
    let (keys, dbf_digest) = dbf_column(dbf, key);
    let bytes = std::fs::read(shp).unwrap_or_else(|e| panic!("{}: {e}", shp.display()));
    let digest = turbo_sha256::sha256(&bytes).to_hex();
    assert_eq!(be_u32(&bytes, 0), 9994, "{}: not a shapefile", shp.display());
    let mut at = 100;
    let mut out = HashMap::new();
    let mut n = 0;
    while at + 8 <= bytes.len() {
        let words = be_u32(&bytes, at + 4) as usize;
        let body = at + 8;
        let shape = le_u32(&bytes, body);
        if shape != 0 {
            let f = |k: usize| f64::from_le_bytes(bytes[body + 4 + 8 * k..body + 12 + 8 * k].try_into().unwrap());
            let (w, s, e, no) = (f(0), f(1), f(2), f(3));
            let down = |v: f64| (v * 1000.0).floor() as i64;
            let up = |v: f64| (v * 1000.0).ceil() as i64;
            out.insert(keys[n].clone(), [down(w), down(s), up(e), up(no)]);
        }
        n += 1;
        at = body + 2 * words;
    }
    assert_eq!(n, keys.len(), "the .shp and .dbf hold different record counts");
    (out, digest, dbf_digest)
}

/// The trimmed values of one character column of a dBASE table, and the
/// table's SHA-256.
fn dbf_column(path: &Path, name: &str) -> (Vec<String>, String) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let digest = turbo_sha256::sha256(&b).to_hex();
    let records = le_u32(&b, 4) as usize;
    let header = usize::from(u16::from_le_bytes([b[8], b[9]]));
    let width = usize::from(u16::from_le_bytes([b[10], b[11]]));
    let mut offset = 1;
    let mut field = None;
    let mut at = 32;
    while b[at] != 0x0d {
        let fname: Vec<u8> = b[at..at + 11].iter().copied().take_while(|&c| c != 0).collect();
        let len = usize::from(b[at + 16]);
        if fname == name.as_bytes() {
            field = Some((offset, len));
        }
        offset += len;
        at += 32;
    }
    let (start, len) = field.unwrap_or_else(|| panic!("{}: no field {name}", path.display()));
    let keys = (0..records)
        .map(|r| {
            let rec = header + r * width;
            String::from_utf8_lossy(&b[rec + start..rec + start + len]).trim().to_string()
        })
        .collect();
    (keys, digest)
}

fn be_u32(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}

/// `v` with a decimal point `places` digits from its end (`-896820`, 4
/// gives `-89.6820`).
pub fn fixed(v: i64, places: u32) -> String {
    let scale = 10i64.pow(places);
    let sign = if v < 0 { "-" } else { "" };
    let a = v.unsigned_abs();
    let s = scale as u64;
    format!("{sign}{}.{:0width$}", a / s, a % s, width = places as usize)
}
