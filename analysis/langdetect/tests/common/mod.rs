//! Helpers shared by the integration tests: finding the goldens and the
//! model, reading decision point dumps, and building small model files.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use turbo_goldens::data_dir;
use turbo_maxent::crc32;

/// The model the goldens were made with.
pub const MODEL_FILE: &str = "langdetect-183.bin";

/// `analysis/goldens/data/langdetect/langdetect-183`.
pub fn golden_dir() -> PathBuf {
    data_dir().join("langdetect").join("langdetect-183")
}

/// Every golden of the language detector, as (set, id, path), sorted.
pub fn goldens() -> Vec<(String, String, PathBuf)> {
    goldens_in(&golden_dir())
}

/// Every golden under `dir`, a model's directory, as (set, id, path),
/// sorted.
pub fn goldens_in(dir: &Path) -> Vec<(String, String, PathBuf)> {
    let mut out = Vec::new();
    for set in std::fs::read_dir(dir).unwrap() {
        let set = set.unwrap().path();
        for f in std::fs::read_dir(&set).unwrap() {
            let path = f.unwrap().path();
            if path.extension().is_some_and(|e| e == "golden") {
                let set = set.file_name().unwrap().to_string_lossy().into_owned();
                let id = path.file_stem().unwrap().to_string_lossy().into_owned();
                out.push((set, id, path));
            }
        }
    }
    out.sort();
    out
}

/// `langdetect-183.bin`, where `turbo-goldens fetch-models` puts it
/// (`turbo_goldens::published_model`). The model is not checked in, so the
/// tests that need it say so and pass without it, unless
/// `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes a missing model fail them.
pub fn model_path() -> Option<PathBuf> {
    turbo_goldens::published_model(MODEL_FILE)
}

/// One decision point of a dump: the contexts as Java modified UTF-8, the
/// value given with each, and each outcome's label and probability.
#[derive(Debug)]
pub struct Point {
    pub contexts: Vec<Vec<u8>>,
    pub values: Vec<f64>,
    pub outcomes: Vec<(String, f64)>,
}

/// Reads a `.dump` file of `LanguageDetectorME.predictLanguages`: its
/// header, then one point with values given. Each check names the line.
pub fn read_dump(path: &Path) -> Vec<Point> {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().enumerate();
    assert_eq!(lines.next().map(|(_, l)| l), Some("dump 1"));
    let mut points = Vec::new();
    let mut current: Option<Point> = None;
    for (n, line) in lines {
        let fields: Vec<&str> = line.split('\t').collect();
        let at = || format!("{}:{}", path.display(), n + 1);
        match fields[0] {
            "point" => {
                assert!(current.is_none(), "{}", at());
                assert_eq!(fields[6], "given", "{}", at());
                current = Some(Point { contexts: Vec::new(), values: Vec::new(), outcomes: Vec::new() });
            }
            "context" => {
                let p = current.as_mut().unwrap();
                p.contexts.push(modified_utf8(&unescape_units(fields[1])));
                p.values.push(cell(fields[2]));
            }
            "outcome" => {
                let p = current.as_mut().unwrap();
                p.outcomes.push((String::from_utf16(&unescape_units(fields[1])).unwrap(), cell(fields[2])));
            }
            "end" => points.push(current.take().unwrap()),
            _ => assert!(current.is_none(), "{}: {line}", at()),
        }
    }
    assert!(current.is_none());
    points
}

/// An `f64` cell: 16 hex digits of the bits, a space, the decimal.
fn cell(s: &str) -> f64 {
    let (bits, decimal) = s.split_once(' ').unwrap();
    let v = f64::from_bits(u64::from_str_radix(bits, 16).unwrap());
    assert!(decimal.parse::<f64>().is_ok_and(|d| d.to_bits() == v.to_bits()) || v.is_nan(), "{s}");
    v
}

/// A `str` cell as UTF-16 code units: `\t`, `\n`, `\r`, `\\` and `\uXXXX`
/// undone, a `\u` escape being one code unit, which may be half a pair.
pub fn unescape_units(s: &str) -> Vec<u16> {
    let mut out = Vec::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('t') => out.push(u16::from(b'\t')),
            Some('n') => out.push(u16::from(b'\n')),
            Some('r') => out.push(u16::from(b'\r')),
            Some('\\') => out.push(u16::from(b'\\')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {s:?}"),
        }
    }
    out
}

/// UTF-16 code units as Java modified UTF-8, one unit at a time.
pub fn modified_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for &u in units {
        turbo_unicode::modified_utf8::push_modified_utf8(&mut out, u32::from(u));
    }
    out
}

/// A string as Java modified UTF-8.
pub fn mutf8(s: &str) -> Vec<u8> {
    modified_utf8(&s.encode_utf16().collect::<Vec<u16>>())
}

/// Java's `DataOutputStream`, for the few fields a GIS model has.
struct Data(Vec<u8>);

impl Data {
    fn utf(&mut self, units: &[u16]) {
        let bytes = modified_utf8(units);
        self.0.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        self.0.extend_from_slice(&bytes);
    }
    fn str(&mut self, s: &str) {
        self.utf(&s.encode_utf16().collect::<Vec<u16>>());
    }
    fn int(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_be_bytes());
    }
    fn double(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_be_bytes());
    }
}

/// A predicate's name as UTF-16 code units and its (outcome, weight)
/// parameters.
pub type Predicate = (Vec<u16>, Vec<(u32, f64)>);

/// A GIS model as `GISModelWriter` writes it: the outcomes, and per
/// predicate (given as UTF-16 code units, so it can hold a lone
/// surrogate) its (outcome, weight) parameters. Each predicate gets an
/// outcome pattern of its own.
pub fn gis_model(outcomes: &[&str], predicates: &[Predicate]) -> Vec<u8> {
    let mut d = Data(Vec::new());
    d.str("GIS");
    d.int(1);
    d.double(1.0);
    d.int(outcomes.len() as i32);
    for o in outcomes {
        d.str(o);
    }
    d.int(predicates.len() as i32);
    for (_, params) in predicates {
        let pattern: Vec<String> =
            std::iter::once("1".to_string()).chain(params.iter().map(|(o, _)| o.to_string())).collect();
        d.str(&pattern.join(" "));
    }
    d.int(predicates.len() as i32);
    for (name, _) in predicates {
        d.utf(name);
    }
    for (_, params) in predicates {
        for &(_, w) in params {
            d.double(w);
        }
    }
    d.0
}

/// The manifest of a language detector model.
pub const MANIFEST: &str = "Language=und\nfactory=opennlp.tools.langdetect.LanguageDetectorFactory\n\
Component-Name=LanguageDetectorME\nOpenNLP-Version=3.0.0-SNAPSHOT\nManifest-Version=1.0\n";

/// A zip archive of stored entries.
pub fn zip(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut z = Vec::new();
    let mut central = Vec::new();
    for (name, data) in entries {
        let offset = z.len() as u32;
        let head = |sig: u32, out: &mut Vec<u8>, central: bool| {
            out.extend_from_slice(&sig.to_le_bytes());
            if central {
                out.extend_from_slice(&20u16.to_le_bytes());
            }
            out.extend_from_slice(&[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
            out.extend_from_slice(&crc32(data).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(name.len() as u16).to_le_bytes());
            out.extend_from_slice(&0u16.to_le_bytes());
        };
        head(0x0403_4b50, &mut z, false);
        z.extend_from_slice(name.as_bytes());
        z.extend_from_slice(data);
        head(0x0201_4b50, &mut central, true);
        central.extend_from_slice(&[0; 10]);
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(name.as_bytes());
    }
    let dir = z.len() as u32;
    z.extend_from_slice(&central);
    z.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    let n = entries.len() as u16;
    z.extend_from_slice(&[0, 0, 0, 0]);
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&n.to_le_bytes());
    z.extend_from_slice(&(central.len() as u32).to_le_bytes());
    z.extend_from_slice(&dir.to_le_bytes());
    z.extend_from_slice(&[0, 0]);
    z
}
