//! Helpers shared by the integration tests: the language detector model
//! the fixtures are scored with, and a small model written by the test
//! (the same writer as turbo-langdetect's tests, copied, since tests of
//! one crate cannot share another's helpers).

#![allow(dead_code)]

use std::path::PathBuf;

use turbo_langdetect::{LanguageDetector, LanguageModel};
use turbo_maxent::{ModelFile, crc32};

/// The model the fixtures are scored with.
pub const MODEL_FILE: &str = "langdetect-183.bin";

/// `langdetect-183.bin`, where `turbo-goldens fetch-models` puts it. It is
/// not checked in, so the tests that need it say so and pass without it,
/// unless `TURBO_TEST_REQUIRE_OPENNLP_MODELS=1` makes a missing model
/// fail them, as CI sets it.
pub fn model_path() -> Option<PathBuf> {
    turbo_goldens::published_model(MODEL_FILE)
}

/// OpenNLP's detector over `langdetect-183.bin`, when it is there.
pub fn published_detector() -> Option<LanguageDetector> {
    let path = model_path()?;
    Some(LanguageDetector::new(LanguageModel::open(ModelFile::read(path).unwrap()).unwrap()))
}

/// A detector over a model of the given outcomes, each predicate (one
/// string) with its (outcome, weight) parameters.
pub fn small_detector(outcomes: &[&str], predicates: &[(&str, &[(u32, f64)])]) -> LanguageDetector {
    let predicates: Vec<Predicate> =
        predicates.iter().map(|(name, params)| (name.encode_utf16().collect(), params.to_vec())).collect();
    let classifier = gis_model(outcomes, &predicates);
    let archive = zip(&[("manifest.properties", MANIFEST.as_bytes()), ("langdetect.model", &classifier)]);
    LanguageDetector::new(LanguageModel::open(ModelFile::open(archive).unwrap()).unwrap())
}

/// UTF-16 code units as Java modified UTF-8, one unit at a time.
pub fn modified_utf8(units: &[u16]) -> Vec<u8> {
    let mut out = Vec::new();
    for &u in units {
        turbo_unicode::modified_utf8::push_modified_utf8(&mut out, u32::from(u));
    }
    out
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
