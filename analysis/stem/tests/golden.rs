//! Compares every stemmer with what OpenNLP returned for the same words
//! (the goldens of `StemGolden.java` in `analysis/goldens-tool`, under
//! `analysis/goldens/data/stem`), and the Snowball stemmers with the
//! samples of the Snowball reference vocabularies in `tests/snowball/`.
//! `README.md` describes the files.

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use turbo_goldens::{ColumnSpec, ColumnType, Golden, GoldenAnchor, Shape};
use turbo_stem::{Algorithm, LightStemmer, NorwegianStandard, SnowballAlgorithm, SnowballStemmer, Stemmer};

/// The goldens' directory: one directory per stemmer, holding one golden
/// per input it read.
fn goldens() -> PathBuf {
    turbo_goldens::data_dir().join("stem")
}

/// The words the stemmers read, `analysis/goldens/data/inputs/stem`.
fn inputs() -> PathBuf {
    turbo_goldens::data_dir().join("inputs/stem")
}

/// The stemmer each directory of goldens was written by.
fn configs() -> Vec<(String, Algorithm)> {
    let mut out = vec![("porter".to_string(), Algorithm::Porter)];
    for a in SnowballAlgorithm::ALL {
        let name = format!("snowball-{}", a.java_name().to_lowercase());
        out.push((name, Algorithm::Snowball(SnowballStemmer::new(a))));
    }
    for a in
        [SnowballAlgorithm::English, SnowballAlgorithm::German, SnowballAlgorithm::Russian, SnowballAlgorithm::Turkish]
    {
        let name = format!("snowball-{}-repeat2", a.java_name().to_lowercase());
        out.push((name, Algorithm::Snowball(SnowballStemmer::with_repeat(a, NonZeroU32::new(2).unwrap()))));
    }
    use LightStemmer as L;
    use NorwegianStandard as N;
    for (name, l) in [
        ("english-minimal", L::EnglishMinimal),
        ("finnish-light", L::FinnishLight),
        ("french-light", L::FrenchLight),
        ("french-minimal", L::FrenchMinimal),
        ("german-light", L::GermanLight),
        ("german-minimal", L::GermanMinimal),
        ("hungarian-light", L::HungarianLight),
        ("italian-light", L::ItalianLight),
        ("norwegian-light-bokmaal", L::NorwegianLight(N::Bokmaal)),
        ("norwegian-light-nynorsk", L::NorwegianLight(N::Nynorsk)),
        ("norwegian-light-both", L::NorwegianLight(N::Both)),
        ("norwegian-minimal-bokmaal", L::NorwegianMinimal(N::Bokmaal)),
        ("norwegian-minimal-nynorsk", L::NorwegianMinimal(N::Nynorsk)),
        ("norwegian-minimal-both", L::NorwegianMinimal(N::Both)),
        ("portuguese-light", L::PortugueseLight),
        ("russian-light", L::RussianLight),
        ("spanish-light", L::SpanishLight),
        ("spanish-minimal", L::SpanishMinimal),
        ("swedish-light", L::SwedishLight),
        ("swedish-minimal", L::SwedishMinimal),
    ] {
        out.push((name.to_string(), Algorithm::Light(l)));
    }
    out
}

/// A word or stem as the stem inputs write it: the backslash, tab, line
/// feed, carriage return, other C0 controls, lone surrogates and a leading
/// `#` are escaped.
fn escape(units: &[u16]) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < units.len() {
        let u = units[i];
        if i == 0 && u == u16::from(b'#') {
            out.push_str("\\u0023");
        } else if (0xD800..0xDC00).contains(&u) && units.get(i + 1).is_some_and(|n| (0xDC00..0xE000).contains(n)) {
            out.extend(char::decode_utf16([u, units[i + 1]]).map(Result::unwrap));
            i += 1;
        } else if u == u16::from(b'\\') {
            out.push_str("\\\\");
        } else if u == 9 {
            out.push_str("\\t");
        } else if u == 10 {
            out.push_str("\\n");
        } else if u == 13 {
            out.push_str("\\r");
        } else if u < 0x20 || (0xD800..0xE000).contains(&u) {
            out.push_str(&format!("\\u{u:04X}"));
        } else {
            out.push(char::from_u32(u32::from(u)).unwrap());
        }
        i += 1;
    }
    out
}

/// The UTF-16 units of a line of an input file.
fn unescape(line: &str) -> Vec<u16> {
    let mut out = Vec::new();
    let mut chars = line.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => out.push(u16::from(b'\\')),
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&hex, 16).unwrap());
            }
            other => panic!("bad escape {other:?} in {line}"),
        }
    }
    out
}

/// The UTF-16 units of a string `TrainGolden.lossless` wrote: a doubled
/// backslash is one, `\uXXXX` is one unit (half a surrogate pair where the
/// string held one), everything else is itself.
fn lossless_units(field: &str) -> Vec<u16> {
    let mut units: Vec<u16> = Vec::with_capacity(field.len());
    let mut chars = field.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut buf));
            continue;
        }
        match chars.next() {
            Some('\\') => units.push(u16::from(b'\\')),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                units.push(u16::from_str_radix(&hex, 16).unwrap_or_else(|_| panic!("{field:?}: bad escape")));
            }
            other => panic!("{field:?}: unexpected {other:?} after a backslash"),
        }
    }
    units
}

/// The columns of `turbo:stems`, a stem golden's one layer.
const STEMS: &[ColumnSpec] = &[
    ColumnSpec::new("word", ColumnType::Str),
    ColumnSpec::new("stem", ColumnType::Str),
    ColumnSpec::new("thrown", ColumnType::Str),
];

fn shapes() -> [Shape; 1] {
    [Shape { id: "turbo:stems".into(), anchor: GoldenAnchor::Document, columns: STEMS }]
}

/// The words of an input file; a line starting with # is a comment.
fn words(path: &Path) -> Vec<Vec<u16>> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().filter(|l| !l.starts_with('#')).map(unescape).collect()
}

#[test]
fn escapes_round_trip() {
    for units in [
        &[0x23, 0x61][..],
        &[0x61, 0x23],
        &[0x5C, 9, 10, 13, 1],
        &[0xD83D, 0xDE00, 0x61],
        &[0xD83D, 0x61, 0xDE00],
        &[0xDBFF],
        &[],
    ] {
        assert_eq!(unescape(&escape(units)), units);
    }
    assert_eq!(escape(&[0xD83D, 0x61]), "\\uD83Da");
}

#[test]
fn stemmers_match_opennlp() {
    let configs = configs();
    let mut dirs: Vec<String> =
        fs::read_dir(goldens()).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    dirs.sort();
    let mut names: Vec<String> = configs.iter().map(|(n, _)| n.clone()).collect();
    names.sort();
    assert_eq!(dirs, names, "every directory of goldens has a stemmer and every stemmer a directory");

    let shapes = shapes();
    let mut failures = Vec::new();
    let mut compared = 0;
    for (name, algorithm) in &configs {
        let dir = goldens().join(name).join("stem");
        let mut files: Vec<PathBuf> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
        files.sort();
        assert!(files.iter().any(|f| f.ends_with("edge.golden")), "{name}: no edge golden");
        let mut out = Vec::new();
        for file in files {
            let g = Golden::read_with(&file, &shapes).unwrap_or_else(|e| panic!("{e}"));
            let input = &g.header.input.id;
            let words = words(&inputs().join(format!("{input}.txt")));
            let l = g.layer("turbo:stems").unwrap();
            let (word, stem, thrown) = (l.strs("word").unwrap(), l.strs("stem").unwrap(), l.strs("thrown").unwrap());
            assert_eq!(words.len(), l.len(), "{name} ({input}): a stem per word");
            for (i, w) in words.iter().enumerate() {
                assert_eq!(&lossless_units(word.get(i)), w, "{name} ({input}): word {i}");
                assert_eq!(thrown.get(i), "", "{name} ({input}): OpenNLP threw on {}", escape(w));
                algorithm.stem_utf16(w, &mut out);
                let want = lossless_units(stem.get(i));
                compared += 1;
                if out != want {
                    failures.push(format!(
                        "{name} ({input}): {} -> {}, OpenNLP {}",
                        escape(w),
                        escape(&out),
                        escape(&want)
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {compared} differ:\n{}",
        failures.len(),
        failures[..failures.len().min(40)].join("\n")
    );
    assert_eq!(compared, 90_713);
}

/// The Snowball algorithm whose reference vocabulary is in each directory
/// of snowball-data, and the name of its sample.
fn snowball_directory(a: SnowballAlgorithm) -> String {
    a.java_name().to_lowercase()
}

#[test]
fn snowball_matches_the_reference_samples() {
    let samples = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/snowball");
    let mut sampled = 0;
    for a in SnowballAlgorithm::ALL {
        let dir = snowball_directory(a);
        let reference = samples.join(format!("{dir}.txt"));
        if !reference.exists() {
            continue;
        }
        sampled += 1;
        let voc = words(&inputs().join(format!("{dir}.txt")));
        let output = fs::read_to_string(&reference).unwrap();
        let output: Vec<&str> = output.lines().collect();
        assert_eq!(voc.len(), output.len(), "{dir}");
        let stemmer = SnowballStemmer::new(a);
        let mut out = Vec::new();
        for (word, want) in voc.iter().zip(output) {
            stemmer.stem_utf16(word, &mut out);
            assert_eq!(String::from_utf16(&out).unwrap(), want, "{dir}: {}", String::from_utf16_lossy(word));
        }
    }
    assert_eq!(sampled, 16);
}
