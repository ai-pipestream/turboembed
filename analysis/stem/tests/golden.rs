//! Compares every stemmer with what OpenNLP returned for the same words
//! (`golden/expected/`, written by `golden/generate.sh`), and the Snowball
//! stemmers with the samples of the Snowball reference vocabularies in
//! `golden/snowball/`. `README.md` describes the files.

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use turbo_stem::{Algorithm, LightStemmer, NorwegianStandard, SnowballAlgorithm, SnowballStemmer, Stemmer};

const PIN: &str = "cdf33c653d5d61d5af7d37ffee189a83bd8392ea";

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("golden")
}

/// The stemmer each file of `expected/` was written by.
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

/// A word or stem as the golden files write it: the backslash, tab, line
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
    let golden = golden();
    let configs = configs();
    let mut files: Vec<String> =
        fs::read_dir(golden.join("expected")).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    files.sort();
    let mut names: Vec<String> = configs.iter().map(|(n, _)| format!("{n}.txt")).collect();
    names.sort();
    assert_eq!(files, names, "every expected file has a stemmer and every stemmer a file");

    let mut failures = Vec::new();
    let mut compared = 0;
    for (name, algorithm) in &configs {
        let text = fs::read_to_string(golden.join("expected").join(format!("{name}.txt"))).unwrap();
        let mut lines = text.lines();
        let header = lines.next().unwrap();
        let (_, inputs) = header.split_once(" over ").unwrap();
        assert!(header.contains(&format!(" at OpenNLP {PIN} ")), "{name}: {header}");
        let mut expected = lines;
        let mut out = Vec::new();
        for input in inputs.split(' ') {
            for word in words(&golden.join("inputs").join(input)) {
                let want = expected.next().unwrap_or_else(|| panic!("{name}: fewer stems than words"));
                algorithm.stem_utf16(&word, &mut out);
                let got = escape(&out);
                compared += 1;
                if got != want {
                    failures.push(format!("{name} ({input}): {} -> {got}, OpenNLP {want}", escape(&word)));
                }
            }
        }
        assert!(expected.next().is_none(), "{name}: more stems than words");
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
/// of snowball-data.
fn snowball_directory(a: SnowballAlgorithm) -> String {
    a.java_name().to_lowercase()
}

#[test]
fn snowball_matches_the_reference_samples() {
    let golden = golden();
    let mut sampled = 0;
    for a in SnowballAlgorithm::ALL {
        let dir = snowball_directory(a);
        let reference = golden.join("snowball").join(format!("{dir}.txt"));
        if !reference.exists() {
            continue;
        }
        sampled += 1;
        let voc = words(&golden.join("inputs").join(format!("{dir}.txt")));
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
