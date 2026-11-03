//! Reviewed stem snapshots: each stemmer's output on the checked-in word lists.
//!
//! Files under `tests/snapshots/<stemmer>/<input>.tsv` hold `word<TAB>stem`
//! lines, written by this crate (`TURBO_BLESS=1 cargo test -p turbo-stem
//! --test snapshots`). No JVM output.

use std::fs;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use turbo_stem::{Algorithm, LightStemmer, NorwegianStandard, SnowballAlgorithm, SnowballStemmer, Stemmer};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn inputs() -> PathBuf {
    root().join("inputs")
}

fn snapshots() -> PathBuf {
    root().join("snapshots")
}

fn configs() -> Vec<(String, Algorithm)> {
    let mut out = vec![("porter".to_string(), Algorithm::Porter)];
    for a in SnowballAlgorithm::ALL {
        out.push((format!("snowball-{}", a.java_name().to_lowercase()), Algorithm::Snowball(SnowballStemmer::new(a))));
    }
    for a in
        [SnowballAlgorithm::English, SnowballAlgorithm::German, SnowballAlgorithm::Russian, SnowballAlgorithm::Turkish]
    {
        out.push((
            format!("snowball-{}-repeat2", a.java_name().to_lowercase()),
            Algorithm::Snowball(SnowballStemmer::with_repeat(a, NonZeroU32::new(2).unwrap())),
        ));
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
        } else if let Some(c) = char::from_u32(u32::from(u)) {
            out.push(c);
        } else {
            out.push_str(&format!("\\u{u:04X}"));
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

fn words(path: &Path) -> Vec<Vec<u16>> {
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines().filter(|l| !l.starts_with('#')).map(unescape).collect()
}

fn write_snapshot(stemmer: &str, input: &str, rows: &[(String, String)]) {
    let dir = snapshots().join(stemmer);
    fs::create_dir_all(&dir).unwrap();
    let mut text = String::new();
    for (w, s) in rows {
        text.push_str(w);
        text.push('\t');
        text.push_str(s);
        text.push('\n');
    }
    fs::write(dir.join(format!("{input}.tsv")), text).unwrap();
}

fn read_snapshot(stemmer: &str, input: &str) -> Vec<(String, String)> {
    let path = snapshots().join(stemmer).join(format!("{input}.tsv"));
    let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .map(|line| {
            let (w, s) = line.split_once('\t').unwrap_or_else(|| panic!("{path:?}: no tab in {line:?}"));
            (w.to_string(), s.to_string())
        })
        .collect()
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
}

#[test]
fn stemmers_match_the_snapshots() {
    let bless = std::env::var_os("TURBO_BLESS").is_some();
    let configs: std::collections::BTreeMap<String, Algorithm> = configs().into_iter().collect();
    let pairs: Vec<(String, String)> = fs::read_to_string(root().join("snapshot-pairs.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (a, b) = l.split_once('\t').unwrap();
            (a.to_string(), b.to_string())
        })
        .collect();
    assert_eq!(pairs.len(), 131, "snapshot-pairs.txt");
    let mut compared = 0usize;
    let mut failures = Vec::new();
    for (name, id) in &pairs {
        let algorithm = configs.get(name).unwrap_or_else(|| panic!("no stemmer {name}"));
        let words = words(&inputs().join(format!("{id}.txt")));
        let mut out = Vec::new();
        let mut rows = Vec::with_capacity(words.len());
        for w in &words {
            algorithm.stem_utf16(w, &mut out);
            rows.push((escape(w), escape(&out)));
        }
        if bless {
            write_snapshot(name, id, &rows);
            compared += rows.len();
            continue;
        }
        let want = read_snapshot(name, id);
        assert_eq!(want.len(), rows.len(), "{name}/{id}: row count");
        for (i, ((gw, gs), (ww, ws))) in rows.iter().zip(want).enumerate() {
            compared += 1;
            if gw != &ww || gs != &ws {
                failures.push(format!("{name}/{id}[{i}]: got {gw}\t{gs}, snapshot {ww}\t{ws}"));
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

#[test]
fn snowball_matches_the_reference_samples() {
    let samples = root().join("snowball");
    let mut sampled = 0;
    for a in SnowballAlgorithm::ALL {
        let dir = a.java_name().to_lowercase();
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
