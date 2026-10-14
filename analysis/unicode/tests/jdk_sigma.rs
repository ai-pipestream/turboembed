//! The JDK's final sigma condition against a running JDK 21: the word
//! classes of `java::word_class` against the root locale's word
//! `BreakIterator` for every assigned code point (`tests/jdk/WordClasses.java`),
//! and `java::to_lowercase` against `String.toLowerCase(Locale.ROOT)` on
//! hundreds of thousands of random short strings holding a capital sigma
//! (`tests/jdk/FinalSigma.java`), in the Basic Multilingual Plane and in
//! every plane.
//!
//! Like `tests/jdk.rs`, the tests run only with `TURBO_TEST_JDK` set and
//! then need `java` and `javac` on the PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_unicode::java::{self, WordClass};

fn run(command: &mut Command) -> String {
    let output = command.output().unwrap_or_else(|e| panic!("{command:?} does not start: {e}"));
    assert!(output.status.success(), "{command:?} failed: {}", String::from_utf8_lossy(&output.stderr));
    String::from_utf8(output.stdout).expect("the JDK prints ASCII")
}

fn jdk_requested() -> bool {
    if std::env::var_os("TURBO_TEST_JDK").is_none() {
        eprintln!("skipped: set TURBO_TEST_JDK to compare with a JDK");
        return false;
    }
    true
}

/// Compiles `tests/jdk/<name>.java` and runs it with `args`.
fn jdk(name: &str, args: &[&str]) -> String {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/jdk/{name}.java"));
    let classes: PathBuf = std::env::temp_dir().join(format!("turbo-unicode-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&classes).unwrap();
    run(Command::new("javac").arg("-d").arg(&classes).arg(&source));
    let output = run(Command::new("java").arg("-cp").arg(&classes).arg(name).args(args));
    let _ = std::fs::remove_dir_all(&classes);
    output
}

/// The class a probe signature names: the sixteen boundary answers of
/// `WordClasses.java`. A supplementary character has its own signatures,
/// as `isBoundary` right after one answers true.
fn class_of(cp: u32, signature: &str) -> WordClass {
    let bmp = cp <= 0xFFFF;
    match (bmp, signature) {
        (true, "0000000000000001") | (false, "0101010101000011") => WordClass::Letter,
        (true, "0000000000000010") => WordClass::Digit,
        (true, "0000000000010000") | (false, "0101010101110011") => WordClass::Mark,
        (true, "0000000000000000") => WordClass::Transparent,
        (true, "0011111111111111") | (false, "0111111111111111") => WordClass::MidWord,
        (true, "1100111111111111") => WordClass::MidNum,
        (true, "0000111111111111") => WordClass::MidBoth,
        (true, "0000101111111111") => WordClass::Dot,
        (_, "1110101111111111") => WordClass::PreNum,
        (true, "1101110111111111") => WordClass::PostNum,
        (true, "0111001101110111") => WordClass::Danda,
        (_, "1111111111111111") | (true, "1111111110111111") => WordClass::Other,
        _ => panic!("U+{cp:04X}: the signature {signature} names no class; the probe found a new behaviour"),
    }
}

#[test]
fn word_classes_match_the_jdk() {
    if !jdk_requested() {
        return;
    }
    let dump = jdk("WordClasses", &[]);
    let mut lines = dump.lines();
    assert_eq!(lines.next().unwrap().trim(), "21", "the classes were probed on JDK 21");
    let mut mismatches = Vec::new();
    let mut checked = 0;
    for line in lines {
        let (hex, signature) = line.split_once(' ').unwrap();
        let cp = u32::from_str_radix(hex, 16).unwrap();
        let c = char::from_u32(cp).unwrap();
        let jdk = class_of(cp, signature);
        let ours = java::word_class(c);
        checked += 1;
        if ours != jdk {
            mismatches.push(format!("U+{cp:04X}: {ours:?} here, {jdk:?} on the JDK ({signature})"));
        }
    }
    assert!(checked > 280_000, "{checked} code points");
    assert!(mismatches.is_empty(), "{} word classes differ:\n{}", mismatches.len(), mismatches.join("\n"));
}

/// Lowercases `text` and reads the sigma decisions as the probe writes them.
fn decisions(text: &str) -> String {
    let mut lower = String::new();
    java::to_lowercase(text, &mut lower);
    text.chars()
        .zip(lower.chars())
        .filter(|(c, _)| *c == '\u{3A3}')
        .map(|(_, l)| if l == '\u{3C2}' { 'F' } else { 'S' })
        .collect()
}

fn compare_random(seed: u64, count: usize, planes: &str) -> Vec<String> {
    let dump = jdk("FinalSigma", &[&seed.to_string(), &count.to_string(), planes]);
    let mut mismatches = Vec::new();
    let mut lines = 0;
    for line in dump.lines() {
        let (cps, jdk) = line.split_once(' ').unwrap();
        let text: String =
            cps.split('-').map(|h| char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap()).collect();
        lines += 1;
        let ours = decisions(&text);
        if ours != jdk {
            mismatches.push(format!("{text:?} ({cps}): {ours} here, {jdk} on the JDK"));
        }
    }
    assert_eq!(lines, count);
    mismatches
}

#[test]
fn final_sigma_matches_the_jdk_on_random_strings() {
    if !jdk_requested() {
        return;
    }
    // Another seed, for a run that wants fresh strings: TURBO_TEST_JDK_SEED.
    let seed: u64 = std::env::var("TURBO_TEST_JDK_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let mut mismatches = compare_random(seed, 200_000, "bmp");
    mismatches.extend(compare_random(seed + 1, 200_000, "all"));
    let shown: Vec<&String> = mismatches.iter().take(40).collect();
    assert!(
        mismatches.is_empty(),
        "{} of 400000 strings lowercase differently (first {}):\n{}",
        mismatches.len(),
        shown.len(),
        shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
}
