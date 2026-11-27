//! The JDK's final sigma condition against a running JDK 21: the word
//! classes of `java::word_class` against the root locale's word
//! `BreakIterator` for every assigned code point (`tests/jdk/WordClasses.java`),
//! and `java::to_lowercase` against `String.toLowerCase(Locale.ROOT)` on
//! hundreds of thousands of random strings holding a capital sigma
//! (`tests/jdk/FinalSigma.java`): short ones from the Basic Multilingual
//! Plane, short ones from every plane, and ones of up to twenty characters
//! from an alphabet that puts dandas, marks, joiners, controls and
//! supplementary characters next to each other.
//!
//! Like `tests/jdk.rs`, the two tests that ask the JDK run only with
//! `TURBO_TEST_JDK` set and then need `java` and `javac` on the PATH.
//! `TURBO_TEST_JDK_SEED` picks the seed of the random strings (1 by
//! default) and `TURBO_TEST_JDK_COUNT` how many of each kind to draw
//! (200000 by default).
//!
//! This file also writes the table of word classes
//! (`src/java/word_classes.rs`): the word class test renders it from the
//! probe's answers and fails when the committed table differs. After a
//! change to the probe, the signatures or the renderer, run
//!
//! ```text
//! TURBO_TEST_JDK=1 TURBO_WRITE_WORD_CLASSES=1 cargo test -p turbo-unicode --test jdk_sigma word_classes
//! ```
//!
//! which writes the table, then run the test again without the second
//! variable. Without a JDK, a test still holds the committed table to the
//! renderer's layout.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use turbo_scratch::Scratch;

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
    let classes = Scratch::at(std::env::temp_dir().join(format!("turbo-unicode-{name}-{}", std::process::id())));
    std::fs::create_dir_all(&classes).unwrap();
    run(Command::new("javac").arg("-d").arg(classes.path()).arg(&source));
    let output = run(Command::new("java").arg("-cp").arg(classes.path()).arg(name).args(args));
    let _ = std::fs::remove_dir_all(&classes);
    output
}

/// The class a probe signature names: the boundary answers of
/// `WordClasses.java`, seventeen for a character of the Basic Multilingual
/// Plane and fourteen, probed at the start of the text, for a supplementary
/// one.
fn class_of(cp: u32, signature: &str) -> WordClass {
    let bmp = cp <= 0xFFFF;
    match (bmp, signature) {
        (true, "00000000000000010") | (false, "00110001000000") => WordClass::Letter,
        (true, "00000000000000100") | (false, "11000010000000") => WordClass::Digit,
        (true, "00000000000100000") | (false, "11111111110000") => WordClass::Mark,
        (true, "00000000000000000") | (false, "01010100000000") => WordClass::Transparent,
        (true, "00111111111111110") | (false, "11110111110111") => WordClass::MidWord,
        // The soft hyphen: a mark after one that joined nothing is a word of its own.
        (true, "00111111111111111") => WordClass::MidWord,
        (true, "11001111111111110") => WordClass::MidNum,
        (true, "00001111111111110") => WordClass::MidBoth,
        (true, "00001011111111110") => WordClass::Dot,
        (true, "11101011111111110") | (false, "11110111101111") => WordClass::PreNum,
        (true, "11011101111111110") => WordClass::PostNum,
        (true, "01110011011101111") => WordClass::Danda,
        (true, "11111111111111110") | (true, "11111111101111110") | (false, "11110111111111") => WordClass::Other,
        (true, "11111111111111111") | (false, "11111111111111") => WordClass::Control,
        _ => panic!("U+{cp:04X}: the signature {signature} names no class; the probe found a new behaviour"),
    }
}

/// The JDK feature version the classes were probed on.
const JDK_FEATURE: &str = "21";

/// The table file for `classes`: each assigned code point with its class,
/// in code point order. Neighbours of one class make one range, a code
/// point missing from `classes` ends a range, and `Other`, the class of
/// every code point the table does not list, is left out. The comments
/// stand in one column, as rustfmt puts them.
fn render(classes: impl IntoIterator<Item = (u32, WordClass)>) -> String {
    let mut ranges: Vec<(u32, u32, WordClass)> = Vec::new();
    for (cp, class) in classes {
        match ranges.last_mut() {
            _ if class == WordClass::Other => {}
            Some((_, last, of)) if *of == class && *last + 1 == cp => *last = cp,
            _ => ranges.push((cp, cp, class)),
        }
    }
    let entries: Vec<String> =
        ranges.iter().map(|&(first, last, class)| format!("(0x{first:04X}, 0x{last:04X}, {}),", class as u8)).collect();
    let width = entries.iter().map(String::len).max().unwrap_or(0);
    let mut out = format!(
        "//! Generated by analysis/unicode/tests/jdk_sigma.rs from the answers of the\n\
         //! root locale's word `BreakIterator` of JDK {JDK_FEATURE}. Do not edit this file by\n\
         //! hand: run that test with TURBO_TEST_JDK and TURBO_WRITE_WORD_CLASSES set\n\
         //! and JDK {JDK_FEATURE} on the PATH.\n\
         //!\n\
         //! What that iterator makes of each code point, as found by probing it:\n\
         //! `analysis/unicode/tests/jdk/WordClasses.java` prints, for every assigned\n\
         //! code point, whether the iterator puts a boundary around it in short\n\
         //! contexts (a supplementary character is probed at the start of the text,\n\
         //! where `isBoundary` after it is not disturbed by the surrogate pair), and\n\
         //! the signature names the class (`word_class` in `sigma.rs` says what each\n\
         //! class does). Code points not listed, the unassigned and most\n\
         //! punctuation, symbols, separators and private use characters among them,\n\
         //! end a word on both sides.\n\
         //!\n\
         //! `(first, last, class)` ranges, sorted, with the class numbers of\n\
         //! `WordClass`.\n\
         pub(crate) static JDK_WORD_CLASSES: &[(u32, u32, u8)] = &[\n"
    );
    for (entry, (_, _, class)) in entries.iter().zip(&ranges) {
        writeln!(out, "    {entry:width$} // {class:?}").unwrap();
    }
    out.push_str("];\n");
    out
}

/// The committed table file and its text.
fn committed() -> (PathBuf, String) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/java/word_classes.rs");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    (path, text)
}

#[test]
fn word_classes_match_the_jdk() {
    if !jdk_requested() {
        return;
    }
    let dump = jdk("WordClasses", &[]);
    let mut lines = dump.lines();
    assert_eq!(lines.next().unwrap().trim(), JDK_FEATURE, "the classes were probed on JDK {JDK_FEATURE}");
    let mut probed = Vec::new();
    let mut mismatches = Vec::new();
    for line in lines {
        let (hex, signature) = line.split_once(' ').unwrap();
        let cp = u32::from_str_radix(hex, 16).unwrap();
        let c = char::from_u32(cp).unwrap();
        let jdk = class_of(cp, signature);
        let ours = java::word_class(c);
        probed.push((cp, jdk));
        if ours != jdk {
            mismatches.push(format!("U+{cp:04X}: {ours:?} here, {jdk:?} on the JDK ({signature})"));
        }
    }
    assert!(probed.len() > 280_000, "{} code points", probed.len());
    let fresh = render(probed);
    let (path, text) = committed();
    let rewrite = std::env::var_os("TURBO_WRITE_WORD_CLASSES").is_some();
    if rewrite {
        std::fs::write(&path, &fresh).unwrap();
    }
    // A rewrite does not reach the table this test was built with.
    let again = if rewrite { "\nsrc/java/word_classes.rs is rewritten; run the test again" } else { "" };
    assert!(mismatches.is_empty(), "{} word classes differ:\n{}{again}", mismatches.len(), mismatches.join("\n"));
    assert!(
        rewrite || text == fresh,
        "src/java/word_classes.rs differs from the JDK's answers; see this file's comment"
    );
}

/// Runs without a JDK: the committed table is what the renderer writes for
/// the classes it holds, so its header, its order, its ranges (none that
/// could join its neighbour) and the class named beside each are the
/// renderer's, and an edit by hand that leaves that form shows here.
#[test]
fn the_committed_table_is_in_the_form_the_renderer_writes() {
    let held = (0..=char::MAX as u32).filter_map(char::from_u32).map(|c| (c as u32, java::word_class(c)));
    let (_, text) = committed();
    assert!(text == render(held), "src/java/word_classes.rs is not as tests/jdk_sigma.rs writes it");
}

#[test]
fn the_renderer_joins_neighbours_of_one_class_and_leaves_other_out() {
    use WordClass::{Control, Letter, Mark, Other};
    let classes = [
        (0x41, Letter),
        (0x42, Letter),
        (0x43, Other),
        (0x44, Letter),
        // U+0045 is missing: an unassigned code point ends the range.
        (0x46, Letter),
        (0x47, Mark),
        (0xE0100, Control),
        (0xE0101, Control),
    ];
    let table = render(classes);
    let entries = "pub(crate) static JDK_WORD_CLASSES: &[(u32, u32, u8)] = &[\n    \
                   (0x0041, 0x0042, 1),    // Letter\n    \
                   (0x0044, 0x0044, 1),    // Letter\n    \
                   (0x0046, 0x0046, 1),    // Letter\n    \
                   (0x0047, 0x0047, 3),    // Mark\n    \
                   (0xE0100, 0xE0101, 12), // Control\n\
                   ];\n";
    assert!(table.ends_with(entries), "{table}");
    assert!(table.starts_with("//! Generated by analysis/unicode/tests/jdk_sigma.rs from "));
}

/// Lowercases `text` and reads the sigma decisions as the probe writes them.
fn decisions(text: &str) -> String {
    let mut lower = String::new();
    java::to_lowercase(text, &mut lower);
    assert_eq!(java::lowercase_cow(text).as_ref(), lower, "lowercase_cow disagrees with to_lowercase on {text:?}");
    text.chars()
        .zip(lower.chars())
        .filter(|(c, _)| *c == '\u{3A3}')
        .map(|(_, l)| if l == '\u{3C2}' { 'F' } else { 'S' })
        .collect()
}

/// Draws `count` strings of the `alphabet` kind of `FinalSigma.java` from
/// `seed` and lists the ones the model lowercases differently.
fn compare_random(seed: u64, count: usize, alphabet: &str) -> Vec<String> {
    let dump = jdk("FinalSigma", &[&seed.to_string(), &count.to_string(), alphabet]);
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
    let setting = |name: &str, default: u64| -> u64 {
        std::env::var(name)
            .ok()
            .map(|s| s.parse().unwrap_or_else(|_| panic!("{name} is not a number")))
            .unwrap_or(default)
    };
    let seed = setting("TURBO_TEST_JDK_SEED", 1);
    let count = setting("TURBO_TEST_JDK_COUNT", 200_000) as usize;
    let mut mismatches = compare_random(seed, count, "bmp");
    mismatches.extend(compare_random(seed + 1, count, "all"));
    mismatches.extend(compare_random(seed + 2, count, "mixed"));
    let shown: Vec<&String> = mismatches.iter().take(40).collect();
    assert!(
        mismatches.is_empty(),
        "{} of {} strings lowercase differently (first {}):\n{}",
        mismatches.len(),
        3 * count,
        shown.len(),
        shown.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n")
    );
}
