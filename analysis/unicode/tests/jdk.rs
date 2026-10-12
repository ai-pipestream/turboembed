//! Compares the Java character class tables with a running JDK.
//!
//! The test compiles tests/jdk/JavaChars.java into a temporary directory,
//! runs it, and checks every code point it prints (the whole BMP and every
//! 257th code point above it) against `turbo_unicode::java::code_point`:
//! general category, the six class tests, the simple lowercase mapping
//! and the script. Code points the JDK reports as unassigned are skipped:
//! the tables follow Unicode 18.0 and JDK 21 Unicode 15.0, so characters
//! added since are known to differ. The list of known differences is for
//! JDK 21. The test runs only when the `TURBO_TEST_JDK` environment
//! variable is set and `java` and `javac` are on the PATH; otherwise it
//! says so and passes.

use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_unicode::GeneralCategory;
use turbo_unicode::java::code_point as java;

/// Runs a command and returns its standard output, or `None` when it
/// cannot be started or fails.
fn run(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    if !output.status.success() {
        panic!("{command:?} failed: {}", String::from_utf8_lossy(&output.stderr));
    }
    Some(String::from_utf8(output.stdout).expect("the JDK prints ASCII"))
}

fn jdk_available() -> bool {
    if std::env::var_os("TURBO_TEST_JDK").is_none() {
        eprintln!("skipped: set TURBO_TEST_JDK to compare with a JDK");
        return false;
    }
    let found = |tool: &str| Command::new(tool).arg("-version").output().is_ok_and(|o| o.status.success());
    if !found("java") || !found("javac") {
        eprintln!("skipped: java or javac is not on the PATH");
        return false;
    }
    true
}

/// Compiles and runs JavaChars.java, returning its output.
fn jdk_dump() -> String {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/jdk/JavaChars.java");
    let classes: PathBuf = std::env::temp_dir().join(format!("turbo-unicode-jdk-{}", std::process::id()));
    std::fs::create_dir_all(&classes).unwrap();
    run(Command::new("javac").arg("-d").arg(&classes).arg(&source)).expect("javac runs");
    let output = run(Command::new("java").arg("-cp").arg(&classes).arg("JavaChars")).expect("java runs");
    let _ = std::fs::remove_dir_all(&classes);
    output
}

/// The answers for one code point, from the JDK or from the tables.
#[derive(Debug, PartialEq)]
struct Answer {
    cp: u32,
    category: GeneralCategory,
    flags: u8,
    lower: u32,
    script: String,
}

fn parse(line: &str) -> Answer {
    let fields: Vec<&str> = line.split(' ').collect();
    let hex = |s: &str| u32::from_str_radix(s, 16).unwrap();
    Answer {
        cp: hex(fields[0]),
        category: java::from_java_type(fields[1].parse().unwrap()).unwrap(),
        flags: u8::from_str_radix(fields[2], 16).unwrap(),
        lower: hex(fields[3]),
        script: fields[4].to_string(),
    }
}

const ALPHABETIC: u8 = 0x02;

/// The same answers from the tables.
fn tables_answer(cp: u32) -> Answer {
    Answer {
        cp,
        category: java::general_category(cp),
        flags: table_flags(cp),
        lower: java::to_lower_case(cp),
        script: java::script(cp).name().to_ascii_uppercase(),
    }
}

/// The flag bits JavaChars.java prints, from the tables.
fn table_flags(cp: u32) -> u8 {
    let bits = [
        java::is_whitespace(cp),
        java::is_alphabetic(cp),
        java::is_upper_case(cp),
        java::is_letter(cp),
        java::is_digit(cp),
        java::is_space_char(cp),
    ];
    bits.iter().enumerate().map(|(i, &bit)| u8::from(bit) << i).sum()
}

/// Code points JDK 21 (Unicode 15.0) assigns but classifies differently from
/// Unicode 18.0, and how: 34 code points, [`KNOWN_DIFFERENCES`]. Any other
/// difference fails the test, and so does a listed code point that does not
/// differ.
///
/// - U+0295 LATIN LETTER PHARYNGEAL VOICED FRICATIVE is `Lo` in 18.0 and `Ll` in
///   15.0. It is a letter either way, and neither upper case nor alphabetic
///   changes.
/// - The 33 combining Latin small letters U+0363..U+036F and U+1DD3..U+1DE6
///   have `Other_Alphabetic`, so `Alphabetic`, in 18.0 and not in 15.0.
///
/// Every other code point the dump lists and both assign has the same
/// category, classes, lowercase mapping and script in both.
fn known_difference(cp: u32, ours: &Answer, jdk: &Answer) -> bool {
    let same_but = |category_differs: bool, flags: u8| {
        (ours.category != jdk.category) == category_differs
            && ours.flags ^ jdk.flags == flags
            && ours.lower == jdk.lower
            && ours.script == jdk.script
    };
    match cp {
        0x0295 => ours.category == GeneralCategory::Lo && jdk.category == GeneralCategory::Ll && same_but(true, 0),
        0x0363..=0x036F | 0x1DD3..=0x1DE6 => same_but(false, ALPHABETIC) && ours.flags & ALPHABETIC != 0,
        _ => false,
    }
}

/// The number of code points [`known_difference`] accepts.
const KNOWN_DIFFERENCES: usize = 1 + (0x036F - 0x0363 + 1) + (0x1DE6 - 0x1DD3 + 1);

#[test]
fn tables_match_the_jdk() {
    if !jdk_available() {
        return;
    }
    let dump = jdk_dump();
    let mut checked = 0;
    let mut unassigned = 0;
    let mut known = 0;
    let mut differences = Vec::new();
    for answer in dump.lines().map(parse) {
        if answer.category == GeneralCategory::Cn {
            unassigned += 1;
            continue;
        }
        checked += 1;
        let cp = answer.cp;
        let ours = tables_answer(cp);
        if ours == answer {
            continue;
        }
        if known_difference(cp, &ours, &answer) {
            known += 1;
        } else {
            differences.push(format!("U+{cp:04X}: tables {ours:?}, JDK {answer:?}"));
        }
    }
    eprintln!(
        "compared {checked} assigned code points, {known} of them known to differ; \
         skipped {unassigned} the JDK reports as unassigned"
    );
    assert!(checked > 60_000, "the dump covers the BMP");
    assert!(differences.is_empty(), "{} code points differ:\n{}", differences.len(), differences.join("\n"));
    assert_eq!(known, KNOWN_DIFFERENCES, "every listed difference occurs");
}
