//! Checks the crate against vectors written by a real JVM
//! (`java/Vectors.java`; the README says how to regenerate them). Every
//! value is compared bit for bit, and the first mismatch fails the test
//! with the input, the expected and the actual value.
//!
//! The one exception is NaN: Java does not specify which NaN bit pattern
//! an operation returns, so where the JVM gave a NaN the test asks for a
//! NaN of any pattern.

use std::path::PathBuf;

use turbo_jvm::chars::{self, Utf16Units};
use turbo_jvm::random::JavaRandom;
use turbo_jvm::strict_math;
use turbo_jvm::text;

/// The records of a vector file: its lines after the header, split on tabs.
fn records(name: &str) -> Vec<(usize, Vec<String>)> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/vectors").join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().enumerate();
    let (_, header) = lines.next().unwrap_or_else(|| panic!("{name} is empty"));
    assert!(header.starts_with("# generator=Vectors.java"), "{name}: unexpected header {header:?}");
    let records: Vec<_> =
        lines.map(|(n, line)| (n + 1, line.split('\t').map(str::to_owned).collect::<Vec<_>>())).collect();
    assert!(!records.is_empty(), "{name} has no records");
    records
}

fn bits(field: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(field, 16).unwrap_or_else(|e| panic!("{field:?}: {e}")))
}

fn same(expected: f64, actual: f64) -> bool {
    if expected.is_nan() { actual.is_nan() } else { expected.to_bits() == actual.to_bits() }
}

fn check_unary(name: &str, f: fn(f64) -> f64) {
    let records = records(name);
    for (line, fields) in &records {
        let (x, expected) = (bits(&fields[0]), bits(&fields[1]));
        let actual = f(x);
        assert!(
            same(expected, actual),
            "{name} line {line}: input {:016x} ({x:e}): expected {:016x} ({expected:e}), got {:016x} ({actual:e})",
            x.to_bits(),
            expected.to_bits(),
            actual.to_bits(),
        );
    }
    eprintln!("{name}: {} values match", records.len());
}

#[test]
fn strict_math_exp_matches_the_jvm() {
    check_unary("strict_math_exp.tsv", strict_math::exp);
}

#[test]
fn strict_math_log_matches_the_jvm() {
    check_unary("strict_math_log.tsv", strict_math::log);
}

/// Formats a double as the vector file does, for comparison as text.
fn hex(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

/// One value from `method` of `r`, formatted as the vector file writes it.
fn draw(r: &mut JavaRandom, method: &str, argument: &str, index: usize) -> String {
    match method {
        "nextInt" => r.next_int().to_string(),
        "nextIntBounded" => r.next_int_bounded(argument.parse().unwrap()).to_string(),
        "nextLong" => r.next_long().to_string(),
        "nextBoolean" => u8::from(r.next_boolean()).to_string(),
        "nextFloat" => hex(r.next_float() as f64),
        "nextDouble" => hex(r.next_double()),
        "nextGaussian" => hex(r.next_gaussian()),
        // The cycle Vectors.java uses for "mixed".
        "mixed" => match index % 4 {
            0 | 2 => hex(r.next_gaussian()),
            1 => r.next_int_bounded(1000).to_string(),
            _ => r.next_long().to_string(),
        },
        "shuffle" => {
            let mut items: Vec<u32> = (0..argument.parse().unwrap()).collect();
            r.shuffle(&mut items);
            items.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        }
        _ => panic!("unknown method {method:?}"),
    }
}

#[test]
fn java_random_matches_the_jvm() {
    let records = records("random.tsv");
    let mut r = JavaRandom::new(0);
    for (line, fields) in &records {
        let [seed, method, argument, index, expected] = fields.as_slice() else {
            panic!("random.tsv line {line}: {fields:?}");
        };
        let seed: i64 = seed.parse().unwrap();
        let index: usize = index.parse().unwrap();
        // Each run of a method starts from a fresh generator, as on the JVM.
        if index == 0 {
            r = JavaRandom::new(seed);
        }
        let actual = draw(&mut r, method, argument, index);
        assert!(
            &actual == expected,
            "random.tsv line {line}: new Random({seed}), {method}({argument}) value {index}: expected {expected}, got {actual}{}",
            float_detail(method, index, expected, &actual),
        );
    }
    eprintln!("random.tsv: {} records match", records.len());
}

/// For a mismatch in a floating point value, both values in decimal too.
fn float_detail(method: &str, index: usize, expected: &str, actual: &str) -> String {
    let float =
        matches!(method, "nextFloat" | "nextDouble" | "nextGaussian") || (method == "mixed" && index.is_multiple_of(2));
    if float { format!(" ({:e} vs {:e})", bits(expected), bits(actual)) } else { String::new() }
}

/// Decodes the vector file's escapes (a backslash, `u` and four hex digits
/// per UTF-16 unit) into a string.
fn unescape(field: &str) -> String {
    let mut units = Vec::new();
    let mut rest = field;
    while let Some(c) = rest.chars().next() {
        if let Some(hex) = rest.strip_prefix("\\u") {
            units.push(u16::from_str_radix(&hex[..4], 16).unwrap());
            rest = &hex[4..];
        } else {
            units.push(c as u16);
            rest = &rest[1..];
        }
    }
    String::from_utf16(&units).unwrap_or_else(|e| panic!("{field:?}: {e}"))
}

#[test]
fn compare_matches_string_compare_to() {
    let records = records("compare.tsv");
    let mut outside_bmp_disagreements = 0;
    for (line, fields) in &records {
        let (a, b) = (unescape(&fields[0]), unescape(&fields[1]));
        let expected: i32 = fields[2].parse().unwrap();
        let actual = chars::compare_to(&a, &b);
        assert_eq!(actual, expected, "compare.tsv line {line}: {a:?} vs {b:?}");
        assert_eq!(chars::compare(&a, &b), expected.cmp(&0), "compare.tsv line {line}: {a:?} vs {b:?}");
        assert_eq!(chars::utf16_len(&a), a.encode_utf16().count());
        assert_eq!(Utf16Units::new(&a).count(), chars::utf16_len(&a));
        if a.cmp(&b) != expected.cmp(&0) {
            outside_bmp_disagreements += 1;
        }
    }
    // The vectors include pairs where Rust's own order differs from
    // Java's, so the test shows the difference matters.
    assert!(outside_bmp_disagreements > 0);
    eprintln!(
        "compare.tsv: {} pairs match, {outside_bmp_disagreements} ordered differently by str::cmp",
        records.len()
    );
}

#[test]
fn whitespace_matches_character_is_whitespace_for_every_code_point() {
    let java: Vec<u32> =
        records("whitespace.tsv").iter().map(|(_, f)| u32::from_str_radix(&f[0], 16).unwrap()).collect();
    for c in (0..=0x10FFFF).filter_map(char::from_u32) {
        let expected = java.binary_search(&(c as u32)).is_ok();
        assert_eq!(chars::is_java_whitespace(c), expected, "U+{:04X}", c as u32);
    }
}

/// Whether Rust's shortest form of `x` has a single significant digit.
fn one_digit(x: f64) -> bool {
    format!("{x:e}").trim_start_matches('-').split('e').next().unwrap().len() == 1
}

#[test]
fn text_conversions_match_the_jvm() {
    let records = records("text.tsv");
    let mut one_digit_replaced = 0;
    let mut non_ascii_digits = 0;
    for (line, fields) in &records {
        let [method, input, expected] = fields.as_slice() else {
            panic!("text.tsv line {line}: {fields:?}");
        };
        let at = format!("text.tsv line {line}: {method}({input:?})");
        match method.as_str() {
            "Double.toString" => {
                let x = bits(input);
                let actual = text::double_to_string(x);
                assert_eq!(&actual, &unescape(expected), "{at}");
                if one_digit(x) && actual.trim_start_matches('-').bytes().filter(u8::is_ascii_digit).count() > 2 {
                    one_digit_replaced += 1;
                }
            }
            "Float.toString" => {
                let x = bits(input) as f32;
                assert_eq!(text::float_to_string(x), unescape(expected), "{at}");
            }
            "Double.parseDouble" | "Float.parseFloat" => {
                let input = unescape(input);
                let actual = if method == "Double.parseDouble" {
                    text::parse_double(&input).map(hex)
                } else {
                    text::parse_float(&input).map(|v| hex(f64::from(v)))
                };
                match actual {
                    Ok(h) if bits(&h).is_nan() => assert!(bits(expected).is_nan(), "{at}"),
                    Ok(h) => assert_eq!(&h, expected, "{at}"),
                    Err(text::ParseError::NotANumber(_)) => assert_eq!(expected, "NumberFormatException", "{at}"),
                }
            }
            "Integer.parseInt" => {
                let input = unescape(input);
                match text::parse_int(&input) {
                    Ok(v) => assert_eq!(&v.to_string(), expected, "{at}"),
                    Err(_) if !input.is_ascii() => {
                        // Java accepts the decimal digits of other scripts; this port does not.
                        assert_ne!(expected, "NumberFormatException", "{at}");
                        non_ascii_digits += 1;
                    }
                    Err(_) => assert_eq!(expected, "NumberFormatException", "{at}"),
                }
            }
            "Boolean.parseBoolean" => {
                assert_eq!(text::parse_boolean(&unescape(input)).to_string(), *expected, "{at}");
            }
            _ => panic!("{at}: unknown method"),
        }
    }
    // The vectors reach the rule that replaces a one digit decimal with a
    // closer two digit one (Double.MIN_VALUE prints as 4.9E-324).
    assert!(one_digit_replaced > 0);
    assert_eq!(non_ascii_digits, 1);
    eprintln!("text.tsv: {} records match", records.len());
}
