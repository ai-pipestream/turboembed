//! Edge cases of the golden reader, writer and hashes that
//! `tests/goldens.rs` and the unit tests do not reach: each way a file can
//! be malformed with the line the error names, round trips of every column
//! type and every escape, the writer's refusals, and the SHA functions on
//! streamed input.

use std::path::PathBuf;

use turbo_analysis_layers::Strs;
use turbo_goldens::*;

#[path = "edge_cases/hashes.rs"]
mod hashes;
#[path = "edge_cases/rows.rs"]
mod rows;
#[path = "edge_cases/writer.rs"]
mod writer;

// ---------------------------------------------------------------------
// Helpers

/// Seven header lines (lines 1 to 7), so the first layer block starts on
/// line 8. The input is 100 bytes long.
const HEADER: &str = "golden 1\n\
generator\tg\t1\n\
opennlp\t0123456789abcdef0123456789abcdef01234567\tv\n\
jdk\t21\t21+1\tV\n\
unicode\t15.0\n\
input\ttest/x\t100\t100\tf03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f\n\
annotator\tC\tm\n";

fn file(body: &str) -> String {
    format!("{HEADER}{body}")
}

/// A `opennlp:sentences` block with these rows.
fn sentences(rows: &[&str]) -> String {
    block("opennlp:sentences", "text", &["start:u32", "end:u32"], rows)
}

fn block(id: &str, anchor: &str, columns: &[&str], rows: &[&str]) -> String {
    let mut s = format!("layer\t{id}\t{anchor}\t{}\t{}\n", rows.len(), columns.join("\t"));
    for r in rows {
        s.push_str(r);
        s.push('\n');
    }
    s.push_str(&format!("end\t{id}\n"));
    s
}

fn tokens(rows: &[&str]) -> String {
    block("opennlp:tokens", "text", &["start:u32", "end:u32"], rows)
}

fn lemmas(rows: &[&str]) -> String {
    block("opennlp:lemmas", "aligned:opennlp:tokens", &["lemma:str"], rows)
}

/// A file with one token and one lemma row `lemma` (line 12).
fn lemma_file(lemma: &str) -> String {
    file(&format!("{}{}", tokens(&["0\t5"]), lemmas(&[lemma])))
}

/// A file with one sentence and one confidence row `value` (line 12).
fn conf_file(value: &str) -> String {
    let conf = block("turbo:confidence<opennlp:sentences>", "aligned:opennlp:sentences", &["value:f64"], &[value]);
    file(&format!("{}{}", sentences(&["0\t5"]), conf))
}

/// Parses `text`, which must fail on `line`, and returns the error.
fn refused(text: &str, line: usize) -> GoldenError {
    let e = Golden::parse(text).expect_err("the file should be refused");
    let shown = e.to_string();
    assert!(shown.starts_with(&format!("line {line}: ")), "expected an error on line {line}, got: {shown}");
    e
}

fn says(e: &GoldenError, parts: &[&str]) {
    let shown = e.to_string();
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in the error: {shown}");
    }
}

fn fixture_text() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zurich.golden")).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------
// The helpers build files the reader accepts

#[test]
fn the_helper_files_are_valid() {
    let g = Golden::parse(&file(&sentences(&["0\t5", "6\t100"]))).unwrap();
    assert_eq!(g.layers.len(), 1);
    assert_eq!(g.layers[0].len(), 2);
    let g = Golden::parse(&lemma_file("go")).unwrap();
    assert_eq!(g.layer("opennlp:lemmas").unwrap().strs("lemma").unwrap().get(0), "go");
    let g = Golden::parse(&conf_file("3fe0000000000000 0.5")).unwrap();
    assert_eq!(g.layer("turbo:confidence<opennlp:sentences>").unwrap().f64s("value").unwrap(), [0.5]);
}

// ---------------------------------------------------------------------
// The first line

#[test]
fn the_first_line_must_be_golden_1() {
    let body = file(&sentences(&["0\t5"]));
    let rest = body.strip_prefix("golden 1\n").unwrap();
    // Missing: the generator record is where the version should be.
    let e = refused(rest, 1);
    assert!(matches!(e, GoldenError::Version { .. }), "{e:?}");
    says(&e, &["expected \"golden 1\"", "generator"]);
    for first in [
        "golden 2",
        "golden 0",
        "golden 01",
        "golden +1",
        "golden 1 ",
        " golden 1",
        "Golden 1",
        "golden  1",
        "golden\t1",
        "golden",
    ] {
        let text = format!("{first}\n{rest}");
        let e = refused(&text, 1);
        assert!(matches!(e, GoldenError::Version { .. }), "{first:?}: {e:?}");
    }
}

#[test]
fn a_byte_order_mark_is_refused_on_line_1() {
    let text = format!("\u{feff}{}", file(&sentences(&["0\t5"])));
    let e = Golden::parse_bytes(text.as_bytes()).unwrap_err();
    assert!(matches!(e, GoldenError::Version { line: 1, .. }), "{e:?}");
    // The BOM is shown so a person can see why.
    assert!(e.to_string().contains("\\u{feff}"), "{e}");
}

#[test]
fn an_empty_file_and_a_lone_line_end() {
    let e = refused("", 1);
    assert!(matches!(e, GoldenError::NoFinalLineEnd { line: 1 }), "{e:?}");
    let e = refused("\n", 1);
    assert!(matches!(e, GoldenError::Version { .. }), "{e:?}");
    let e = Golden::parse_bytes(b"").unwrap_err();
    assert!(matches!(e, GoldenError::NoFinalLineEnd { line: 1 }), "{e:?}");
    // The first line alone.
    let e = refused("golden 1\n", 2);
    assert!(matches!(e, GoldenError::Eof { .. }), "{e:?}");
    says(&e, &["the generator record"]);
}

// ---------------------------------------------------------------------
// Header fields

#[test]
fn a_tab_in_a_header_field_changes_the_field_count() {
    let lines: Vec<&str> = HEADER.lines().collect();
    // (line, replacement, fields found, record named in the message)
    let cases = [
        (2, "generator\tg\tx\t1", 4, "generator <name> <version>"),
        (3, "opennlp\t0123456789abcdef0123456789abcdef01234567\tv\tw", 4, "opennlp <commit> <maven version>"),
        (4, "jdk\t21\t21+1\tV\tW", 5, "jdk <java.version>"),
        (5, "unicode\t15\t0", 3, "unicode <version>"),
        (7, "annotator\tC\tm\tn", 4, "annotator <class> <method>"),
    ];
    for (line, with, found, named) in cases {
        let mut l = lines.clone();
        l[line - 1] = with;
        let e = refused(&(l.join("\n") + "\n" + &sentences(&["0\t5"])), line);
        assert!(matches!(&e, GoldenError::Fields { found: f, .. } if *f == found), "line {line}: {e:?}");
        says(&e, &[named]);
    }
}

#[test]
fn a_carriage_return_anywhere_names_its_line() {
    let base = file(&sentences(&["0\t5", "6\t9"]));
    // In a header field, in a layer record, in a row, and in the end line.
    for (needle, line) in
        [("jdk\t21\t21+1\tV", 4), ("layer\topennlp:sentences", 8), ("6\t9", 10), ("end\topennlp:sentences", 11)]
    {
        let text = base.replacen(needle, &format!("{needle}\r"), 1);
        let e = refused(&text, line);
        assert!(matches!(e, GoldenError::CarriageReturn { .. }), "{needle:?}: {e:?}");
    }
    // A CR inside a field, before its tab.
    let text = base.replacen("generator\tg\t1", "generator\tg\r\t1", 1);
    refused(&text, 2);
    // A lone CR at the very end instead of the line end is also a CR.
    let text = base.trim_end_matches('\n').to_string() + "\r";
    refused(&text, 11);
}

#[test]
fn header_numbers_and_hashes_are_strict() {
    let sha = "f03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f";
    let commit = "0123456789abcdef0123456789abcdef01234567";
    let with_line = |line: usize, with: &str| {
        let mut l: Vec<String> = HEADER.lines().map(String::from).collect();
        l[line - 1] = with.to_string();
        l.join("\n") + "\n" + &sentences(&["0\t5"])
    };
    // u32 forms in the generator version and the input lengths.
    for bad in ["+1", " 1", "1 ", "01", "-1", "", "4294967296", "1.0", "0x1", "１"] {
        let e = refused(&with_line(2, &format!("generator\tg\t{bad}")), 2);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
        says(&e, &["generator version"]);
        let e = refused(&with_line(6, &format!("input\ttest/x\t{bad}\t100\t{sha}")), 6);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
        let e = refused(&with_line(6, &format!("input\ttest/x\t100\t{bad}\t{sha}")), 6);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
    }
    // The largest u32 is fine (the file has no layer needing it).
    Golden::parse(&with_line(6, &format!("input\ttest/x\t4294967295\t4294967295\t{sha}"))).unwrap();
    // Commit: 39 and 41 digits, uppercase, a non-hex digit, empty.
    for bad in [&commit[..39], &format!("{commit}0"), &commit.to_uppercase(), &format!("{}g", &commit[..39]), ""] {
        let e = refused(&with_line(3, &format!("opennlp\t{bad}\tv")), 3);
        says(&e, &["40 lowercase hex digits"]);
    }
    // SHA-256: 63 and 65 digits, uppercase.
    for bad in [&sha[..63], &format!("{sha}0"), &sha.to_uppercase()] {
        let e = refused(&with_line(6, &format!("input\ttest/x\t100\t100\t{bad}")), 6);
        says(&e, &["64 lowercase hex digits"]);
    }
    // Input names need both a set and an id.
    for bad in ["x", "test/", "/x", "/"] {
        let e = refused(&with_line(6, &format!("input\t{bad}\t100\t100\t{sha}")), 6);
        says(&e, &["<set>/<id>"]);
    }
    // Unicode versions.
    for bad in ["15", "15.", ".0", "15.0.1", "+15.0", "015.0", "15.00", "a.b", "15,0", "-1.0"] {
        let e = refused(&with_line(5, &format!("unicode\t{bad}")), 5);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
    }
    let g = Golden::parse(&with_line(5, "unicode\t16.0")).unwrap();
    assert_eq!(g.header.unicode, UnicodeVersion { major: 16, minor: 0 });
    // Empty text fields are refused, and a space inside one is not.
    let e = refused(&with_line(4, "jdk\t21\t\tV"), 4);
    says(&e, &["java.vm.version", "a non-empty field"]);
    let g = Golden::parse(&with_line(4, "jdk\t21.0.5\t21.0.5+11-LTS\tEclipse Adoptium")).unwrap();
    assert_eq!(g.header.jdk.vendor, "Eclipse Adoptium");
}

#[test]
fn model_records_are_hashed_and_ordered() {
    let sha512 = "ab".repeat(64);
    let with_models = |models: &str| {
        let lines: Vec<&str> = HEADER.lines().collect();
        let mut text = lines[..6].join("\n") + "\n" + models + lines[6] + "\n";
        text.push_str(&sentences(&["0\t5"]));
        text
    };
    let ok = with_models(&format!("model\ta.bin\t{sha512}\nmodel\tb.bin\t{sha512}\n"));
    let g = Golden::parse(&ok).unwrap();
    assert_eq!(g.header.models.iter().map(|m| m.file.as_str()).collect::<Vec<_>>(), ["a.bin", "b.bin"]);
    assert_eq!(g.header.models[0].sha512, [0xab; 64]);
    // 127 and 129 digits, uppercase.
    for bad in [sha512[..127].to_string(), format!("{sha512}0"), sha512.to_uppercase()] {
        let e = refused(&with_models(&format!("model\ta.bin\t{bad}\n")), 7);
        says(&e, &["128 lowercase hex digits"]);
    }
    // A model record after the annotator record is not a layer record.
    let late = format!("{}model\ta.bin\t{sha512}\n{}", HEADER, sentences(&["0\t5"]));
    let e = refused(&late, 8);
    assert!(matches!(e, GoldenError::Record { .. }), "{e:?}");
    says(&e, &["expected a layer record"]);
    // A config record before the annotator record is not accepted either.
    let early = with_models("config\tk\tv\n");
    let e = refused(&early, 7);
    says(&e, &["expected a model record or the annotator record"]);
}

// ---------------------------------------------------------------------
// Config records

#[test]
fn config_keys_must_be_sorted_and_unique() {
    let with_config = |config: &str| {
        let lines: Vec<&str> = HEADER.lines().collect();
        lines.join("\n") + "\n" + config + &sentences(&["0\t5"])
    };
    let g = Golden::parse(&with_config("config\ta\t1\nconfig\tb\t2\nconfig\tc\t3\n")).unwrap();
    assert_eq!(g.header.config.len(), 3);
    // Out of order: reported on the key that is too small.
    let e = refused(&with_config("config\tb\t1\nconfig\ta\t2\n"), 9);
    assert!(matches!(&e, GoldenError::ConfigOrder { key, previous, .. } if key == "a" && previous == "b"), "{e:?}");
    // Duplicate.
    let e = refused(&with_config("config\ta\t1\nconfig\ta\t2\n"), 9);
    assert!(matches!(e, GoldenError::ConfigOrder { .. }), "{e:?}");
    // Out of order after a good pair: line 10 is the third record.
    let e = refused(&with_config("config\ta\t1\nconfig\tc\t2\nconfig\tb\t3\n"), 10);
    assert!(matches!(e, GoldenError::ConfigOrder { .. }), "{e:?}");
    // A prefix sorts before the longer key; upper case before lower case.
    Golden::parse(&with_config("config\tA\t1\nconfig\ta\t1\nconfig\tab\t1\n")).unwrap();
    refused(&with_config("config\tab\t1\nconfig\ta\t1\n"), 9);
    refused(&with_config("config\ta\t1\nconfig\tA\t1\n"), 9);
    // The order is by the unescaped key: "\\t" is a tab, which sorts before "a".
    Golden::parse(&with_config("config\t\\t\t1\nconfig\ta\t1\n")).unwrap();
    // Wrong field counts.
    let e = refused(&with_config("config\ta\n"), 8);
    assert!(matches!(e, GoldenError::Fields { found: 2, .. }), "{e:?}");
    let e = refused(&with_config("config\ta\t1\t2\n"), 8);
    assert!(matches!(e, GoldenError::Fields { found: 4, .. }), "{e:?}");
    // Bad escapes name the field.
    let e = refused(&with_config("config\ta\\q\t1\n"), 8);
    says(&e, &["config key"]);
    let e = refused(&with_config("config\ta\tx\\q\n"), 8);
    says(&e, &["config value"]);
    let e = refused(&with_config("config\ta\tx\u{1}y\n"), 8);
    says(&e, &["config value", "U+0001 must be escaped"]);
    // An empty value, and an empty key, are accepted and kept.
    let g = Golden::parse(&with_config("config\t\t\nconfig\ta\t\n")).unwrap();
    assert_eq!(g.header.config[0], ConfigEntry { key: String::new(), value: String::new() });
    assert_eq!(g.header.config[1].value, "");
}

// ---------------------------------------------------------------------
// Layer blocks

#[test]
fn a_layer_id_must_be_a_known_shape() {
    for id in [
        "opennlp:sentence",
        "OPENNLP:SENTENCES",
        "opennlp:sentences ",
        "",
        "turbo:confidence<opennlp:pos>",
        "turbo:view2",
    ] {
        let text = file(&format!("layer\t{id}\ttext\t0\tstart:u32\tend:u32\nend\t{id}\n"));
        let e = refused(&text, 8);
        assert!(matches!(e, GoldenError::UnknownLayer { .. }), "{id:?}: {e:?}");
        // The message lists what is known.
        says(&e, &["opennlp:sentences", "opennlp:tokens", "turbo:view", "turbo:confidence<opennlp:tokens>"]);
    }
}

#[test]
fn duplicate_blocks_are_refused_where_the_second_starts() {
    let text = file(&format!("{}{}{}", sentences(&["0\t5"]), tokens(&["0\t5"]), sentences(&["0\t5"])));
    let e = refused(&text, 14);
    assert!(matches!(&e, GoldenError::Duplicate { layer, .. } if layer == "opennlp:sentences"), "{e:?}");
    // Two blocks that differ in everything but the id are still duplicates.
    let text = file(&format!("{}{}", sentences(&["0\t5"]), sentences(&[])));
    refused(&text, 11);
}

#[test]
fn an_aligned_layer_needs_its_anchor_earlier_in_the_file() {
    let pos = |n: usize| {
        let rows: Vec<String> = (0..n).map(|_| "NOUN\t3ff0000000000000 1.0".to_string()).collect();
        let rows: Vec<&str> = rows.iter().map(String::as_str).collect();
        block("opennlp:pos", "aligned:opennlp:tokens", &["tag:str", "prob:f64"], &rows)
    };
    // The anchor is there, so this reads.
    Golden::parse(&file(&format!("{}{}", tokens(&["0\t5", "6\t9"]), pos(2)))).unwrap();
    // The anchor comes later.
    let text = file(&format!("{}{}", pos(2), tokens(&["0\t5", "6\t9"])));
    let e = refused(&text, 8);
    assert!(matches!(&e, GoldenError::AnchorLayer { anchor, .. } if anchor == "opennlp:tokens"), "{e:?}");
    says(&e, &["expected opennlp:tokens earlier in the file"]);
    // The anchor is not in the file at all.
    let e = refused(&file(&pos(2)), 8);
    assert!(matches!(e, GoldenError::AnchorLayer { .. }), "{e:?}");
    // Another layer of the file is not the anchor: the type decides it.
    let other =
        block("opennlp:pos", "aligned:opennlp:sentences", &["tag:str", "prob:f64"], &["NOUN\t3ff0000000000000 1.0"]);
    let e = refused(&file(&format!("{}{}", sentences(&["0\t5"]), other)), 11);
    assert!(matches!(e, GoldenError::Anchor { .. }), "{e:?}");
    says(&e, &["anchored to aligned:opennlp:sentences, expected aligned:opennlp:tokens"]);
    // Row counts must be the anchor's, in both directions, and the message
    // names the layer, the anchor and both counts.
    for n in [0, 1, 3] {
        let e = refused(&file(&format!("{}{}", tokens(&["0\t5", "6\t9"]), pos(n))), 12);
        assert!(matches!(&e, GoldenError::AlignedRows { expected: 2, found, .. } if *found == n), "{n}: {e:?}");
        says(&e, &["opennlp:pos", "opennlp:tokens", &format!("declares {n} rows, expected 2")]);
    }
    // Both empty is consistent.
    Golden::parse(&file(&format!("{}{}", tokens(&[]), pos(0)))).unwrap();
    // A layer anchored to the text or document cannot name an anchor layer, and an
    // aligned layer must name one.
    let e = refused(&file(&block("opennlp:tokens", "aligned:opennlp:sentences", &["start:u32", "end:u32"], &[])), 8);
    assert!(matches!(e, GoldenError::Anchor { .. }), "{e:?}");
    let e = refused(&file(&block("opennlp:tokens", "aligned", &["start:u32", "end:u32"], &[])), 8);
    says(&e, &["layer anchor", "text, document or aligned:<layer id>"]);
    let e = refused(&file(&block("opennlp:tokens", "TEXT", &["start:u32", "end:u32"], &[])), 8);
    assert!(matches!(e, GoldenError::Field { .. }), "{e:?}");
}

#[test]
fn layer_records_with_the_wrong_shape_of_fields() {
    // No column list at all.
    let e = refused(&file("layer\topennlp:tokens\ttext\t0\nend\topennlp:tokens\n"), 8);
    assert!(matches!(e, GoldenError::Fields { found: 4, .. }), "{e:?}");
    // A column without a type, with an empty name, with a type in upper case.
    for col in ["start", ":u32", "start:U32", "start:u64", "start:u32:x", "start::u32"] {
        let text = file(&format!("layer\topennlp:tokens\ttext\t0\t{col}\tend:u32\nend\topennlp:tokens\n"));
        let e = refused(&text, 8);
        assert!(matches!(e, GoldenError::Field { .. }), "{col:?}: {e:?}");
    }
    // Too many columns, too few, the wrong type.
    let e = refused(&file("layer\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\tx:u32\nend\topennlp:tokens\n"), 8);
    assert!(matches!(e, GoldenError::Columns { .. }), "{e:?}");
    let e = refused(&file("layer\topennlp:tokens\ttext\t0\tstart:u32\nend\topennlp:tokens\n"), 8);
    assert!(matches!(e, GoldenError::Columns { .. }), "{e:?}");
    let e = refused(&file("layer\topennlp:tokens\ttext\t0\tstart:u32\tend:f64\nend\topennlp:tokens\n"), 8);
    says(&e, &["[start:u32, end:f64]", "expected [start:u32, end:u32]"]);
    // Row counts that are not u32s.
    for n in ["-1", "+1", "01", "1.0", "", "4294967296", "x"] {
        let text = file(&format!("layer\topennlp:tokens\ttext\t{n}\tstart:u32\tend:u32\nend\topennlp:tokens\n"));
        let e = refused(&text, 8);
        assert!(matches!(e, GoldenError::Field { .. }), "{n:?}: {e:?}");
        says(&e, &["layer row count"]);
    }
    // A record that is not a layer, where a layer should start.
    let e = refused(&file("lay\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\n"), 8);
    assert!(matches!(e, GoldenError::Record { .. }), "{e:?}");
}
