//! Edge cases of the golden reader, writer and hashes that
//! `tests/goldens.rs` and the unit tests do not reach: each way a file can
//! be malformed with the line the error names, round trips of every column
//! type and every escape, the writer's refusals, and the SHA functions on
//! streamed input.

use std::path::PathBuf;

use turbo_analysis_layers::Strs;
use turbo_goldens::*;

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

#[test]
fn blocks_end_with_their_own_end_line() {
    // The wrong id.
    let text = file("layer\topennlp:tokens\ttext\t1\tstart:u32\tend:u32\n0\t5\nend\topennlp:sentences\n");
    let e = refused(&text, 10);
    assert!(matches!(e, GoldenError::Record { .. }), "{e:?}");
    says(&e, &["end\\topennlp:tokens", "after 1 rows", "end\\topennlp:sentences"]);
    // No id, an id with a trailing space, a prefix of the id, an extra field.
    for end in [
        "end",
        "end\t",
        "end\topennlp:tokens ",
        "end\topennlp:token",
        "end\topennlp:tokens\tx",
        "END\topennlp:tokens",
        " end\topennlp:tokens",
    ] {
        let text = file(&format!("layer\topennlp:tokens\ttext\t1\tstart:u32\tend:u32\n0\t5\n{end}\n"));
        let e = refused(&text, 10);
        assert!(matches!(e, GoldenError::Record { .. }), "{end:?}: {e:?}");
    }
    // More rows than declared: the extra row is where `end` should be.
    let text = file("layer\topennlp:tokens\ttext\t1\tstart:u32\tend:u32\n0\t5\n6\t9\nend\topennlp:tokens\n");
    let e = refused(&text, 10);
    assert!(matches!(e, GoldenError::Record { .. }), "{e:?}");
    says(&e, &["after 1 rows"]);
    // Fewer rows than declared: the end line is where row 1 should be.
    let text = file("layer\topennlp:tokens\ttext\t2\tstart:u32\tend:u32\n0\t5\nend\topennlp:tokens\n");
    let e = refused(&text, 10);
    assert!(matches!(e, GoldenError::Rows { declared: 2, found: 1, .. }), "{e:?}");
    // No rows declared but one present.
    let text = file("layer\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\n0\t5\nend\topennlp:tokens\n");
    refused(&text, 9);
    // No rows declared, none present: fine.
    let g = Golden::parse(&file(&tokens(&[]))).unwrap();
    assert!(g.layers[0].is_empty());
    assert_eq!(g.layers[0].column_names(), ["start:u32", "end:u32"]);
}

#[test]
fn blank_lines_are_not_ignored() {
    let ok = file(&format!("{}{}", sentences(&["0\t5"]), tokens(&["0\t5"])));
    Golden::parse(&ok).unwrap();
    // Between blocks.
    let text = ok.replacen("end\topennlp:sentences\n", "end\topennlp:sentences\n\n", 1);
    let e = refused(&text, 11);
    says(&e, &["expected a layer record"]);
    // At the end of the file.
    let text = format!("{ok}\n");
    let e = refused(&text, 14);
    assert!(matches!(e, GoldenError::Record { .. }), "{e:?}");
    // Inside the header.
    let text = ok.replacen("unicode\t15.0\n", "unicode\t15.0\n\n", 1);
    refused(&text, 6);
    // Where a row should be: a one column layer's empty row is the empty
    // string, but a two column layer's is a field count error.
    let text = file("layer\topennlp:tokens\ttext\t1\tstart:u32\tend:u32\n\nend\topennlp:tokens\n");
    let e = refused(&text, 9);
    assert!(matches!(e, GoldenError::Fields { found: 1, .. }), "{e:?}");
}

#[test]
fn a_file_that_ends_inside_a_block_names_the_missing_line() {
    let whole = file(&format!("{}{}", sentences(&["0\t5", "6\t9"]), tokens(&["0\t5"])));
    let lines: Vec<&str> = whole.lines().collect();
    // After the layer record (line 8): row 0 is missing.
    let cut = lines[..8].join("\n") + "\n";
    let e = refused(&cut, 9);
    assert!(matches!(e, GoldenError::Eof { .. }), "{e:?}");
    says(&e, &["row 0 of opennlp:sentences"]);
    // After row 0: row 1 is missing.
    let cut = lines[..9].join("\n") + "\n";
    says(&refused(&cut, 10), &["row 1 of opennlp:sentences"]);
    // After the last row: the end line is missing.
    let cut = lines[..10].join("\n") + "\n";
    let e = refused(&cut, 11);
    assert!(matches!(e, GoldenError::Eof { .. }), "{e:?}");
    says(&e, &["end\topennlp:sentences"]);
    // After the first block: a good file.
    let cut = lines[..11].join("\n") + "\n";
    assert_eq!(Golden::parse(&cut).unwrap().layers.len(), 1);
    // Without the line end of the last line.
    let cut = lines[..10].join("\n");
    assert!(matches!(refused(&cut, 10), GoldenError::NoFinalLineEnd { .. }));
}

/// Every prefix of the fixture, cut at any byte, is refused except the
/// ones that end a layer block; none panics. Every file with one line
/// removed, emptied or doubled is also handled without a panic.
#[test]
fn truncations_and_line_mutations_never_panic() {
    let text = fixture_text();
    let bytes = text.as_bytes();
    let mut accepted = 0;
    for n in 0..=bytes.len() {
        let result = Golden::parse_bytes(&bytes[..n]);
        let ends_a_block = n > 0 && bytes[n - 1] == b'\n' && {
            let last = bytes[..n - 1].rsplit(|&b| b == b'\n').next().unwrap();
            last.starts_with(b"end\t")
        };
        match result {
            Ok(g) => {
                assert!(ends_a_block, "a prefix of {n} bytes was accepted but does not end a block");
                assert!(!g.layers.is_empty());
                accepted += 1;
            }
            Err(e) => {
                assert!(!ends_a_block || n < 700, "a prefix of {n} bytes ends a block and was refused: {e}");
                assert!(!e.to_string().is_empty());
            }
        }
    }
    // One per block of the fixture.
    assert_eq!(accepted, 8);

    let lines: Vec<&str> = text.lines().collect();
    for i in 0..lines.len() {
        for variant in 0..4 {
            let mut l: Vec<String> = lines.iter().map(|s| s.to_string()).collect();
            match variant {
                0 => {
                    l.remove(i);
                }
                1 => l[i] = String::new(),
                2 => l.insert(i, lines[i].to_string()),
                _ => l[i] = format!("{}\t{}", lines[i], lines[i]),
            }
            // Either outcome is fine; what matters is that it returns.
            if let Err(e) = Golden::parse(&(l.join("\n") + "\n")) {
                let shown = e.to_string();
                assert!(shown.starts_with("line "), "{shown}");
            }
        }
    }
}

// ---------------------------------------------------------------------
// Rows: u32 fields

#[test]
fn u32_fields_in_rows_are_decimal_with_no_sign_or_padding() {
    for bad in [
        "+1",
        " 1",
        "1 ",
        "01",
        "00",
        "-0",
        "-1",
        "",
        "1_0",
        "0x1",
        "1e3",
        "1.0",
        "１",
        "4294967296",
        "99999999999999999999",
        "٣",
    ] {
        let e = refused(&file(&sentences(&[&format!("{bad}\t50")])), 9);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
        says(&e, &["start", "a u32 in decimal"]);
        let e = refused(&file(&sentences(&[&format!("0\t{bad}")])), 9);
        assert!(matches!(e, GoldenError::Field { .. }), "{bad:?}: {e:?}");
        says(&e, &["end"]);
    }
    // The offending value is shown as found.
    let e = refused(&file(&sentences(&["+1\t5"])), 9);
    says(&e, &["found \"+1\""]);
    // 0 and the largest u32 are fine in a column no span check reads.
    let rows = ["0\t0\t0\t4294967295\tperson\t3ff0000000000000 1.0"];
    let ents = block(
        "opennlp:entities",
        "text",
        &["start:u32", "end:u32", "token_start:u32", "token_end:u32", "kind:str", "prob:f64"],
        &rows,
    );
    let g = Golden::parse(&file(&ents)).unwrap();
    assert_eq!(g.layers[0].u32s("token_end").unwrap(), [u32::MAX]);
}

// ---------------------------------------------------------------------
// Rows: spans

#[test]
fn spans_are_checked_against_the_input_and_each_other() {
    // end < start.
    let e = refused(&file(&sentences(&["0\t5", "10\t9"])), 10);
    assert!(matches!(e, GoldenError::Span { .. }), "{e:?}");
    says(&e, &["opennlp:sentences", "the span 10..9 ends before it starts"]);
    // end past the input (100 bytes): by one.
    let e = refused(&file(&sentences(&["0\t101"])), 9);
    says(&e, &["the span 0..101 ends past the input's 100 bytes"]);
    // A span starting past the input is also past it at its end.
    let e = refused(&file(&sentences(&["101\t101"])), 9);
    says(&e, &["ends past the input's 100 bytes"]);
    // The largest u32 end.
    let e = refused(&file(&sentences(&["0\t4294967295"])), 9);
    says(&e, &["ends past"]);
    // Decreasing starts, with the previous row's end beyond the next start.
    let e = refused(&file(&sentences(&["10\t20", "5\t30"])), 10);
    says(&e, &["the span 5..30 starts before the row before it"]);
    // Starts are compared with the start before, not the end: overlap is fine.
    // Equal starts, empty spans and an end at exactly the input length are fine.
    let g = Golden::parse(&file(&sentences(&["10\t20", "10\t30", "15\t15", "15\t100", "100\t100"]))).unwrap();
    assert_eq!(g.layers[0].len(), 5);
    // Empty at 0 and at the end of the text.
    Golden::parse(&file(&sentences(&["0\t0"]))).unwrap();
    // The first bad row is the one named, not a later one.
    let e = refused(&file(&sentences(&["0\t5", "6\t200", "3\t2"])), 10);
    says(&e, &["6..200"]);
    // The same check holds for the other text anchored layers.
    let ents = |rows: &[&str]| {
        block(
            "opennlp:entities",
            "text",
            &["start:u32", "end:u32", "token_start:u32", "token_end:u32", "kind:str", "prob:f64"],
            rows,
        )
    };
    let row = |s: u32, e: u32| format!("{s}\t{e}\t0\t1\tperson\t3ff0000000000000 1.0");
    refused(&file(&ents(&[&row(9, 5)])), 9);
    refused(&file(&ents(&[&row(0, 101)])), 9);
    refused(&file(&ents(&[&row(10, 12), &row(9, 12)])), 10);
    let view = |rows: &[&str]| block("turbo:view", "text", &["start:u32", "end:u32", "with:str"], rows);
    refused(&file(&view(&["5\t4\tx"])), 9);
    refused(&file(&view(&["0\t101\tx"])), 9);
    refused(&file(&view(&["4\t5\tx", "3\t5\ty"])), 10);
    Golden::parse(&file(&view(&["4\t4\tx", "4\t4\ty", "4\t5\t"]))).unwrap();
}

#[test]
fn spans_of_aligned_layers_are_not_a_thing() {
    // Aligned and document layers have no start and end columns, so there is
    // nothing to check; the language layer's rows may be in any order.
    let lang = block(
        "opennlp:language",
        "document",
        &["code:str", "confidence:f64"],
        &["eng\t3fe0000000000000 0.5", "deu\t3fe0000000000001 0.5", "eng\t3fe0000000000000 0.5"],
    );
    let g = Golden::parse(&file(&lang)).unwrap();
    assert_eq!(g.layers[0].anchor, GoldenAnchor::Document);
    assert_eq!(g.layers[0].len(), 3);
}

#[test]
fn rows_with_the_wrong_number_of_fields() {
    for (row, found) in [("", 1), ("1", 1), ("1\t2\t3", 3), ("1\t2\t", 3), ("\t", 2), ("1\t2\t3\t4", 4)] {
        let e = refused(&file(&sentences(&[row])), 9);
        if found == 2 {
            assert!(matches!(e, GoldenError::Field { .. }), "{row:?}: {e:?}");
        } else {
            assert!(matches!(&e, GoldenError::Fields { found: f, .. } if *f == found), "{row:?}: {e:?}");
            says(&e, &["<start:u32> <end:u32>"]);
        }
    }
}

// ---------------------------------------------------------------------
// Rows: f64 fields

#[test]
fn f64_fields_are_16_lowercase_hex_digits_a_space_and_a_decimal() {
    let bad = [
        // Uppercase hex.
        "3FE0000000000000 0.5",
        "3fE0000000000000 0.5",
        // 15 and 17 digits.
        "3fe000000000000 0.5",
        "03fe0000000000000 0.5",
        // Not hex, a prefix, a sign.
        "3fe000000000000g 0.5",
        "0x3fe000000000000 0.5",
        "+3fe000000000000 0.5",
        // No space: the bits alone, the bits then a tab-less other separator.
        "3fe0000000000000",
        "3fe0000000000000,0.5",
        "3fe0000000000000_0.5",
        // The decimal alone.
        "0.5",
        "",
        " 3fe0000000000000 0.5",
    ];
    for b in bad {
        let e = refused(&conf_file(b), 12);
        assert!(matches!(e, GoldenError::Field { .. }), "{b:?}: {e:?}");
        says(&e, &["value", "16 lowercase hex digits of bits"]);
    }
    // A tab between the bits and the decimal makes a second field.
    let e = refused(&conf_file("3fe0000000000000\t0.5"), 12);
    assert!(matches!(e, GoldenError::Fields { found: 2, .. }), "{e:?}");
}

#[test]
fn only_the_bits_are_read_from_an_f64_field() {
    let bits = |s: &str| {
        let g = Golden::parse(&conf_file(s)).unwrap();
        g.layer("turbo:confidence<opennlp:sentences>").unwrap().f64s("value").unwrap()[0].to_bits()
    };
    assert_eq!(bits("3fe0000000000000 0.5"), 0.5f64.to_bits());
    // What follows the space is for a person, so it can be anything,
    // including nothing, and the bits win when it disagrees.
    assert_eq!(bits("3fe0000000000000 7.25"), 0.5f64.to_bits());
    assert_eq!(bits("3fe0000000000000 not a number"), 0.5f64.to_bits());
    assert_eq!(bits("3fe0000000000000 "), 0.5f64.to_bits());
    // Every kind of value is carried by its bits.
    assert_eq!(bits("8000000000000000 -0.0"), (-0.0f64).to_bits());
    assert_eq!(bits("7ff8000000000001 NaN"), 0x7ff8000000000001);
    assert_eq!(bits("fff8000000000000 NaN"), 0xfff8000000000000);
    assert_eq!(bits("7ff0000000000001 NaN"), 0x7ff0000000000001);
    assert_eq!(bits("0000000000000001 4.9E-324"), 1);
    assert_eq!(bits("ffffffffffffffff NaN"), u64::MAX);
    assert_eq!(bits("0000000000000000 0.0"), 0);
    assert_eq!(bits("7ff0000000000000 Infinity"), f64::INFINITY.to_bits());
    assert_eq!(bits("fff0000000000000 -Infinity"), f64::NEG_INFINITY.to_bits());
}

// ---------------------------------------------------------------------
// Rows: str fields

#[test]
fn bad_escapes_in_a_str_field_are_refused_on_their_line() {
    // (field, text the message must contain)
    let bad: &[(&str, &str)] = &[
        ("a\\xb", "found \\x"),
        ("\\U0041", "found \\U"),
        ("\\/", "found \\/"),
        ("\\'", "expected one of"),
        ("\\0", "found \\0"),
        ("\\b", "found \\b"),
        ("\\f", "found \\f"),
        ("\\ ", "found \\ "),
        ("\\é", "found \\é"),
        ("abc\\", "a backslash ends the field"),
        ("\\u00E9", "four lowercase hex digits"),
        ("\\u00eG", "four lowercase hex digits"),
        ("\\u12", "four lowercase hex digits"),
        ("\\u", "four lowercase hex digits"),
        ("\\u+041", "four lowercase hex digits"),
        ("\\u 041", "four lowercase hex digits"),
        ("\\uD83D\\uDE00", "four lowercase hex digits"),
        ("\\ud83d\\uDE00", "four lowercase hex digits"),
        ("\\ud83d", "surrogate without its other half"),
        ("\\ude00", "surrogate without its other half"),
        ("\\ude00\\ud83d", "surrogate without its other half"),
        ("\\ud83d\\ud83d", "surrogate without its other half"),
        ("\\ud83dx", "surrogate without its other half"),
        ("\\ud83d\\u0041", "surrogate without its other half"),
        ("\\ud83d\\n", "surrogate without its other half"),
        ("x\\udbff", "surrogate without its other half"),
        ("a\u{1}b", "U+0001 must be escaped"),
        ("a\u{1f}b", "U+001F must be escaped"),
        ("\u{0}", "U+0000 must be escaped"),
        ("a\u{8}", "U+0008 must be escaped"),
    ];
    for &(field, part) in bad {
        let e = refused(&lemma_file(field), 12);
        assert!(matches!(e, GoldenError::Escape { .. }), "{field:?}: {e:?}");
        says(&e, &["lemma", part]);
    }
    // The byte offset in the field is shown.
    says(&refused(&lemma_file("ab\\xc"), 12), &["at byte 2 of the field"]);
    // The same escapes are refused in a tag column with the column's name.
    let pos =
        block("opennlp:pos", "aligned:opennlp:tokens", &["tag:str", "prob:f64"], &["N\\qN\t3ff0000000000000 1.0"]);
    let e = refused(&file(&format!("{}{}", tokens(&["0\t5"]), pos)), 12);
    says(&e, &["tag: at byte 1"]);
}

#[test]
fn good_escapes_and_characters_that_are_not_escaped() {
    let read = |field: &str| {
        let g = Golden::parse(&lemma_file(field)).unwrap();
        g.layer("opennlp:lemmas").unwrap().strs("lemma").unwrap().get(0).to_string()
    };
    assert_eq!(read("a\\tb\\nc\\rd\\\\e"), "a\tb\nc\rd\\e");
    assert_eq!(read("\\u0000"), "\0");
    assert_eq!(read("\\u001f"), "\u{1f}");
    // The reader accepts an escape the writer would not write.
    assert_eq!(read("\\u0041"), "A");
    assert_eq!(read("\\u00e9\\u4e2d"), "é中");
    assert_eq!(read("\\u0009"), "\t");
    assert_eq!(read("\\uffff"), "\u{ffff}");
    assert_eq!(read("\\ue000"), "\u{e000}");
    // A backslash followed by an escape letter is two escapes, not one.
    assert_eq!(read("\\\\n"), "\\n");
    assert_eq!(read("\\\\\\n"), "\\\n");
    assert_eq!(read("\\\\u0041"), "\\u0041");
    // Raw DEL, NEL, line and paragraph separators, non-BMP and combining
    // characters, a space, an empty field.
    for s in ["\u{7f}", "\u{85}", "\u{2028}\u{2029}", "😀", "e\u{301}", " ", "  x  ", "\u{feff}", "\u{ffff}", ""] {
        assert_eq!(read(s), s);
    }
    // A surrogate pair written as two escapes is one character.
    assert_eq!(read("\\ud83d\\ude00"), "😀");
    assert_eq!(read("a\\ud83d\\ude00b"), "a😀b");
    assert_eq!(read("\\ud83d\\ude00\\ud83d\\ude01"), "😀😁");
    assert_eq!(read("\\ud800\\udc00"), "\u{10000}");
    assert_eq!(read("\\udbff\\udfff"), "\u{10ffff}");
    assert_eq!(read("\\ud83d\\ude00").chars().count(), 1);
    assert_eq!(read("\\ud83d\\ude00").len(), 4);
    // The next escape after a pair is read on its own.
    assert_eq!(read("\\ud83d\\ude00\\n"), "😀\n");
}

// ---------------------------------------------------------------------
// Row count and memory

/// A file that declares 4294967295 rows and has none must be refused with
/// an error, not by trying to allocate room for them.
#[test]
fn a_huge_declared_row_count_is_refused_without_allocating_for_it() {
    let text = file("layer\topennlp:tokens\ttext\t4294967295\tstart:u32\tend:u32\n");
    let e = Golden::parse(&text).unwrap_err();
    assert!(matches!(e, GoldenError::Eof { .. }), "{e:?}");
}

// ---------------------------------------------------------------------
// Input text

#[test]
fn the_input_text_must_match_in_encoding_and_utf16_length() {
    let dir = scratch("reader-edge-input");
    std::fs::create_dir_all(dir.join("inputs/test")).unwrap();
    let write_input = |bytes: &[u8]| std::fs::write(dir.join("inputs/test/x.txt"), bytes).unwrap();
    let golden_for = |bytes: &[u8], utf16: u32| {
        let sha = to_hex_lower(&sha256(bytes));
        let text = HEADER.replacen(
            "input\ttest/x\t100\t100\tf03e4a09022fa42c68b4fbab2730c195ceb468707eb0b2598a7b1e91abc4918f",
            &format!("input\ttest/x\t{}\t{utf16}\t{sha}", bytes.len()),
            1,
        ) + &sentences(&["0\t0"]);
        Golden::parse(&text).unwrap()
    };
    // A text with a character outside the BMP: "a😀b" is 6 bytes and 4 UTF-16 units.
    let text = "a😀b".as_bytes();
    write_input(text);
    assert_eq!(golden_for(text, 4).input_text(&dir).unwrap(), "a😀b");
    // The UTF-16 length counted as code points or as bytes is refused.
    for wrong in [3, 5, 6] {
        let e = golden_for(text, wrong).input_text(&dir).unwrap_err();
        assert!(matches!(e, GoldenError::Input { what: "length in UTF-16 units", .. }), "{e:?}");
        says(&e, &["is 4, expected"]);
    }
    // Bytes that are not UTF-8, with a header that matches their length and hash.
    let bad = b"a\xffb";
    write_input(bad);
    let e = golden_for(bad, 3).input_text(&dir).unwrap_err();
    assert!(matches!(e, GoldenError::Input { what: "encoding", .. }), "{e:?}");
    says(&e, &["invalid UTF-8 at byte 1"]);
    // An empty input.
    write_input(b"");
    assert_eq!(golden_for(b"", 0).input_text(&dir).unwrap(), "");
    // A BOM is part of the text: 3 bytes, 1 unit.
    let bom = "\u{feff}x".as_bytes();
    write_input(bom);
    assert_eq!(golden_for(bom, 2).input_text(&dir).unwrap(), "\u{feff}x");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn to_hex_lower(b: &[u8]) -> String {
    hex::to_hex(b)
}

// ---------------------------------------------------------------------
// Round trips

fn f64_specials() -> Vec<f64> {
    vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        0.1,
        0.1 + 0.2,
        1.0 / 3.0,
        2.0 / 3.0,
        f64::NAN,
        -f64::NAN,
        f64::from_bits(0x7ff8000000000001),
        f64::from_bits(0xfff8000000000001),
        f64::from_bits(0x7ff0000000000001),
        f64::from_bits(0xfff0000000000001),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000fffffffffffff),
        f64::MAX,
        f64::MIN,
        f64::EPSILON,
        1e-3,
        9.999999999999999e-4,
        1e7,
        9999999.0,
        123456.789,
        1e23,
        9.223372036854776e18,
        1e-5,
        100.0,
        1.0e300,
        1.0e-300,
    ]
}

fn escape_strings() -> Vec<String> {
    let mut v: Vec<String> = (0u32..0x20).map(|c| format!("<{}>", char::from_u32(c).unwrap())).collect();
    v.extend(
        [
            "\\",
            "\\\\",
            "\\n",
            "\\u0041",
            "\\ud83d\\ude00",
            "tab\t",
            "\tlead",
            "a\tb\tc",
            "line\r\nend",
            "\n",
            "\r",
            "\t",
            "",
            " ",
            "  ",
            "a b",
            "é",
            "日本語",
            "Ａ",
            "😀",
            "😀😁",
            "\u{10000}",
            "\u{10ffff}",
            "\u{ffff}",
            "\u{fffe}",
            "\u{e000}",
            "\u{d7ff}",
            "\u{7f}",
            "\u{80}",
            "\u{85}",
            "\u{a0}",
            "\u{2028}",
            "\u{2029}",
            "\u{feff}",
            "e\u{301}",
            "\"quote\"",
            "'",
            "/",
            "a\\",
            "\\a",
            "\0",
            "\0\0",
            "x\0y",
            "tab\\tnot",
            "<>",
            "&amp;",
            "\u{1}\u{2}\u{3}",
        ]
        .map(String::from),
    );
    v
}

fn strs(items: &[String]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

/// A golden holding every column type in several layers, with every
/// special f64 and every escape.
fn full_golden() -> Golden {
    let escapes = escape_strings();
    let floats = f64_specials();
    let n = escapes.len();
    let at = |i: usize| floats[i % floats.len()];
    let u32s: Vec<u32> = (0..n as u32).collect();
    let ends: Vec<u32> = u32s.iter().map(|&i| i + 3).collect();
    let layers = vec![
        GoldenLayer {
            id: "opennlp:tokens".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s.clone()) },
                Column { name: "end".into(), data: ColumnData::U32(ends.clone()) },
            ],
        },
        GoldenLayer {
            id: "turbo:confidence<opennlp:tokens>".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![Column { name: "value".into(), data: ColumnData::F64((0..n).map(at).collect()) }],
        },
        GoldenLayer {
            id: "opennlp:pos".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![
                Column { name: "tag".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "prob".into(), data: ColumnData::F64((0..n).map(|i| at(i + 7)).collect()) },
            ],
        },
        GoldenLayer {
            id: "opennlp:lemmas".into(),
            anchor: GoldenAnchor::Aligned("opennlp:tokens".into()),
            columns: vec![Column { name: "lemma".into(), data: ColumnData::Str(strs(&escapes)) }],
        },
        GoldenLayer {
            id: "opennlp:entities".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s.clone()) },
                Column { name: "end".into(), data: ColumnData::U32(ends.clone()) },
                Column { name: "token_start".into(), data: ColumnData::U32(vec![u32::MAX; n]) },
                Column {
                    name: "token_end".into(),
                    data: ColumnData::U32((0..n as u32).map(|i| u32::MAX - i).collect()),
                },
                Column { name: "kind".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "prob".into(), data: ColumnData::F64((0..n).map(|i| at(i + 3)).collect()) },
            ],
        },
        GoldenLayer {
            id: "opennlp:language".into(),
            anchor: GoldenAnchor::Document,
            columns: vec![
                Column { name: "code".into(), data: ColumnData::Str(strs(&escapes)) },
                Column { name: "confidence".into(), data: ColumnData::F64((0..n).map(|i| at(i + 11)).collect()) },
            ],
        },
        GoldenLayer {
            id: "turbo:view".into(),
            anchor: GoldenAnchor::Text,
            columns: vec![
                Column { name: "start".into(), data: ColumnData::U32(u32s) },
                Column { name: "end".into(), data: ColumnData::U32(ends) },
                Column { name: "with".into(), data: ColumnData::Str(strs(&escapes)) },
            ],
        },
    ];
    Golden {
        header: Header {
            generator: Generator { name: "turbo goldens".into(), version: u32::MAX },
            opennlp: OpenNlp { commit: [0xab; 20], version: "3.0.0-PRE-SNAPSHOT".into() },
            jdk: Jdk {
                version: "21.0.5".into(),
                vm_version: "21.0.5+11-LTS".into(),
                vendor: "Eclipse Adoptium".into(),
            },
            unicode: UnicodeVersion { major: 16, minor: 0 },
            input: Input { set: "s".into(), id: "an id".into(), utf8_len: 1000, utf16_len: 900, sha256: [7; 32] },
            models: vec![
                Model { file: "a.bin".into(), sha512: [0xff; 64] },
                Model { file: "b c.bin".into(), sha512: [0x00; 64] },
            ],
            annotator: AnnotatorRef { class: "a.b.C".into(), method: "m".into() },
            config: vec![
                ConfigEntry { key: String::new(), value: String::new() },
                ConfigEntry { key: "a".into(), value: "1".into() },
                ConfigEntry { key: "b\t".into(), value: "x\ny\\z\r".into() },
                ConfigEntry { key: "é".into(), value: "\u{1}\u{1f}".into() },
                ConfigEntry { key: "😀".into(), value: "😀\t😀".into() },
            ],
            error: None,
        },
        layers,
    }
}

#[test]
fn every_column_type_special_value_and_escape_round_trips() {
    let g = full_golden();
    let text = g.to_text().unwrap();
    let back = Golden::parse(&text).unwrap();
    assert_eq!(back, g);
    // And again: the second text is the first, byte for byte.
    assert_eq!(back.to_text().unwrap(), text);
    // Bytes, not only text.
    let mut bytes = Vec::new();
    g.write(&mut bytes).unwrap();
    assert_eq!(bytes, text.as_bytes());
    assert_eq!(Golden::parse_bytes(&bytes).unwrap(), g);

    // f64 values come back with the same bits (PartialEq on ColumnData
    // compares bits, and this checks it did).
    let floats = f64_specials();
    let conf = back.layer("turbo:confidence<opennlp:tokens>").unwrap().f64s("value").unwrap();
    for (i, v) in conf.iter().enumerate() {
        assert_eq!(v.to_bits(), floats[i % floats.len()].to_bits(), "row {i}");
    }
    assert!(conf.iter().any(|v| v.is_nan() && v.to_bits() == 0xfff8000000000001));
    assert!(conf.iter().any(|v| v.to_bits() == (-0.0f64).to_bits()));

    // Strings come back as they were.
    let tags = back.layer("opennlp:pos").unwrap().strs("tag").unwrap();
    for (i, s) in escape_strings().iter().enumerate() {
        assert_eq!(tags.get(i), s, "row {i}");
    }
    assert_eq!(back.header.config, g.header.config);
    assert_eq!(back.header.input.id, "an id");
    assert_eq!(back.header.generator.version, u32::MAX);
}

#[test]
fn the_written_form_has_no_stray_control_characters() {
    let text = full_golden().to_text().unwrap();
    assert!(text.ends_with('\n'));
    assert!(!text.contains('\r'));
    // Every line has only tabs as control characters, and each row has the
    // fields its layer declares.
    let mut columns = 0;
    for (i, line) in text.lines().enumerate() {
        assert!(!line.chars().any(|c| c < ' ' && c != '\t'), "line {}: {line:?}", i + 1);
        let fields: Vec<&str> = line.split('\t').collect();
        match fields[0] {
            "layer" => columns = fields.len() - 4,
            "end" | "golden 1" | "generator" | "opennlp" | "jdk" | "unicode" | "input" | "model" | "annotator"
            | "config" => {}
            _ => assert_eq!(fields.len(), columns, "line {}: {line:?}", i + 1),
        }
    }
    // Checks on a few of the escapes in the text.
    assert!(text.contains("\n<\\u0000>\t"));
    assert!(text.contains("\n<\\t>\t"));
    assert!(text.contains("\n<\\u001f>\t"));
    assert!(text.contains("\nline\\r\\nend\t"));
    assert!(text.contains("\n\\\\u0041\t"));
    assert!(text.contains("\n\\\\ud83d\\\\ude00\t"));
    assert!(text.contains("\n😀😁\t"));
    assert!(text.contains("\n\u{7f}\t"));
    assert!(text.contains("\nconfig\t\t\n"));
    assert!(text.contains("\nconfig\tb\\t\tx\\ny\\\\z\\r\n"));
}

#[test]
fn java_decimals_for_values_in_every_format() {
    let g = full_golden();
    let text = g.to_text().unwrap();
    // Java's Double.toString on JDK 19 and later, for the values above.
    for (bits, shown) in [
        (0.0f64.to_bits(), "0.0"),
        ((-0.0f64).to_bits(), "-0.0"),
        (1.0f64.to_bits(), "1.0"),
        (0.1f64.to_bits(), "0.1"),
        ((0.1f64 + 0.2).to_bits(), "0.30000000000000004"),
        (f64::NAN.to_bits(), "NaN"),
        (0x7ff8000000000001, "NaN"),
        (0xfff0000000000001, "NaN"),
        (f64::INFINITY.to_bits(), "Infinity"),
        (f64::NEG_INFINITY.to_bits(), "-Infinity"),
        (f64::MIN_POSITIVE.to_bits(), "2.2250738585072014E-308"),
        (1u64, "4.9E-324"),
        (f64::MAX.to_bits(), "1.7976931348623157E308"),
        (f64::MIN.to_bits(), "-1.7976931348623157E308"),
        (f64::EPSILON.to_bits(), "2.220446049250313E-16"),
        (1e-3f64.to_bits(), "0.001"),
        (1e7f64.to_bits(), "1.0E7"),
        (9999999.0f64.to_bits(), "9999999.0"),
        (123456.789f64.to_bits(), "123456.789"),
        (1e23f64.to_bits(), "1.0E23"),
        (9.223372036854776e18f64.to_bits(), "9.223372036854776E18"),
        (1e-5f64.to_bits(), "1.0E-5"),
        (100.0f64.to_bits(), "100.0"),
    ] {
        let line = format!("{bits:016x} {shown}\n");
        assert!(text.contains(&line), "{line:?} is not in the written golden");
    }
}

#[test]
fn a_surrogate_pair_in_a_file_reads_as_one_character_and_writes_as_utf8() {
    let g = Golden::parse(&lemma_file("\\ud83d\\ude00")).unwrap();
    let lemma = g.layer("opennlp:lemmas").unwrap().strs("lemma").unwrap();
    assert_eq!(lemma.get(0), "😀");
    let again = g.to_text().unwrap();
    assert!(again.contains("\n😀\n"), "{again}");
    assert!(!again.contains("\\ud83d"));
    assert_eq!(Golden::parse(&again).unwrap(), g);
    // In a config value, too.
    let lines: Vec<&str> = HEADER.lines().collect();
    let text = lines.join("\n") + "\nconfig\tk\t\\ud83d\\ude00\\ud83d\\ude01\n" + &sentences(&["0\t1"]);
    let g = Golden::parse(&text).unwrap();
    assert_eq!(g.header.config[0].value, "😀😁");
}

#[test]
fn a_cloned_golden_writes_the_same_bytes() {
    // Reading, cloning, writing and reading again changes nothing.
    let g = Golden::parse(&fixture_text()).unwrap();
    let c = g.clone();
    assert_eq!(c.to_text().unwrap(), fixture_text());
    assert_eq!(Golden::parse(&c.to_text().unwrap()).unwrap(), g);
}

// ---------------------------------------------------------------------
// Writer refusals

type Change = Box<dyn Fn(&mut Golden, &str)>;
type Edit = Box<dyn Fn(&mut Golden)>;

fn refuses_write(g: &Golden, parts: &[&str]) {
    let e = g.to_text().expect_err("the writer should refuse");
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{e}");
    let shown = e.to_string();
    for p in parts {
        assert!(shown.contains(p), "{p:?} is not in {shown:?}");
    }
    // `write` refuses too, and writes nothing.
    let mut out = Vec::new();
    assert!(g.write(&mut out).is_err());
    assert!(out.is_empty());
}

#[test]
fn the_writer_refuses_a_tab_or_line_end_in_any_header_field() {
    for bad in ["a\tb", "a\nb", "a\rb", "\t", "\n"] {
        let fields: Vec<(&str, Change)> = vec![
            ("generator", Box::new(|g, s| g.header.generator.name = s.into())),
            ("opennlp", Box::new(|g, s| g.header.opennlp.version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.vm_version = s.into())),
            ("jdk", Box::new(|g, s| g.header.jdk.vendor = s.into())),
            ("input", Box::new(|g, s| g.header.input.set = s.into())),
            ("input", Box::new(|g, s| g.header.input.id = s.into())),
            ("model", Box::new(|g, s| g.header.models[0].file = s.into())),
            ("annotator", Box::new(|g, s| g.header.annotator.class = s.into())),
            ("annotator", Box::new(|g, s| g.header.annotator.method = s.into())),
        ];
        for (record, set) in fields {
            let mut g = full_golden();
            set(&mut g, bad);
            refuses_write(&g, &[&format!("the {record} record"), "a tab or a line end"]);
        }
    }
    // Config keys and values are escaped, so they are fine.
    let g = full_golden();
    assert!(g.header.config.iter().any(|c| c.key.contains('\t')));
    g.to_text().unwrap();
}

#[test]
fn the_writer_refuses_a_slash_in_the_input_set() {
    let mut g = full_golden();
    g.header.input.set = "a/b".into();
    refuses_write(&g, &["the input set", "'/'"]);
    // A slash in the id is fine: the reader splits at the first one.
    let mut g = full_golden();
    g.header.input.id = "a/b".into();
    let back = Golden::parse(&g.to_text().unwrap()).unwrap();
    assert_eq!((back.header.input.set.as_str(), back.header.input.id.as_str()), ("s", "a/b"));
}

#[test]
fn the_writer_refuses_bad_layers() {
    // A tab or a line end in the layer id.
    for bad in ["a\tb", "a\nb", "a\rb"] {
        let mut g = full_golden();
        g.layers[0].id = bad.into();
        refuses_write(&g, &["the layer id", "a tab or a line end"]);
    }
    // No columns.
    let mut g = full_golden();
    g.layers[0].columns.clear();
    refuses_write(&g, &["opennlp:tokens has no columns"]);
    // Columns of different lengths, in any of the columns after the first,
    // naming the column.
    for (layer, col) in [(0, 1), (2, 1), (4, 5), (6, 2)] {
        let mut g = full_golden();
        let l = &mut g.layers[layer];
        let n = l.len();
        l.columns[col].data = match l.columns[col].data.ty() {
            ColumnType::U32 => ColumnData::U32(vec![0; n - 1]),
            ColumnType::F64 => ColumnData::F64(vec![0.0; n + 1]),
            ColumnType::Str => ColumnData::Str(Strs::default()),
        };
        let name = g.layers[layer].columns[col].name.clone();
        let id = g.layers[layer].id.clone();
        refuses_write(&g, &[&format!("the layer {id} has {n} rows in its first column"), &format!("in {name}")]);
    }
    // A column name with a tab, a line end or a colon.
    for bad in ["a\tb", "a\nb", "a\rb", "a:b"] {
        let mut g = full_golden();
        g.layers[0].columns[0].name = bad.into();
        refuses_write(&g, &["the column name"]);
    }
    // A shorter first column than the others is caught the same way.
    let mut g = full_golden();
    g.layers[0].columns[0].data = ColumnData::U32(vec![]);
    refuses_write(&g, &["has 0 rows in its first column"]);
}

#[test]
fn the_writer_refuses_config_keys_out_of_order_or_repeated() {
    let mut g = full_golden();
    g.header.config.swap(1, 2);
    refuses_write(&g, &["the config key", "keys are sorted, each once"]);
    let mut g = full_golden();
    g.header.config[2].key = "a".into();
    refuses_write(&g, &["the config key \"a\" comes after \"a\""]);
    // An empty list and one entry are fine.
    let mut g = full_golden();
    g.header.config.clear();
    g.to_text().unwrap();
    g.header.config.push(ConfigEntry { key: "k".into(), value: "v".into() });
    g.to_text().unwrap();
}

#[test]
fn the_writer_writes_a_layer_with_no_rows_and_a_file_with_no_layers() {
    let mut g = full_golden();
    for l in &mut g.layers {
        for c in &mut l.columns {
            c.data = ColumnData::empty(c.data.ty());
        }
    }
    let text = g.to_text().unwrap();
    assert!(text.contains("layer\topennlp:tokens\ttext\t0\tstart:u32\tend:u32\nend\topennlp:tokens\n"));
    assert_eq!(Golden::parse(&text).unwrap(), g);
    // A golden with no layers can be written, and the reader refuses it.
    g.layers.clear();
    let text = g.to_text().unwrap();
    assert!(matches!(Golden::parse(&text).unwrap_err(), GoldenError::NoLayers { .. }));
}

#[test]
fn the_writer_refuses_a_tab_or_line_end_in_an_aligned_anchor() {
    for bad in ["a\tb", "a\nb", "a\rb"] {
        let mut g = full_golden();
        g.layers[1].anchor = GoldenAnchor::Aligned(bad.into());
        let e = g.to_text().expect_err("the writer should refuse an anchor that breaks the record");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    }
}

#[test]
fn what_the_writer_writes_the_reader_reads_for_empty_fields() {
    let cases: Vec<Edit> = vec![
        Box::new(|g| g.header.generator.name = String::new()),
        Box::new(|g| g.header.opennlp.version = String::new()),
        Box::new(|g| g.header.jdk.vendor = String::new()),
        Box::new(|g| g.header.input.set = String::new()),
        Box::new(|g| g.header.input.id = String::new()),
        Box::new(|g| g.header.models[0].file = String::new()),
        Box::new(|g| g.header.annotator.class = String::new()),
        Box::new(|g| g.layers[0].columns[0].name = String::new()),
        Box::new(|g| g.layers[1].anchor = GoldenAnchor::Aligned(String::new())),
    ];
    for (i, change) in cases.iter().enumerate() {
        let mut g = full_golden();
        change(&mut g);
        // Either the writer refuses, or the reader reads what was written.
        if let Ok(text) = g.to_text() {
            assert!(Golden::parse(&text).is_ok(), "case {i}: the writer wrote a file the reader refuses");
        }
    }
}

/// The writer accepts a layer id the reader has no shape for; that is the
/// writer being generic, and the reader is where the id is checked.
#[test]
fn the_writer_does_not_check_layer_ids_but_the_reader_does() {
    let mut g = full_golden();
    g.layers[0].id = "my:layer".into();
    let text = g.to_text().unwrap();
    let e = Golden::parse(&text).unwrap_err();
    assert!(matches!(e, GoldenError::UnknownLayer { .. }), "{e:?}");
}

// ---------------------------------------------------------------------
// Hashes

/// 1,000,000 repetitions of "a", fed in pieces of odd sizes that cycle,
/// so block edges fall in every position of a piece.
fn million_a_in_pieces(sizes: &[usize]) -> (Sha256, Sha512) {
    let (mut a, mut b) = (Sha256::new(), Sha512::new());
    let buf = vec![b'a'; *sizes.iter().max().unwrap()];
    let (mut left, mut i) = (1_000_000usize, 0);
    while left > 0 {
        let n = sizes[i % sizes.len()].min(left);
        a.update(&buf[..n]);
        b.update(&buf[..n]);
        left -= n;
        i += 1;
    }
    (a, b)
}

const MILLION_A_256: &str = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
const MILLION_A_512: &str = "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973eb\
                             de0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b";

#[test]
fn one_million_a_streamed_in_odd_pieces() {
    for sizes in [
        &[1usize][..],
        &[7],
        &[61],
        &[63, 65],
        &[127, 129, 1],
        &[3, 1000, 17, 4093],
        &[55, 56, 57, 111, 112, 113],
        &[1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987],
        &[100_000],
        &[999_999, 1],
    ] {
        let (a, b) = million_a_in_pieces(sizes);
        assert_eq!(hex::to_hex(&a.finish()), MILLION_A_256, "sha256 in pieces {sizes:?}");
        assert_eq!(hex::to_hex(&b.finish()), MILLION_A_512, "sha512 in pieces {sizes:?}");
    }
}

#[test]
fn published_vectors_by_streaming_and_in_one_call() {
    let m56 = b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    let m112 = b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    let v256: [(&[u8], &str); 4] = [
        (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
        (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
        (m56, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
        (m112, "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"),
    ];
    for (msg, want) in v256 {
        assert_eq!(hex::to_hex(&sha256(msg)), want);
        let mut h = Sha256::new();
        for b in msg {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(hex::to_hex(&h.finish()), want, "byte by byte, {} bytes", msg.len());
    }
    let v512: [(&[u8], &str); 4] = [
        (
            b"",
            "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
        ),
        (
            b"abc",
            "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
        ),
        (
            m56,
            "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445",
        ),
        (
            m112,
            "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909",
        ),
    ];
    for (msg, want) in v512 {
        assert_eq!(hex::to_hex(&sha512(msg)), want);
        let mut h = Sha512::new();
        for b in msg {
            h.update(std::slice::from_ref(b));
        }
        assert_eq!(hex::to_hex(&h.finish()), want, "byte by byte, {} bytes", msg.len());
    }
}

#[test]
fn empty_updates_and_defaults_change_nothing() {
    let mut a = Sha256::default();
    a.update(b"");
    a.update(b"ab");
    a.update(b"");
    a.update(b"c");
    a.update(b"");
    assert_eq!(a.finish(), sha256(b"abc"));
    let mut b = Sha512::default();
    b.update(b"");
    b.update(b"abc");
    b.update(b"");
    assert_eq!(b.finish(), sha512(b"abc"));
    assert_eq!(Sha256::new().finish(), sha256(b""));
    assert_eq!(Sha512::new().finish(), sha512(b""));
}

#[test]
fn messages_around_the_padding_edges_agree_with_streaming() {
    // Lengths on both sides of every block and padding edge, hashed whole
    // and as two pieces split at every position.
    let data: Vec<u8> = (0..520u32).map(|i| (i * 31 + 7) as u8).collect();
    for len in (0..=140).chain([191, 192, 193, 255, 256, 257, 383, 384, 385, 511, 512, 519]) {
        let msg = &data[..len];
        for split in [0, 1, len / 2, len.saturating_sub(1), len] {
            let (x, y) = msg.split_at(split.min(len));
            let mut a = Sha256::new();
            a.update(x);
            a.update(y);
            assert_eq!(a.finish(), sha256(msg), "sha256 of {len} bytes split at {split}");
            let mut b = Sha512::new();
            b.update(x);
            b.update(y);
            assert_eq!(b.finish(), sha512(msg), "sha512 of {len} bytes split at {split}");
        }
    }
}
