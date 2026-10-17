use super::*;

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
