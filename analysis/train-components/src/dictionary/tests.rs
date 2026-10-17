use super::*;

fn entries(tokens: &[&str]) -> Vec<Vec<String>> {
    tokens.iter().map(|t| vec![t.to_string()]).collect()
}

#[test]
fn put_collapses_equal_entries_by_the_dictionarys_case_rule() {
    let d = AbbreviationDictionary::new(false, entries(&["Dr.", "dr.", "Mr."])).unwrap();
    assert_eq!(d.entries(), &entries(&["Dr.", "Mr."])[..]);
    assert!(d.lookup().contains("DR."));
    let d = AbbreviationDictionary::new(true, entries(&["Dr.", "Dr.", "Mr."])).unwrap();
    assert_eq!(d.len(), 2);
    assert!(d.lookup().contains("Dr.") && !d.lookup().contains("dr."));
    assert!(matches!(
        AbbreviationDictionary::new(true, entries(&["Dr.", "dr."])),
        Err(Error::CaseVariants { first, second }) if first == "Dr." && second == "dr."
    ));
    assert!(matches!(AbbreviationDictionary::new(false, vec![Vec::new()]), Err(Error::EmptyEntry)));
}

#[test]
fn read_rejects_entry_with_no_tokens() {
    let xml = b"<dictionary><entry></entry></dictionary>";
    assert!(matches!(AbbreviationDictionary::read(xml), Err(Error::EmptyEntry)));
    assert!(matches!(write_entries_xml(false, [XmlEntry { tokens: &[], tags: None }]), Err(Error::EmptyEntry)));
}

#[test]
fn a_token_holding_a_tab_is_not_the_two_tokens_around_it() {
    // `StringList.equals` compares token counts first.
    let d = AbbreviationDictionary::new(false, vec![vec!["a\tb".to_string()], vec!["a".to_string(), "b".to_string()]])
        .unwrap();
    assert_eq!(d.len(), 2);
    let d = AbbreviationDictionary::new(true, vec![vec!["A\tb".to_string()], vec!["a".to_string(), "b".to_string()]])
        .unwrap();
    assert_eq!(d.len(), 2);
    let mut scratch = String::new();
    assert!(d.lookup().contains_tokens(&["a", "b"], &mut scratch));
}

#[test]
fn entries_put_directly_are_looked_up_as_given() {
    // `put` does not trim, unlike reading XML, so a padded token is an
    // entry no whitespace-cut token ever matches.
    let d = AbbreviationDictionary::new(false, vec![vec![" Dr. ".to_string()]]).unwrap();
    assert_eq!(d.entries(), &[vec![" Dr. ".to_string()]][..]);
    assert!(d.lookup().contains(" dr. ") && !d.lookup().contains("Dr."));
    // Read from XML, the same token is trimmed on both sides.
    let d = AbbreviationDictionary::read(b"<dictionary><entry><token> Dr. </token></entry></dictionary>").unwrap();
    assert_eq!(d.entries(), &[vec!["Dr.".to_string()]][..]);
    assert!(d.lookup().contains("Dr.") && !d.lookup().contains(" Dr. "));
}

#[test]
fn reading_collapses_case_variants_even_when_case_sensitive() {
    let xml = b"<dictionary case_sensitive=\"true\"><entry><token>Dr.</token></entry>\
        <entry><token>dr.</token></entry><entry><token>et</token><token>al.</token></entry></dictionary>";
    let d = AbbreviationDictionary::read(xml).unwrap();
    assert!(d.is_case_sensitive());
    assert_eq!(d.entries(), &[vec!["Dr.".to_string()], vec!["et".to_string(), "al.".to_string()]][..]);
}

#[test]
fn the_xml_is_indented_as_the_jdk_indents_it() {
    let d = AbbreviationDictionary::new(false, vec![vec!["et".to_string(), "al.".to_string()]]).unwrap();
    let xml = String::from_utf8(d.serialize().unwrap()).unwrap();
    assert_eq!(
        xml,
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"false\">\n    <entry>\n        \
         <token>et</token>\n        <token>al.</token>\n    </entry>\n</dictionary>\n"
    );
    let empty = AbbreviationDictionary::new(true, Vec::new()).unwrap();
    assert_eq!(
        String::from_utf8(empty.serialize().unwrap()).unwrap(),
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"true\"/>\n"
    );
}

#[test]
fn text_is_escaped_as_the_serializer_escapes_it() {
    let mut out = String::new();
    push_text(&mut out, "a&b<c>d\"e'f\r\t\n\u{1F600}").unwrap();
    // The emoji is outside the Basic Multilingual Plane, so the JDK
    // serializer writes it as a decimal character reference (the
    // `train-events` goldens' `special-characters` dictionary shows it).
    assert_eq!(out, "a&amp;b&lt;c&gt;d\"e'f&#13;\t\n&#128512;");
    let d = AbbreviationDictionary::new(false, vec![vec!["\u{1F600}.".to_string()]]).unwrap();
    assert!(d.lookup().contains("\u{1F600}."));
    // U+007F and the C1 controls are references too, as
    // `ToStream.accumDefaultEscape` writes them; U+00A0 is not.
    let mut out = String::new();
    push_text(&mut out, "a\u{7F}b n\u{85}l c\u{9F}d e\u{A0}f").unwrap();
    assert_eq!(out, "a&#127;b n&#133;l c&#159;d e\u{A0}f");
    assert!(matches!(push_text(&mut out, "\u{1}"), Err(Error::InvalidXmlCharacter(1))));
    let mut out = String::new();
    push_attribute(&mut out, "a&b<c>d\"e'f\r\t\n\u{85}").unwrap();
    assert_eq!(out, "a&amp;b&lt;c&gt;d&quot;e'f&#13;&#9;&#10;&#133;");
    assert!(matches!(push_attribute(&mut out, "\u{1}"), Err(Error::InvalidXmlCharacter(1))));
    assert!(matches!(push_text(&mut out, "\u{FFFE}"), Err(Error::InvalidXmlCharacter(0xFFFE))));
    // A dictionary with such a token cannot be made, as its lookup
    // side is read from the XML.
    assert!(matches!(
        AbbreviationDictionary::new(false, vec![vec!["\u{2}".to_string()]]),
        Err(Error::InvalidXmlCharacter(2))
    ));
}

#[test]
fn entries_come_out_in_hash_set_order() {
    // The fixture en-abbrev.bin's dictionary, read in the order its
    // training program put the entries, serializes in the order the
    // fixture holds them (tests/fixtures.rs checks the bytes against
    // the file).
    let d = AbbreviationDictionary::new(
        false,
        [
            vec!["Mr."],
            vec!["Mrs."],
            vec!["Dr."],
            vec!["Prof."],
            vec!["St."],
            vec!["Inc."],
            vec!["U.S."],
            vec!["e.g."],
            vec!["i.e."],
            vec!["etc."],
            vec!["a.m."],
            vec!["p.m."],
            vec!["et", "al."],
        ]
        .into_iter()
        .map(|e| e.into_iter().map(str::to_string).collect()),
    )
    .unwrap();
    let order: Vec<String> = d.java_order().unwrap().iter().map(|e| e.join(" ")).collect();
    assert_eq!(
        order,
        ["Dr.", "Inc.", "U.S.", "a.m.", "Mr.", "Prof.", "St.", "p.m.", "Mrs.", "etc.", "et al.", "e.g.", "i.e."]
    );
}

#[test]
fn parse_one_entry_per_line_matches_jvm_dictionary_builder_tool() {
    let lines = "New York\n  Los   Angeles \n\n   \r\nnew york\rSt.\tLouis\nSan Francisco\n";
    let dict = AbbreviationDictionary::parse_one_entry_per_line(lines).unwrap();
    assert!(!dict.is_case_sensitive());
    assert_eq!(
        dict.entries(),
        &[
            vec!["New".to_string(), "York".to_string()],
            vec!["Los".to_string(), "Angeles".to_string()],
            vec!["St.\tLouis".to_string()],
            vec!["San".to_string(), "Francisco".to_string()],
        ][..]
    );
    let xml = build_dictionary_xml(lines).unwrap();
    let expected = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"false\">\n",
        "    <entry>\n",
        "        <token>Los</token>\n",
        "        <token>Angeles</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>San</token>\n",
        "        <token>Francisco</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>St.\tLouis</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>New</token>\n",
        "        <token>York</token>\n",
        "    </entry>\n",
        "</dictionary>\n",
    );
    assert_eq!(String::from_utf8(xml.clone()).unwrap(), expected);
    // Byte-for-byte round-trip through XML read + serialize.
    let round_trip = AbbreviationDictionary::read(&xml).unwrap().serialize().unwrap();
    assert_eq!(round_trip, xml);
}

#[test]
fn from_census_names_matches_jvm_census_dictionary_creator_tool() {
    let census = concat!(
        "SMITH          1.006  1.006      1\n",
        "JOHNSON        0.810  1.816      2\n",
        "MCDONALD       0.140 32.149    101\n",
        "SMITH          0.500 32.649    102\n",
        "MC             0.010 32.659    103\n",
        "\u{00C9}LAN           0.005 32.664    104\n",
        "\n",
        "WILLIAMS       0.699  2.515      3\n",
    );
    let dict = AbbreviationDictionary::from_census_names(census).unwrap();
    assert!(dict.is_case_sensitive());
    assert_eq!(dict.entries(), &entries(&["Smith", "Johnson", "McDonald", "Mc", "\u{00C9}lan"])[..]);
    let xml = build_census_dictionary_xml(census).unwrap();
    let expected = concat!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?><dictionary case_sensitive=\"true\">\n",
        "    <entry>\n",
        "        <token>McDonald</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>Smith</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>Mc</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>\u{00C9}lan</token>\n",
        "    </entry>\n",
        "    <entry>\n",
        "        <token>Johnson</token>\n",
        "    </entry>\n",
        "</dictionary>\n",
    );
    assert_eq!(String::from_utf8(xml.clone()).unwrap(), expected);
    let round_trip = AbbreviationDictionary::read(&xml).unwrap().serialize().unwrap();
    assert_eq!(round_trip, xml);
}

#[test]
fn from_census_names_refuses_names_that_differ_only_in_case() {
    let census = "MCDONALD 0.140 32.149 101\nMcDONALD 0.140 32.289 102\n";
    assert!(matches!(AbbreviationDictionary::from_census_names(census), Err(Error::CaseVariants { .. })));
}

#[test]
fn from_census_names_reports_leading_space_error() {
    assert!(matches!(AbbreviationDictionary::from_census_names(" SMITH 1.006\n"), Err(Error::Format(_))));
}
