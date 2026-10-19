use super::*;

fn events(input: &str) -> Result<Vec<XmlEvent>, WordNetError> {
    let mut parser = XmlPullParser::new(input, "test.xml");
    let mut out = Vec::new();
    loop {
        let event = parser.next_event()?;
        if let XmlEvent::Eof { .. } = event {
            return Ok(out);
        }
        out.push(event);
    }
}

fn start_names(events: &[XmlEvent]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|e| match e {
            XmlEvent::Start { local_name, .. } => Some(local_name.as_str()),
            _ => None,
        })
        .collect()
}

fn text(input: &str) -> Result<String, WordNetError> {
    unescape_xml(input, "test.xml", 1, false)
}

fn assert_err_contains(result: Result<impl std::fmt::Debug, WordNetError>, fragment: &str) {
    let err = result.expect_err("expected a parse error");
    assert!(err.to_string().contains(fragment), "expected '{fragment}' in '{err}'");
}

#[test]
fn numeric_character_references_decode_decimal_and_hex() {
    assert_eq!("A", text("&#65;").unwrap());
    assert_eq!("A", text("&#x41;").unwrap());
    assert_eq!("\u{e9}", text("&#xe9;").unwrap());
    assert_eq!("\u{1F600}", text("&#x1F600;").unwrap());
    assert_eq!("\u{1F600}", text("&#128512;").unwrap());
    assert_eq!("\t\n\r", text("&#9;&#xA;&#13;").unwrap());
}

#[test]
fn hex_character_references_need_a_lowercase_x() {
    assert_err_contains(text("&#X41;"), "Invalid character reference '&#X41;'");
    assert_eq!("\u{AB}", text("&#xAb;").unwrap(), "hex digits themselves may be either case");
}

#[test]
fn literal_characters_outside_the_xml_char_production_are_rejected() {
    for (doc, line) in [
        ("<r>a\u{0}b</r>", 1),
        ("<r>\n\n\u{1}</r>", 3),
        ("<r a=\"\u{8}\"/>", 1),
        ("<r><!-- \u{1F} --></r>", 1),
        ("<r><![CDATA[\u{B}]]></r>", 1),
        ("<r>\u{FFFE}</r>", 1),
        ("\u{C}<r/>", 1),
    ] {
        assert_err_contains(events(doc), &format!("line {line}: Illegal XML character"));
    }
    assert!(events("\u{feff}<r a=\"\t\">\t\n\u{D7FF}\u{E000}\u{FFFD}\u{10FFFF}</r>").is_ok());
}

#[test]
fn numeric_character_references_reject_signs_and_empty_digits() {
    for reference in ["&#+65;", "&#x+41;", "&#-65;", "&#x-41;", "&#;", "&#x;", "&# 65;", "&#x 41;", "&#65 ;"] {
        assert!(text(reference).is_err(), "{reference} must be rejected");
    }
}

#[test]
fn numeric_character_references_reject_code_points_outside_the_xml_char_production() {
    for reference in ["&#0;", "&#x0;", "&#x1;", "&#8;", "&#xB;", "&#xC;", "&#x1F;", "&#xFFFE;", "&#xFFFF;"] {
        assert_err_contains(text(reference), "not a legal XML character");
    }
    for reference in ["&#xD800;", "&#xDFFF;", "&#x110000;", "&#99999999999;"] {
        assert!(text(reference).is_err(), "{reference} must be rejected");
    }
    for (reference, expected) in [
        ("&#x20;", '\u{20}'),
        ("&#xD7FF;", '\u{D7FF}'),
        ("&#xE000;", '\u{E000}'),
        ("&#xFFFD;", '\u{FFFD}'),
        ("&#x10000;", '\u{10000}'),
        ("&#x10FFFF;", '\u{10FFFF}'),
    ] {
        assert_eq!(expected.to_string(), text(reference).unwrap(), "{reference}");
    }
}

#[test]
fn unknown_and_unterminated_entity_references_are_rejected() {
    assert_err_contains(text("&nbsp;"), "Unknown XML entity reference '&nbsp;'");
    assert_err_contains(text("fish & chips"), "Unescaped '&'");
    assert_err_contains(text("&amp"), "Unescaped '&'");
}

#[test]
fn multibyte_text_survives_next_to_references_and_attribute_normalization() {
    assert_eq!("caf\u{e9}\u{e9}&\u{fc}", text("caf&#xE9;\u{e9}&amp;\u{fc}").unwrap());
    let attr = unescape_xml("\u{65e5}\t\u{672c}\n&#9;\u{8a9e}", "test.xml", 1, true).unwrap();
    assert_eq!("\u{65e5} \u{672c} \t\u{8a9e}", attr);
    let events = events("<r a=\"\u{1F600}&lt;\u{e9}\">\u{4e2d}&gt;\u{6587}</r>").unwrap();
    let XmlEvent::Start { attrs, .. } = &events[0] else { panic!("expected start") };
    assert_eq!("\u{1F600}<\u{e9}", attrs[0].value);
    let XmlEvent::Text(body) = &events[1] else { panic!("expected text") };
    assert_eq!("\u{4e2d}>\u{6587}", body);
}

#[test]
fn internal_entities_are_never_expanded() {
    let mut doctype = String::from("<!DOCTYPE r [\n<!ENTITY lol0 \"lol\">\n");
    for i in 1..10 {
        let prev = format!("&lol{};", i - 1);
        doctype.push_str(&format!("<!ENTITY lol{i} \"{}\">\n", prev.repeat(10)));
    }
    doctype.push_str("]>\n");
    let doc = format!("{doctype}<r>&lol9;</r>");
    assert_err_contains(events(&doc), "Unknown XML entity reference '&lol9;'");
    let attr_doc = format!("{doctype}<r a=\"&lol9;\"/>");
    assert_err_contains(events(&attr_doc), "Unknown XML entity reference '&lol9;'");
}

#[test]
fn doctype_internal_subset_may_hold_comments_and_processing_instructions_with_quotes_and_brackets() {
    for subset in [
        "<!-- don't panic -->",
        "<!-- a ] in a comment -->",
        "<!-- a ]> in a comment -->",
        "<?pi it's \"odd\" ]> ?>",
        "<!ENTITY e \"]>\"> <!-- ' -->",
        "<!ATTLIST r a CDATA '\"]>'>",
        "<!ENTITY \u{e9} \"\u{65e5}\u{1F600}\"> <!-- \u{e9}' --> <?\u{e9} ?>",
    ] {
        let doc = format!("<!DOCTYPE r [ {subset} ]>\n<r a='x'>it's</r>");
        let events = events(&doc).unwrap_or_else(|e| panic!("{subset}: {e}"));
        assert_eq!(vec!["r"], start_names(&events), "{subset}");
        let XmlEvent::Text(body) = &events[1] else { panic!("expected text for {subset}") };
        assert_eq!("it's", body);
    }
}

#[test]
fn doctype_is_only_legal_once_before_the_root_element() {
    assert_err_contains(events("<r><!DOCTYPE r></r>"), "DOCTYPE");
    assert_err_contains(events("<r/><!DOCTYPE r>"), "DOCTYPE");
    assert_err_contains(events("<!DOCTYPE r><!DOCTYPE r><r/>"), "DOCTYPE");
    assert!(events("<?xml version=\"1.0\"?>\n<!-- c -->\n<!DOCTYPE r SYSTEM \"r.dtd\">\n<r/>").is_ok());
}

#[test]
fn markup_declarations_outside_a_doctype_are_rejected() {
    assert_err_contains(events("<!ELEMENT r ANY><r/>"), "<!ELEMENT");
    assert_err_contains(events("<r><!ENTITY e \"x\"></r>"), "<!ENTITY");
    assert_err_contains(events("<r><![IGNORE[ x ]]></r>"), "<![IGNORE");
}

#[test]
fn a_second_root_element_is_rejected() {
    assert_err_contains(events("<r/><s/>"), "second root element <s>");
    assert_err_contains(events("<r></r>\n<r></r>"), "second root element <r>");
    assert!(events("<r/>\n<!-- trailing comment -->\n<?pi trailing?>\n").is_ok());
}

#[test]
fn unclosed_constructs_are_reported_with_their_opening_line() {
    assert_err_contains(events("<r>\n<!-- never closed"), "line 2: Unclosed XML comment");
    assert_err_contains(events("<r>\n\n<?pi never closed"), "line 3: Unclosed XML processing instruction");
    assert_err_contains(events("<r><![CDATA[ never closed"), "Unclosed CDATA section");
    assert_err_contains(events("<!DOCTYPE r [ <!ENTITY e \"x\">"), "Unclosed XML declaration or DOCTYPE");
    assert_err_contains(events("<!DOCTYPE r [ <!-- ]> never closed"), "Unclosed XML comment");
    assert_err_contains(events("<r a=\"open"), "Unclosed attribute value for 'a'");
    assert_err_contains(events("<r a=\"x\""), "Unclosed start tag <r>");
    assert_err_contains(events("<r>\n<s>\n</s>"), "line 3: Unclosed XML element <r>");
    assert_err_contains(events("<r></s>"), "Mismatched closing tag </s>, expected </r>");
    assert_err_contains(events("<r>]]></r>"), "Unexpected ']]>'");
}

#[test]
fn line_numbers_count_newlines_inside_multibyte_text_comments_and_cdata() {
    let doc = "<r>\n\u{e9}\u{1F600}\n<!-- \n\u{4e2d} -->\n<![CDATA[\n]]>\n<bad attr></r>";
    assert_err_contains(events(doc), "line 7: Expected '=' after attribute 'attr'");
}

#[test]
fn namespaces_resolve_through_ancestors_and_reject_unbound_prefixes() {
    let events = events("<r xmlns=\"urn:d\" xmlns:p=\"urn:p\"><p:c p:a=\"1\" b=\"2\"/><d/></r>").unwrap();
    let resolved: Vec<(&str, &str)> = events
        .iter()
        .filter_map(|e| match e {
            XmlEvent::Start { namespace_uri, local_name, .. } => Some((namespace_uri.as_str(), local_name.as_str())),
            _ => None,
        })
        .collect();
    assert_eq!(vec![("urn:d", "r"), ("urn:p", "c"), ("urn:d", "d")], resolved);
    let XmlEvent::Start { attrs, .. } = &events[1] else { panic!("expected start") };
    assert_eq!(("urn:p", "a"), (attrs[0].namespace_uri.as_str(), attrs[0].local_name.as_str()));
    assert_eq!(("", "b"), (attrs[1].namespace_uri.as_str(), attrs[1].local_name.as_str()));

    assert_err_contains(self::events("<q:r/>"), "Unbound XML namespace prefix 'q'");
    assert_err_contains(
        self::events("<r xmlns:a=\"urn:x\" xmlns:b=\"urn:x\" a:k=\"1\" b:k=\"2\"/>"),
        "Duplicate expanded attribute",
    );
}

#[test]
fn comments_reject_double_hyphens_and_trailing_hyphen() {
    assert_err_contains(events("<r><!-- a -- b --></r>"), "Illegal '--' inside XML comment");
    assert_err_contains(events("<r><!-----></r>"), "Illegal '--' inside XML comment");
    assert_err_contains(events("<!DOCTYPE r [ <!-- a -- b --> ]><r/>"), "Illegal '--' inside XML comment");
}

#[test]
fn attributes_require_whitespace_and_reject_illegal_namespace_declarations() {
    assert_err_contains(events("<r a=\"1\"b=\"2\"/>"), "Expected whitespace between attributes");
    assert_err_contains(events("\u{A0}<r/>"), "Unexpected text outside root XML element");
    assert!(events("<r/>\u{a0}").is_err());
    assert!(events("<r>\u{a0}</r>").is_ok());
    assert_err_contains(events("<r xmlns:p=\"\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:a:b=\"urn:x\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:1bad=\"urn:test\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:-bad=\"urn:test\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:p!bad=\"urn:test\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:xmlns=\"urn:x\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns:xml=\"urn:x\"/>"), "Illegal namespace declaration");
    assert_err_contains(events("<r xmlns=\"http://www.w3.org/XML/1998/namespace\"/>"), "Illegal default namespace");
    assert!(events("<r xmlns:\u{e9}_1-a.b=\"urn:test\"><\u{e9}_1-a.b:c/></r>").is_ok());
}
