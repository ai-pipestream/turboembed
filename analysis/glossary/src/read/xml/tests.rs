use super::*;

const NS: &str = "urn:test";

/// Every event as text: `<local known id lang>`, `</>`, the text.
fn events(src: &str) -> Result<Vec<String>, ReadError> {
    let mut x = Xml::new(src, 0, NS);
    let mut out = Vec::new();
    loop {
        match x.next()? {
            Event::Start(s) => out.push(format!("<{} {} {:?} {:?}>", s.local, s.known, s.id, s.lang)),
            Event::End { .. } => out.push("</>".into()),
            Event::Text(t) => out.push(t.into_owned()),
            Event::Eof => return Ok(out),
        }
    }
}

#[test]
fn namespaces_attributes_and_references() {
    let src = "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<!DOCTYPE r [<!ENTITY e \"x&#38;#38;y\"><!ELEMENT r ANY>]>\
               <r xmlns=\"urn:test\" xmlns:o=\"urn:other\"><o:a id=\"1\" xml:lang=\"en\t \"/>\
               <b>a&amp;b&e;<![CDATA[<c>]]>\r\nd</b></r>";
    let got = events(src).unwrap();
    assert_eq!(
        got,
        [
            "<r true None None>",
            "<a false Some(\"1\") Some(\"en  \")>",
            "</>",
            "<b true None None>",
            "a&bx&y",
            "<c>",
            "\nd",
            "</>",
            "</>"
        ]
    );
}

#[test]
fn refuses_what_is_not_well_formed() {
    let bad = [
        "",
        "<a>",
        "<a></b>",
        "<a/><b/>",
        "<a x='1' x='2'/>",
        "<a x='<'/>",
        "<p:a/>",
        "<a>&nope;</a>",
        "<a>&#0;</a>",
        "<a>]]></a>",
        "<a><!-- a -- b --></a>",
        "<?xml version='1.0' encoding='ISO-8859-1'?><a/>",
        "<a/><?xml version='1.0'?>",
        "<!DOCTYPE a [<!ENTITY e '<b/>'>]><a>&e;</a>",
        "<!DOCTYPE a [<!ENTITY e '&e;'>]><a>&e;</a>",
        "<a xmlns:p='u' p:x='1' xmlns:q='u' q:x='2'/>",
        "<a xmlns:xml='urn:x'/>",
        "<a xmlns:p='http://www.w3.org/XML/1998/namespace'/>",
        "<a xmlns='http://www.w3.org/XML/1998/namespace'/>",
        "<a xmlns:xmlns='urn:x'/>",
        "<a xmlns:p='http://www.w3.org/2000/xmlns/'/>",
        "text<a/>",
        "<a/>text",
    ];
    for src in bad {
        assert!(events(src).is_err(), "{src:?}");
    }
    assert!(events("<a xmlns:xml='http://www.w3.org/XML/1998/namespace'/>").is_ok());
    let err = events("<!DOCTYPE a [\n<!ENTITY e SYSTEM 'file:///etc/passwd'>]><a/>").unwrap_err();
    assert_eq!(err, ReadError::ExternalEntity { line: 2, name: "e".into() });
}

#[test]
fn lines_count_on_and_back() {
    let src = "<a>\r\n<b/>\r<c/>\n\n<d/></a>";
    let x = Xml::new(src, 0, NS);
    let want = |at: usize| {
        let b = &src.as_bytes()[..at];
        1 + (0..at).filter(|&i| b[i] == b'\n' || (b[i] == b'\r' && b.get(i + 1) != Some(&b'\n'))).count() as u32
    };
    for at in (0..=src.len()).chain([4, 0, src.len(), 3, 5]) {
        assert_eq!(x.line_at(at), want(at), "{at}");
    }
    let mut x = Xml::new(src, 0, NS);
    let mut lines = Vec::new();
    loop {
        match x.next().unwrap() {
            Event::Start(s) => lines.push((s.local, s.line)),
            Event::Eof => break,
            _ => {}
        }
    }
    assert_eq!(lines, [("a", 1), ("b", 2), ("c", 3), ("d", 5)]);
}

#[test]
fn many_attributes_and_nested_bindings() {
    let mut src = String::from("<a");
    for i in 0..40 {
        src.push_str(&format!(" x{i}='{i}'"));
    }
    assert!(events(&format!("{src}/>")).is_ok());
    let err = events(&format!("{src} x17='again'/>")).unwrap_err();
    assert!(err.to_string().contains("x17 appears twice"), "{err}");
    let mut src = String::from("<a");
    for i in 0..40 {
        src.push_str(&format!(" xmlns:p{i}='u' p{i}:x='{i}'"));
    }
    let err = events(&format!("{src}/>")).unwrap_err();
    assert!(err.to_string().contains("p1:x appears twice"), "{err}");
    // A prefix bound again inside, then out of scope again.
    let src = "<a xmlns:p='urn:test'><p:b xmlns:p='urn:other'><p:c/></p:b><p:d/></a>";
    assert_eq!(
        events(src).unwrap(),
        [
            "<a true None None>",
            "<b false None None>",
            "<c false None None>",
            "</>",
            "</>",
            "<d true None None>",
            "</>",
            "</>"
        ]
    );
    assert!(events("<a><p:b xmlns:p='u'/><p:c/></a>").is_err());
}

#[test]
fn carriage_returns_from_references_stay() {
    let src = "<!DOCTYPE a [<!ENTITY e 'x&#13;y\r\nz'>]><a v='&e;'>&e;&#13;\r\n</a>";
    let mut x = Xml::new(src, 0, NS);
    let Event::Start(start) = x.next().unwrap() else { panic!() };
    assert!(start.id.is_none());
    let mut text = String::new();
    while let Event::Text(t) = x.next().unwrap() {
        text.push_str(&t);
    }
    assert_eq!(text, "x\ry\nz\r\n");
    let mut x = Xml::new("<!DOCTYPE a [<!ENTITY e 'x&#13;&#10;y'>]><a id='&e;&#13;'/>", 0, NS);
    let Event::Start(start) = x.next().unwrap() else { panic!() };
    assert_eq!(start.id.as_deref(), Some("x  y\r"));
}

#[test]
fn expansions_stop_at_a_total_size() {
    let big = "x".repeat(1 << 20);
    let mut src = format!("<!DOCTYPE a [<!ENTITY b '{big}'>]><a>");
    for _ in 0..60 {
        src.push_str("&b;");
    }
    src.push_str("</a>");
    let err = events(&src).unwrap_err();
    assert!(err.to_string().contains("entity expansions add more than"), "{err}");
    src.replace_range(src.len() - 4 - 3 * 20.., "</a>");
    assert!(events(&src).is_ok());
}

#[test]
fn the_billion_laughs_stop() {
    let mut src = String::from("<!DOCTYPE a [<!ENTITY l0 'ha'>");
    for i in 1..10 {
        let prev = format!("&l{};", i - 1);
        src.push_str(&format!("<!ENTITY l{i} '{}'>", prev.repeat(10)));
    }
    src.push_str("]><a>&l9;</a>");
    let err = events(&src).unwrap_err();
    assert!(err.to_string().contains("too many entity expansions"), "{err}");
}
