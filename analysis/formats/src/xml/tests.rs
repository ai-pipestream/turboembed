use super::*;

/// Records the events as text: `<name a=v>`, `</name>`, and text.
#[derive(Default)]
struct Log(String);

impl Handler for Log {
    fn start_element(&mut self, name: &str, attributes: &[Attribute<'_>], _: usize) -> Result<()> {
        self.0.push('<');
        self.0.push_str(name);
        for a in attributes {
            self.0.push_str(&format!(" {}={}", a.name, a.value));
        }
        self.0.push('>');
        Ok(())
    }
    fn end_element(&mut self, name: &str, _: usize) -> Result<()> {
        self.0.push_str(&format!("</{name}>"));
        Ok(())
    }
    fn characters(&mut self, text: &str) {
        self.0.push_str(text);
    }
}

fn log(xml: &str) -> String {
    let mut l = Log::default();
    parse(xml, &mut l).unwrap();
    l.0
}

fn error(xml: &str) -> (usize, &'static str) {
    match parse(xml, &mut Log::default()) {
        Err(Error::Xml { offset, message }) => (offset, message),
        other => panic!("expected an XML error, got {other:?}"),
    }
}

#[test]
fn elements_attributes_and_text() {
    assert_eq!(
        log("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<!-- c --><d a='1' b=\"x y\"><e/>t<?pi data?></d>\n"),
        "<d a=1 b=x y><e></e>t</d>"
    );
}

#[test]
fn references_and_cdata() {
    assert_eq!(
        log("<d>&lt;&gt;&amp;&apos;&quot;&#65;&#x42;&#x1F600;<![CDATA[<&>]]></d>"),
        "<d><>&'\"AB\u{1F600}<&></d>"
    );
}

#[test]
fn line_ends_and_attribute_whitespace_are_normalized() {
    assert_eq!(log("<d a=\"1\r\n2\t3&#10;4\">x\r\ny\rz</d>"), "<d a=1 2 3\n4>x\ny\nz</d>");
}

#[test]
fn prefixes_are_dropped_and_xmlns_is_not_reported() {
    assert_eq!(log("<n:d xmlns:n=\"urn:x\" n:a=\"1\"></n:d>"), "<d a=1></d>");
}

#[test]
fn qualified_names_keep_their_prefixes() {
    let mut l = Log::default();
    parse_with("<n:d xmlns:n=\"urn:x\" xml:id=\"1\"><e/></n:d>", Names::Qualified, &mut l).unwrap();
    assert_eq!(l.0, "<n:d xml:id=1><e></e></n:d>");
}

#[test]
fn malformed_documents_name_the_offset() {
    assert_eq!(error("<!DOCTYPE d><d/>"), (0, "document type declarations are not allowed"));
    assert_eq!(error("<d></e>"), (3, "an end tag does not match its start tag"));
    assert_eq!(error("<d/><e/>"), (4, "a second root element"));
    assert_eq!(error("<d/>x"), (4, "text outside the root element"));
    assert_eq!(error("<d>&nbsp;</d>"), (3, "an entity other than the five XML defines"));
    assert_eq!(error("<d>&#0;</d>"), (3, "a character reference to a character XML does not allow"));
    assert_eq!(error("<d>\u{1}</d>"), (3, "a character XML does not allow"));
    assert_eq!(error("<d a='1' a='2'/>"), (9, "an attribute is given twice"));
    // Past the scanned attributes the names are looked up in a set,
    // which the next tag starts again.
    let many: String = (0..40).map(|i| format!(" a{i}='x'")).collect();
    let twice = format!("<d{many} a3='y'/>");
    assert_eq!(error(&twice), (2 + many.len() + 1, "an attribute is given twice"));
    let tags = format!("<d{many}><e{many}/><e a1='1' a2='2'/></d>");
    assert!(parse(&tags, &mut Log::default()).is_ok());
    assert_eq!(error("<d a='<'/>"), (6, "'<' in an attribute value"));
    assert_eq!(error("<d a=1/>"), (5, "an attribute value must be quoted"));
    assert_eq!(error("<d>"), (3, "the document ends inside an element"));
    assert_eq!(error(""), (0, "the document has no root element"));
    assert_eq!(error("<d><!-- a -- b --></d>"), (3, "a comment contains \"--\""));
    assert_eq!(error("<d>]]></d>"), (3, "\"]]>\" in text"));
    assert_eq!(error("<d/><?xml version='1.0'?>"), (4, "an XML declaration that is not at the start of the document"));
}

/// Records the node boundaries a tree builder is given.
#[derive(Default)]
struct Nodes(Vec<String>);

impl Handler for Nodes {
    fn start_element(&mut self, name: &str, _: &[Attribute<'_>], offset: usize) -> Result<()> {
        self.0.push(format!("<{name}@{offset}"));
        Ok(())
    }
    fn end_element(&mut self, name: &str, offset: usize) -> Result<()> {
        self.0.push(format!("/{name}@{offset}"));
        Ok(())
    }
    fn characters(&mut self, text: &str) {
        self.0.push(format!("'{text}'"));
    }
    fn element_closed(&mut self, end: usize) {
        self.0.push(format!("closed@{end}"));
    }
    fn text_run(&mut self, start: usize, end: usize) {
        self.0.push(format!("text {start}..{end}"));
    }
    fn cdata_section(&mut self, start: usize, end: usize) {
        self.0.push(format!("cdata {start}..{end}"));
    }
    fn comment(&mut self, start: usize, end: usize) {
        self.0.push(format!("comment {start}..{end}"));
    }
    fn processing_instruction(&mut self, target: &str, start: usize, end: usize) {
        self.0.push(format!("pi {target} {start}..{end}"));
    }
}

#[test]
fn tree_builders_get_the_bytes_of_each_node() {
    let mut n = Nodes::default();
    parse("<!--x--><d>a&amp;b<e/><![CDATA[c]]><?p q?></d >", &mut n).unwrap();
    assert_eq!(
        n.0,
        [
            "comment 0..8",
            "<d@8",
            "'a'",
            "'&'",
            "'b'",
            "text 11..18",
            "<e@18",
            "/e@18",
            "closed@22",
            "'c'",
            "cdata 22..35",
            "pi p 35..42",
            "/d@42",
            "closed@47",
        ]
    );
}

#[test]
fn encodings() {
    assert_eq!(decode(b"\xEF\xBB\xBF<d/>").unwrap(), "<d/>");
    assert_eq!(
        decode(b"<?xml version='1.0' encoding='ISO-8859-1'?><d>\xe9</d>").unwrap(),
        "<?xml version='1.0' encoding='ISO-8859-1'?><d>\u{e9}</d>"
    );
    assert!(
        matches!(decode(b"<?xml version='1.0' encoding='Shift_JIS'?><d/>"), Err(Error::XmlEncoding(e)) if &*e == "Shift_JIS")
    );
    assert!(matches!(decode(b"<d>\xe9</d>"), Err(Error::Xml { offset: 3, .. })));
    assert!(matches!(decode(b"\xFF\xFE<\x00"), Err(Error::XmlEncoding(_))));
}
