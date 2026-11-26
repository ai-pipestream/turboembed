//! The SVG header, and the SVG scanner against the JDK's StAX parser.

use super::forms::utf16;
use super::*;

// ---------------------------------------------------------------------------
// AssetSvgHeaderTest

const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

fn image() -> String {
    String::from_utf8(svg()).unwrap()
}

fn assert_svg(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "svg", "image/svg+xml", transport);
}

#[test]
fn svg_image() {
    let image = image();
    let ns = SVG_NAMESPACE;
    for xml in [
        image.clone(),
        format!("<?xml version='1.0'?>{image}"),
        format!("\u{FEFF}{image}"),
        format!(" \r\n\t<!-- Survey icon --><?creator fieldwork?>{image}"),
        format!("<drawing:svg xmlns:drawing='{ns}'/>"),
        format!("<svg1:svg xmlns:svg1='{ns}'/>"),
        format!("<svg:svg xmlns:svg='{ns}'/>"),
        format!("<\u{3b1}:svg xmlns:\u{3b1}='{ns}'/>"),
        format!("<svg data-note='&gt; &quot; &amp;' xmlns='{ns}'/>"),
        "<svg xmlns='http://www.w3.org/2000/sv&#x67;'/>".to_string(),
        format!("<!DOCTYPE svg SYSTEM 'urn:opennlp:unused-dtd'>{image}"),
    ] {
        for transport in TRANSPORTS {
            assert_svg(xml.as_bytes(), transport);
        }
    }
}

#[test]
fn svg_encoding() {
    for charset in ["UTF-8", "UTF-16", "UTF-16LE", "UTF-16BE"] {
        let xml = format!("<?xml version='1.0' encoding='{charset}'?>{}", image());
        let bytes = match charset {
            "UTF-8" => xml.into_bytes(),
            // Java's UTF-16 encoder writes a big-endian byte-order mark.
            "UTF-16" => utf16(&format!("\u{FEFF}{xml}"), true),
            "UTF-16LE" => utf16(&xml, false),
            _ => utf16(&xml, true),
        };
        assert_svg(&bytes, "standard");
        assert_svg(&bytes, "uri");
    }
}

#[test]
fn svg_not_svg() {
    for xml in [
        "<svg-notes>Survey</svg-notes>",
        "<svgNotes/>",
        "<svg/>",
        "<svg xmlns='urn:survey'/>",
        "<svg xmlns='http://www.w3.org/2000/svg-notes'/>",
        "<svg xmlns:other='http://www.w3.org/2000/svg'/>",
        "<svg xmlns='HTTP://www.w3.org/2000/svg'/>",
        "<svg xmlns='http://www.w3.org/2000/svg' xmlns='urn:survey'/>",
        "<svg xmlns='http://www.w3.org/2000/svg' a='1' a='2'/>",
        "<svg:notes xmlns:svg='http://www.w3.org/2000/svg'/>",
        "<svg:not-svg xmlns:svg='http://www.w3.org/2000/svg'/>",
        "<svg:svg/>",
        "<svg xmlns=http://www.w3.org/2000/svg/>",
        "<svg xmlns='http://www.w3.org/2000/svg'",
        "<svg",
        "<svg ",
        "<svg\0 xmlns='http://www.w3.org/2000/svg'/>",
        "<!-- <svg xmlns='http://www.w3.org/2000/svg'/> --><notes/>",
        "<notes><svg xmlns='http://www.w3.org/2000/svg'/></notes>",
        "<SVG xmlns='http://www.w3.org/2000/svg'/>",
    ] {
        assert_unrecognized_default(xml.as_bytes());
    }
}

#[test]
fn svg_later_namespace() {
    let xml = format!("<svg data-note='{}' xmlns='{SVG_NAMESPACE}'/>", "survey ".repeat(1000));
    assert_svg(xml.as_bytes(), "mime76");
}

#[test]
fn svg_header_limit() {
    let (prefix, suffix) = ("<svg data-note='", format!("' xmlns='{SVG_NAMESPACE}'/>"));
    let xml = format!("{prefix}{}{suffix}", "x".repeat(64 * 1024 - prefix.len() - suffix.len()));
    assert_svg(xml.as_bytes(), "uri");
    assert_unrecognized_default(format!("{prefix}x{}", &xml[prefix.len()..]).as_bytes());
}

// testBoundedSourceRead: a root tag that ends past the 64 KiB limit is
// not SVG. (That the scan reads no further than the limit is a property
// of the streaming decoder, which stops after `MAX_HEADER_BYTES`.)
#[test]
fn svg_bounded_source_read() {
    let xml = format!("<svg data-note='{}' xmlns='{SVG_NAMESPACE}'/>", "x".repeat(128 * 1024));
    assert_unrecognized_default(xml.as_bytes());
}

#[test]
fn svg_incomplete_image() {
    assert_svg(format!("<svg xmlns='{SVG_NAMESPACE}'><invalid").as_bytes(), "standard");
}

#[test]
fn svg_other_xml() {
    let bytes = b"<?xml version='1.0'?><notes/>";
    assert_identified(&AssetDetector::new(), bytes, "xml", "application/xml", "uri");
}

#[test]
fn svg_declared_type() {
    let bytes = b"<svg-notes>Survey</svg-notes>";
    let text = format!("data:image/svg+xml;base64,{}", std64(bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("svg+xml", "image/svg+xml"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// testNoExternalRequests and testExternalNamespace: the scanner has no
// way to fetch anything, so a stylesheet or DTD URL changes nothing, and
// an attribute default an external DTD would supply is never applied.
#[test]
fn svg_external_references_are_never_read() {
    let url = "http://127.0.0.1:9/unused.dtd";
    let xml = format!("<?xml-stylesheet href='{url}'?><!DOCTYPE svg SYSTEM '{url}'>{}", image());
    assert_svg(xml.as_bytes(), "uri");
    assert_unrecognized_default(b"<!DOCTYPE svg SYSTEM 'file:///tmp/namespace.dtd'><svg/>");
}

#[test]
fn svg_internal_namespace_entity() {
    let xml = format!("<!DOCTYPE svg [<!ENTITY ns '{SVG_NAMESPACE}'>]><svg xmlns='&ns;'/>");
    assert_unrecognized_default(xml.as_bytes());
}

#[test]
fn svg_concurrent_detection() {
    std::thread::scope(|scope| {
        for thread in 0..4 {
            scope.spawn(move || {
                for index in (thread..32).step_by(4) {
                    assert_svg(format!("<!-- request {index} -->{}", image()).as_bytes(), "uri");
                    assert_unrecognized_default(b"<svg xmlns='urn:not-an-image'/>");
                }
            });
        }
    });
}

#[test]
fn svg_manual_example() {
    let text = format!("{DATA_URI}{}", std64(&svg()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("svg", "image/svg+xml"));
    assert_eq!(a.decode(&text).unwrap(), svg());
}

// ---------------------------------------------------------------------------
// The SVG scanner against the JDK's StAX parser, which OpenNLP's
// `SvgHeader` runs with DTD support off. Each expectation was read from
// that parser.

#[test]
fn svg_prolog_follows_the_jdk_parser() {
    let root = format!("<svg xmlns='{SVG_NAMESPACE}'/>");
    let cases: [(&str, bool); 31] = [
        // The internal subset is skipped unread up to its first `]`.
        ("<!DOCTYPE svg [<!A<TLIST>]>", true),
        ("<!DOCTYPE svg [ ' ]>", true),
        ("<!DOCTYPE svg [ %pe; garbage ]>", true),
        // A character XML does not allow: the JDK parser throws
        // MissingResourceException, which `SvgHeader` does not catch; here
        // the payload is not SVG.
        ("<!DOCTYPE svg [ \u{1} ]>", false),
        ("<!DOCTYPE svg [ ' ]>' ]>", false),
        ("<!DOCTYPE svg [ ]] ]>", false),
        ("<!DOCTYPE svg []  >", true),
        ("<!DOCTYPE svg [] x>", false),
        ("<!DOCTYPE svg [<!-- ]> -->]>", false),
        ("<!DOCTYPE svg[]>", true),
        ("<!DOCTYPE svg SYSTEM 'c'[]>", true),
        // Public identifiers hold PubidChar only; system literals anything.
        ("<!DOCTYPE svg PUBLIC 'a]b' 'c'>", false),
        ("<!DOCTYPE svg PUBLIC 'a\tb' 'c'>", false),
        ("<!DOCTYPE svg PUBLIC \"a'b\" 'c'>", true),
        ("<!DOCTYPE svg SYSTEM 'a]b<\t'>", true),
        // White space is required between the identifiers and after SYSTEM.
        ("<!DOCTYPE svg PUBLIC 'ab''c'>", false),
        ("<!DOCTYPE svg SYSTEM'c'>", false),
        ("<!DOCTYPE svg SYSTEM 'c'><!DOCTYPE svg SYSTEM 'c'>", false),
        // A processing instruction target may hold a colon, not be `xml`.
        ("<?x:l data?>", true),
        ("<?XmL data?>", false),
        // Encoding names: the JDK's IANA names and aliases, not Java's.
        ("<?xml version='1.0' encoding='utf-8'?>", true),
        ("<?xml version='1.0' encoding='UTF8'?>", false),
        ("<?xml version='1.0' encoding='ISO8859-1'?>", false),
        ("<?xml version='1.0' encoding='latin1'?>", true),
        ("<?xml version='1.0' encoding='L1'?>", true),
        ("<?xml version='1.0' encoding='CP819'?>", true),
        ("<?xml version='1.0' encoding='US'?>", true),
        ("<?xml version='1.0' encoding='IBM367'?>", true),
        ("<?xml version='1.0' encoding='ANSI_X3.4-1968'?>", true),
        ("<?xml version='1.0' encoding='UTF-16'?>", false),
        ("<?xml version='1.0' encoding='ISO_646.IRV:1991'?>", false),
    ];
    for (prolog, is_svg) in cases {
        let bytes = format!("{prolog}{root}").into_bytes();
        let text = format!("{DATA_URI}{}", std64(&bytes));
        let found = detect(&text);
        let svg = found.first().is_some_and(|a| a.format == AssetFormat::Svg);
        assert_eq!(svg, is_svg, "{prolog:?}: {found:?}");
    }
}

#[test]
fn svg_qualified_names_follow_the_jdk_parser() {
    let ns = SVG_NAMESPACE;
    let cases = [
        // A colon in first place is part of the name.
        (format!("<svg xmlns='{ns}' :b='1'/>"), true),
        (format!("<svg xmlns='{ns}' :='1'/>"), true),
        (format!("<svg xmlns='{ns}' :b='1' b='2'/>"), true),
        (format!("<svg xmlns='{ns}' :xmlns='1'/>"), true),
        (format!("<svg xmlns='{ns}' :b='1' :b='2'/>"), false),
        (format!("<:svg xmlns='{ns}'/>"), false),
        (format!("<svg :xmlns='{ns}'/>"), false),
        // After it, the first colon splits the prefix, which must be bound.
        (format!("<svg xmlns='{ns}' :a:b='1'/>"), false),
        (format!("<svg xmlns='{ns}' ::b='1'/>"), false),
        (format!("<svg xmlns='{ns}' b:='1'/>"), false),
        (format!("<svg xmlns='{ns}' a::b='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:b:c='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:1='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:b='1' xmlns:a='u'/>"), true),
        (format!("<svg: xmlns='{ns}'/>"), false),
        (format!("<a:svg:b xmlns:a='{ns}'/>"), false),
        (format!("<svg xmlns:='{ns}'/>"), false),
        (format!("<\u{e9}:svg xmlns:\u{e9}='{ns}'/>"), true),
    ];
    for (xml, is_svg) in cases {
        let text = format!("{DATA_URI}{}", std64(xml.as_bytes()));
        let svg = detect(&text).first().is_some_and(|a| a.format == AssetFormat::Svg);
        assert_eq!(svg, is_svg, "{xml}");
    }
}

/// Names follow XML 1.0 fifth edition. The JDK applies the fourth
/// edition's tables to XML 1.0 and rejects these; see the differences in
/// docs/normalizers.md.
#[test]
fn svg_names_follow_the_fifth_edition() {
    for name in ["\u{2C30}", "a\u{2C30}", "\u{10000}", "\u{2070}", "\u{FDF0}"] {
        let xml = format!("<svg xmlns='{SVG_NAMESPACE}' {name}='1'/>");
        assert_svg(xml.as_bytes(), "uri");
    }
}

/// Declared encodings other than UTF-8, UTF-16, US-ASCII and ISO-8859-1
/// are not read, so the payload is XML but not SVG; the JDK reads them.
#[test]
fn svg_other_declared_encodings_are_not_read() {
    for encoding in ["windows-1252", "ISO-8859-2", "Shift_JIS", "EBCDIC-CP-US"] {
        let xml = format!("<?xml version='1.0' encoding='{encoding}'?><svg xmlns='{SVG_NAMESPACE}'/>");
        assert_identified(&AssetDetector::new(), xml.as_bytes(), "xml", "application/xml", "uri");
    }
}

/// The scan stops at the end of the root start tag, so bytes after it do
/// not matter. The JDK decodes ahead and rejects a four-byte UTF-8
/// sequence above U+10FFFF within about 8 KB after the tag.
#[test]
fn svg_bytes_after_the_root_tag_do_not_matter() {
    let root = format!("<svg xmlns='{SVG_NAMESPACE}'/>");
    for tail in [&b"\xF4\x90\x80\x80"[..], b"\xF6\xB4\xB6\xAB", b"\xFF\xFF", b"\0"] {
        assert_svg(&cat(root.as_bytes(), tail), "uri");
    }
}
