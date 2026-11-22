use std::error::Error as _;

use crate::masc::{AnnotationFile, MascChunkFiles, MascCorpus, MascDocument, MascError, MascFiles, read_document};
use crate::xml::Error as XmlError;

#[test]
fn malformed_annotation_xml_preserves_byte_offsets_and_refused_encodings() {
    let valid_seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/></graph>"#;
    let valid_s = br#"<graph><region anchors="0 1"/></graph>"#;
    let check_parse = |files: MascFiles<'_>, expected_file: AnnotationFile, expected: XmlError| {
        let err = MascDocument::parse("t.hdr", files).unwrap_err();
        let source = err.source().and_then(|s| s.downcast_ref::<XmlError>()).cloned();
        assert_eq!(source.as_ref(), Some(&expected));
        match err {
            MascError::Annotation { file, cause } => assert_eq!((file, cause), (expected_file, expected)),
            other => panic!("expected Annotation error, got {other:?}"),
        }
    };

    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(br#"<graph><region xml:id="seg-r0" anchors="0 1"></wrong></graph>"#),
            sentences: Some(valid_s),
            penn: None,
            named_entities: None,
        },
        AnnotationFile::Segments,
        XmlError::Xml { offset: 45, message: "an end tag does not match its start tag" },
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(valid_seg),
            sentences: Some(b"<graph>\xFF</graph>"),
            penn: None,
            named_entities: None,
        },
        AnnotationFile::Sentences,
        XmlError::Xml { offset: 7, message: "the document is not valid UTF-8" },
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(valid_seg),
            sentences: Some(valid_s),
            penn: Some(b"<?xml version='1.0' encoding='US-ASCII'?><graph>\x80</graph>"),
            named_entities: None,
        },
        AnnotationFile::Penn,
        XmlError::Xml { offset: 48, message: "a byte above 0x7F in a US-ASCII document" },
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(br#"<graph>  <region xml:id="bad" anchors="0 1"/></graph>"#),
            sentences: Some(valid_s),
            penn: None,
            named_entities: None,
        },
        AnnotationFile::Segments,
        XmlError::Xml { offset: 9, message: "a region id is not seg-r and digits" },
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(b"\xFF\xFE<\x00g\x00"),
            sentences: Some(valid_s),
            penn: None,
            named_entities: None,
        },
        AnnotationFile::Segments,
        XmlError::XmlEncoding("UTF-16".into()),
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(valid_seg),
            sentences: Some(b"<?xml version='1.0' encoding='Shift_JIS'?><graph/>"),
            penn: None,
            named_entities: None,
        },
        AnnotationFile::Sentences,
        XmlError::XmlEncoding("Shift_JIS".into()),
    );
    check_parse(
        MascFiles {
            text: b"a",
            segments: Some(valid_seg),
            sentences: Some(valid_s),
            penn: None,
            named_entities: Some(b"\xFE\xFF\x00<\x00g"),
        },
        AnnotationFile::NamedEntities,
        XmlError::XmlEncoding("UTF-16".into()),
    );

    let mut doc = MascDocument::parse(
        "t.hdr",
        MascFiles { text: b"a", segments: Some(valid_seg), sentences: Some(valid_s), penn: None, named_entities: None },
    )
    .unwrap();
    let chunk_err = doc.add_chunks(MascChunkFiles { noun: Some(b"<!DOCTYPE graph><graph/>"), verb: None }).unwrap_err();
    assert!(matches!(
        chunk_err,
        MascError::Annotation {
            file: AnnotationFile::NounChunks,
            cause: XmlError::Xml { offset: 0, message: "document type declarations are not allowed" },
        }
    ));
    let logical_err = doc.add_logical(Some(b"<graph>&unknown;</graph>")).unwrap_err();
    assert!(matches!(
        logical_err,
        MascError::Annotation {
            file: AnnotationFile::Logical,
            cause: XmlError::Xml { offset: 7, message: "an entity other than the five XML defines" },
        }
    ));
}

#[test]
fn corpus_and_header_readers_report_xml_byte_offsets_and_refused_encodings() {
    let dir =
        turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("turbo-formats-masc-xml-{}", std::process::id())));
    std::fs::create_dir_all(&dir).unwrap();
    let hdr = dir.join("d.hdr");

    for (raw, expected, formatted) in [
        (
            b"<documentHeader><primaryData f.id='f.text' loc='d.txt'/></wrong>".as_slice(),
            XmlError::Xml { offset: 56, message: "an end tag does not match its start tag" },
            "XML at byte 56: an end tag does not match its start tag",
        ),
        (
            b"\xFE\xFF\x00<".as_slice(),
            XmlError::XmlEncoding("UTF-16".into()),
            "XML encoding \"UTF-16\" is not UTF-8, US-ASCII or ISO-8859-1",
        ),
        (
            b"<?xml version=\"1.0\" encoding=\"EUC-JP\"?><documentHeader/>".as_slice(),
            XmlError::XmlEncoding("EUC-JP".into()),
            "XML encoding \"EUC-JP\" is not UTF-8, US-ASCII or ISO-8859-1",
        ),
    ] {
        std::fs::write(&hdr, raw).unwrap();
        for err in [read_document(&hdr).unwrap_err(), MascCorpus::read(&dir).unwrap_err()] {
            assert_eq!(err.source().map(ToString::to_string).as_deref(), Some(formatted));
            match err {
                MascError::Header { cause, .. } => assert_eq!(cause, expected),
                other => panic!("expected Header error, got {other:?}"),
            }
        }
    }

    std::fs::write(
        &hdr,
        br#"<documentHeader><primaryData f.id="f.text" loc="d.txt"/><annotation f.id="f.seg" loc="d-seg.xml"/><annotation f.id="f.s" loc="d-s.xml"/></documentHeader>"#,
    )
    .unwrap();
    std::fs::write(dir.join("d.txt"), b"a").unwrap();
    std::fs::write(dir.join("d-seg.xml"), b"<graph><region/></graph>").unwrap();
    std::fs::write(dir.join("d-s.xml"), b"<?xml version='1.0' encoding='Shift_JIS'?><graph/>").unwrap();
    let corpus = MascCorpus::read(&dir).unwrap();
    assert!(corpus.documents().is_empty());
    assert_eq!(corpus.skipped().len(), 1);
    assert!(matches!(
        &corpus.skipped()[0].error,
        MascError::Annotation {
            file: AnnotationFile::Segments,
            cause: XmlError::Xml { offset: 7, message: "a region id is not seg-r and digits" },
        }
    ));

    std::fs::write(dir.join("d-seg.xml"), br#"<graph><region xml:id="seg-r0" anchors="0 1"/></graph>"#).unwrap();
    let corpus = MascCorpus::read(&dir).unwrap();
    assert_eq!(corpus.skipped().len(), 1);
    assert!(matches!(
        &corpus.skipped()[0].error,
        MascError::Annotation { file: AnnotationFile::Sentences, cause: XmlError::XmlEncoding(enc) } if enc == "Shift_JIS"
    ));
    std::fs::remove_dir_all(&dir).unwrap();
}
