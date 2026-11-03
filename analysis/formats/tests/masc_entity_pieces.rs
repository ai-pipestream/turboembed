//! Project-authored annotations exercised through the public XML parser.
use turbo_formats::masc::{MascDocument, MascError, MascFiles};

fn document(cut: bool) -> MascDocument {
    let text = "😀 Ada Lovelace Paris";
    let segments = br#"<graph>
        <region xml:id="seg-r0" anchors="0 2"/>
        <region xml:id="seg-r1" anchors="3 6"/>
        <region xml:id="seg-r2" anchors="7 15"/>
        <region xml:id="seg-r3" anchors="16 21"/>
    </graph>"#;
    let sentences: &[u8] = if cut {
        br#"<graph><region anchors="0 6"/><region anchors="7 21"/></graph>"#
    } else {
        br#"<graph><region anchors="0 21"/></graph>"#
    };
    let penn = br#"<graph>
        <node xml:id="penn-n0"><link targets="seg-r0"/></node>
        <node xml:id="penn-n1"><link targets="seg-r1"/></node>
        <node xml:id="penn-n2"><link targets="seg-r2"/></node>
        <node xml:id="penn-n3"><link targets="seg-r3"/></node>
    </graph>"#;
    let ne = br#"<graph>
        <a ref="ne-n42" label="person"/><a ref="ne-n43" label="location"/>
        <edge from="ne-n42" to="penn-n2"/><edge from="ne-n42" to="penn-n1"/>
        <edge from="ne-n43" to="penn-n3"/>
    </graph>"#;
    MascDocument::parse(
        "authored.hdr",
        MascFiles {
            text: text.as_bytes(),
            segments: Some(segments),
            sentences: Some(sentences),
            penn: Some(penn),
            named_entities: Some(ne),
        },
    )
    .unwrap()
}

#[test]
fn cut_pieces_keep_identity_and_exact_unicode_offsets() {
    let doc = document(true);
    let left = &doc.sentences()[0].entity_pieces()[0];
    let right = &doc.sentences()[1].entity_pieces()[0];
    assert_eq!((left.id, right.id), (42, 42));
    assert!(left.continues && right.continues);
    assert_eq!(left.entity.tokens, [1]);
    assert_eq!(right.entity.tokens, [0]);
    for (sentence, piece, expected) in [(&doc.sentences()[0], left, "Ada"), (&doc.sentences()[1], right, "Lovelace")] {
        let spans = doc.token_spans(sentence).unwrap();
        let span = spans[piece.entity.start as usize];
        assert_eq!(&doc.text()[span.start as usize..span.end as usize], expected);
    }
    let normal = &doc.sentences()[1].entity_pieces()[1];
    assert_eq!(normal.id, 43);
    assert!(!normal.continues);
    assert_eq!(normal.entity.kind.as_deref(), Some("location"));
    assert!(matches!(doc.require_complete_entities(), Err(MascError::CutEntity {
        path, entity: 42, start: 3, end: 6
    }) if path.to_str() == Some("authored.hdr")));
}

#[test]
fn a_whole_entity_keeps_identity_without_continuation() {
    let doc = document(false);
    doc.require_complete_entities().unwrap();
    let pieces = doc.sentences()[0].entity_pieces();
    assert_eq!(pieces.iter().map(|p| (p.id, p.continues)).collect::<Vec<_>>(), [(42, false), (43, false)]);
    assert_eq!(pieces[0].entity.tokens, [1, 2]);
    assert_eq!(pieces[0].entity, doc.sentences()[0].entities()[0]);
}

#[test]
fn document_view_recovers_a_cut_entity_without_changing_source_or_sentences() {
    let doc = document(true);
    let all = doc.document_tokens().unwrap();
    assert_eq!((all.start, all.end), (0, 21));
    assert_eq!(all.entity_pieces().iter().map(|p| (p.id, p.continues)).collect::<Vec<_>>(), [(42, false), (43, false)]);
    assert_eq!(all.entities()[0].tokens, [1, 2]);
    let spans = doc.token_spans(all).unwrap();
    assert_eq!(&doc.text()[spans[1].start as usize..spans[2].end as usize], "Ada Lovelace");
    assert_eq!(spans[0].end, 4);
    assert_eq!(doc.sentences().len(), 2);
    assert!(doc.sentences()[0].entity_pieces()[0].continues);
    assert!(doc.require_complete_entities().is_err());
}
