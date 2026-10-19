use super::align::remove_overlaps;
use super::*;

fn name(start: u32, end: u32) -> MascName {
    MascName { start, end, kind: None }
}

fn spans(names: &[MascName]) -> Vec<(u32, u32)> {
    names.iter().map(|n| (n.start, n.end)).collect()
}

#[test]
fn one_name_is_kept() {
    assert_eq!(remove_overlaps(&[name(0, 2)]), None);
    assert_eq!(remove_overlaps(&[]), None);
}

#[test]
fn names_apart_lose_all_but_the_first() {
    // the left index reaches the right, the name contains itself and is
    // marked, and the last name is dropped as well
    let kept = remove_overlaps(&[name(0, 1), name(2, 3), name(4, 6)]).unwrap();
    assert_eq!(spans(&kept), [(0, 1)]);
}

#[test]
fn the_shorter_of_two_overlapping_names_is_marked() {
    // (1, 4) contains (2, 3): the right is marked, then the left is
    // compared with (5, 6), which moves it on to itself
    let kept = remove_overlaps(&[name(1, 4), name(2, 3), name(5, 6), name(7, 8)]).unwrap();
    assert_eq!(spans(&kept), [(1, 4)]);
    // the longer name on the right marks the left one
    let kept = remove_overlaps(&[name(0, 2), name(1, 5), name(9, 9)]).unwrap();
    assert_eq!(spans(&kept), [(1, 5)]);
}

#[test]
fn overlapping_names_can_survive() {
    // next to each other, the right copy is marked
    let kept = remove_overlaps(&[name(0, 2), name(0, 2)]).unwrap();
    assert_eq!(spans(&kept), [(0, 2)]);
    // a shorter name before the copies crosses both and is marked twice,
    // so the walk moves the right index past the copies without
    // comparing them with each other; both stay, and `NameSample`
    // refuses them
    let kept = remove_overlaps(&[name(0, 2), name(1, 4), name(1, 4), name(9, 10)]).unwrap();
    assert_eq!(spans(&kept), [(1, 4), (1, 4)]);
    // the same with two names of one start but different ends, which a
    // `HashMap` of entity ids can order shorter first
    let kept = remove_overlaps(&[name(0, 2), name(1, 5), name(1, 6), name(20, 21)]).unwrap();
    assert_eq!(spans(&kept), [(1, 5), (1, 6)]);
}

#[test]
fn a_name_ends_one_past_a_token_only_when_past_the_end_so_far() {
    let doc = |penn: &str, ne: &str| {
        let text = b"a b c d e";
        let seg = br#"<graph><region xml:id="seg-r0" anchors="0 1"/><region xml:id="seg-r1" anchors="2 3"/><region xml:id="seg-r2" anchors="4 5"/><region xml:id="seg-r3" anchors="6 7"/><region xml:id="seg-r4" anchors="8 9"/></graph>"#;
        let s = br#"<graph><region anchors="0 9"/></graph>"#;
        MascDocument::parse(
            "t.hdr",
            MascFiles {
                text,
                segments: Some(seg),
                sentences: Some(s),
                penn: Some(penn.as_bytes()),
                named_entities: Some(ne.as_bytes()),
            },
        )
        .unwrap()
    };
    let mut penn = String::from("<graph>");
    for i in 0..5 {
        penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
    }
    penn.push_str("</graph>");
    let names = |edges: &[u32]| {
        let mut ne = String::from(r#"<graph><a ref="ne-n0" label="x"/>"#);
        for e in edges {
            ne.push_str(&format!(r#"<edge from="ne-n0" to="penn-n{e}"/>"#));
        }
        ne.push_str("</graph>");
        spans(doc(&penn, &ne).sentences()[0].names())
    };
    assert_eq!(names(&[3]), [(3, 4)]);
    // tokens 3 and 4: 4 is not past the end 4 the token 3 gave
    assert_eq!(names(&[3, 4]), [(3, 4)]);
    assert_eq!(names(&[1, 3]), [(1, 4)]);
    // the first token alone is empty
    assert_eq!(names(&[0]), [(0, 0)]);
    assert_eq!(names(&[4, 2]), [(2, 5)]);
    // Listed first to last, an even number of tokens loses the last
    // and an odd number is whole.
    assert_eq!(names(&[1, 2, 3, 4]), [(1, 4)]);
    assert_eq!(names(&[2, 3, 4]), [(2, 5)]);
    // Listed last to first, as MASC lists most entities, the first
    // token visited sets the end.
    assert_eq!(names(&[4, 3]), [(3, 5)]);
    assert_eq!(names(&[4, 3, 2, 1]), [(1, 5)]);
    // At the sentence's start the token 0 does not move the end from
    // 0, so the parity flips; last to first is whole there too.
    assert_eq!(names(&[0, 1]), [(0, 2)]);
    assert_eq!(names(&[0, 1, 2]), [(0, 2)]);
    assert_eq!(names(&[2, 1, 0]), [(0, 3)]);
}

#[test]
fn an_entity_in_no_sentence_is_kept_with_why() {
    // Seven tokens; the first region touches the next, so the reader
    // drops it, and the last three tokens are in no region.
    let text = b"a b c d e f g";
    let mut seg = String::from("<graph>");
    let mut penn = String::from("<graph>");
    for i in 0..7 {
        seg.push_str(&format!(r#"<region xml:id="seg-r{i}" anchors="{} {}"/>"#, 2 * i, 2 * i + 1));
        penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
    }
    seg.push_str("</graph>");
    penn.push_str("</graph>");
    let s = br#"<graph><region anchors="0 3"/><region anchors="3 7"/></graph>"#;
    // ne-n0: `a`, in the dropped region; ne-n1: `c`, in the sentence;
    // ne-n2: `e`, in no region; ne-n3: `f` and `b`, one token in the
    // dropped region; ne-n4: `g`, no label.
    let ne = r#"<graph><a ref="ne-n0" label="person"/><a ref="ne-n1" label="org"/><a ref="ne-n2" label="date"/>
        <a ref="ne-n3" label="location"/>
        <edge from="ne-n0" to="penn-n0"/><edge from="ne-n1" to="penn-n2"/><edge from="ne-n2" to="penn-n4"/>
        <edge from="ne-n3" to="penn-n5"/><edge from="ne-n3" to="penn-n1"/><edge from="ne-n4" to="penn-n6"/></graph>"#;
    let doc = MascDocument::parse(
        "t.hdr",
        MascFiles {
            text,
            segments: Some(seg.as_bytes()),
            sentences: Some(s),
            penn: Some(penn.as_bytes()),
            named_entities: Some(ne.as_bytes()),
        },
    )
    .unwrap();
    assert_eq!(doc.sentences().len(), 1);
    assert_eq!(doc.sentences()[0].entities().len(), 1);
    let unplaced = |kind: Option<&str>, cause| MascUnplacedEntity { kind: kind.map(str::to_owned), cause };
    assert_eq!(
        doc.unplaced_entities(),
        [
            unplaced(Some("person"), Unplaced::DroppedSentence),
            unplaced(Some("date"), Unplaced::OutsideSentences),
            unplaced(Some("location"), Unplaced::DroppedSentence),
            unplaced(None, Unplaced::OutsideSentences),
        ]
    );
}

#[test]
fn an_entity_spans_its_first_token_to_one_past_its_last_and_none_is_removed() {
    let text = b"a b c d e. f g";
    let mut seg = String::from("<graph>");
    let mut penn = String::from("<graph>");
    for (i, at) in [0, 2, 4, 6, 8, 11, 13].iter().enumerate() {
        seg.push_str(&format!(r#"<region xml:id="seg-r{i}" anchors="{at} {}"/>"#, at + 1));
        penn.push_str(&format!(r#"<node xml:id="penn-n{i}"><link targets="seg-r{i}"/></node>"#));
    }
    seg.push_str("</graph>");
    penn.push_str("</graph>");
    let s = br#"<graph><region anchors="0 10"/><region anchors="11 14"/></graph>"#;
    // ne-n0: tokens 0 and 1; ne-n1: token 0, inside ne-n0; ne-n2: tokens
    // 4 and 2, with a gap; ne-n3: tokens 4 and 5, across the sentence
    // end; ne-n4: no label.
    let ne = r#"<graph><a ref="ne-n0" label="person"/><a ref="ne-n1" label="org"/><a ref="ne-n2" label="date"/>
        <a ref="ne-n3" label="location"/>
        <edge from="ne-n0" to="penn-n1"/><edge from="ne-n0" to="penn-n0"/><edge from="ne-n1" to="penn-n0"/>
        <edge from="ne-n2" to="penn-n4"/><edge from="ne-n2" to="penn-n2"/>
        <edge from="ne-n3" to="penn-n4"/><edge from="ne-n3" to="penn-n5"/><edge from="ne-n4" to="penn-n6"/></graph>"#;
    let doc = MascDocument::parse(
        "t.hdr",
        MascFiles {
            text,
            segments: Some(seg.as_bytes()),
            sentences: Some(s),
            penn: Some(penn.as_bytes()),
            named_entities: Some(ne.as_bytes()),
        },
    )
    .unwrap();
    let entity = |start, end, kind: Option<&str>, tokens: &[u32]| MascEntity {
        start,
        end,
        kind: kind.map(str::to_owned),
        tokens: tokens.to_vec(),
    };
    let first = doc.sentences()[0].entities();
    assert_eq!(
        first,
        [
            entity(0, 2, Some("person"), &[0, 1]),
            entity(0, 1, Some("org"), &[0]),
            entity(2, 5, Some("date"), &[2, 4]),
            entity(4, 5, Some("location"), &[4]),
        ]
    );
    assert!(first[0].is_contiguous() && !first[2].is_contiguous());
    // OpenNLP's names of the same sentence lose most of this.
    assert_ne!(spans(doc.sentences()[0].names()), [(0, 2), (0, 1), (2, 5), (4, 5)]);
    assert_eq!(doc.sentences()[1].entities(), [entity(0, 1, Some("location"), &[0]), entity(1, 2, None, &[1])]);
}

#[test]
fn entity_is_contiguous_checks_reversed_bounds_duplicates_and_exact_range() {
    assert!(!MascEntity { start: 3, end: 1, kind: None, tokens: vec![] }.is_contiguous());
    assert!(!MascEntity { start: 0, end: 2, kind: None, tokens: vec![0, 0] }.is_contiguous());
    assert!(!MascEntity { start: 0, end: 1, kind: None, tokens: vec![5] }.is_contiguous());
    assert!(!MascEntity { start: 0, end: 2, kind: None, tokens: vec![1, 0] }.is_contiguous());
    assert!(MascEntity { start: 2, end: 5, kind: None, tokens: vec![2, 3, 4] }.is_contiguous());
    assert!(MascEntity { start: 4, end: 4, kind: None, tokens: vec![] }.is_contiguous());
}

#[test]
fn unplaced_entity_checks_all_quarks_when_quark_id_is_repeated() {
    // `seg-r0` appears first at 0..1 inside the dropped sentence region 0..2,
    // and again at 4..5 outside all sentence regions.
    let text = b"a b c";
    let seg = br#"<graph>
        <region xml:id="seg-r0" anchors="0 1"/>
        <region xml:id="seg-r1" anchors="2 3"/>
        <region xml:id="seg-r0" anchors="4 5"/>
    </graph>"#;
    let s = br#"<graph><region anchors="0 2"/><region anchors="2 3"/></graph>"#;
    let penn = br#"<graph>
        <node xml:id="penn-n0"><link targets="seg-r0"/></node>
        <node xml:id="penn-n1"><link targets="seg-r1"/></node>
    </graph>"#;
    let ne = br#"<graph>
        <a ref="ne-n0" label="person"/>
        <edge from="ne-n0" to="penn-n0"/>
    </graph>"#;
    let doc = MascDocument::parse(
        "t.hdr",
        MascFiles { text, segments: Some(seg), sentences: Some(s), penn: Some(penn), named_entities: Some(ne) },
    )
    .unwrap();
    assert_eq!(
        doc.unplaced_entities(),
        [MascUnplacedEntity { kind: Some("person".into()), cause: Unplaced::DroppedSentence }]
    );
}
