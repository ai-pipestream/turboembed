use super::*;

const TEXT: &str = r#"<teiCorpus><TEI><text xml:id="t"><body><div xml:id="d"><p xml:id="p1">Ala ma kota. Kot.</p><ab xml:id="p2">Dwa</ab></div></body></text></TEI></teiCorpus>"#;

fn seg(id: &str, p: &str, offset: u32, len: u32) -> String {
    format!(r#"<seg xml:id="{id}" corresp="text.xml#string-range({p},{offset},{len})"/>"#)
}

fn segmentation(sentences: &[(&str, String)]) -> String {
    let mut s = String::from("<teiCorpus><TEI><text><body><p>");
    for (id, segs) in sentences {
        s.push_str(&format!(r#"<s xml:id="{id}">{segs}</s>"#));
    }
    s.push_str("</p></body></text></TEI></teiCorpus>");
    s
}

fn read(seg: &str) -> Vec<Result<String, String>> {
    let s = NkjpSegmentation::parse(seg).unwrap();
    let t = NkjpText::parse(TEXT).unwrap();
    NkjpReader::new(s, &t).map(|r| r.map(|s| s.document().to_owned()).map_err(|e| e.to_string())).collect()
}

#[test]
fn sentences_are_cut_from_the_paragraphs() {
    let src = segmentation(&[
        ("s1", seg("a", "p1", 0, 3) + &seg("b", "p1", 7, 5)),
        ("s2", seg("c", "p1", 13, 3) + &seg("d", "p2", 0, 3)),
        ("s3", String::new()),
    ]);
    assert_eq!(read(&src), [Ok("Ala ma kota. Kot Dwa  ".to_owned())]);
}

#[test]
fn duplicate_sentence_ids_do_not_invert_sample_source_span() {
    let src =
        segmentation(&[("s1", seg("a", "p1", 0, 3)), ("s2", seg("b", "p2", 0, 3)), ("s1", seg("c", "p1", 13, 3))]);
    assert_eq!(read(&src), [Ok("Kot Dwa ".to_owned())]);
}

#[test]
fn anonymous_sentences_are_kept_independently() {
    let xml = r#"<teiCorpus><TEI><text><body><p><s><seg xml:id="a" corresp="text.xml#string-range(p1,0,3)"/></s><s><seg xml:id="b" corresp="text.xml#string-range(p1,4,2)"/></s></p></body></text></TEI></teiCorpus>"#;
    let parsed = NkjpSegmentation::parse(xml).unwrap();
    assert_eq!(parsed.sentences.len(), 2);
    assert_eq!(parsed.sentences[0].id, None);
    assert_eq!(parsed.sentences[1].id, None);
    assert_eq!(read(xml), [Ok("Ala ma ".to_owned())]);
}

#[test]
fn a_sentence_that_throws_loses_the_sample_so_far() {
    let src = segmentation(&[
        ("s1", seg("a", "p1", 0, 3)),
        ("s2", seg("b", "p9", 0, 3)),
        ("s3", seg("c", "p2", 1, 9)),
        ("s4", seg("d", "p2", 0, 3)),
    ]);
    assert_eq!(
        read(&src),
        [
            Err("Cannot invoke \"String.substring(int, int)\" because \"currentParagraph\" is null".to_owned()),
            Err("Range [1, 10) out of bounds for length 3".to_owned()),
            Ok("Dwa ".to_owned()),
        ]
    );
}

#[test]
fn three_number_pointers_and_bad_ones() {
    let s =
        NkjpSegmentation::parse(&segmentation(&[("s", r#"<seg xml:id="a" corresp="t#r(p1,0,13,3)"/>"#.to_owned())]))
            .unwrap();
    assert_eq!(s.sentences[0].segments[0].1.offset, 13);
    let e = NkjpSegmentation::parse(&segmentation(&[("s", r#"<seg xml:id="a" corresp="t#r(p1,x,3)"/>"#.into())]))
        .unwrap_err();
    assert_eq!(
        (e.java_class(), e.to_string().as_str()),
        (Some("java.lang.NumberFormatException"), "For input string: \"x\"")
    );
    let e = NkjpSegmentation::parse(&segmentation(&[("s", r#"<seg xml:id="a" corresp="t#r(p1,3)"/>"#.into())]))
        .unwrap_err();
    assert_eq!(e.to_string(), "Failed to parse NKJP document");
}
