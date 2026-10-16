use super::*;

/// A word line with the form, the universal tag and the `MISC` column.
fn w(id: &str, form: &str, misc: &str) -> String {
    format!("{id}\t{form}\t_\tX\tx{form}\t_\t0\troot\t_\t{misc}\n")
}

fn docs(src: &str) -> Vec<Result<CorefDocument<'_>, FormatError>> {
    ConlluCorefReader::new(src, Tagset::U).unwrap().collect()
}

fn one(src: &str) -> CorefDocument<'_> {
    let mut d = docs(src);
    assert_eq!(d.len(), 1);
    d.remove(0).unwrap()
}

fn mentions<'d>(d: &'d CorefDocument<'_>) -> Vec<(u32, &'d str)> {
    d.mentions().iter().map(|m| (m.chain, &d.text()[m.text.range()])).collect()
}

fn error(src: &str) -> String {
    docs(src).remove(0).unwrap_err().to_string()
}

#[test]
fn nested_mentions_and_text() {
    let src = format!(
        "# newdoc id = d\n# speaker = Ann\n{}{}{}{}\n{}{}",
        w("1", "Mary", "Entity=(e1-person-1-)"),
        w("2", "saw", "_"),
        w("3", "her", "Entity=(e2-person-2-(e1-person-1-)"),
        w("4", "dog", "Entity=e2)|SpaceAfter=No"),
        w("1", "It", "Entity=(e2-animal-1-)"),
        w("2", "ran", "SpaceAfter=No"),
    );
    let d = one(&src);
    assert_eq!(d.text(), "Mary saw her dog\nIt ran");
    // In text order; the chains are numbered by first mention.
    assert_eq!(mentions(&d), [(0, "Mary"), (1, "her dog"), (0, "her"), (1, "It")]);
    assert_eq!(d.clusters().len(), 2);
    assert_eq!(d.clusters()[1].mentions, [1, 3]);
    let m = &d.mentions()[1];
    assert_eq!(m.entity().as_str(), "e2");
    assert_eq!(m.tokens, 2..4);
    assert_eq!(&src[m.source.range()], "her\t_\tX\txher\t_\t0\troot\t_\tEntity=(e2-person-2-(e1-person-1-)\n4\tdog");
    assert_eq!(&src[m.open.span.range()], "(e2-person-2-");
    assert_eq!(&src[m.close.as_ref().unwrap().span.range()], "e2)");
    assert_eq!(m.open.attribute_values().collect::<Vec<_>>(), ["person", "2", ""]);
    let s: Vec<(&str, Option<&str>)> =
        d.sentences().iter().map(|s| (&d.text()[s.text.range()], s.speaker.as_ref().map(|f| f.as_str()))).collect();
    assert_eq!(s, [("Mary saw her dog", Some("Ann")), ("It ran", None)]);
    assert_eq!(d.tokens()[3].tag.as_str(), "X");
    assert_eq!(&src[d.tokens()[3].form.span.range()], "dog");
}

#[test]
fn equal_spans_keep_the_order_entities_appear_in() {
    // Two mentions of the same span: e9 appears first, so it is first and
    // gets chain 0 though e1 opened its mention on the same token.
    let src = format!("{}{}", w("1", "a", "Entity=(e9(e1"), w("2", "b", "Entity=e1)e9)"));
    let d = one(&src);
    let ids: Vec<&str> = d.mentions().iter().map(|m| m.entity().as_str()).collect();
    assert_eq!(ids, ["e9", "e1"]);
    assert_eq!(mentions(&d), [(0, "a b"), (1, "a b")]);
}

#[test]
fn a_close_takes_the_innermost_open_mention_of_its_entity() {
    // Parts of a discontinuous mention around a nested mention of the same
    // entity: the first close pops the nested one, whatever part it names.
    let src = format!(
        "{}{}{}{}",
        w("1", "a", "Entity=(e5[1/2]-x"),
        w("2", "b", "Entity=(e5-y"),
        w("3", "c", "Entity=e5[1/2])"),
        w("4", "d", "Entity=e5)"),
    );
    let d = one(&src);
    assert_eq!(mentions(&d), [(0, "a b c d"), (0, "b c")]);
    let outer = &d.mentions()[0];
    assert_eq!(outer.open.part, Some(entity::Part { index: 1, total: 2 }));
    assert_eq!(outer.close.as_ref().unwrap().part, None);
}

#[test]
fn multiword_tokens_and_empty_nodes() {
    let src = format!(
        "{}{}{}{}{}",
        w("1-2", "del", "SpaceAfter=No"),
        w("1", "de", "_"),
        w("2", "el", "Entity=(e1)"),
        w("2.1", "pro", "Entity=(e2)"),
        w("3", "mar", "_"),
    );
    let d = one(&src);
    assert_eq!(d.text(), "de el mar");
    assert_eq!(mentions(&d), [(0, "el")]);
    assert_eq!(
        d.warnings(),
        [
            CorefWarning::MultiwordSpaceAfter { line: 1 },
            CorefWarning::EmptyNodeEntity {
                line: 4,
                value: Span::new(src.find("(e2)").unwrap() as u32, src.find("(e2)").unwrap() as u32 + 4)
            }
        ]
    );
}

#[test]
fn documents_split_at_newdoc() {
    let src = format!(
        "# newdoc id = a\n# only comments\n\n# newdoc id = b\n{}\n# newdoc id = c\n{}# newdocument\n{}",
        w("1", "x", "_"),
        w("1", "y", "_"),
        w("1", "z", "_"),
    );
    let d = docs(&src);
    let texts: Vec<String> = d.iter().map(|d| d.as_ref().unwrap().text().to_owned()).collect();
    // `# newdocument` starts with `# newdoc` too; the first document has no
    // word and is passed over.
    assert_eq!(texts, ["x", "y", "z"]);
}

#[test]
fn refusals() {
    assert_eq!(error("# speaker = \u{3000}\n1\ta\n"), "speaker label must not be blank");
    assert_eq!(error("1\ta\n"), "expected 10 columns: 1\ta");
    assert_eq!(error(&w("1", " ", "_")), "column 2 must not be blank: 1\t \t_\tX\tx \t_\t0\troot\t_\t_");
    for bad in ["01", "0", "2-2", "3-2", "1-2-3", "a", "1.0", ".1", "0-1"] {
        assert_eq!(error(&w(bad, "a", "_")), format!("invalid token id: {bad}"), "{bad}");
    }
    assert!(matches!(word_id("0.1"), Id::EmptyNode));
    assert!(matches!(word_id("9-10"), Id::Multiword));
    assert_eq!(error(&w("2", "a", "_")), "expected token id 1 but found 2");
    assert_eq!(error(&w("1", "a", "Entity=")), "entity id must not be empty");
    assert_eq!(error(&w("1", "a", "Entity=e1")), "entity close is missing ')': e1");
    assert_eq!(error(&w("1", "a", "Entity=e1)")), "entity e1 is closed but not open");
    assert_eq!(
        error(&format!("{}\n{}", w("1", "a", "Entity=(e1"), w("1", "b", "Entity=e1)"))),
        "entity e1 crosses a sentence boundary"
    );
    assert_eq!(error(&w("1", "a", "Entity=(e1")), "entity e1 is never closed");
    // A byte order mark is part of the first line.
    assert_eq!(error("\u{feff}# newdoc\n"), "expected 10 columns: \u{feff}# newdoc");
}

#[test]
fn an_error_ends_only_its_document() {
    let src = format!("{}# newdoc\n{}", w("1", "a", "Entity=(e1"), w("1", "b", "Entity=(e1)"));
    let d = docs(&src);
    assert!(d[0].is_err());
    assert_eq!(d[1].as_ref().unwrap().text(), "b");
}

#[test]
fn open_entities_are_named_in_hash_map_order() {
    // The expected names are what Java 21 prints for these maps. "Aa" and
    // "BB" have the same hash code and "zz" shares their bin;
    // `computeIfAbsent` puts each new key at the head of its bin.
    assert_eq!(error(&w("1", "a", "Entity=(BB(e1(Aa")), "entity Aa is never closed");
    assert_eq!(error(&w("1", "a", "Entity=(BB(e1(Aa(zz")), "entity zz is never closed");
    // Thirteen entities fill a table of 16 bins. `computeIfAbsent` doubles
    // it only on its next call, which moves e3 ahead of e80.
    let ids: Vec<String> = (0..13).map(|i| format!("e{}", i * 7 + 3)).collect();
    let opens: String = ids.iter().map(|id| format!("({id}")).collect();
    let closes: String = ids.iter().filter(|id| *id != "e3" && *id != "e80").map(|id| format!("{id})")).collect();
    let first = w("1", "a", &format!("Entity={opens}"));
    let second = format!("{first}{}", w("2", "b", &format!("Entity={closes}")));
    assert_eq!(error(&second), "entity e80 is never closed");
    let reopened = format!("{first}{}", w("2", "b", &format!("Entity={closes}(e3")));
    assert_eq!(error(&reopened), "entity e3 is never closed");
}

#[test]
fn speaker_comments_without_words_carry_over() {
    let src = format!("# speaker = A\n\n{}\n{}", w("1", "x", "_"), w("1", "y", "_"));
    let d = one(&src);
    let s: Vec<Option<&str>> = d.sentences().iter().map(|s| s.speaker.as_ref().map(|f| f.as_str())).collect();
    assert_eq!(s, [Some("A"), None]);
}

#[test]
fn unicode_offsets() {
    let src = format!("{}{}", w("1", "Ça", "Entity=(é1"), w("2", "𝔘", "Entity=é1)"));
    let d = one(&src);
    assert_eq!(d.text(), "Ça 𝔘");
    let m = &d.mentions()[0];
    assert_eq!(m.text, Span::new(0, 8));
    assert_eq!(m.entity().as_str(), "é1");
    assert_eq!(&src[d.tokens()[1].form.span.range()], "𝔘");
}

#[test]
fn many_mentions() {
    // A long sentence of nested mentions, all closed on the last word.
    let n = 2_000;
    let mut src = String::new();
    for i in 1..=n {
        src.push_str(&w(&i.to_string(), "w", &format!("Entity=(e{i}")));
    }
    let closes: String = (1..=n).rev().map(|i| format!("e{i})")).collect();
    src.push_str(&w(&(n + 1).to_string(), "end", &format!("Entity={closes}")));
    let d = one(&src);
    assert_eq!(d.mentions().len(), n);
    assert_eq!(d.mentions()[0].entity().as_str(), "e1");
    assert_eq!(d.clusters().len(), n);
}

#[test]
fn crossing_mentions_close_by_entity() {
    // e1 opens before e2 but closes first: each close takes its own
    // entity's mention, so the two mentions cross.
    let src = format!("{}{}{}", w("1", "a", "Entity=(e1(e2"), w("2", "b", "Entity=e1)"), w("3", "c", "Entity=e2)"));
    let d = one(&src);
    assert_eq!(mentions(&d), [(0, "a b c"), (1, "a b")]);
    let ids: Vec<&str> = d.clusters().iter().map(|c| c.entity.as_str()).collect();
    assert_eq!(ids, ["e2", "e1"]);
}
