use super::*;
use crate::ner::RuleTokenizer;

fn read(source: &str) -> Vec<Result<String, String>> {
    MucReader::new(source, &RuleTokenizer::default(), WhitespaceMode::Unicode)
        .unwrap()
        .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
        .collect()
}

#[test]
fn reads_content_elements_and_names() {
    let text = "<DOC><HL>Big <ENAMEX TYPE=\"ORGANIZATION\">Acme</ENAMEX> news</HL>\n<s> On <TIMEX TYPE=\"DATE\">Monday</TIMEX> . </s></DOC>";
    assert_eq!(
        read(text),
        [Ok("\nBig <START:organization> Acme <END> news".to_owned()), Ok("On <START:date> Monday <END> .".to_owned())]
    );
}

#[test]
fn text_outside_content_and_after_the_last_tag_is_not_read() {
    assert_eq!(read("lost <s>a b</s> lost <s></s> lost"), [Ok("a b".to_owned()), Ok("".to_owned())]);
}

#[test]
fn double_closing_brackets_are_one() {
    // a `>` outside a tag is an error, so the replacement matters in
    // tags, and the spans after it still name the source
    let text = "<s>>a <ENAMEX TYPE=\"PERSON\">>b</ENAMEX></s>";
    let samples: Vec<_> =
        MucReader::new(text, &RuleTokenizer::default(), WhitespaceMode::Unicode).unwrap().map(Result::unwrap).collect();
    assert_eq!(samples[0].to_string(), "a <START:person> b <END>");
    let tokens = samples[0].tokens();
    assert_eq!(&text[tokens[1].span.range()], "b");
    assert!(tokens.iter().all(Field::is_borrowed));
    assert_eq!(&text[samples[0].source().range()], text);
    assert_eq!(read("<s>a >> b</s>"), [Err("Did not expect > char!".to_owned())]);
}

#[test]
fn attributes_need_double_quotes() {
    assert_eq!(type_attribute("<ENAMEX TYPE=\"PERSON\">", WhitespaceMode::Unicode).as_deref(), Some("PERSON"));
    assert_eq!(type_attribute("<ENAMEX TYPE=PERSON>", WhitespaceMode::Unicode), None);
    assert_eq!(
        type_attribute("<ENAMEX TYPE=\"DATE\" TYPE=\"TIME\">", WhitespaceMode::Unicode).as_deref(),
        Some("TIME")
    );
    // the space after `=` starts another key, which takes the quote
    assert_eq!(type_attribute("<ENAMEX TYPE = \"MONEY\">", WhitespaceMode::Unicode), None);
    assert_eq!(type_attribute("<ENAMEX TYPE =\"MONEY\">", WhitespaceMode::Unicode).as_deref(), Some("MONEY"));
    assert_eq!(type_attribute("<ENAMEX\u{a0}TYPE=\"MONEY\">", WhitespaceMode::Unicode).as_deref(), Some("MONEY"));
    assert_eq!(type_attribute("<ENAMEX\u{a0}TYPE=\"MONEY\">", WhitespaceMode::Legacy).as_deref(), Some("MONEY"));
    assert_eq!(type_attribute("<ENAMEX\u{85}TYPE=\"MONEY\">", WhitespaceMode::Legacy), None);
}

#[test]
fn tag_names() {
    let m = WhitespaceMode::Unicode;
    assert_eq!(tag_name("<s>", m), "s");
    assert_eq!(tag_name("</s>", m), "s");
    assert_eq!(tag_name("<ENAMEX TYPE=\"X\">", m), "ENAMEX");
    assert_eq!(tag_name("<>", m), "");
    assert_eq!(tag_name("</>", m), "");
    assert_eq!(tag_name("< s>", m), "");
    assert_eq!(tag_name("<a/>", m), "a/");
}

#[test]
fn an_error_comes_before_the_samples_made_ahead_of_it() {
    assert_eq!(
        read("<s>a</s><s>b <ENAMEX TYPE=\"THING\">c</ENAMEX></s><s>d</s>"),
        [
            Err("Unknown timex, numex or namex type: THING, expected one of [PERSON, ORGANIZATION, LOCATION, DATE, TIME, MONEY, PERCENT]".to_owned()),
            Ok("a ".to_owned()),
        ]
    );
    assert_eq!(read("<s>a</s> > <s>b</s>"), [Err("Did not expect > char!".to_owned()), Ok("a ".to_owned())]);
    assert_eq!(read("<s <s>"), [Err("Did not expect < char!".to_owned())]);
    assert_eq!(read("<s>a</s><s"), [Err("Did not find matching > char!".to_owned()), Ok("a ".to_owned())]);
    assert_eq!(read("<s>a</ENAMEX></s>"), [Err("".to_owned())]);
    assert_eq!(read("<s><ENAMEX>a</ENAMEX></s>"), [Err("".to_owned())]);
}

#[test]
fn names_across_samples() {
    assert_eq!(
        read("<s>a b <ENAMEX TYPE=\"PERSON\">c</s><s>d</ENAMEX></s>"),
        [Err("start index must not be larger than end index: start=2, end=1".to_owned()), Ok("a b c".to_owned())]
    );
    assert_eq!(
        read("<s><ENAMEX TYPE=\"PERSON\">a</s><s>b c</ENAMEX></s>"),
        [Ok("a ".to_owned()), Ok("<START:person> b c <END>".to_owned())]
    );
    assert_eq!(
        read("<s>a</s><ENAMEX TYPE=\"PERSON\">x</ENAMEX><s>b</s>"),
        [Ok("a ".to_owned()), Ok("<START:person> <END> b ".to_owned())]
    );
}

#[test]
fn nested_names_overlap() {
    let text = "<s><ENAMEX TYPE=\"ORGANIZATION\"><ENAMEX TYPE=\"LOCATION\">a</ENAMEX> b</ENAMEX></s>";
    assert_eq!(
        read(text),
        [Err("name spans [0..2) organization and [0..1) location are overlapped in file: null, sentence: [a, b]"
            .to_owned())]
    );
}

#[test]
fn documents_end_at_one_without_samples() {
    let t = RuleTokenizer::default();
    let read = |docs: &[&'static str]| -> Vec<Result<String, String>> {
        MucSamples::new(docs.iter().copied(), &t, WhitespaceMode::Unicode)
            .map(|r| r.map(|s| s.to_string()).map_err(|e| e.to_string()))
            .collect()
    };
    assert_eq!(
        read(&["<s>a</s>", "<s", "<s>b</s>"]),
        [Ok("a ".to_owned()), Err("Did not find matching > char!".to_owned()), Ok("b ".to_owned())]
    );
    assert_eq!(read(&["<s>a</s>", "<DOC></DOC>", "<s>b</s>"]), [Ok("a ".to_owned())]);
}

#[test]
fn files_come_in_reverse_name_order() {
    let dir = std::env::temp_dir().join(format!("turbo-muc-files-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("d.sgm")).unwrap();
    for name in ["a.sgm", "B.SGM", "c.Sgm", "e.txt", "sgm"] {
        std::fs::write(dir.join(name), "").unwrap();
    }
    let names: Vec<_> =
        muc_files(&dir).unwrap().iter().map(|p| p.file_name().unwrap().to_str().unwrap().to_owned()).collect();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(names, ["c.Sgm", "a.sgm", "B.SGM"]);
}

#[test]
fn errors_name_their_lines() {
    let text = "<s>a</s>\r\n\n<s>\rb <ENAMEX TYPE=\"X\">";
    let e = MucReader::new(text, &RuleTokenizer::default(), WhitespaceMode::Unicode).unwrap().next().unwrap();
    assert_eq!(e.unwrap_err().line(), Some(4));
}
