use super::*;

fn read(s: &str) -> Vec<Result<Sentence<'_>, FormatError>> {
    ConlluReader::new(s).unwrap().collect()
}

fn line(id: &str, form: &str, lemma: &str, upos: &str) -> String {
    format!("{id}\t{form}\t{lemma}\t{upos}\t_\t_\t_\t_\t_\t_\n")
}

#[test]
fn comments_and_multiword_tokens() {
    let src = format!(
        "\n\n# newdoc id = d1\n  # sent_id = s1 \n# text = Vamos al mar.\n# text_en = We go\n# text_es =\n#newpar\n{}{}{}{}{}\n\n",
        line("1", "Vamos", "ir", "VERB"),
        line("2-3", "al", "_", "_"),
        line("2", "a", "a", "ADP"),
        line("3", "el", "_", "DET"),
        line("4", "mar", "mar", "NOUN"),
    );
    let sentences = read(&src);
    assert_eq!(sentences.len(), 1);
    let s = sentences[0].as_ref().unwrap();
    assert!(s.new_document && s.new_paragraph);
    assert_eq!(s.document_id.as_ref().unwrap().as_str(), "d1");
    assert_eq!(s.sentence_id.as_ref().unwrap().as_str(), "s1");
    assert_eq!(s.text.as_ref().unwrap().as_str(), "Vamos al mar.");
    let langs: Vec<(&str, &str)> = s.text_lang.iter().map(|t| (t.language.as_str(), t.text.as_str())).collect();
    assert_eq!(langs, [("en", "We go"), ("es", "")]);
    let forms: Vec<&str> = s.word_lines.iter().map(|l| l.form.as_str()).collect();
    assert_eq!(forms, ["Vamos", "al", "mar"]);
    let al = &s.word_lines[1];
    assert_eq!(al.lemma.as_str(), "a");
    assert!(al.lemma.is_borrowed());
    assert_eq!(&src[al.lemma.span.range()], "a");
    assert_eq!(al.upos.as_str(), "ADP+DET");
    assert!(!al.upos.is_borrowed());
    assert_eq!(&src[al.upos.span.range()], "ADP\t_\t_\t_\t_\t_\t_\n3\tel\t_\tDET");
    assert_eq!(al.xpos.as_str(), "");
    assert_eq!(al.line, 10);
}

#[test]
fn a_line_of_spaces_is_a_word_line() {
    let src = format!("{} \n{}", line("1", "a", "a", "X"), line("1", "b", "b", "X"));
    let sentences = read(&src);
    assert_eq!(sentences.len(), 1);
    assert_eq!(sentences[0], Err(FormatError::ConlluColumns { line: 2, text: " ".into() }));
}

#[test]
fn multiword_ids() {
    assert!(matches!(multiword_range("1-2"), Ok((1, 2))));
    assert!(matches!(multiword_range("3-3"), Ok((3, 3))));
    for bad in ["-2", "1-", "01-2", "1-02", "1-2-3", "a-2", "1-2a", "2147483648-2147483649", "1--2"] {
        assert!(matches!(multiword_range(bad), Err(RangeError::Invalid)), "{bad}");
    }
    assert!(matches!(multiword_range("2147483647-2147483647"), Ok((2147483647, 2147483647))));
    assert!(matches!(multiword_range("3-2"), Err(RangeError::Backwards)));
}

#[test]
fn overlapping_ranges_remove_past_the_end() {
    let src = format!(
        "{}{}{}{}",
        line("1-2", "ab", "_", "_"),
        line("1-2", "ab", "_", "_"),
        line("1", "a", "a", "X"),
        line("2", "b", "b", "Y")
    );
    assert_eq!(read(&src)[0], Err(FormatError::MultiwordRemoval { line: 1, index: 3, len: 2 }));
}

#[test]
fn ranges_out_of_order_remove_other_lines() {
    // The words of 3-4 are at places 4 and 5 and those of 1-2 at 1 and
    // 2; removing 2, 1, 5 and 4 in that order removes the words of 1-2
    // and then the lines that moved to places 5 and 4: 6 and 5.
    let src = format!(
        "{}{}{}{}{}{}{}{}",
        line("3-4", "cd", "_", "_"),
        line("1", "a", "a", "X"),
        line("2", "b", "b", "X"),
        line("1-2", "ab", "_", "_"),
        line("3", "c", "c", "Y"),
        line("4", "d", "d", "Y"),
        line("5", "e", "e", "Z"),
        line("6", "f", "f", "Z"),
    );
    let s = read(&src).remove(0).unwrap();
    let forms: Vec<&str> = s.word_lines.iter().map(|l| l.form.as_str()).collect();
    assert_eq!(forms, ["cd", "ab", "c", "d"]);
}

#[test]
fn missing_words_and_bad_languages() {
    let src = format!("{}{}", line("1-2", "ab", "_", "_"), line("1", "a", "a", "X"));
    assert_eq!(read(&src)[0], Err(FormatError::MissingMultiwordLine { line: 1, id: "1-2".into(), missing: 2 }));
    let src = format!("# text_X = a\n{}", line("1", "a", "a", "X"));
    assert_eq!(read(&src)[0], Err(FormatError::InvalidLanguage { line: 1, key: "text_X".into() }));
    let key = "text_e text_abcd";
    let piece = Piece { text: key, span: Span::new(0, key.len() as u32) };
    assert_eq!(text_language(piece).map(|p| p.text), Some("abc"));
}
