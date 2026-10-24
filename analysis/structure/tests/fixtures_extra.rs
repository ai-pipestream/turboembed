mod support;
use support::*;
use turbo_structure::Syntax::Plain as PLAIN;

#[test]
fn item_indentation_counts_from_inside_the_quotes() {
    // CommonMark examples 259 and 260: the quote markers take five columns
    // on one line and three on another, and an item's continuation is
    // measured from where the quotes' content starts.
    check(
        "   > > 1.  one\n>>\n>>     two\n",
        MD,
        r########"
    Quote 1 "> > 1.  one\n>>\n>>     two"
      Quote 2 "> 1.  one\n>>\n>>     two"
        LooseList 1 "1.  one\n>>\n>>     two"
          ListItem 1 "1.  one\n>>\n>>     two"
            Paragraph 0 "one"
            Paragraph 0 "two"
    "########,
    );
    check(
        ">>- one\n>>\n  >  > two\n",
        MD,
        r########"
    Quote 1 ">>- one\n>>\n  >  > two"
      Quote 2 ">- one\n>>\n  >  > two"
        List 1 "- one"
          ListItem 1 "- one"
            Paragraph 0 "one"
        Paragraph 0 "two"
    "########,
    );
}

#[test]
fn crlf_and_lone_cr_end_lines() {
    let text = "# T\r\n\r\npara one\r\nline two\r\n\r\n- a\r- b\r\n";
    check(
        text,
        MD,
        r########"
    Section 1 "# T\r\n\r\npara one\r\nline two\r\n\r\n- a\r- b"
      Heading 1 "# T"
      Paragraph 0 "para one\r\nline two"
      List 1 "- a\r- b"
        ListItem 1 "- a"
          Paragraph 0 "a"
        ListItem 1 "- b"
          Paragraph 0 "b"
    "########,
    );
}

#[test]
fn unicode_text_and_whitespace() {
    // A line of U+3000 or U+00A0 is blank; a byte order mark is in no
    // block; spans end on whole characters.
    let text = "\u{FEFF}# Über\n\nΚαλημέρα κόσμε 😀\u{A0}\n\u{3000}\n- 東京\n- naïve\n";
    check(
        text,
        MD,
        r########"
    Section 1 "# Über\n\nΚαλημέρα κόσμε 😀\u{a0}\n\u{3000}\n- 東京\n- naïve"
      Heading 1 "# Über"
      Paragraph 0 "Καλημέρα κόσμε 😀"
      List 1 "- 東京\n- naïve"
        ListItem 1 "- 東京"
          Paragraph 0 "東京"
        ListItem 1 "- naïve"
          Paragraph 0 "naïve"
    "########,
    );
}

#[test]
fn empty_and_whitespace_only_documents_have_no_blocks() {
    for syntax in [MD, PLAIN] {
        for text in ["", " ", "\n\n", "\r\n", " \t\r\n\u{3000}\n\u{85}", "\u{FEFF}", "\u{FEFF}\n  \n"] {
            assert_eq!(outline(text, syntax), "", "{text:?}");
        }
    }
    // A thematic break alone is no block either.
    assert_eq!(outline("***\n---\n", MD), "");
}

#[test]
fn plain_text() {
    let text = "INTRODUCTION\n\n    This contract, indented with spaces,\nis between two parties.\n\nTerms\n-----\n1. The first term.\n2. The second.\n\n• bullet one\n• bullet two\n\n# not a heading\n";
    check(
        text,
        PLAIN,
        r########"
    Section 1 "INTRODUCTION\n\n    This contract, indented with spaces,\nis between two parties.\n\nTerms\n-----\n1. The first term.\n2. The second.\n\n• bullet one\n• bullet two\n\n# not a heading"
      Heading 1 "INTRODUCTION"
      Paragraph 0 "This contract, indented with spaces,\nis between two parties."
      Section 2 "Terms\n-----\n1. The first term.\n2. The second.\n\n• bullet one\n• bullet two\n\n# not a heading"
        Heading 2 "Terms\n-----"
        List 1 "1. The first term.\n2. The second."
          ListItem 1 "1. The first term."
            Paragraph 0 "The first term."
          ListItem 1 "2. The second."
            Paragraph 0 "The second."
        List 1 "• bullet one\n• bullet two"
          ListItem 1 "• bullet one"
            Paragraph 0 "bullet one"
          ListItem 1 "• bullet two"
            Paragraph 0 "bullet two"
        Paragraph 0 "# not a heading"
    "########,
    );
}

#[test]
fn the_same_text_as_markdown_has_code_and_no_capital_heading() {
    let text = "INTRODUCTION\n\n    This contract, indented with spaces,\nis between two parties.\n";
    check(
        text,
        MD,
        r########"
    Paragraph 0 "INTRODUCTION"
    Code 0 "This contract, indented with spaces,"
    Paragraph 0 "is between two parties."
    "########,
    );
}

#[test]
fn plain_text_headings_are_conservative() {
    // An underline under a paragraph of several lines is a separator; a
    // line in capitals over another in capitals, or ending like a
    // sentence, or inside a quote, is text.
    let text = "Some text\nmore text\n=====\nNext\n\nTHE END.\n\nWARNING\nREAD THIS\n\n> QUOTED HEADING\n";
    check(
        text,
        PLAIN,
        r########"
    Paragraph 0 "Some text\nmore text"
    Paragraph 0 "Next"
    Paragraph 0 "THE END."
    Paragraph 0 "WARNING\nREAD THIS"
    Quote 1 "> QUOTED HEADING"
      Paragraph 0 "QUOTED HEADING"
    "########,
    );
}

#[test]
fn a_plain_heading_in_capitals_over_its_paragraph() {
    let text =
        "INTRODUCTION\nThis report covers the year.\nIt has two parts.\n\nTERMS AND CONDITIONS\n1. The first term.\n";
    check(
        text,
        PLAIN,
        r########"
    Section 1 "INTRODUCTION\nThis report covers the year.\nIt has two parts."
      Heading 1 "INTRODUCTION"
      Paragraph 0 "This report covers the year.\nIt has two parts."
    Section 1 "TERMS AND CONDITIONS\n1. The first term."
      Heading 1 "TERMS AND CONDITIONS"
      List 1 "1. The first term."
        ListItem 1 "1. The first term."
          Paragraph 0 "The first term."
    "########,
    );
}

#[test]
fn a_letter_closing_opens_no_section() {
    // "OK" has too few letters for a heading; a name and a country in
    // capitals with nothing after them are a signature.
    let text = "Dear Sam,\n\nThe plan is fine.\n\nOK\n\nSincerely,\nJOHN SMITH\n\nJANE DOE\n\nUSA\n";
    check(
        text,
        PLAIN,
        r########"
    Paragraph 0 "Dear Sam,"
    Paragraph 0 "The plan is fine."
    Paragraph 0 "OK"
    Paragraph 0 "Sincerely,\nJOHN SMITH"
    Paragraph 0 "JANE DOE"
    Paragraph 0 "USA"
    "########,
    );
}

#[test]
fn a_signature_delimiter_is_no_underline() {
    // "-- " is the line before an email signature, not a setext underline
    // in plain text; "---" still is one.
    let text = "Thanks\n-- \nJohn Smith\n\nNotes\n---\nmore\n";
    check(
        text,
        PLAIN,
        r########"
    Paragraph 0 "Thanks\n-- \nJohn Smith"
    Section 2 "Notes\n---\nmore"
      Heading 2 "Notes\n---"
      Paragraph 0 "more"
    "########,
    );
}

#[test]
fn a_year_is_no_plain_list_item() {
    // Right under an item, the year's line continues the item's text, as
    // any unindented line does; after a blank line it is a paragraph. A
    // number of up to three digits is still an item.
    let text = "- first\n- second\n2024. A great year\n\n2025. Another\n\n100. The hundredth\n";
    check(
        text,
        PLAIN,
        r########"
    List 1 "- first\n- second\n2024. A great year"
      ListItem 1 "- first"
        Paragraph 0 "first"
      ListItem 1 "- second\n2024. A great year"
        Paragraph 0 "second\n2024. A great year"
    Paragraph 0 "2025. Another"
    List 1 "100. The hundredth"
      ListItem 1 "100. The hundredth"
        Paragraph 0 "The hundredth"
    "########,
    );
    check(
        "1. a\n2024. b\n",
        MD,
        r########"
    List 1 "1. a\n2024. b"
      ListItem 1 "1. a"
        Paragraph 0 "a"
      ListItem 1 "2024. b"
        Paragraph 0 "b"
    "########,
    );
}

#[test]
fn markdown_front_matter_is_in_no_block() {
    let text = "\u{FEFF}---\ntitle: A\ntags: [x]\n...  \n# Body\n\ntext\n";
    check(
        text,
        MD,
        r########"
    Section 1 "# Body\n\ntext"
      Heading 1 "# Body"
      Paragraph 0 "text"
    "########,
    );
    // Closed with "---" as well; without a closing line, or not at the
    // start, it is no front matter. Plain text has none.
    check(
        "---\na: 1\n---\ntext\n",
        MD,
        r########"
    Paragraph 0 "text"
    "########,
    );
    check(
        "---\na: 1\n",
        MD,
        r########"
    Paragraph 0 "a: 1"
    "########,
    );
    check(
        "text\n\n---\na: 1\n---\n",
        MD,
        r########"
    Paragraph 0 "text"
    Section 2 "a: 1\n---"
      Heading 2 "a: 1\n---"
    "########,
    );
    check(
        "---\na: 1\n---\ntext\n",
        PLAIN,
        r########"
    Section 2 "a: 1\n---\ntext"
      Heading 2 "a: 1\n---"
      Paragraph 0 "text"
    "########,
    );
}
