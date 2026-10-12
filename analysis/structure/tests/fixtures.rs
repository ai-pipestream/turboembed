//! Hand-labelled documents: each one's whole structure, written out as an
//! outline, one block per line, indented under its parent, with its kind,
//! its depth and the exact text its span covers. Every structure is also
//! added to a document, so the layer's own check (a tree in text order)
//! and the span checks (in bounds, on character boundaries) run on it.

use turbo_analysis::{Document, Key, Provenance, ViewId};
use turbo_analysis_layers::Structure;
use turbo_structure::{Syntax, detect};

fn outline(text: &str, syntax: Syntax) -> String {
    let s = detect(text, syntax, ViewId::SOURCE);
    let mut level = vec![0usize; s.spans.len()];
    let mut out = String::new();
    for (i, span) in s.spans.iter() {
        let i = i.get();
        level[i] = s.parent[i].map_or(0, |p| level[p.get()] + 1);
        let pad = "  ".repeat(level[i]);
        out.push_str(&format!("{pad}{:?} {} {:?}\n", s.kind[i], s.depth[i], &text[span.range()]));
    }
    let mut doc = Document::new(text);
    doc.add(&Key::<Structure>::DEFAULT, s, Provenance::by("test")).unwrap();
    out
}

#[track_caller]
fn check(text: &str, syntax: Syntax, want: &str) {
    let want: String = want.lines().filter(|l| !l.trim().is_empty()).map(|l| format!("{}\n", &l[4..])).collect();
    assert_eq!(outline(text, syntax), want, "\n{}", outline(text, syntax));
}

const MD: Syntax = Syntax::Markdown;
const PLAIN: Syntax = Syntax::Plain;

#[test]
fn a_markdown_document() {
    let text = "# Title\n\nIntro para\ntwo lines.\n\n## Part A\n\n- one\n- two\n  - nested\n- three\n\n## Part B\n\n> quoted\n> text\n";
    check(
        text,
        MD,
        r########"
    Section 1 "# Title\n\nIntro para\ntwo lines.\n\n## Part A\n\n- one\n- two\n  - nested\n- three\n\n## Part B\n\n> quoted\n> text"
      Heading 1 "# Title"
      Paragraph 0 "Intro para\ntwo lines."
      Section 2 "## Part A\n\n- one\n- two\n  - nested\n- three"
        Heading 2 "## Part A"
        List 1 "- one\n- two\n  - nested\n- three"
          ListItem 1 "- one"
            Paragraph 0 "one"
          ListItem 1 "- two\n  - nested"
            Paragraph 0 "two"
            List 2 "- nested"
              ListItem 2 "- nested"
                Paragraph 0 "nested"
          ListItem 1 "- three"
            Paragraph 0 "three"
      Section 2 "## Part B\n\n> quoted\n> text"
        Heading 2 "## Part B"
        Quote 1 "> quoted\n> text"
          Paragraph 0 "quoted\n> text"
    "########,
    );
}

#[test]
fn setext_and_atx_headings_make_one_hierarchy() {
    // "Para / with lines" underlined with dashes is a level 2 heading, so
    // it closes both the level 3 section and the level 2 one around it.
    let text = "Title\n=====\n\nSub\n---\ntext\n### Three ###\nPara\nwith lines\n---\n";
    check(
        text,
        MD,
        r########"
    Section 1 "Title\n=====\n\nSub\n---\ntext\n### Three ###\nPara\nwith lines\n---"
      Heading 1 "Title\n====="
      Section 2 "Sub\n---\ntext\n### Three ###"
        Heading 2 "Sub\n---"
        Paragraph 0 "text"
        Section 3 "### Three ###"
          Heading 3 "### Three ###"
      Section 2 "Para\nwith lines\n---"
        Heading 2 "Para\nwith lines\n---"
    "########,
    );
}

#[test]
fn headings_below_the_top_open_sections_at_the_top() {
    // No level 1 heading: the level 2 sections are the top level, and a
    // later level 1 heading closes them. Text before the first heading is
    // in no section; "#nope" and "#######" are not headings.
    let text = "preamble\n\n## A\n#nope\n## B\n# C\n####### x\n";
    check(
        text,
        MD,
        r########"
    Paragraph 0 "preamble"
    Section 2 "## A\n#nope"
      Heading 2 "## A"
      Paragraph 0 "#nope"
    Section 2 "## B"
      Heading 2 "## B"
    Section 1 "# C\n####### x"
      Heading 1 "# C"
      Paragraph 0 "####### x"
    "########,
    );
}

#[test]
fn code_fences_hold_heading_and_table_lines() {
    let text = "# Real\n\n```md\n# not a heading\n- not a list\n| a | b |\n|---|---|\n```\n\n~~~\n```\nstill code\n~~~\n    indented code\n    # also code\n\nafter\n";
    check(
        text,
        MD,
        r########"
    Section 1 "# Real\n\n```md\n# not a heading\n- not a list\n| a | b |\n|---|---|\n```\n\n~~~\n```\nstill code\n~~~\n    indented code\n    # also code\n\nafter"
      Heading 1 "# Real"
      Code 0 "```md\n# not a heading\n- not a list\n| a | b |\n|---|---|\n```"
      Code 0 "~~~\n```\nstill code\n~~~"
      Code 0 "indented code\n    # also code"
      Paragraph 0 "after"
    "########,
    );
}

#[test]
fn an_unclosed_fence_runs_to_the_end_of_its_container() {
    check(
        "```\ncode\n\n# x\n",
        MD,
        r########"
    Code 0 "```\ncode\n\n# x"
    "########,
    );
    // Inside a quote, a line without ">" ends the quote and the fence.
    check(
        "> ```\n> # in code\n# out\n",
        MD,
        r########"
    Quote 1 "> ```\n> # in code"
      Code 0 "```\n> # in code"
    Section 1 "# out"
      Heading 1 "# out"
    "########,
    );
}

#[test]
fn indented_code_keeps_inner_blank_lines_but_not_trailing_ones() {
    check(
        "    a\n\n    b\n\n\nc\n",
        MD,
        r########"
    Code 0 "a\n\n    b"
    Paragraph 0 "c"
    "########,
    );
    // An indented line cannot interrupt a paragraph.
    check(
        "para\n    more\n",
        MD,
        r########"
    Paragraph 0 "para\n    more"
    "########,
    );
}

#[test]
fn nested_and_mixed_lists() {
    let text =
        "1. first\n2. second\n   - inner a\n   - inner b\n\n     continued inner b\n3) other list\n-\tTab item\n";
    check(
        text,
        MD,
        r########"
    List 1 "1. first\n2. second\n   - inner a\n   - inner b\n\n     continued inner b"
      ListItem 1 "1. first"
        Paragraph 0 "first"
      ListItem 1 "2. second\n   - inner a\n   - inner b\n\n     continued inner b"
        Paragraph 0 "second"
        List 2 "- inner a\n   - inner b\n\n     continued inner b"
          ListItem 2 "- inner a"
            Paragraph 0 "inner a"
          ListItem 2 "- inner b\n\n     continued inner b"
            Paragraph 0 "inner b"
            Paragraph 0 "continued inner b"
    List 1 "3) other list"
      ListItem 1 "3) other list"
        Paragraph 0 "other list"
    List 1 "-\tTab item"
      ListItem 1 "-\tTab item"
        Paragraph 0 "Tab item"
    "########,
    );
}

#[test]
fn list_items_with_tabs_and_lazy_lines() {
    // A tab indents a nested item past its parent's content; a line not
    // indented enough continues the paragraph lazily.
    let text = "- a\n\t- b\nlazy\n- c\n";
    check(
        text,
        MD,
        r########"
    List 1 "- a\n\t- b\nlazy\n- c"
      ListItem 1 "- a\n\t- b\nlazy"
        Paragraph 0 "a"
        List 2 "- b\nlazy"
          ListItem 2 "- b\nlazy"
            Paragraph 0 "b\nlazy"
      ListItem 1 "- c"
        Paragraph 0 "c"
    "########,
    );
}

#[test]
fn what_may_interrupt_a_paragraph() {
    // An ordered item interrupts a paragraph only when it starts at 1, an
    // empty item never does (a lone "-" underlines a heading instead), and
    // a thematic break always does.
    check("a\n2. b\n", MD, "    Paragraph 0 \"a\\n2. b\"\n");
    check(
        "a\n1. b\n",
        MD,
        r########"
    Paragraph 0 "a"
    List 1 "1. b"
      ListItem 1 "1. b"
        Paragraph 0 "b"
    "########,
    );
    check(
        "a\n-\nb\n",
        MD,
        r########"
    Section 2 "a\n-\nb"
      Heading 2 "a\n-"
      Paragraph 0 "b"
    "########,
    );
    check(
        "a\n* * *\nb\n",
        MD,
        r########"
    Paragraph 0 "a"
    Paragraph 0 "b"
    "########,
    );
}

#[test]
fn a_thematic_break_ends_a_list() {
    check(
        "- a\n---\n- c\n",
        MD,
        r########"
    List 1 "- a"
      ListItem 1 "- a"
        Paragraph 0 "a"
    List 1 "- c"
      ListItem 1 "- c"
        Paragraph 0 "c"
    "########,
    );
    // A blank line does not: the items are one loose list.
    check(
        "- a\n\n- c\n",
        MD,
        r########"
    List 1 "- a\n\n- c"
      ListItem 1 "- a"
        Paragraph 0 "a"
      ListItem 1 "- c"
        Paragraph 0 "c"
    "########,
    );
}

#[test]
fn an_item_starts_with_at_most_one_blank_line() {
    check(
        "-\n  foo\n",
        MD,
        r########"
    List 1 "-\n  foo"
      ListItem 1 "-\n  foo"
        Paragraph 0 "foo"
    "########,
    );
    check(
        "-\n\n  foo\n",
        MD,
        r########"
    List 1 "-"
      ListItem 1 "-"
    Paragraph 0 "foo"
    "########,
    );
}

#[test]
fn tables_with_escaped_pipes_and_pipes_in_code() {
    let text =
        "Intro line\n| Name | Cmd |\n|------|:---:|\n| grep | `a | b` |\n| esc  | x \\| y |\n| short |\n\nafter\n";
    check(
        text,
        MD,
        r########"
    Paragraph 0 "Intro line"
    Table 0 "| Name | Cmd |\n|------|:---:|\n| grep | `a | b` |\n| esc  | x \\| y |\n| short |"
      TableRow 0 "| Name | Cmd |"
        TableCell 0 "Name"
        TableCell 0 "Cmd"
      TableRow 0 "| grep | `a | b` |"
        TableCell 0 "grep"
        TableCell 0 "`a | b`"
      TableRow 0 "| esc  | x \\| y |"
        TableCell 0 "esc"
        TableCell 0 "x \\| y"
      TableRow 0 "| short |"
        TableCell 0 "short"
    Paragraph 0 "after"
    "########,
    );
}

#[test]
fn a_table_needs_a_header_as_wide_as_its_delimiter_row() {
    check("a | b\n|---|\n", MD, "    Paragraph 0 \"a | b\\n|---|\"\n");
    // Without a pipe the dashes underline a heading.
    check(
        "a | b\n---\n",
        MD,
        r########"
    Section 2 "a | b\n---"
      Heading 2 "a | b\n---"
    "########,
    );
    // A table ends at a block that starts, and empty cells keep their place.
    check(
        "x | y\n--|--\n| | 2\n# h\n",
        MD,
        r########"
    Table 0 "x | y\n--|--\n| | 2"
      TableRow 0 "x | y"
        TableCell 0 "x"
        TableCell 0 "y"
      TableRow 0 "| | 2"
        TableCell 0 ""
        TableCell 0 "2"
    Section 1 "# h"
      Heading 1 "# h"
    "########,
    );
}

#[test]
fn quotes_nest_and_take_lazy_lines() {
    let text = "> # In quote\n> para\nlazy line\n> > deeper\n>\n> - item\n";
    check(
        text,
        MD,
        r########"
    Quote 1 "> # In quote\n> para\nlazy line\n> > deeper\n>\n> - item"
      Heading 1 "# In quote"
      Paragraph 0 "para\nlazy line"
      Quote 2 "> deeper"
        Paragraph 0 "deeper"
      List 1 "- item"
        ListItem 1 "- item"
          Paragraph 0 "item"
    "########,
    );
}

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
        List 1 "1.  one\n>>\n>>     two"
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
