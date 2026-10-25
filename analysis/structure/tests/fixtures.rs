mod support;
use support::*;

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
        LooseList 2 "- inner a\n   - inner b\n\n     continued inner b"
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
    LooseList 1 "- a\n\n- c"
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
