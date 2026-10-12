//! Every document of up to three lines from a set of tricky lines, in both
//! syntaxes and with both line breaks: the layer is always accepted by a
//! document (a tree in text order, spans in bounds and on character
//! boundaries), every span is trimmed, every section starts with its
//! heading, and the same text always gives the same layer. Then a few
//! inputs built to be slow or deep.

use turbo_analysis::{Document, Key, Provenance, ViewId};
use turbo_analysis_layers::{Block, Structure};
use turbo_structure::{Syntax, detect};

const LINES: &[&str] = &[
    "",
    " ",
    "a",
    "  b",
    "    c",
    "\tc",
    "# h",
    "## h",
    "===",
    "---",
    "- x",
    "  - y",
    "1. n",
    "2) m",
    "-",
    "> q",
    ">",
    "> - z",
    "```",
    "~~~",
    "| a | b |",
    "|---|---|",
    "a|b",
    "`x|y`|z",
    "• w",
    "é",
    "\u{3000}",
    "INTRO",
    "    - deep",
];

fn holds(text: &str, syntax: Syntax) {
    let s = detect(text, syntax, ViewId::SOURCE);
    for (i, span) in s.spans.iter() {
        let t = &text[span.range()];
        assert_eq!(t.len(), t.trim().len(), "{text:?} {syntax:?}: block {i:?} is not trimmed");
        if s.kind[i.get()] == Block::Section {
            let next = i.get() + 1;
            assert_eq!(s.kind.get(next), Some(&Block::Heading), "{text:?}: a section without its heading");
            assert_eq!(s.spans.get(turbo_analysis::Ix::new(next as u32)).start, span.start);
        }
    }
    let again = detect(text, syntax, ViewId::SOURCE);
    assert_eq!(s.spans.columns(), again.spans.columns());
    assert_eq!(s.kind, again.kind);
    assert_eq!(s.parent, again.parent);
    assert_eq!(s.depth, again.depth);
    let mut doc = Document::new(text);
    if let Err(e) = doc.add(&Key::<Structure>::DEFAULT, s, Provenance::by("test")) {
        panic!("{text:?} {syntax:?}: {e}");
    }
}

#[test]
fn every_three_line_document() {
    let mut n = 0;
    for a in LINES {
        for b in LINES {
            for c in LINES {
                for brk in ["\n", "\r\n"] {
                    let text = [*a, *b, *c].join(brk);
                    holds(&text, Syntax::Markdown);
                    holds(&text, Syntax::Plain);
                    n += 1;
                }
            }
        }
    }
    assert_eq!(n, LINES.len().pow(3) * 2);
}

#[test]
fn deep_nesting_is_one_pass() {
    // Ten thousand nested quotes, then lines that continue them lazily.
    let mut text = "> ".repeat(10_000);
    text.push_str("a\n");
    for _ in 0..10_000 {
        text.push_str("lazy\n");
    }
    let s = detect(&text, Syntax::Markdown, ViewId::SOURCE);
    assert_eq!(s.spans.len(), 10_001);
    assert_eq!(s.depth[9_999], u8::MAX);
    holds(&text, Syntax::Markdown);

    // Nested list items, each indented two more columns.
    let text: String = (0..1_000).map(|i| format!("{}- x\n", "  ".repeat(i))).collect();
    let s = detect(&text, Syntax::Markdown, ViewId::SOURCE);
    assert!(s.kind.iter().filter(|&&k| k == Block::ListItem).count() > 1);
    holds(&text, Syntax::Markdown);
}

#[test]
fn many_backtick_runs_in_a_table_row() {
    // Runs of every length from 1 to 400, none closed: each is looked up,
    // not scanned for.
    let mut row = String::from("| ");
    for n in 1..=400 {
        row.push_str(&"`".repeat(n));
        row.push_str(" | ");
    }
    let text = format!("h | i\n-|-\n{row}\n");
    let s = detect(&text, Syntax::Markdown, ViewId::SOURCE);
    assert_eq!(s.kind.iter().filter(|&&k| k == Block::TableCell).count(), 2 + 400);
    holds(&text, Syntax::Markdown);
}

#[test]
fn a_long_document() {
    let mut text = String::new();
    for i in 0..5_000 {
        text.push_str(&format!(
            "## Part {i}\n\nSome text, a | b.\n\n- one\n- two\n\n```\n# x\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        ));
    }
    let s = detect(&text, Syntax::Markdown, ViewId::SOURCE);
    let sections = s.spans.iter().filter(|&(i, _)| s.kind[i.get()] == Block::Section).count();
    assert_eq!(sections, 5_000);
    holds(&text, Syntax::Markdown);
}
