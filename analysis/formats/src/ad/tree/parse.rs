use turbo_unicode::java::is_letter_or_digit;

use super::{
    BIZARRE_TAG_SEPARATOR, BRACKET_CLOSE, BRACKET_OPEN, Element, GROUP_OPEN, LEVEL_MARK, Leaf, Scan, TAG_SEPARATOR,
    Tag, is_quote,
};
use crate::source::Piece;

/// The element of one tree line and its level, or `None` for a line that
/// is no element (`SentenceParser.getElement`).
pub(crate) fn parse_line(line: Piece<'_>) -> Option<(u32, Element<'_>)> {
    let mut scan = Scan::new(line);
    if let Some(node) = parse_node(&scan) {
        return Some(node);
    }
    if let Some(leaf) = parse_leaf(&mut scan) {
        return Some(leaf);
    }
    let text = line.text;
    if let Some((level, lexeme)) = parse_punctuation(line) {
        let leaf = Leaf {
            syntactic: None,
            functional: None,
            lemma: None,
            secondary: None,
            morphological: None,
            lexeme,
            line: line.span,
        };
        return Some((level, Element::Leaf(leaf)));
    }
    if text == "_" || text.starts_with("<lixo") || text.starts_with("pause") {
        return None;
    }
    if text.as_bytes().first() == Some(&LEVEL_MARK) {
        if let Some(leaf) = parse_bizarre_leaf(&mut scan) {
            return Some(leaf);
        }
        let last = text.rfind('=').expect("the line starts with one");
        if last + 1 == text.len() {
            return None;
        }
        let lexeme = line.slice(last + 1, text.len());
        if is_word_with_markup(lexeme.text) {
            return None;
        }
        // Java's level is the index after the last `=`, in UTF-16 units.
        let level = turbo_jvm::chars::utf16_len(&text[..last]) as u32 + 2;
        return Some((level, Element::Leaf(empty_tagged(lexeme, line))));
    }
    // "Couldn't parse leaf": the whole line is the lexeme.
    Some((1, Element::Leaf(empty_tagged(line, line))))
}

/// A leaf whose syntactic, functional and morphological tags are empty
/// strings, which Java makes of a line no rule reads.
fn empty_tagged<'a>(lexeme: Piece<'a>, line: Piece<'a>) -> Leaf<'a> {
    let empty = lexeme.slice(0, 0);
    Leaf {
        syntactic: Some(empty),
        functional: Some(empty),
        lemma: None,
        secondary: None,
        morphological: Some(empty),
        lexeme,
        line: line.span,
    }
}

/// `parseNode`: the level prefix, a tag with a colon, an optional part in
/// parentheses and optional tag groups.
fn parse_node<'a>(scan: &Scan<'a>) -> Option<(u32, Element<'a>)> {
    let prefix = scan.level_prefix_end();
    let mut tag = scan.level_and_tag(prefix, TAG_SEPARATOR, Some(prefix));
    while let Some(t) = tag {
        if scan.is_node_tail(t.end) {
            let node = Element::Node { tag: Some(scan.piece(t.start, t.end)), children: Vec::new() };
            return Some((t.start as u32 + 1, node));
        }
        tag = scan.level_and_tag(prefix, TAG_SEPARATOR, t.start.checked_sub(1));
    }
    None
}

/// `parseLeaf`: the level prefix, a syntactic tag, a colon, a functional
/// tag, and in parentheses a quoted lemma, secondary tags and a
/// morphological tag, then whitespace and the lexeme.
fn parse_leaf<'a>(scan: &mut Scan<'a>) -> Option<(u32, Element<'a>)> {
    let prefix = scan.level_prefix_end();
    let mut tag = scan.level_and_tag(prefix, TAG_SEPARATOR, Some(prefix));
    while let Some(t) = tag {
        let b = scan.bytes;
        if t.end + 1 < b.len()
            && b[t.end] == GROUP_OPEN
            && is_quote(b[t.end + 1])
            && let Some(leaf) = parse_leaf_after_tag(scan, t)
        {
            return Some(leaf);
        }
        tag = scan.level_and_tag(prefix, TAG_SEPARATOR, t.start.checked_sub(1));
    }
    None
}

/// The rest of a leaf line after its tag: the longest lemma after which
/// the rest still reads.
fn parse_leaf_after_tag<'a>(scan: &mut Scan<'a>, tag: Tag) -> Option<(u32, Element<'a>)> {
    let lemma_start = tag.end + 2;
    let mut lemma_end = scan.len().checked_sub(1)?;
    while lemma_end > lemma_start {
        if is_quote(scan.bytes[lemma_end]) {
            let tags_start = scan.skip_ws(lemma_end + 1);
            if let Some((tags_end, end)) = scan.secondary_tags(tags_start) {
                let leaf = Leaf {
                    syntactic: Some(scan.piece(tag.start, tag.separator)),
                    functional: Some(scan.piece(tag.separator + 1, tag.end)),
                    lemma: Some(scan.piece(lemma_start, lemma_end)),
                    secondary: Some(scan.piece(tags_start, tags_end)),
                    morphological: scan.morphology(end),
                    lexeme: scan.piece(end.lexeme_start, scan.len()),
                    line: scan.line.span,
                };
                return Some((tag.start as u32 + 1, Element::Leaf(leaf)));
            }
        }
        lemma_end -= 1;
    }
    None
}

/// `parseBizarreLeaf`: a leaf line with `=` in place of the colon of its
/// tag, then in parentheses an optional quoted lemma and an optional
/// morphological tag, whitespace and the lexeme.
fn parse_bizarre_leaf<'a>(scan: &mut Scan<'a>) -> Option<(u32, Element<'a>)> {
    let prefix = scan.level_prefix_end();
    let mut tag = scan.level_and_tag(prefix, BIZARRE_TAG_SEPARATOR, Some(prefix));
    while let Some(t) = tag {
        if scan.bytes.get(t.end) == Some(&GROUP_OPEN)
            && let Some(leaf) = parse_bizarre_leaf_after_tag(scan, t)
        {
            return Some(leaf);
        }
        tag = scan.level_and_tag(prefix, BIZARRE_TAG_SEPARATOR, t.start.checked_sub(1));
    }
    None
}

fn parse_bizarre_leaf_after_tag<'a>(scan: &mut Scan<'a>, tag: Tag) -> Option<(u32, Element<'a>)> {
    let open = tag.end + 1;
    let mut lemma = None;
    let mut end = None;
    if open < scan.len() && is_quote(scan.bytes[open]) {
        let mut lemma_end = scan.len() - 1;
        while lemma_end > open + 1 && end.is_none() {
            if is_quote(scan.bytes[lemma_end]) {
                end = scan.leaf_end(lemma_end + 1);
                if end.is_some() {
                    lemma = Some(scan.piece(open + 1, lemma_end));
                }
            }
            lemma_end -= 1;
        }
    }
    let end = match end {
        Some(end) => end,
        None => scan.leaf_end(open)?,
    };
    let whole = scan.piece(tag.start, tag.end);
    let (syntactic, functional) = match whole.text.rfind(':') {
        Some(colon) if colon > 0 && colon + 1 < whole.text.len() => {
            (whole.slice(0, colon), Some(whole.slice(colon + 1, whole.text.len())))
        }
        _ => (whole, None),
    };
    let leaf = Leaf {
        syntactic: Some(syntactic),
        functional,
        lemma,
        secondary: None,
        morphological: scan.morphology(end),
        lexeme: scan.piece(end.lexeme_start, scan.len()),
        line: scan.line.span,
    };
    Some((tag.start as u32 + 1, Element::Leaf(leaf)))
}

/// `parsePunctuationLine`: a line with no letter, decimal digit or
/// underscore. Its level is one more than its leading `=`, and its lexeme
/// the rest; a line of `=` alone has the last one as its lexeme and the
/// count as its level.
fn parse_punctuation(line: Piece<'_>) -> Option<(u32, Piece<'_>)> {
    let text = line.text;
    if text.is_empty() || text.chars().any(|c| is_letter_or_digit(c) || c == '_') {
        return None;
    }
    let equals = text.bytes().take_while(|&b| b == LEVEL_MARK).count();
    if equals == text.len() {
        return Some((equals as u32, line.slice(equals - 1, equals)));
    }
    Some((equals as u32 + 1, line.slice(equals, text.len())))
}

/// `isWordWithMarkup`: a lexeme that starts with a letter, a digit or an
/// underscore and has a period or an angle bracket after that.
fn is_word_with_markup(lexeme: &str) -> bool {
    let mut chars = lexeme.chars();
    match chars.next() {
        Some(c) if is_letter_or_digit(c) || c == '_' => {}
        _ => return false,
    }
    chars.as_str().bytes().any(|b| b == b'.' || b == BRACKET_OPEN || b == BRACKET_CLOSE)
}
