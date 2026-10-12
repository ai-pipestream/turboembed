//! The line starts that open or close a block, each tested at `p`, the
//! first byte after a line's indentation (at most three columns of it),
//! with `e` the end of the line's content. Every marker is ASCII, except
//! the bullet `•` that plain text takes, so the offsets they give are on
//! character boundaries.

/// An ATX heading's level: one to six `#`, then a space, a tab or the end
/// of the line.
pub(crate) fn atx(b: &[u8], p: usize, e: usize) -> Option<u8> {
    let n = b[p..e].iter().take_while(|&&x| x == b'#').count();
    let after = p + n;
    ((1..=6).contains(&n) && (after == e || b[after] == b' ' || b[after] == b'\t')).then_some(n as u8)
}

/// A code fence: three or more backticks or tildes. A backtick fence's
/// info string holds no backtick, so a line like "```a``b" is text.
pub(crate) fn fence(b: &[u8], p: usize, e: usize) -> Option<(u8, usize)> {
    let ch = b[p];
    if ch != b'`' && ch != b'~' {
        return None;
    }
    let n = b[p..e].iter().take_while(|&&x| x == ch).count();
    if n < 3 || (ch == b'`' && b[p + n..e].contains(&b'`')) {
        return None;
    }
    Some((ch, n))
}

/// Whether the line closes a fence opened with `len` of `ch`: at least as
/// many of the same character and nothing after them but spaces and tabs.
pub(crate) fn closes_fence(b: &[u8], p: usize, e: usize, ch: u8, len: usize) -> bool {
    let n = b[p..e].iter().take_while(|&&x| x == ch).count();
    n >= len && b[p + n..e].iter().all(|&x| x == b' ' || x == b'\t')
}

/// A thematic break: three or more of one of `-`, `*`, `_`, with only
/// spaces and tabs between and after them.
#[cfg(test)]
fn thematic(b: &[u8], p: usize, e: usize) -> bool {
    thematic_scan(b, p, e) == Ok(true)
}

/// Whether a thematic break is at `p`, or `Err` with the first byte that
/// is not the break's character, a space or a tab, when one comes before
/// `e`: a scan from any later byte of the same character stops there too.
pub(crate) fn thematic_scan(b: &[u8], p: usize, e: usize) -> Result<bool, usize> {
    let ch = b[p];
    if ch != b'-' && ch != b'*' && ch != b'_' {
        return Ok(false);
    }
    let mut n = 0;
    for (i, &x) in b[p..e].iter().enumerate() {
        if x == ch {
            n += 1;
        } else if x != b' ' && x != b'\t' {
            return Err(p + i);
        }
    }
    Ok(n >= 3)
}

/// A setext underline's heading level and the length of its run: a run of
/// `=` (level 1) or `-` (level 2) and nothing after it but spaces and
/// tabs.
pub(crate) fn setext(b: &[u8], p: usize, e: usize) -> Option<(u8, usize)> {
    let level = match b[p] {
        b'=' => 1,
        b'-' => 2,
        _ => return None,
    };
    let n = b[p..e].iter().take_while(|&&x| x == b[p]).count();
    b[p + n..e].iter().all(|&x| x == b' ' || x == b'\t').then_some((level, n))
}

/// A list item marker.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct ListMarker {
    /// Items with the same kind are one list: the bullet character, or
    /// the delimiter after an ordered item's number.
    pub kind: char,
    /// An ordered item's number, which must be 1 for the item to interrupt
    /// a paragraph.
    pub number: Option<u32>,
    pub bytes: usize,
    pub cols: usize,
}

/// A bullet (`-`, `+`, `*`, and `•` in plain text) or one to nine digits
/// (one to three in plain text, where a year starting a sentence is no
/// item) and `.` or `)`, followed by a space, a tab or the end of the
/// line.
pub(crate) fn list_marker(text: &str, p: usize, e: usize, plain: bool) -> Option<ListMarker> {
    let b = text.as_bytes();
    let (kind, number, bytes, cols) = match b[p] {
        c @ (b'-' | b'+' | b'*') => (c as char, None, 1, 1),
        b'0'..=b'9' => {
            let digits = b[p..e].iter().take_while(|x| x.is_ascii_digit()).count();
            let delim = *b.get(p + digits).filter(|_| p + digits < e)?;
            if digits > if plain { 3 } else { 9 } || (delim != b'.' && delim != b')') {
                return None;
            }
            let number = text[p..p + digits].parse().ok()?;
            (delim as char, Some(number), digits + 1, digits + 1)
        }
        _ if plain && text[p..e].starts_with('•') => ('•', None, '•'.len_utf8(), 1),
        _ => return None,
    };
    let after = p + bytes;
    (after == e || b[after] == b' ' || b[after] == b'\t').then_some(ListMarker { kind, number, bytes, cols })
}

/// Whether a line of plain text is a heading by its look alone: short, in
/// capitals (at least three letters, an upper case one among them, no
/// lower case one, so "OK" is not one), and not ending like a sentence or a label. Scripts without
/// case never qualify.
pub(crate) fn caps_heading(line: &str) -> bool {
    if line.chars().count() > 80 || line.ends_with(['.', ',', ';', ':']) {
        return false;
    }
    let (mut letters, mut upper) = (0, false);
    for c in line.chars() {
        if c.is_lowercase() {
            return false;
        }
        if c.is_alphabetic() {
            letters += 1;
            upper |= c.is_uppercase();
        }
    }
    letters >= 3 && upper
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at<T>(s: &str, f: impl Fn(&[u8], usize, usize) -> T) -> T {
        f(s.as_bytes(), 0, s.len())
    }

    #[test]
    fn atx_headings() {
        assert_eq!(at("# a", atx), Some(1));
        assert_eq!(at("###### a", atx), Some(6));
        assert_eq!(at("#", atx), Some(1));
        assert_eq!(at("##\ta", atx), Some(2));
        assert_eq!(at("####### a", atx), None);
        assert_eq!(at("#hashtag", atx), None);
    }

    #[test]
    fn fences() {
        assert_eq!(at("```rust", fence), Some((b'`', 3)));
        assert_eq!(at("~~~~", fence), Some((b'~', 4)));
        assert_eq!(at("~~~ a`b", fence), Some((b'~', 3)));
        assert_eq!(at("``", fence), None);
        assert_eq!(at("``` a`b", fence), None);
        let b = b"````  ";
        assert!(closes_fence(b, 0, b.len(), b'`', 3));
        assert!(!closes_fence(b, 0, b.len(), b'`', 5));
        assert!(!closes_fence(b"``` x", 0, 5, b'`', 3));
        assert!(!closes_fence(b"~~~", 0, 3, b'`', 3));
    }

    #[test]
    fn breaks_and_underlines() {
        assert!(at("---", thematic));
        assert!(at("* * *", thematic));
        assert!(at("_\t_ _  ", thematic));
        assert!(!at("--", thematic));
        assert!(!at("--- a", thematic));
        assert!(!at("-*-", thematic));
        assert_eq!(at("===", setext), Some((1, 3)));
        assert_eq!(at("- ", setext), Some((2, 1)));
        assert_eq!(at("= =", setext), None);
        assert_eq!(at("--- a", setext), None);
    }

    #[test]
    fn list_markers() {
        let m = |s: &str, plain| list_marker(s, 0, s.len(), plain);
        assert_eq!(m("- a", false), Some(ListMarker { kind: '-', number: None, bytes: 1, cols: 1 }));
        assert_eq!(m("*", false).map(|x| x.kind), Some('*'));
        assert_eq!(m("12. a", false), Some(ListMarker { kind: '.', number: Some(12), bytes: 3, cols: 3 }));
        assert_eq!(m("1)\tb", false).map(|x| (x.kind, x.number)), Some((')', Some(1))));
        assert_eq!(m("123456789. a", false).map(|x| x.number), Some(Some(123_456_789)));
        assert_eq!(m("1234567890. a", false), None);
        assert_eq!(m("123. a", true).map(|x| x.number), Some(Some(123)));
        assert_eq!(m("2024. a", true), None);
        assert_eq!(m("2024. a", false).map(|x| x.number), Some(Some(2024)));
        assert_eq!(m("-a", false), None);
        assert_eq!(m("1.5", false), None);
        assert_eq!(m("12", false), None);
        assert_eq!(m("• a", false), None);
        assert_eq!(m("• a", true), Some(ListMarker { kind: '•', number: None, bytes: 3, cols: 1 }));
    }

    #[test]
    fn capital_headings() {
        assert!(caps_heading("INTRODUCTION"));
        assert!(caps_heading("PART II: THE ÉTUDES OF 1840"));
        assert!(caps_heading("WHY?"));
        assert!(!caps_heading("NOTE:"));
        assert!(!caps_heading("THE END."));
        assert!(!caps_heading("Introduction"));
        assert!(!caps_heading("A"));
        assert!(!caps_heading("OK"));
        assert!(caps_heading("TAX"));
        assert!(!caps_heading("1999"));
        assert!(!caps_heading("東京"));
        assert!(!caps_heading(&"A".repeat(81)));
    }
}
