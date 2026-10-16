//! The ids in a sentence's metadata: `ADMetadata`.
//!
//! The metadata starts with the id before the sentence's text, such as
//! `CP12-3` or `1001`: ASCII letters and hyphens, then the text id's
//! digits. The paragraph id is the number after the first `p=` that has
//! one.

const PARAGRAPH_PREFIX: &str = "p=";
const SOURCE_PREFIX: &str = "source=\"";

/// The text id and the paragraph id.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct TextAndParagraph {
    pub text: i32,
    pub paragraph: i32,
}

/// Where the digits of the two ids are, as byte ranges.
struct IdSpans {
    text: (usize, usize),
    paragraph: (usize, usize),
}

fn end_of_digits(meta: &str, from: usize) -> usize {
    from + meta.as_bytes()[from..].iter().take_while(|b| b.is_ascii_digit()).count()
}

fn scan(meta: &str) -> Option<IdSpans> {
    let b = meta.as_bytes();
    let text_start = b.iter().take_while(|&&c| c.is_ascii_alphabetic() || c == b'-').count();
    let text_end = end_of_digits(meta, text_start);
    if text_end == text_start {
        return None;
    }
    let mut from = text_end;
    loop {
        let prefix = from + meta[from..].find(PARAGRAPH_PREFIX)?;
        let start = prefix + PARAGRAPH_PREFIX.len();
        let end = end_of_digits(meta, start);
        if end > start {
            return Some(IdSpans { text: (text_start, text_end), paragraph: (start, end) });
        }
        from = prefix + 1;
    }
}

/// The digits as a Java `int`, or `None` when they do not fit.
fn parse_digits(digits: &str) -> Option<i32> {
    digits.bytes().try_fold(0i32, |v, d| v.checked_mul(10)?.checked_add(i32::from(d - b'0')))
}

/// `parseTextAndParagraph`: both ids, or `None` when one is missing or
/// does not fit an `int`.
pub(crate) fn text_and_paragraph(meta: &str) -> Option<TextAndParagraph> {
    let s = scan(meta)?;
    let text = parse_digits(&meta[s.text.0..s.text.1])?;
    let paragraph = parse_digits(&meta[s.paragraph.0..s.paragraph.1])?;
    Some(TextAndParagraph { text, paragraph })
}

/// `textPrefix`: the letters and hyphens before the text id, which name
/// the text in the literary corpora; `None` when there are none or an id
/// is missing.
pub(crate) fn text_prefix(meta: &str) -> Option<&str> {
    let s = scan(meta)?;
    (s.text.0 > 0).then(|| &meta[..s.text.0])
}

/// `source`: the text between the first `source="` and the next `"`.
pub(crate) fn source(meta: &str) -> Option<&str> {
    let start = meta.find(SOURCE_PREFIX)? + SOURCE_PREFIX.len();
    let end = start + meta[start..].find('"')?;
    Some(&meta[start..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids() {
        let ids = text_and_paragraph("CP12-3 p=4 title src").unwrap();
        assert_eq!((ids.text, ids.paragraph), (12, 4));
        assert_eq!(text_and_paragraph("1001 p= p=x p=7").unwrap().paragraph, 7);
        assert_eq!(text_and_paragraph("A p=1"), None);
        assert_eq!(text_and_paragraph("1 p=2147483648"), None);
        assert_eq!(text_and_paragraph("2147483647 p=0").unwrap().text, i32::MAX);
        assert_eq!(text_prefix("LIT-ab3 p=1"), Some("LIT-ab"));
        assert_eq!(text_prefix("3 p=1"), None);
        assert_eq!(source("x source=\"abc\" y"), Some("abc"));
        assert_eq!(source("x source=\"abc"), None);
    }
}
