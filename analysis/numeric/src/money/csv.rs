//! The text layer of the ECB history file: lines as `BufferedReader`
//! splits them and fields as `EcbFxRates.fields` splits and trims them.

use turbo_unicode::java;

/// The lines of `text`: split at `\n`, `\r` or `\r\n`, with no empty line
/// after a final terminator, as `BufferedReader.readLine` returns them.
pub(crate) fn lines(text: &str) -> impl Iterator<Item = &str> {
    let mut rest = text;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        match rest.find(['\n', '\r']) {
            None => {
                let line = rest;
                rest = "";
                Some(line)
            }
            Some(end) => {
                let line = &rest[..end];
                let terminator = if rest[end..].starts_with("\r\n") { 2 } else { 1 };
                rest = &rest[end + terminator..];
                Some(line)
            }
        }
    })
}

/// `String.isBlank`: empty or only `Character.isWhitespace` characters.
pub(crate) fn is_blank(line: &str) -> bool {
    line.chars().all(java::is_whitespace)
}

/// Splits a line at every comma and trims each field as `String.trim`
/// does, removing every character up to U+0020 at both ends. A control
/// character below U+0020 that is not `Character.isWhitespace` (tab, the
/// vertical tab, form feed and the information separators U+001C to
/// U+001F) refuses the line; the error is its byte offset.
pub(crate) fn fields<'a>(line: &'a str, out: &mut Vec<&'a str>) -> Result<(), usize> {
    out.clear();
    if let Some(at) = line.bytes().position(|b| b < b' ' && !java::is_whitespace(char::from(b))) {
        return Err(at);
    }
    out.extend(line.split(',').map(|f| f.trim_matches(|c: char| c <= ' ')));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_terminators() {
        let all: Vec<&str> = lines("a\nb\r\nc\rd").collect();
        assert_eq!(all, ["a", "b", "c", "d"]);
        let all: Vec<&str> = lines("a\n\nb\n").collect();
        assert_eq!(all, ["a", "", "b"]);
        assert_eq!(lines("").count(), 0);
        let all: Vec<&str> = lines("\n").collect();
        assert_eq!(all, [""]);
        let all: Vec<&str> = lines("a\r\r\nb").collect();
        assert_eq!(all, ["a", "", "b"]);
    }

    #[test]
    fn fields_keep_empties_and_trim() {
        let mut f = Vec::new();
        fields("Date, USD ,\tJPY\u{1C},", &mut f).unwrap();
        assert_eq!(f, ["Date", "USD", "JPY", ""]);
        fields("", &mut f).unwrap();
        assert_eq!(f, [""]);
        // U+00A0 is no padding.
        fields("\u{A0}USD", &mut f).unwrap();
        assert_eq!(f, ["\u{A0}USD"]);
        assert_eq!(fields("a,\u{0}b", &mut f), Err(2));
        assert_eq!(fields("\u{7}", &mut f), Err(0));
        // DEL and U+0085 are not below U+0020: kept, not refused.
        fields("USD\u{7F}", &mut f).unwrap();
        assert_eq!(f, ["USD\u{7F}"]);
    }

    #[test]
    fn blank_lines() {
        assert!(is_blank(""));
        assert!(is_blank(" \t\u{1F}\u{2003}"));
        assert!(!is_blank("\u{A0}"));
        assert!(!is_blank("\u{85}"));
        assert!(!is_blank(" x "));
    }
}
