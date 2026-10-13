//! Converting a set's own offsets to UTF-8 byte offsets into the text as
//! loaded, and the newline translation Python applies when it reads a
//! text file.

/// Byte offsets of every code point boundary of a text, so an offset in
/// code points (a Python `str` index) maps to a byte offset in O(1).
pub struct CodePoints {
    /// `bytes[i]` is the byte offset of code point `i`; the last entry is
    /// the text's length.
    bytes: Vec<usize>,
}

impl CodePoints {
    pub fn new(text: &str) -> CodePoints {
        let mut bytes: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        bytes.push(text.len());
        CodePoints { bytes }
    }

    /// The text's length in code points.
    pub fn len(&self) -> usize {
        self.bytes.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The byte offset of code point offset `cp`; `None` past the end.
    pub fn byte(&self, cp: usize) -> Option<usize> {
        self.bytes.get(cp).copied()
    }
}

/// The text as Python's `open(path).read()` returns it: `\r\n` and a lone
/// `\r` become `\n` (universal newlines). Sets whose offsets were taken
/// from text read that way are offsets into this text, not the file.
pub fn universal_newlines(text: String) -> String {
    if !text.contains('\r') {
        return text;
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_points_to_bytes() {
        // é is 2 bytes, the emoji 4 bytes (and 2 UTF-16 units), each one
        // code point.
        let t = "aé😀b";
        let cp = CodePoints::new(t);
        assert_eq!(cp.len(), 4);
        assert_eq!((0..=4).map(|i| cp.byte(i).unwrap()).collect::<Vec<_>>(), vec![0, 1, 3, 7, 8]);
        assert_eq!(cp.byte(5), None);
        assert_eq!(&t[cp.byte(2).unwrap()..cp.byte(3).unwrap()], "😀");
    }

    #[test]
    fn newlines_as_python_reads_them() {
        assert_eq!(universal_newlines("a\r\nb\rc\n\r\n".into()), "a\nb\nc\n\n");
        assert_eq!(universal_newlines("plain".into()), "plain");
    }
}
