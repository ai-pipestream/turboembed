//! Where a piece of a rejoined chunk lies in the text.
//!
//! A chunk's words are rejoined with single spaces before tokenizing, as
//! `AbstractDL` rejoins them, while the text may part them with longer or
//! multi-byte whitespace. Each end of a piece is mapped through the word
//! it falls in: a start on a joining space moves to the next word's start,
//! an end on one moves back to the previous word's end. So a piece that
//! crosses a word boundary spans the real gap in the text, one wholly on a
//! joining space covers no text, and every mapped byte is a word's byte or
//! a word's end, on a character boundary of the text.

/// One word of a chunk: its start in the rejoined chunk, its start in the
/// text, and its length in bytes (the same in both).
pub type Word = (usize, usize, usize);

/// The bytes in the text of the piece `s` of the rejoined chunk whose
/// words are `words`, in order; none for a piece of no text.
pub fn place(words: &[Word], s: [u32; 2]) -> Option<(usize, usize)> {
    let (a, b) = (s[0] as usize, s[1] as usize);
    if a >= b {
        return None;
    }
    let i = words.partition_point(|w| w.0 <= a).checked_sub(1)?;
    let (j, t, n) = words[i];
    let start = if a - j < n { t + a - j } else { words.get(i + 1).map_or(t + n, |w| w.1) };
    let (j, t, n) = words[words.partition_point(|w| w.0 < b).checked_sub(1)?];
    let end = t + (b - j).min(n);
    (start < end).then_some((start, end))
}

#[cfg(test)]
mod tests {
    use super::*;

    // "Zoë  Ann\u{3000}Lee" rejoined as "Zoë Ann Lee".
    const TEXT: &str = "Zoë  Ann\u{3000}Lee";
    const WORDS: [Word; 3] = [(0, 0, 4), (5, 6, 3), (9, 12, 3)];

    fn at(a: u32, b: u32) -> Option<&'static str> {
        place(&WORDS, [a, b]).map(|(s, e)| &TEXT[s..e])
    }

    #[test]
    fn pieces_inside_a_word_shift_to_it() {
        assert_eq!(at(0, 4), Some("Zoë"));
        assert_eq!(at(2, 4), Some("ë"));
        assert_eq!(at(5, 8), Some("Ann"));
        assert_eq!(at(10, 12), Some("ee"));
    }

    #[test]
    fn pieces_on_or_across_a_joining_space_map_each_end_to_its_word() {
        // Wholly on a joining space: no text.
        assert_eq!(at(4, 5), None);
        assert_eq!(at(8, 9), None);
        // Starting on a space: from the next word.
        assert_eq!(at(4, 8), Some("Ann"));
        // Ending on a space: to the word before it.
        assert_eq!(at(5, 9), Some("Ann"));
        // Across words: the real gap in the text, whatever it holds.
        assert_eq!(at(2, 7), Some("ë  An"));
        assert_eq!(at(6, 11), Some("nn\u{3000}Le"));
        assert_eq!(at(0, 12), Some(TEXT));
        // Past the last word or empty.
        assert_eq!(at(12, 20), None);
        assert_eq!(at(3, 3), None);
        assert_eq!(place(&[], [0, 1]), None);
    }
}
