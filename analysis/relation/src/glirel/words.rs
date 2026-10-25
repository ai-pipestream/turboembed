//! GLiREL's words: the matches of `\w+(?:[-_]\w+)*|\S` under Python's `re`,
//! where `\w` is a character `str.isalnum()` accepts (a letter or a
//! number, by general category) or `_`. Marks, joiners and every other
//! character that is not whitespace are words of their own. Each word keeps
//! its byte range in the text; the text keeps its case.

use turbo_unicode::category::general_category;
use turbo_unicode::whitespace::is_python_space;

/// One word: its byte range in the text.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Word {
    pub start: usize,
    pub end: usize,
}

fn is_word(c: char) -> bool {
    let g = general_category(c);
    g.is_letter() || g.is_number() || c == '_'
}

/// The length in bytes of the run of word characters at the start of `s`.
fn run(s: &str) -> usize {
    s.char_indices().find(|&(_, c)| !is_word(c)).map_or(s.len(), |(i, _)| i)
}

/// The words of `text`, in order.
pub fn split(text: &str) -> Vec<Word> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(c) = text[at..].chars().next() {
        if is_python_space(c) {
            at += c.len_utf8();
            continue;
        }
        let rest = &text[at..];
        let mut len = run(rest);
        if len == 0 {
            len = c.len_utf8();
        } else {
            // Joined by a single `-` or `_` before more word characters.
            while rest[len..].starts_with(['-', '_']) {
                let more = run(&rest[len + 1..]);
                if more == 0 {
                    break;
                }
                len += 1 + more;
            }
        }
        out.push(Word { start: at, end: at + len });
        at += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<&str> {
        split(s).into_iter().map(|w| &s[w.start..w.end]).collect()
    }

    #[test]
    fn words_and_punctuation() {
        assert_eq!(
            texts("Steve Jobs founded Apple, in 1976."),
            ["Steve", "Jobs", "founded", "Apple", ",", "in", "1976", "."]
        );
        assert_eq!(texts("state-of-the-art x--y _z a_"), ["state-of-the-art", "x", "-", "-", "y", "_z", "a_"]);
        assert_eq!(texts("cafe\u{301} a\u{200d}b"), ["cafe", "\u{301}", "a", "\u{200d}", "b"]);
        assert_eq!(texts("\u{1c}x\u{a0}y"), ["x", "y"]);
    }
}
