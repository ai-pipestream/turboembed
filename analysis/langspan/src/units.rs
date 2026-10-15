//! The units the detector labels: words, each in one script group.
//!
//! Without a token layer a unit is a stretch of letters inside one
//! whitespace-separated chunk: punctuation ends it, except an apostrophe,
//! a hyphen, a middle dot or a full stop with a letter on both sides
//! (`don't`, `e-mail`, `col·lecció`, `z.B`), and so does a change of script
//! group (`iPhone是` is two units). With a token layer the same rules run
//! inside each token, so no unit crosses a token boundary. Scripts written
//! without spaces (Han, kana, Thai) make long units without a token layer:
//! a clause each, cut at punctuation.
//!
//! A chunk (or token) that OpenNLP's detector would not see, because its
//! normalizers remove it (a URL, an e-mail address, a hashtag, a
//! `@mention`, digits), makes no unit.

use turbo_normalize::language_detection::prepare;
use turbo_unicode::Script;
use turbo_unicode::category::general_category;
use turbo_unicode::props::white_space;
use turbo_unicode::script::script;

use crate::scripts::group;

/// One unit: bytes `[start, end)` of the text, its script group, and how
/// many letters (characters of a script group) it has.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Unit {
    pub start: u32,
    pub end: u32,
    pub group: Script,
    pub letters: u32,
}

/// Characters that join two letters into one word.
fn joiner(c: char) -> bool {
    matches!(
        c,
        '\'' | '\u{2019}' | '\u{02BC}' | '-' | '\u{2010}' | '\u{2011}' | '.' | '\u{00B7}' | '\u{200C}' | '\u{200D}'
    )
}

fn wordy(c: char) -> bool {
    let cat = general_category(c);
    cat.is_letter() || cat.is_mark()
}

/// The units of `text`, in text order, into `out` (cleared first). With
/// `tokens` (start and end columns of a token layer over the same text,
/// in text order) units stay inside tokens; a token that overlaps the one
/// before it is skipped.
pub fn units(text: &str, tokens: Option<(&[u32], &[u32])>, out: &mut Vec<Unit>) {
    out.clear();
    match tokens {
        Some((starts, ends)) => {
            // A token that starts before the one before it ends (tokens out
            // of order, or overlapping) is skipped, so units stay in order.
            let mut done = 0;
            for (&s, &e) in starts.iter().zip(ends) {
                if s >= done {
                    chunk(text, s as usize, e as usize, out);
                    done = e;
                }
            }
        }
        None => {
            let mut start = None;
            for (i, c) in text.char_indices() {
                match (white_space(c), start) {
                    (true, Some(s)) => {
                        chunk(text, s, i, out);
                        start = None;
                    }
                    (false, None) => start = Some(i),
                    _ => {}
                }
            }
            if let Some(s) = start {
                chunk(text, s, text.len(), out);
            }
        }
    }
}

/// The units of `text[from..to]`.
fn chunk(text: &str, from: usize, to: usize, out: &mut Vec<Unit>) {
    let piece = &text[from..to];
    if !prepare(piece).chars().any(|c| wordy(c) && group(script(c)).is_some()) {
        return;
    }
    let mut open: Option<Unit> = None;
    let mut chars = piece.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let at = (from + i) as u32;
        let end = at + c.len_utf8() as u32;
        if wordy(c) {
            let g = group(script(c));
            match (&mut open, g) {
                (Some(u), Some(g)) if g != u.group => {
                    out.push(*u);
                    open = Some(Unit { start: at, end, group: g, letters: 1 });
                }
                (Some(u), g) => {
                    u.end = end;
                    u.letters += u32::from(g.is_some());
                }
                (None, Some(g)) => open = Some(Unit { start: at, end, group: g, letters: 1 }),
                // A mark or a Common letter with no letter before it.
                (None, None) => {}
            }
            continue;
        }
        let joins = joiner(c) && open.is_some() && chars.peek().is_some_and(|&(_, next)| wordy(next));
        if joins {
            if let Some(u) = &mut open {
                u.end = end;
            }
        } else if let Some(u) = open.take() {
            out.push(u);
        }
    }
    if let Some(u) = open {
        out.push(u);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words<'a>(text: &'a str, tokens: Option<(&[u32], &[u32])>) -> Vec<(&'a str, Script)> {
        let mut out = Vec::new();
        units(text, tokens, &mut out);
        out.iter().map(|u| (&text[u.start as usize..u.end as usize], u.group)).collect()
    }

    #[test]
    fn words_keep_inner_apostrophes_and_hyphens_and_drop_outer_punctuation() {
        let w = words("\"Don't\" (e-mail), l'homme... z.B. col·lecció", None);
        let w: Vec<&str> = w.into_iter().map(|(s, _)| s).collect();
        assert_eq!(w, ["Don't", "e-mail", "l'homme", "z.B", "col·lecció"]);
    }

    #[test]
    fn a_change_of_script_group_starts_a_unit_and_kana_go_with_han() {
        let w = words("iPhone是 東京ひらがなカタカナ Привет", None);
        assert_eq!(
            w,
            [
                ("iPhone", Script::Latin),
                ("是", Script::Han),
                ("東京ひらがなカタカナ", Script::Han),
                ("Привет", Script::Cyrillic)
            ]
        );
    }

    #[test]
    fn marks_stay_with_their_letter_and_count_no_letter() {
        let mut out = Vec::new();
        // "é" as e and a combining acute.
        units("cafe\u{301} नमस्ते", None, &mut out);
        assert_eq!(out.len(), 2);
        assert_eq!((out[0].start, out[0].end, out[0].letters), (0, 6, 4));
        assert_eq!(out[1].group, Script::Devanagari);
    }

    #[test]
    fn urls_mentions_hashtags_and_digits_make_no_unit() {
        let w = words("see https://example.com/path and #tagged @someone 2024 mail@example.org ok", None);
        let w: Vec<&str> = w.into_iter().map(|(s, _)| s).collect();
        assert_eq!(w, ["see", "and", "ok"]);
    }

    #[test]
    fn with_tokens_no_unit_crosses_a_token() {
        let text = "don't stop";
        // Tokens "do", "n't", "stop".
        let w = words(text, Some((&[0, 2, 6], &[2, 5, 10])));
        let w: Vec<&str> = w.into_iter().map(|(s, _)| s).collect();
        assert_eq!(w, ["do", "n't", "stop"]);
    }

    #[test]
    fn a_token_overlapping_the_one_before_it_is_skipped() {
        let w = words("ab cd ef", Some((&[3, 0, 6], &[5, 2, 8])));
        let w: Vec<&str> = w.into_iter().map(|(s, _)| s).collect();
        assert_eq!(w, ["cd", "ef"]);
    }

    #[test]
    fn cjk_clauses_are_cut_at_punctuation() {
        let w = words("请勿倒置，小心轻放。", None);
        let w: Vec<&str> = w.into_iter().map(|(s, _)| s).collect();
        assert_eq!(w, ["请勿倒置", "小心轻放"]);
    }
}
