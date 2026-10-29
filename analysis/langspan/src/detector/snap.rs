//! Span boundary snapping, inter-word gap classification, and dominant
//! script calculation for [`SpanDetector`](super::SpanDetector).

use turbo_unicode::Script;
use turbo_unicode::category::{GeneralCategory, general_category};
use turbo_unicode::props::white_space;
use turbo_unicode::script::script;

use crate::decode::Gap;
use crate::scripts::group;
use crate::units::Unit;

/// Characters that end a sentence, for texts without a sentence layer.
fn sentence_final(c: char) -> bool {
    matches!(
        c,
        '.' | '!'
            | '?'
            | '\u{2026}'
            | '\u{3002}'
            | '\u{FF01}'
            | '\u{FF1F}'
            | '\u{FF0E}'
            | '\u{061F}'
            | '\u{06D4}'
            | '\u{0964}'
            | '\u{0965}'
            | '\u{0E2F}'
    )
}

fn line_break(c: char) -> bool {
    matches!(c, '\n' | '\r' | '\u{0B}' | '\u{0C}' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// What separates the word ending at `end` from the one starting at
/// `start`.
pub(super) fn gap(text: &str, end: u32, start: u32, sentence_starts: Option<&[u32]>) -> Gap {
    let between = &text[end as usize..start as usize];
    if between.chars().any(line_break) {
        return Gap::Sentence;
    }
    match sentence_starts {
        Some(starts) => {
            let i = starts.partition_point(|&s| s < end);
            if starts.get(i).is_some_and(|&s| s <= start) {
                return Gap::Sentence;
            }
        }
        None => {
            if between.chars().any(sentence_final) {
                return Gap::Sentence;
            }
        }
    }
    if between.chars().any(|c| general_category(c).is_punctuation()) { Gap::Punctuation } else { Gap::Space }
}

/// Punctuation that opens what follows it: opening brackets and initial
/// quotation marks (Ps, Pi), the Spanish inverted marks and the straight
/// quotation marks, which open as often as they close.
fn opening(c: char) -> bool {
    matches!(c, '"' | '\'' | '\u{00BF}' | '\u{00A1}')
        || matches!(general_category(c), GeneralCategory::Ps | GeneralCategory::Pi)
}

/// Where a run whose first word starts at `start` begins: at the start of
/// a sentence between the word before it (which ends at `floor`) and it;
/// or else back over the opening punctuation right before it (a quotation
/// mark or bracket), when whitespace or the start of the text comes
/// before that punctuation; and then back to the start of the token it
/// falls inside, never before `floor`.
///
/// Anything else before the word stays with the run before: sentence-final
/// punctuation (`休息。But`), closing brackets and quotation marks, and
/// opening marks with no space before them, which close what precedes
/// them as often as not (German `„Hallo!“Then`).
pub(super) fn snap_start(
    text: &str,
    start: u32,
    floor: u32,
    sentence_starts: Option<&[u32]>,
    tokens: Option<(&[u32], &[u32])>,
) -> u32 {
    let mut s = match sentence_starts.and_then(|starts| {
        let i = starts.partition_point(|&s| s <= start);
        (i > 0 && starts[i - 1] >= floor).then(|| starts[i - 1])
    }) {
        Some(sent_start) => sent_start,
        None => {
            let mut s = start as usize;
            while s > floor as usize {
                let c = text[..s].chars().next_back().expect("a character before a position above the floor");
                if !opening(c) {
                    break;
                }
                s -= c.len_utf8();
            }
            if text[..s].chars().next_back().is_some_and(|c| !white_space(c)) {
                s = start as usize;
            }
            s as u32
        }
    };
    if let Some((starts, ends)) = tokens {
        let i = starts.partition_point(|&t| t <= s);
        if i > 0 && starts[i - 1] < s && s < ends[i - 1] && starts[i - 1] >= floor {
            s = starts[i - 1];
        }
    }
    s
}

/// Where the last run ends: after its last word and the characters after
/// it up to whitespace (closing punctuation), and then on to the end of
/// the token it falls inside.
pub(super) fn snap_end(text: &str, end: u32, tokens: Option<(&[u32], &[u32])>) -> u32 {
    let rest = &text[end as usize..];
    let mut e = end + rest.chars().take_while(|&c| !white_space(c)).map(|c| c.len_utf8() as u32).sum::<u32>();
    if let Some((starts, ends)) = tokens {
        let i = starts.partition_point(|&t| t < e);
        if i > 0 && ends[i - 1] > e {
            e = ends[i - 1];
        }
    }
    e
}

/// The script most letters of `units` are in; the first such in text
/// order on a tie.
pub(super) fn dominant_script(text: &str, units: &[Unit], counts: &mut Vec<(Script, u32)>) -> Script {
    counts.clear();
    for u in units {
        for c in text[u.start as usize..u.end as usize].chars() {
            let s = script(c);
            if group(s).is_none() {
                continue;
            }
            match counts.iter_mut().find(|(x, _)| *x == s) {
                Some((_, n)) => *n += 1,
                None => counts.push((s, 1)),
            }
        }
    }
    let mut best = counts.first().map_or(units[0].group, |&(s, _)| s);
    let mut most = 0;
    for &(s, n) in counts.iter() {
        if n > most {
            best = s;
            most = n;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaps_rank_line_breaks_and_sentences_over_punctuation_over_spaces() {
        let text = "aa bb, cc. dd\nee";
        assert_eq!(gap(text, 2, 3, None), Gap::Space);
        assert_eq!(gap(text, 5, 7, None), Gap::Punctuation);
        assert_eq!(gap(text, 9, 11, None), Gap::Sentence);
        assert_eq!(gap(text, 13, 14, None), Gap::Sentence);
        // With a sentence layer only its starts are sentence boundaries,
        // and a line break still is one.
        assert_eq!(gap(text, 9, 11, Some(&[0])), Gap::Punctuation);
        assert_eq!(gap(text, 5, 7, Some(&[0, 7])), Gap::Sentence);
        assert_eq!(gap(text, 13, 14, Some(&[0])), Gap::Sentence);
    }

    #[test]
    fn a_run_starts_at_a_sentence_start_or_its_opening_punctuation() {
        let text = "ab. - \"cd\"";
        // Without a sentence layer: back over the quotation mark only.
        assert_eq!(snap_start(text, 7, 2, None, None), 6);
        // A sentence starting at the dash takes the run's start with it.
        assert_eq!(snap_start(text, 7, 2, Some(&[0, 4]), None), 4);
        // A quotation mark with no space before it is left where it is.
        assert_eq!(snap_start("ab\"cd", 3, 2, None, None), 3);
        assert_eq!(snap_start("ab \"cd", 4, 2, None, None), 3);
        // A sentence start before the floor is not used.
        assert_eq!(snap_start(text, 7, 2, Some(&[0]), None), 6);
    }

    #[test]
    fn closing_and_sentence_final_punctuation_stay_with_the_run_before() {
        // (text, end of the word before, start of the run's first word)
        for (text, floor, start) in [("休息。But tomorrow", 6, 9), ("today!Привет", 5, 6), ("„Hallo!“Then", 8, 12)]
        {
            assert_eq!(snap_start(text, start, floor, None, None), start, "{text}");
        }
        // Opening marks after whitespace go with the run.
        assert_eq!(snap_start("gestern (I was", 9, 7, None, None), 8);
        assert_eq!(snap_start("nur „Hallo", 7, 3, None, None), 4);
        assert_eq!(snap_start("dijo ¿«Qué", 9, 4, None, None), 5);
        // Closing punctuation before an opening mark stops the walk.
        assert_eq!(snap_start("ab.)(cd", 5, 2, None, None), 5);
    }

    #[test]
    fn a_run_inside_a_token_starts_at_the_token() {
        // One token "ab cd" spanning a space: the run of "cd" starts with it.
        assert_eq!(snap_start("ab cd", 3, 0, None, Some((&[0], &[5]))), 0);
        // A token that starts before the floor does not move it.
        assert_eq!(snap_start("ab cd", 3, 2, None, Some((&[0], &[5]))), 3);
        // A sentence start inside a token still snaps back to the token start.
        assert_eq!(snap_start("ab - cd", 5, 2, Some(&[0, 4]), Some((&[0, 3], &[2, 7]))), 3);
        // The last run ends after closing punctuation, then at its token's end.
        assert_eq!(snap_end("ab.\" cd", 2, None), 4);
        assert_eq!(snap_end("ab.\" cd", 2, Some((&[0], &[7]))), 7);
        assert_eq!(snap_end("ab", 2, Some((&[0], &[2]))), 2);
    }
}
