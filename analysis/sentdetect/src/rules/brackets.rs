//! Matched brackets and quotation marks. A boundary between an opening
//! mark and the mark that closes it is not taken, so a quoted or
//! parenthesized sentence stays inside the sentence that holds it.

use turbo_unicode::SentenceBreak::Sp;

use super::units::{Unit, is_para_sep};

/// The pairs of opening and closing marks. A mark can open one pair and
/// close another: `“` closes `„` in German and opens `”` in English, `»`
/// opens `«` in Danish and closes `«` in French.
const PAIRS: &[(char, char)] = &[
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
    ('\u{FF08}', '\u{FF09}'), // fullwidth parentheses
    ('\u{FF3B}', '\u{FF3D}'), // fullwidth square brackets
    ('\u{FF5B}', '\u{FF5D}'), // fullwidth curly brackets
    ('\u{300C}', '\u{300D}'), // 「 」
    ('\u{300E}', '\u{300F}'), // 『 』
    ('\u{300A}', '\u{300B}'), // 《 》
    ('\u{3008}', '\u{3009}'), // 〈 〉
    ('\u{3010}', '\u{3011}'), // 【 】
    ('\u{3014}', '\u{3015}'), // 〔 〕
    ('\u{AB}', '\u{BB}'),     // « »
    ('\u{BB}', '\u{AB}'),     // » «
    ('\u{2039}', '\u{203A}'), // ‹ ›
    ('\u{203A}', '\u{2039}'), // › ‹
    ('\u{201E}', '\u{201C}'), // „ “
    ('\u{201E}', '\u{201D}'), // „ ”
    ('\u{201A}', '\u{2018}'), // ‚ ‘
    ('\u{201A}', '\u{2019}'), // ‚ ’
    ('\u{201C}', '\u{201D}'), // “ ”
    ('\u{2018}', '\u{2019}'), // ‘ ’
    ('\u{201D}', '\u{201D}'), // ” ”
    ('"', '"'),
    ('\'', '\''),
];

/// Whether a mark is also used inside words or alone, so that it opens
/// only before a word and closes only after one. Brackets are never
/// ambiguous.
fn is_ambiguous(c: char) -> bool {
    !matches!(
        c,
        '(' | ')'
            | '['
            | ']'
            | '{'
            | '}'
            | '\u{FF08}'..='\u{FF09}'
            | '\u{FF3B}'
            | '\u{FF3D}'
            | '\u{FF5B}'
            | '\u{FF5D}'
            | '\u{3008}'..='\u{3015}'
    )
}

/// How deep the search for an opener goes when a closer does not match
/// the innermost one; deeper unmatched openers are left open.
const SEARCH_DEPTH: usize = 8;

/// Marks the units a pair protects: for every matched pair, the units
/// after the opener up to and including the closer get a positive count in
/// `depth`, which has one entry per unit and one more.
pub(crate) fn protect(text: &str, units: &[Unit], stack: &mut Vec<(char, u32)>, depth: &mut Vec<i32>) {
    stack.clear();
    depth.clear();
    depth.resize(units.len() + 1, 0);
    let char_at = |i: usize| units.get(i).map(|u| text[u.start as usize..].chars().next().unwrap_or(' '));
    for (i, u) in units.iter().enumerate() {
        if is_para_sep(u.class) {
            stack.clear();
            continue;
        }
        let c = char_at(i).unwrap_or(' ');
        if !PAIRS.iter().any(|&(o, k)| o == c || k == c) {
            continue;
        }
        let before = if i == 0 { None } else { char_at(i - 1) };
        let after = char_at(i + 1);
        let word_before = before.is_some_and(char::is_alphanumeric);
        let space_before = before.is_none_or(char::is_whitespace) || i > 0 && units[i - 1].class == Sp;
        let word_after = after.is_some_and(char::is_alphanumeric);
        let space_after = after.is_none_or(char::is_whitespace);
        let ambiguous = is_ambiguous(c);
        if ambiguous && word_before && word_after {
            continue; // an apostrophe inside a word
        }
        let may_close = !ambiguous || !space_before;
        let may_open = !ambiguous || (!space_after && !word_before);
        if may_close
            && let Some(at) = stack.iter().rev().take(SEARCH_DEPTH).position(|&(open, _)| PAIRS.contains(&(open, c)))
        {
            let (_, open) = stack[stack.len() - 1 - at];
            stack.truncate(stack.len() - 1 - at);
            depth[open as usize + 1] += 1;
            depth[i + 1] -= 1;
            continue;
        }
        if may_open && PAIRS.iter().any(|&(o, _)| o == c) {
            stack.push((c, i as u32));
        }
    }
    let mut level = 0;
    for d in depth.iter_mut() {
        level += *d;
        *d = level;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::profile::ROOT;
    use crate::rules::units::classify;
    use crate::rules::{LineBreaks, Rules};

    /// The protected stretches of `text` as character strings.
    fn protected(text: &str) -> Vec<String> {
        let mut units = Vec::new();
        classify(text, Rules::ALL, &ROOT, LineBreaks::Sentence, &mut units);
        let (mut stack, mut depth) = (Vec::new(), Vec::new());
        protect(text, &units, &mut stack, &mut depth);
        let mut out = Vec::new();
        let mut cur = String::new();
        for (i, u) in units.iter().enumerate() {
            let c = text[u.start as usize..].chars().next().unwrap();
            if depth[i] > 0 {
                cur.push(c);
            } else if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
        out
    }

    #[test]
    fn pairs() {
        assert_eq!(protected("a (b. C) d"), ["b. C)"]);
        assert_eq!(protected("He said \"Stop. Go.\" Then"), ["Stop. Go.\""]);
        assert_eq!(protected("„Ja. Nein.“ Er"), ["Ja. Nein.“"]);
        assert_eq!(protected("«Oui. Non.» Il"), ["Oui. Non.»"]);
        assert_eq!(protected("»Ja. Nej.« Han"), ["Ja. Nej.«"]);
        assert_eq!(protected("“One. Two.” Three"), ["One. Two.”"]);
        assert_eq!(protected("「はい。いいえ。」と"), ["はい。いいえ。」"]);
    }

    #[test]
    fn apostrophes_are_not_quotes() {
        assert_eq!(protected("Don't go. It's late. 'Yes. No.' ok"), ["Yes. No.'"]);
        assert_eq!(protected("the Joneses' car. Then 'x"), Vec::<String>::new());
        assert_eq!(protected("rock ’n’ roll. Then"), Vec::<String>::new());
    }

    #[test]
    fn unmatched_and_paragraphs() {
        assert_eq!(protected("a (b. C d"), Vec::<String>::new());
        assert_eq!(protected("a (b.\nC) d"), Vec::<String>::new());
        // A closer that skips an unclosed opener still matches.
        assert_eq!(protected("(a [b. C) d"), ["a [b. C)"]);
    }
}
