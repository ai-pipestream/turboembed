//! List items. An item marker (`1.`, `2)`, `(3)`, `a.`, `b)`, a bullet)
//! starts a sentence, and the full stop inside a marker ends none, so each
//! item is one sentence even when the items are written on one line.
//!
//! A bullet counts at a line start, or inside a line when its paragraph
//! holds another of the same bullet, and a number or letter right after a
//! counted bullet counts with it. A marker with `)`, `.)` or parentheses
//! counts at a line start or next to a neighbour of its sequence.
//!
//! A number or letter with a bare full stop is also prose (`plan A. Then
//! plan B.`, `Der 1. und 2. Platz`, `J. K. Rowling`), so it counts only as
//! part of a list: at a line start unless a single letter with a full stop
//! or the end of the text follows (`J. K.`, `x. A`); inside a line when it
//! continues by one a counted marker of its sequence in the same paragraph
//! (`1. One 2. Two`), or when it is a `1.` or `a.` whose `2.` or `b.` comes
//! later in the paragraph and it follows a sentence end, a colon or an
//! earlier counted marker of the paragraph (`this. 1. One 2. Two`).

use turbo_unicode::SentenceBreak::Sp;

use super::units::{Unit, is_para_sep};

/// Bullets that mark an item anywhere.
const BULLETS: &[char] = &[
    '\u{2022}', // •
    '\u{2023}', // ‣
    '\u{2043}', // ⁃
    '\u{25E6}', // ◦
    '\u{25AA}', // ▪
    '\u{25CF}', // ●
    '\u{25CB}', // ○
    '\u{25A0}', // ■
    '\u{25A1}', // □
    '\u{2219}', // ∙
];

/// Bullets that mark an item only at the start of a line, since inside a
/// line they are dashes and operators.
const LINE_BULLETS: &[char] = &['-', '*', '+', '\u{2013}'];

/// What a marker counts with.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Kind {
    /// A number, with the delimiter that follows it: `.`, `)`, `.)` or
    /// `(` before.
    Number(Delimiter),
    /// One ASCII letter, with its delimiter.
    Letter(Delimiter),
    /// A bullet character.
    Bullet(char),
}

/// How a number or letter is delimited.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Delimiter {
    Stop,
    Paren,
    StopParen,
    Enclosed,
}

/// One marker.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Marker {
    /// The unit it starts at.
    pub start: u32,
    /// The unit its item's text starts at, after the spaces.
    pub content: u32,
    pub kind: Kind,
    /// The number, or the letter's position in the alphabet.
    pub value: u32,
    /// Whether it starts a line (or the text).
    pub line_start: bool,
    /// Whether it follows a bullet, as `9.` in `• 9. Item`.
    pub after_bullet: bool,
    /// The paragraph it is in, counted from 0.
    pub paragraph: u32,
}

/// A marker's place in the sequence index: paragraph, kind, value, start.
pub(crate) type Key = (u32, Kind, u32, u32);

impl Marker {
    fn key(&self) -> Key {
        (self.paragraph, self.kind, self.value, self.start)
    }

    /// The keys of the markers in this paragraph of this kind with
    /// `value`. `index` holds the keys of every marker found, sorted, so
    /// this is a binary search.
    fn with_value<'a>(&self, index: &'a [Key], value: u32) -> &'a [Key] {
        let at = (self.paragraph, self.kind, value);
        let lo = index.partition_point(|k| (k.0, k.1, k.2) < at);
        let hi = index.partition_point(|k| (k.0, k.1, k.2) <= at);
        &index[lo..hi]
    }

    /// Whether a bullet written inside a line counts: when its paragraph
    /// holds another of the same bullet.
    fn bullet_repeats(&self, index: &[Key]) -> bool {
        let lo = index.partition_point(|k| (k.0, k.1) < (self.paragraph, self.kind));
        let hi = index.partition_point(|k| (k.0, k.1) <= (self.paragraph, self.kind));
        hi - lo >= 2
    }

    /// Whether the paragraph holds the next value of this marker's
    /// sequence after it.
    fn followed(&self, index: &[Key]) -> bool {
        self.with_value(index, self.value + 1).last().is_some_and(|k| k.3 > self.start)
    }

    /// Whether the paragraph holds the previous value of this marker's
    /// sequence before it.
    fn preceded(&self, index: &[Key]) -> bool {
        self.value > 0 && self.with_value(index, self.value - 1).first().is_some_and(|k| k.3 < self.start)
    }
}

/// The last accepted marker of each number or letter kind: its value and
/// paragraph. There are eight such kinds, so a linear search is enough.
#[derive(Default)]
struct Last(Vec<(Kind, u32, u32)>);

impl Last {
    fn get(&self, kind: Kind) -> Option<(u32, u32)> {
        self.0.iter().find(|e| e.0 == kind).map(|e| (e.1, e.2))
    }

    fn set(&mut self, m: &Marker) {
        match self.0.iter_mut().find(|e| e.0 == m.kind) {
            Some(e) => (e.1, e.2) = (m.value, m.paragraph),
            None => self.0.push((m.kind, m.value, m.paragraph)),
        }
    }
}

/// Appends to `out` the accepted markers of the text, in text order.
pub(crate) fn markers(
    text: &str,
    units: &[Unit],
    found: &mut Vec<Marker>,
    index: &mut Vec<Key>,
    out: &mut Vec<Marker>,
) {
    found.clear();
    index.clear();
    out.clear();
    let char_at = |i: usize| units.get(i).map(|u| text[u.start as usize..].chars().next().unwrap_or(' '));
    let mut paragraph = 0;
    let mut line_start = true;
    let mut i = 0;
    while i < units.len() {
        let c = char_at(i).unwrap_or(' ');
        if is_para_sep(units[i].class) {
            paragraph += 1;
            line_start = true;
            i += 1;
            continue;
        }
        if matches!(c, '\n' | '\r' | '\u{85}' | '\u{2028}') {
            line_start = true;
            i += 1;
            continue;
        }
        if units[i].class == Sp {
            i += 1;
            continue;
        }
        let at_word = i == 0 || units[i - 1].class == Sp || is_para_sep(units[i - 1].class);
        if at_word && let Some((m, after)) = parse(text, units, i, line_start) {
            let start = i as u32;
            let mut next = after;
            found.push(Marker { paragraph, start, ..m });
            // A number or letter right after a bullet is part of its item.
            if let Kind::Bullet(_) = m.kind
                && let Some((inner, inner_after)) = parse(text, units, m.content as usize, false)
                && !matches!(inner.kind, Kind::Bullet(_))
            {
                found.push(Marker { paragraph, start: m.content, after_bullet: true, ..inner });
                next = inner_after;
            }
            i = next;
            line_start = false;
            continue;
        }
        line_start = false;
        i += 1;
    }
    index.extend(found.iter().map(Marker::key));
    index.sort_unstable();
    let mut last = Last::default();
    let mut bullet_accepted = false;
    // The paragraph of the last accepted marker.
    let mut listing = None;
    for m in found.iter() {
        let accepted = match m.kind {
            Kind::Bullet(_) => m.line_start || m.bullet_repeats(index),
            _ if m.after_bullet => bullet_accepted,
            // A parenthesis is not prose punctuation: such a marker counts
            // at a line start, or next to a neighbour of its sequence.
            Kind::Number(d) | Kind::Letter(d) if d != Delimiter::Stop => {
                m.line_start || m.followed(index) || m.preceded(index)
            }
            _ => {
                let continues =
                    last.get(m.kind).is_some_and(|(v, p)| v + 1 == m.value && (m.line_start || p == m.paragraph));
                if m.line_start {
                    continues || !initial_follows(text, units, m.content as usize)
                } else {
                    let starts = m.value == 1
                        && m.followed(index)
                        && (listing == Some(m.paragraph) || after_sentence_end(text, units, m.start as usize));
                    continues || starts
                }
            }
        };
        if let Kind::Bullet(_) = m.kind {
            bullet_accepted = accepted;
        }
        if accepted {
            if !matches!(m.kind, Kind::Bullet(_)) {
                last.set(m);
            }
            listing = Some(m.paragraph);
            out.push(*m);
        }
    }
}

/// Whether the word at unit `i` is a single letter that ends the text or
/// has a full stop after it, as `K` in `J. K. Rowling` or `A` in `x. A`:
/// the letter before it is then an initial, not a list marker.
fn initial_follows(text: &str, units: &[Unit], i: usize) -> bool {
    let char_at = |j: usize| units.get(j).map(|u| text[u.start as usize..].chars().next().unwrap_or(' '));
    char_at(i).is_some_and(|c| c.is_alphabetic())
        && match char_at(i + 1) {
            None => true,
            Some(c) => c == '.' || units[i + 1].class == Sp && char_at(i + 2).is_none(),
        }
}

/// Whether the text before unit `i`, past spaces and closing quotes and
/// brackets, ends with a sentence terminator or a colon.
fn after_sentence_end(text: &str, units: &[Unit], i: usize) -> bool {
    let before = &text[..units[i].start as usize];
    let before = before.trim_end().trim_end_matches(['"', '\'', '\u{201D}', '\u{2019}', ')', ']']);
    before.ends_with(['.', '!', '?', ':'])
}

/// Parses a marker starting at unit `i`. Returns it (its `start` and
/// `paragraph` are filled in by the caller) and the unit after it.
fn parse(text: &str, units: &[Unit], i: usize, line_start: bool) -> Option<(Marker, usize)> {
    let char_at = |j: usize| units.get(j).map(|u| text[u.start as usize..].chars().next().unwrap_or(' '));
    let is_space = |j: usize| units.get(j).is_some_and(|u| u.class == Sp);
    let marker = |kind, value, end: usize| {
        // The item's text starts after the spaces.
        let mut content = end;
        while is_space(content) {
            content += 1;
        }
        Some((
            Marker { start: 0, content: content as u32, kind, value, line_start, after_bullet: false, paragraph: 0 },
            content,
        ))
    };
    let c = char_at(i)?;
    if BULLETS.contains(&c) || (line_start && LINE_BULLETS.contains(&c)) {
        let next = char_at(i + 1);
        // A bullet stands alone or right before a numbered marker (`⁃9.`).
        let alone = is_space(i + 1);
        let before_number = BULLETS.contains(&c) && next.is_some_and(|n| n.is_ascii_digit());
        if alone || before_number {
            return marker(Kind::Bullet(c), 0, i + 1);
        }
        return None;
    }
    let (enclosed, first) = if c == '(' { (true, i + 1) } else { (false, i) };
    let mut j = first;
    let mut value = 0u32;
    let letter = char_at(j).filter(char::is_ascii_alphabetic);
    if let Some(l) = letter {
        value = (l.to_ascii_lowercase() as u32) - ('a' as u32) + 1;
        j += 1;
    } else {
        while j - first < 3 && char_at(j).is_some_and(|d| d.is_ascii_digit()) {
            value = value * 10 + char_at(j)?.to_digit(10)?;
            j += 1;
        }
        if j == first || char_at(j).is_some_and(|d| d.is_ascii_digit()) {
            return None;
        }
    }
    let delimiter = match (enclosed, char_at(j), char_at(j + 1)) {
        (true, Some(')'), _) => (Delimiter::Enclosed, j + 1),
        (false, Some('.'), Some(')')) => (Delimiter::StopParen, j + 2),
        (false, Some('.'), _) => (Delimiter::Stop, j + 1),
        (false, Some(')'), _) => (Delimiter::Paren, j + 1),
        _ => return None,
    };
    let (delimiter, end) = delimiter;
    if !is_space(end) {
        return None;
    }
    let kind = if letter.is_some() { Kind::Letter(delimiter) } else { Kind::Number(delimiter) };
    marker(kind, value, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::profile::ROOT;
    use crate::rules::units::classify;
    use crate::rules::{LineBreaks, Rules};

    /// The accepted markers as (start character offset, text of the marker).
    fn accepted(text: &str) -> Vec<String> {
        let mut units = Vec::new();
        classify(text, Rules::ALL, &ROOT, LineBreaks::Paragraph, &mut units);
        let (mut found, mut out) = (Vec::new(), Vec::new());
        markers(text, &units, &mut found, &mut Vec::new(), &mut out);
        out.iter()
            .map(|m| {
                let s = units[m.start as usize].start as usize;
                let e = units.get(m.content as usize).map_or(text.len(), |u| u.start as usize);
                text[s..e].trim_end().to_owned()
            })
            .collect()
    }

    #[test]
    fn sequences_inside_a_line() {
        assert_eq!(accepted("1. The first item 2. The second item"), ["1.", "2."]);
        assert_eq!(accepted("1.) One 2.) Two"), ["1.)", "2.)"]);
        assert_eq!(accepted("a. One b. Two c. Three"), ["a.", "b.", "c."]);
        assert_eq!(accepted("(1) One (2) Two"), ["(1)", "(2)"]);
        assert_eq!(accepted("• 9. One • 10. Two"), ["•", "9.", "•", "10."]);
        assert_eq!(accepted("⁃9. One ⁃10. Two"), ["⁃", "9.", "⁃", "10."]);
    }

    #[test]
    fn single_inline_numbers_are_not_markers() {
        assert_eq!(accepted("It was in 1990. Then 2. came"), Vec::<String>::new());
        assert_eq!(accepted("See p. 5. Then"), Vec::<String>::new());
        assert_eq!(accepted("Points 1. and 3. only"), Vec::<String>::new());
        assert_eq!(accepted("a - b * c"), Vec::<String>::new());
        assert_eq!(accepted("1.5 and 2.5"), Vec::<String>::new());
        // Consecutive values inside a line, with nothing at a line start.
        assert_eq!(accepted("plan A. Then plan B. Both"), Vec::<String>::new());
        assert_eq!(accepted("Der 1. und 2. Platz."), Vec::<String>::new());
        assert_eq!(accepted("Mary A. and John B. came"), Vec::<String>::new());
    }

    #[test]
    fn chains_reach_back_to_an_anchor() {
        // `J.` starts the line but is not a first value; `K.` has no anchor.
        assert_eq!(accepted("J. K. Rowling"), Vec::<String>::new());
        assert_eq!(accepted("x. A"), Vec::<String>::new());
        // A line start anchors the inline values that follow it by one.
        assert_eq!(accepted("1. One 2. Two 4. Four"), ["1.", "2."]);
        // A later line continues the sequence.
        assert_eq!(accepted("1. One\n2. Two\n3. Three"), ["1.", "2.", "3."]);
        // Parentheses mark a list at a line start from any value.
        assert_eq!(accepted("c) Three d) Four"), ["c)", "d)"]);
    }

    #[test]
    fn line_starts() {
        assert_eq!(accepted("Items:\n1. One\n- Two\n* Three"), ["1.", "-", "*"]);
        assert_eq!(accepted("1. Dies ist eine Punkteliste."), ["1."]);
    }
}
