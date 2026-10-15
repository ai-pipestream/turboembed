//! List items. An item marker (`1.`, `2)`, `(3)`, `a.`, `b)`, a bullet)
//! starts a sentence, and the full stop inside a marker ends none, so each
//! item is one sentence even when the items are written on one line.
//!
//! A marker counts when it starts a line, or, written inside a line, when
//! the same paragraph holds the next or the previous marker of its
//! sequence (`1.` and `2.`, `a)` and `b)`, two of the same bullet). One
//! inline number with a full stop after it (`in 1990. Then`, `p. 5.`) is
//! not a marker.

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

    /// Whether this marker starts a sentence: at the start of a line, or
    /// with a neighbour of its sequence in the same paragraph. `index`
    /// holds the keys of every marker found, sorted, so each question is a
    /// binary search and a text with many markers stays linear in them.
    fn accepted(&self, index: &[Key]) -> bool {
        if self.line_start {
            return true;
        }
        let (p, kind) = (self.paragraph, self.kind);
        // The keys with this paragraph and kind, and this value when given.
        let range = |value: Option<u32>| {
            let lo = index.partition_point(|k| (k.0, k.1, k.2) < (p, kind, value.unwrap_or(0)));
            let hi = index.partition_point(|k| (k.0, k.1, k.2) <= (p, kind, value.unwrap_or(u32::MAX)));
            &index[lo..hi]
        };
        match kind {
            Kind::Bullet(_) => range(None).len() >= 2,
            _ => {
                let before =
                    self.value.checked_sub(1).is_some_and(|v| range(Some(v)).first().is_some_and(|k| k.3 < self.start));
                let after =
                    self.value.checked_add(1).is_some_and(|v| range(Some(v)).last().is_some_and(|k| k.3 > self.start));
                before || after
            }
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
    // A marker after a bullet counts when its bullet does.
    for (k, m) in found.iter().enumerate() {
        let owner = if m.after_bullet { &found[k - 1] } else { m };
        if owner.accepted(index) {
            out.push(*m);
        }
    }
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
    }

    #[test]
    fn line_starts() {
        assert_eq!(accepted("Items:\n1. One\n- Two\n* Three"), ["1.", "-", "*"]);
        assert_eq!(accepted("1. Dies ist eine Punkteliste."), ["1."]);
    }
}
