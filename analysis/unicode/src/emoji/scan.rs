//! Finding emoji sequences in text.
//!
//! [`scan`] reports the possible emoji of UTS #51 (definition ED-20) from
//! left to right, each as the byte range it covers and the kind of
//! sequence. At each position it tries, in this order:
//!
//! 1. Two regional indicators: a flag ([`EmojiKindHint::Flag`]). Pairs are
//!    taken from the left, so three indicators are a flag and a lone
//!    indicator.
//! 2. `0`-`9`, `#` or `*`, an optional U+FE0F and U+20E3: a keycap
//!    ([`EmojiKindHint::Keycap`]). These base characters have the Emoji
//!    property but are never reported without U+20E3.
//! 3. An element, then any number of U+200D ZERO WIDTH JOINER each
//!    followed by another element ([`EmojiKindHint::Zwj`] when there is at
//!    least one joiner). An element is a character with the Emoji property
//!    (a lone regional indicator included), or an Extended_Pictographic
//!    character followed by U+FE0F, then optionally U+FE0F, optionally an
//!    Emoji_Modifier ([`EmojiKindHint::Modifier`]) and optionally a tag
//!    sequence, one or more of U+E0020 to U+E007E ended by U+E007F
//!    ([`EmojiKindHint::Tag`]). A tag run without U+E007F is not part of
//!    the element. A character followed by U+FE0E (text presentation
//!    selector) is not an element, since the text asks for text
//!    presentation.
//!
//! A joiner that is not followed by an element ends the sequence before
//! the joiner. Everything else is skipped.
//!
//! Text-default emoji are reported without U+FE0F: U+2639 FROWNING FACE,
//! U+00A9 COPYRIGHT SIGN and U+2122 TRADE MARK SIGN alone are each one
//! [`EmojiKindHint::Basic`] sequence, because they have the Emoji
//! property. A caller that wants only characters shown as emoji by default
//! checks the first character with [`crate::props::emoji_presentation`]
//! and looks for U+FE0F in the range. Extended_Pictographic characters
//! without the Emoji property (reserved code points in the pictographic
//! blocks, for example) are reported only when U+FE0F follows them.
//!
//! The scanner does not check that a sequence is RGI; [`super::is_rgi`]
//! does that. Every row of emoji-test.txt, whatever its status, is found
//! as one sequence covering the row exactly.

use std::ops::Range;

use crate::props::bits;
use crate::tables::props::{EMOJI, EMOJI_MODIFIER, EXTENDED_PICTOGRAPHIC, REGIONAL_INDICATOR};

const ZWJ: char = '\u{200D}';
const TEXT_SELECTOR: char = '\u{FE0E}';
const EMOJI_SELECTOR: char = '\u{FE0F}';
const KEYCAP: char = '\u{20E3}';
const FIRST_TAG: char = '\u{E0020}';
const LAST_TAG: char = '\u{E007E}';
const CANCEL_TAG: char = '\u{E007F}';

/// The shape of an emoji sequence found by [`scan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EmojiKindHint {
    /// One emoji character, with an optional U+FE0F.
    Basic,
    /// A keycap sequence: `0`-`9`, `#` or `*`, an optional U+FE0F, U+20E3.
    Keycap,
    /// Two regional indicators.
    Flag,
    /// An emoji followed by a skin tone modifier.
    Modifier,
    /// An emoji followed by a tag sequence, such as a subdivision flag.
    Tag,
    /// Elements joined by U+200D ZERO WIDTH JOINER.
    Zwj,
}

/// The emoji sequences of `text` from left to right, as the module
/// documentation describes.
pub fn scan(text: &str) -> Scan<'_> {
    Scan { text, at: 0 }
}

/// The iterator [`scan`] returns.
pub struct Scan<'t> {
    text: &'t str,
    at: usize,
}

impl Iterator for Scan<'_> {
    type Item = (Range<usize>, EmojiKindHint);

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.text.as_bytes();
        while self.at < bytes.len() {
            let b = bytes[self.at];
            if b.is_ascii() && !is_keycap_base(b as char) {
                self.at += 1;
                continue;
            }
            let start = self.at;
            let rest = &self.text[start..];
            if let Some((len, kind)) = sequence_at(rest) {
                self.at += len;
                return Some((start..start + len, kind));
            }
            self.at += rest.chars().next().map_or(1, char::len_utf8);
        }
        None
    }
}

fn is_keycap_base(c: char) -> bool {
    matches!(c, '0'..='9' | '#' | '*')
}

fn first(s: &str) -> Option<char> {
    s.chars().next()
}

/// The emoji sequence at the start of `s`: its length in bytes and kind.
fn sequence_at(s: &str) -> Option<(usize, EmojiKindHint)> {
    let c = first(s)?;
    if is_keycap_base(c) {
        return keycap(s);
    }
    if bits(c) & REGIONAL_INDICATOR != 0 && first(&s[4..]).is_some_and(|n| bits(n) & REGIONAL_INDICATOR != 0) {
        return Some((8, EmojiKindHint::Flag));
    }
    let head = element(s)?;
    let mut len = head.len;
    let mut joined = false;
    while first(&s[len..]) == Some(ZWJ) {
        let Some(next) = element(&s[len + ZWJ.len_utf8()..]) else { break };
        len += ZWJ.len_utf8() + next.len;
        joined = true;
    }
    let kind = if joined {
        EmojiKindHint::Zwj
    } else if head.tag {
        EmojiKindHint::Tag
    } else if head.modifier {
        EmojiKindHint::Modifier
    } else {
        EmojiKindHint::Basic
    };
    Some((len, kind))
}

/// A keycap base, an optional U+FE0F and U+20E3.
fn keycap(s: &str) -> Option<(usize, EmojiKindHint)> {
    let mut len = 1;
    if first(&s[len..]) == Some(EMOJI_SELECTOR) {
        len += EMOJI_SELECTOR.len_utf8();
    }
    (first(&s[len..]) == Some(KEYCAP)).then_some((len + KEYCAP.len_utf8(), EmojiKindHint::Keycap))
}

/// One element of a sequence, as the module documentation defines it.
struct Element {
    len: usize,
    modifier: bool,
    tag: bool,
}

fn element(s: &str) -> Option<Element> {
    let c = first(s)?;
    if is_keycap_base(c) {
        return None;
    }
    let mut len = c.len_utf8();
    let next = first(&s[len..]);
    let selected = next == Some(EMOJI_SELECTOR);
    let props = bits(c);
    if next == Some(TEXT_SELECTOR) || (props & EMOJI == 0 && !(selected && props & EXTENDED_PICTOGRAPHIC != 0)) {
        return None;
    }
    if selected {
        len += EMOJI_SELECTOR.len_utf8();
    }
    let modifier = first(&s[len..]).is_some_and(|m| bits(m) & EMOJI_MODIFIER != 0);
    if modifier {
        len += 4;
    }
    let tag_len = tag_sequence(&s[len..]);
    len += tag_len;
    Some(Element { len, modifier, tag: tag_len > 0 })
}

/// The length of a terminated tag sequence at the start of `s`, 0 when
/// there is none.
fn tag_sequence(s: &str) -> usize {
    let mut len = 0;
    for c in s.chars() {
        match c {
            CANCEL_TAG if len > 0 => return len + CANCEL_TAG.len_utf8(),
            FIRST_TAG..=LAST_TAG => len += c.len_utf8(),
            _ => return 0,
        }
    }
    0
}
