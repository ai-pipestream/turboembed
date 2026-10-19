//! Decoding flag emoji to region codes, following OpenNLP's `EmojiFlags`.

const FIRST_REGIONAL_INDICATOR: u32 = 0x1F1E6;
const LAST_REGIONAL_INDICATOR: u32 = 0x1F1FF;
const WAVING_BLACK_FLAG: char = '\u{1F3F4}';
const CANCEL_TAG: char = '\u{E007F}';
/// A tag character is the ASCII character at its code point minus this offset.
const TAG_OFFSET: u32 = 0xE0000;

/// The region code a flag emoji encodes: two regional indicators decode to
/// an ISO 3166-1 alpha-2 code (U+1F1E9 U+1F1EA is `"DE"`), and U+1F3F4
/// followed by tag letters and digits and U+E007F decodes to an ISO 3166-2
/// code in uppercase with a hyphen after the region (`"GB-ENG"`).
///
/// `sequence` must be exactly one flag; anything else, including a lone
/// regional indicator, three regional indicators, an unterminated tag
/// sequence or a tag code shorter than three characters, gives `None`.
/// Whether the code is assigned is not checked: U+1F1FD U+1F1FD gives
/// `"XX"`.
pub fn region(sequence: &str) -> Option<String> {
    let mut chars = sequence.chars();
    let first = chars.next()?;
    if is_regional_indicator(first) {
        let second = chars.next().filter(|&c| is_regional_indicator(c))?;
        if chars.next().is_some() {
            return None;
        }
        return Some([letter(first), letter(second)].iter().collect());
    }
    if first == WAVING_BLACK_FLAG {
        return tag_region(chars.as_str());
    }
    None
}

fn is_regional_indicator(c: char) -> bool {
    (FIRST_REGIONAL_INDICATOR..=LAST_REGIONAL_INDICATOR).contains(&(c as u32))
}

fn letter(indicator: char) -> char {
    (b'A' + (indicator as u32 - FIRST_REGIONAL_INDICATOR) as u8) as char
}

/// Decodes the tag characters after U+1F3F4: lowercase tag letters and tag
/// digits ended by CANCEL TAG and nothing after it.
fn tag_region(tags: &str) -> Option<String> {
    let mut code = String::with_capacity(8);
    let mut chars = tags.chars();
    loop {
        let c = chars.next()?;
        if c == CANCEL_TAG {
            break;
        }
        let ascii = char::from_u32((c as u32).checked_sub(TAG_OFFSET)?)?;
        if !ascii.is_ascii_lowercase() && !ascii.is_ascii_digit() {
            return None;
        }
        if code.len() == 2 {
            code.push('-');
        }
        code.push(ascii.to_ascii_uppercase());
    }
    let region_is_letters = code.chars().take(2).all(|c| c.is_ascii_uppercase());
    (chars.next().is_none() && code.len() >= 4 && region_is_letters).then_some(code)
}
