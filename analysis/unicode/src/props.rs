//! Binary character properties from PropList.txt,
//! DerivedCoreProperties.txt and emoji-data.txt.

use crate::tables::props::*;
use crate::trie::lookup;

/// The property bits of `c`, one per constant of `tables::props`.
#[inline]
pub(crate) fn bits(c: char) -> u32 {
    lookup(&INDEX, &BLOCKS, c)
}

#[inline]
fn has(c: char, bit: u32) -> bool {
    bits(c) & bit != 0
}

/// White_Space (PropList.txt).
pub fn white_space(c: char) -> bool {
    has(c, WHITE_SPACE)
}

/// Default_Ignorable_Code_Point (DerivedCoreProperties.txt).
pub fn default_ignorable(c: char) -> bool {
    has(c, DEFAULT_IGNORABLE)
}

/// Dash (PropList.txt).
pub fn dash(c: char) -> bool {
    has(c, DASH)
}

/// Quotation_Mark (PropList.txt).
pub fn quotation_mark(c: char) -> bool {
    has(c, QUOTATION_MARK)
}

/// Alphabetic (DerivedCoreProperties.txt).
pub fn alphabetic(c: char) -> bool {
    has(c, ALPHABETIC)
}

/// Lowercase (DerivedCoreProperties.txt).
pub fn lowercase(c: char) -> bool {
    has(c, LOWERCASE)
}

/// Uppercase (DerivedCoreProperties.txt).
pub fn uppercase(c: char) -> bool {
    has(c, UPPERCASE)
}

/// Cased (DerivedCoreProperties.txt): Lowercase, Uppercase or Lt.
pub fn cased(c: char) -> bool {
    has(c, CASED)
}

/// Case_Ignorable (DerivedCoreProperties.txt).
pub fn case_ignorable(c: char) -> bool {
    has(c, CASE_IGNORABLE)
}

/// Regional_Indicator (PropList.txt): U+1F1E6 to U+1F1FF.
pub fn regional_indicator(c: char) -> bool {
    has(c, REGIONAL_INDICATOR)
}

/// Variation_Selector (PropList.txt).
pub fn variation_selector(c: char) -> bool {
    has(c, VARIATION_SELECTOR)
}

/// Emoji (emoji-data.txt).
pub fn emoji(c: char) -> bool {
    has(c, EMOJI)
}

/// Emoji_Presentation (emoji-data.txt): shown as emoji by default.
pub fn emoji_presentation(c: char) -> bool {
    has(c, EMOJI_PRESENTATION)
}

/// Emoji_Modifier (emoji-data.txt): the five skin tone modifiers.
pub fn emoji_modifier(c: char) -> bool {
    has(c, EMOJI_MODIFIER)
}

/// Emoji_Modifier_Base (emoji-data.txt).
pub fn emoji_modifier_base(c: char) -> bool {
    has(c, EMOJI_MODIFIER_BASE)
}

/// Emoji_Component (emoji-data.txt).
pub fn emoji_component(c: char) -> bool {
    has(c, EMOJI_COMPONENT)
}

/// Extended_Pictographic (emoji-data.txt).
pub fn extended_pictographic(c: char) -> bool {
    has(c, EXTENDED_PICTOGRAPHIC)
}
