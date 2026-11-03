//! Unicode L and N ranges used by tokenizer pre-tokenizers.
//! Regenerate with bpe::tests::write_classes.
mod letters_a;
mod letters_b;
mod numbers;
#[cfg(test)]
pub(crate) mod tests;
mod word;
/// First and last scalar of each Unicode letter run.
pub const LETTERS: &[(u32, u32)] = &letter_ranges();
/// First and last scalar of each Unicode number run.
pub const NUMBERS: &[(u32, u32)] = &numbers::RANGES;
const COUNT: usize = letters_a::RANGES.len() + letters_b::RANGES.len();
const fn letter_ranges() -> [(u32, u32); COUNT] {
    let mut out = [(0, 0); COUNT];
    let mut i = 0;
    while i < letters_a::RANGES.len() {
        out[i] = letters_a::RANGES[i];
        i += 1;
    }
    let mut j = 0;
    while j < letters_b::RANGES.len() {
        out[i + j] = letters_b::RANGES[j];
        j += 1;
    }
    out
}

/// Added-token boundaries use the regex engine's word class, separately
/// from the older category tables used by BertNormalizer.
pub fn is_word(c: char) -> bool {
    let u = c as u32;
    match word::DIFFERENCES.binary_search_by(|&(a, b, _)| {
        if b < u {
            std::cmp::Ordering::Less
        } else if a > u {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => word::DIFFERENCES[i].2,
        Err(_) => current_word(c),
    }
}

fn current_word(c: char) -> bool {
    let category = turbo_unicode::category::general_category(c);
    turbo_unicode::props::alphabetic(c)
        || category.is_mark()
        || matches!(category, turbo_unicode::GeneralCategory::Nd | turbo_unicode::GeneralCategory::Pc)
        || matches!(c, '\u{200c}' | '\u{200d}')
}
