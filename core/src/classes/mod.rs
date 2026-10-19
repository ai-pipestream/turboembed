//! Unicode L and N ranges used by tokenizer pre-tokenizers.
//! Regenerate with bpe::tests::write_classes.
mod letters_a;
mod letters_b;
mod numbers;
#[cfg(test)]
pub(crate) mod tests;
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
