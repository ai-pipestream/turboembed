//! A deterministic generator of texts that stress the annotators: the
//! corpus `turbo-goldens fuzz` runs through the JVM and the Rust ports, and
//! the one the offset invariants are checked on in the ordinary tests.
//!
//! `corpus(seed, count)` gives `count` inputs; input `index` of a seed is
//! the same whatever `count` is and whatever came before it, so one input
//! can be made again on its own (`input(seed, index)`) to reproduce a
//! failure. The texts come from the grammar in `grammar.rs`: sentences of
//! a small English grammar with abbreviations, numbers, money, quantities,
//! URLs and quotes in them, lines of other scripts (CJK, Arabic and Hebrew
//! with bidi controls, Devanagari, Thai), emoji sequences, very long
//! tokens, empty and whitespace only lines, CR LF, and then zero width and
//! format characters, combining marks and Unicode spaces dropped in. The
//! random source is SplitMix64 (`rng.rs`), written here, so the corpus
//! depends on no other crate.

mod grammar;
mod rng;

pub use rng::Rng;

/// One generated input: its id, which is its file name in an input set
/// (`s<seed>-<index>`, five digits), and its text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuzzInput {
    pub seed: u64,
    pub index: u64,
    pub id: String,
    pub text: String,
}

/// The id of input `index` of a corpus made with `seed`.
pub fn input_id(seed: u64, index: u64) -> String {
    format!("s{seed}-{index:05}")
}

/// The seed and index an id names, when it is one `input_id` made.
pub fn parse_input_id(id: &str) -> Option<(u64, u64)> {
    let (seed, index) = id.strip_prefix('s')?.split_once('-')?;
    Some((seed.parse().ok()?, index.parse().ok()?))
}

/// Input `index` of the corpus made with `seed`.
pub fn input(seed: u64, index: u64) -> FuzzInput {
    let mut rng = Rng::for_input(seed, index);
    let text = grammar::Builder::new(&mut rng).document();
    FuzzInput { seed, index, id: input_id(seed, index), text }
}

/// The first `count` inputs of the corpus made with `seed`.
pub fn corpus(seed: u64, count: u64) -> Vec<FuzzInput> {
    (0..count).map(|i| input(seed, i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_corpus_is_deterministic_and_prefix_stable() {
        let a = corpus(42, 20);
        let b = corpus(42, 30);
        assert_eq!(a, b[..20]);
        assert_eq!(input(42, 7), a[7]);
        assert_ne!(corpus(43, 20), a);
        assert_eq!(a[3].id, "s42-00003");
        assert_eq!(parse_input_id("s42-00003"), Some((42, 3)));
        assert_eq!(parse_input_id("en-00003"), None);
    }

    #[test]
    fn the_texts_are_varied() {
        let texts = corpus(1, 400);
        let distinct: std::collections::BTreeSet<&str> = texts.iter().map(|t| t.text.as_str()).collect();
        assert!(distinct.len() > 390, "{} distinct texts of 400", distinct.len());
        let has = |f: fn(&str) -> bool| texts.iter().filter(|t| f(&t.text)).count();
        assert!(has(|t| t.is_empty() || t.chars().all(char::is_whitespace)) > 0, "an empty or blank text");
        assert!(has(|t| t.contains("\r\n")) > 10, "CR LF line ends");
        assert!(has(|t| t.contains('\u{200B}') || t.contains('\u{200D}')) > 10, "zero width characters");
        assert!(has(|t| t.contains('\u{202E}') || t.contains('\u{2067}')) > 5, "bidi controls");
        assert!(has(|t| t.chars().any(|c| c as u32 > 0xFFFF)) > 40, "characters outside the BMP");
        assert!(has(|t| t.contains('\u{00A0}') || t.contains('\u{3000}')) > 20, "Unicode spaces");
        assert!(has(|t| t.split_whitespace().any(|w| w.len() > 300)) > 10, "very long tokens");
        assert!(has(|t| t.contains("Dr.") || t.contains("e.g.")) > 30, "abbreviations");
        assert!(has(|t| t.contains('$') || t.contains('€')) > 50, "money");
        assert!(has(|t| t.contains("http")) > 30, "URLs");
        assert!(has(|t| t.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))) > 40, "CJK");
        assert!(has(|t| t.chars().any(|c| ('\u{0600}'..='\u{06FF}').contains(&c))) > 15, "Arabic");
        assert!(has(|t| t.chars().any(|c| ('\u{0E00}'..='\u{0E7F}').contains(&c))) > 5, "Thai");
        assert!(has(|t| t.chars().any(|c| ('\u{0900}'..='\u{097F}').contains(&c))) > 5, "Devanagari");
    }

    #[test]
    fn every_text_is_in_the_unicode_15_repertoire_and_has_no_noncharacter() {
        // The generator's character lists are fixed, so this checks them
        // once: nothing from a block assigned after Unicode 15.0, and no
        // noncharacter, which the Java side would decode but a reader
        // should never see in an input.
        for t in corpus(3, 300) {
            for c in t.text.chars() {
                let u = c as u32;
                assert!(!(0xFDD0..=0xFDEF).contains(&u) && (u & 0xFFFE) != 0xFFFE, "noncharacter {u:04X}");
                assert!(!(0x2FFC..=0x2FFF).contains(&u), "{u:04X} is new in Unicode 15.1");
                assert!(!(0x31EF..=0x31EF).contains(&u), "{u:04X} is new in Unicode 15.1");
                assert!(u != 0x1C89 && u != 0x1C8A, "{u:04X} is new in Unicode 16.0");
            }
        }
    }
}
