//! Every word boundary of the dictionary segmenter falls on an extended
//! grapheme cluster boundary.

use turbo_tokenize::complex::{ComplexScript, bundled, complex_script};
use turbo_unicode::grapheme_boundaries;

/// A small deterministic generator (xorshift64*), so the cases are the same on every run.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// The characters of each script's runs, then a few that interrupt them.
fn alphabets() -> Vec<Vec<char>> {
    let mut by_script = vec![Vec::new(); 4];
    for c in (0..=0xFFFFu32).filter_map(char::from_u32) {
        if let Some(s) = complex_script(c) {
            by_script[ComplexScript::ALL.iter().position(|&x| x == s).unwrap()].push(c);
        }
    }
    by_script
}

const OTHERS: [char; 8] = [' ', 'a', '1', '\u{200C}', '\u{200D}', '\u{0300}', '.', '\u{0E50}'];

fn words(script: ComplexScript) -> Vec<String> {
    let mut words = Vec::new();
    bundled().dictionary(script).unwrap().for_each_word(|w| words.push(w.to_owned()));
    words
}

/// Every segment starts where the one before ended, the segments cover
/// the text, and every boundary is a grapheme cluster boundary.
fn check(text: &str) {
    let graphemes: Vec<usize> = grapheme_boundaries(text).collect();
    let mut previous = 0u32;
    bundled().for_each_segment(text, |start, end| {
        assert_eq!(start, previous, "{text:?}");
        assert!(start < end, "{text:?}");
        previous = end;
        assert!(graphemes.binary_search(&(end as usize)).is_ok(), "{end} in {text:?} ({})", text.escape_unicode());
    });
    assert_eq!(previous as usize, text.len(), "{text:?}");
}

#[test]
fn random_characters() {
    let alphabets = alphabets();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for _ in 0..200_000 {
        let alphabet = &alphabets[rng.below(4)];
        let len = 1 + rng.below(12);
        let text: String =
            (0..len)
                .map(|_| {
                    if rng.below(10) == 0 {
                        OTHERS[rng.below(OTHERS.len())]
                    } else {
                        alphabet[rng.below(alphabet.len())]
                    }
                })
                .collect();
        check(&text);
    }
}

#[test]
fn words_with_damage() {
    let alphabets = alphabets();
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    for (i, &script) in ComplexScript::ALL.iter().enumerate() {
        let words = words(script);
        for _ in 0..50_000 {
            let mut text = String::new();
            for _ in 0..1 + rng.below(4) {
                let word = &words[rng.below(words.len())];
                let chars: Vec<char> = word.chars().collect();
                // Drop, repeat or insert a character now and then.
                for (j, &c) in chars.iter().enumerate() {
                    match rng.below(12) {
                        0 => {}
                        1 => {
                            text.push(c);
                            text.push(c);
                        }
                        2 => {
                            text.push(c);
                            text.push(alphabets[i][rng.below(alphabets[i].len())]);
                        }
                        _ => text.push(c),
                    }
                    if j == 0 && rng.below(20) == 0 {
                        text.push(OTHERS[rng.below(OTHERS.len())]);
                    }
                }
            }
            check(&text);
        }
    }
}
