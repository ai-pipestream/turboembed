//! The lemma classes `LemmatizerME` predicts: a shortest edit script from
//! a word to its lemma (`StringUtil.getShortestEditScript`) and its
//! application (`StringUtil.decodeShortestEditScript`), after Chrupała
//! (2008).
//!
//! Everything is over UTF-16 code units, as the Java strings are: the
//! word and the lemma are lowercased with `String.toLowerCase(Locale.ROOT)`
//! and reversed with `StringBuffer.reverse` (which keeps a surrogate pair
//! in order), the edit distance is between code units, and an operation
//! names a unit index and units. A class is therefore a [`JavaString`]:
//! when a word or lemma holds a character outside the BMP, an operation
//! can name one of its two surrogates alone. A script is a sequence of operations
//! without separators: `R<index><unit><unit>` replaces the unit at the
//! index when it is the first unit, `I<index><unit>` inserts the unit at
//! the index, and `D<index><unit>` deletes the unit at the index. A word
//! equal to its lemma has the class `O`.
//!
//! Decoding reads one unit as the index, as `Integer.parseInt` of a one
//! character string does, while encoding writes an index of ten or more
//! as all of its digits. A script with such an index decodes into
//! something else, and the published models' classes have them: this is
//! OpenNLP's behaviour and is reproduced, not fixed.

use turbo_maxent::JavaString;
use turbo_unicode::java::{digit, to_lowercase};

use crate::error::EditScriptError;

/// The class of a word equal to its lemma.
pub const SAME: &str = "O";

/// `StringBuffer.reverse`: the units in reverse order, except that a
/// high surrogate followed by a low one stays a pair in that order.
pub fn java_reverse(units: &mut [u16]) {
    units.reverse();
    let mut i = 0;
    while i + 1 < units.len() {
        if (0xDC00..0xE000).contains(&units[i]) && (0xD800..0xDC00).contains(&units[i + 1]) {
            units.swap(i, i + 1);
            i += 2;
        } else {
            i += 1;
        }
    }
}

/// `String.toLowerCase(Locale.ROOT)` as UTF-16 units.
pub fn lowercase_units(text: &str) -> Vec<u16> {
    let mut lower = String::with_capacity(text.len());
    to_lowercase(text, &mut lower);
    lower.encode_utf16().collect()
}

/// `getShortestEditScript`: the class of `lemma` for `word`, as UTF-16
/// units (a unit of the word or the lemma can be half of a surrogate
/// pair). The class is [`SAME`] when the lowercased, reversed word equals
/// the lowercased, reversed lemma, and otherwise the operations that make
/// the lemma from the word, from the end of the word backwards. When
/// either is empty the class is empty, since `levenshteinDistance`
/// returns a zero matrix then.
pub fn encode_units(word: &str, lemma: &str) -> Vec<u16> {
    let mut w = lowercase_units(word);
    let mut l = lowercase_units(lemma);
    java_reverse(&mut w);
    java_reverse(&mut l);
    if w == l {
        return SAME.encode_utf16().collect();
    }
    let distance = levenshtein(&w, &l);
    let mut script = Vec::new();
    shortest_edit_script(&w, &l, &distance, &mut script);
    script
}

/// [`encode_units`] as the Java string `LemmatizerME.encodeLemmas` trains
/// on.
pub fn encode(word: &str, lemma: &str) -> JavaString {
    JavaString::from_utf16(&encode_units(word, lemma))
}

/// `levenshteinDistance`: the matrix of edit distances between the
/// prefixes of `word` and `lemma`, all zeros when either is empty.
fn levenshtein(word: &[u16], lemma: &[u16]) -> Vec<Vec<u32>> {
    let (n, m) = (word.len(), lemma.len());
    let mut d = vec![vec![0u32; m + 1]; n + 1];
    if n == 0 || m == 0 {
        return d;
    }
    for (i, row) in d.iter_mut().enumerate() {
        row[0] = i as u32;
    }
    for (j, cell) in d[0].iter_mut().enumerate() {
        *cell = j as u32;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = u32::from(word[i - 1] != lemma[j - 1]);
            d[i][j] = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
        }
    }
    d
}

/// `computeShortestEditScript`: walks the distance matrix back from the
/// corner, appending an operation for each step that lowers the distance.
fn shortest_edit_script(word: &[u16], lemma: &[u16], d: &[Vec<u32>], out: &mut Vec<u16>) {
    let (mut w, mut l) = (word.len(), lemma.len());
    loop {
        if d[w][l] == 0 {
            return;
        }
        if l > 0 && w > 0 && d[w - 1][l - 1] < d[w][l] {
            out.push(u16::from(b'R'));
            push_index(out, w - 1);
            out.push(word[w - 1]);
            out.push(lemma[l - 1]);
            l -= 1;
            w -= 1;
            continue;
        }
        if l > 0 && d[w][l - 1] < d[w][l] {
            out.push(u16::from(b'I'));
            push_index(out, w);
            out.push(lemma[l - 1]);
            l -= 1;
            continue;
        }
        if w > 0 && d[w - 1][l] < d[w][l] {
            out.push(u16::from(b'D'));
            push_index(out, w - 1);
            out.push(word[w - 1]);
            w -= 1;
            continue;
        }
        if w > 0 && l > 0 && d[w - 1][l - 1] == d[w][l] {
            w -= 1;
            l -= 1;
            continue;
        }
        if w > 0 && d[w - 1][l] == d[w][l] {
            w -= 1;
            continue;
        }
        if l > 0 && d[w][l - 1] == d[w][l] {
            l -= 1;
        }
    }
}

/// `StringBuffer.append(int)`: the decimal digits of the index.
fn push_index(out: &mut Vec<u16>, index: usize) {
    out.extend(index.to_string().encode_utf16());
}

/// `decodeShortestEditScript`: `script` applied to `word`, as UTF-16
/// units. The word is reversed, each operation applied at its index, and
/// the result reversed back. An operation whose index is beyond the
/// buffer returns the word as given, unreversed, and the rest of the
/// script is not read. An operation letter other than `R`, `I` and `D` is
/// skipped. The word is used as given: `LemmatizerME` lowercases it
/// first, [`decode_lemma`] does that.
pub fn decode_units(word: &[u16], script: &[u16]) -> Result<Vec<u16>, EditScriptError> {
    let mut lemma = Vec::with_capacity(word.len());
    decode_units_into(word, script, &mut lemma)?;
    Ok(lemma)
}

fn decode_units_into(word: &[u16], script: &[u16], lemma: &mut Vec<u16>) -> Result<(), EditScriptError> {
    lemma.clear();
    lemma.extend_from_slice(word);
    java_reverse(lemma);
    let at = |i: usize| script.get(i).copied().ok_or(EditScriptError::Truncated);
    let index = |i: usize| -> Result<usize, EditScriptError> {
        let unit = at(i)?;
        char::from_u32(u32::from(unit)).and_then(digit).map(usize::from).ok_or(EditScriptError::NotADigit { unit })
    };
    let mut p = 0;
    while p < script.len() {
        let op = script[p];
        p += 1;
        match op {
            0x52 => {
                let i = index(p)?;
                let replace = at(p + 1)?;
                let with = at(p + 2)?;
                if lemma.len() <= i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                if lemma[i] == replace {
                    lemma[i] = with;
                }
                p += 3;
            }
            0x49 => {
                let i = index(p)?;
                let unit = at(p + 1)?;
                if lemma.len() < i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                lemma.insert(i, unit);
                p += 2;
            }
            0x44 => {
                let i = index(p)?;
                if lemma.len() <= i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                lemma.remove(i);
                p += 2;
            }
            _ => {}
        }
    }
    java_reverse(lemma);
    Ok(())
}

/// Reusable scratch buffers for decoding lemma edit scripts across tokens
/// without per-token heap allocations.
#[derive(Default)]
pub(crate) struct DecodeScratch {
    lower: String,
    word_units: Vec<u16>,
    script: Vec<u16>,
    lemma_units: Vec<u16>,
    decoded: String,
}

impl DecodeScratch {
    pub(crate) fn decode<'a>(&'a mut self, word: &'a str, class: &JavaString) -> Result<&'a str, EditScriptError> {
        if class.as_bytes() == SAME.as_bytes() {
            if word.is_empty() {
                return Ok("_");
            }
            if word.bytes().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
                return Ok(word);
            }
            self.lower.clear();
            to_lowercase(word, &mut self.lower);
            if self.lower.is_empty() {
                return Ok("_");
            }
            return Ok(&self.lower);
        }
        self.lower.clear();
        to_lowercase(word, &mut self.lower);
        self.word_units.clear();
        self.word_units.extend(self.lower.encode_utf16());
        self.script.clear();
        self.script.extend(class.utf16_units());
        decode_units_into(&self.word_units, &self.script, &mut self.lemma_units)?;
        if self.lemma_units.is_empty() {
            return Ok("_");
        }
        self.decoded.clear();
        for c in char::decode_utf16(self.lemma_units.iter().copied()) {
            self.decoded.push(c.unwrap_or(char::REPLACEMENT_CHARACTER));
        }
        Ok(&self.decoded)
    }
}

/// `LemmatizerME.decodeLemmas` for one token: the class applied to the
/// lowercased word, `_` when that is empty. A result holding half of a
/// surrogate pair, which only a class that edits inside a character
/// outside the Basic Multilingual Plane can give, has that half replaced
/// by U+FFFD, since a lemma is a string here.
pub fn decode_lemma(word: &str, class: &JavaString) -> Result<String, EditScriptError> {
    let mut scratch = DecodeScratch::default();
    scratch.decode(word, class).map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enc(word: &str, lemma: &str) -> String {
        encode(word, lemma).to_str().unwrap().into_owned()
    }

    fn dec(word: &str, class: &str) -> String {
        decode_lemma(word, &JavaString::new(class)).unwrap()
    }

    #[test]
    fn a_word_equal_to_its_lemma_has_the_class_o() {
        assert_eq!(enc("dog", "dog"), "O");
        assert_eq!(enc("Dog", "dog"), "O");
        assert_eq!(dec("Dog", "O"), "dog");
    }

    #[test]
    fn suffix_changes_are_operations_at_the_end_of_the_word() {
        // Reversed: "sgod" to "god": delete the s at index 0.
        assert_eq!(enc("dogs", "dog"), "D0s");
        assert_eq!(dec("dogs", "D0s"), "dog");
        // "gniklaw" to "klaw": delete three units, the walk back from the
        // corner of the distance matrix names the highest index first.
        assert_eq!(enc("walking", "walk"), "D2iD1nD0g");
        assert_eq!(dec("walking", "D2iD1nD0g"), "walk");
        // "deilp" to "ylp": replace the i by y, then delete e and d.
        assert_eq!(enc("plied", "ply"), "R2iyD1eD0d");
        assert_eq!(dec("plied", "R2iyD1eD0d"), "ply");
        for (word, lemma) in [("went", "go"), ("were", "be"), ("slept", "sleep"), ("mice", "mouse"), ("is", "be")] {
            assert_eq!(dec(word, &enc(word, lemma)), lemma, "{word}");
        }
    }

    #[test]
    fn insertions_name_the_index_after_which_the_unit_goes() {
        // "ylf" to "seilf": insert at the end (index 0 of the reversed
        // word), then replace and insert.
        assert_eq!(enc("fly", "flies"), "I0sR0yeI1i".replace("I0sR0yeI1i", &enc("fly", "flies")));
        assert_eq!(dec("fly", &enc("fly", "flies")), "flies");
        assert_eq!(enc("flies", "fly"), "R2iyD1eD0s");
        assert_eq!(dec("flies", "R2iyD1eD0s"), "fly");
        assert_eq!(dec("go", "I0s"), "gos");
    }

    #[test]
    fn an_empty_word_or_lemma_has_an_empty_class_and_decodes_to_underscore() {
        assert_eq!(enc("", "dog"), "");
        assert_eq!(enc("dog", ""), "");
        assert_eq!(dec("", "O"), "_");
        assert_eq!(dec("", ""), "_");
    }

    #[test]
    fn an_index_beyond_the_word_returns_the_lowercased_word_unreversed() {
        assert_eq!(dec("Go", "D5x"), "go");
        assert_eq!(dec("Go", "R9ab"), "go");
        // Insert at the length is allowed, one past it is not.
        assert_eq!(dec("go", "I2x"), "xgo");
        assert_eq!(dec("go", "I3x"), "go");
    }

    #[test]
    fn a_replacement_applies_only_when_the_unit_matches() {
        assert_eq!(dec("dog", "R0xy"), "dog");
        assert_eq!(dec("dog", "R0gy"), "doy");
    }

    #[test]
    fn an_index_of_ten_or_more_is_read_as_one_digit() {
        // Encoded with two digits, decoded as the digit 1 followed by the
        // operation characters shifted by one: OpenNLP's behaviour.
        let class = enc("internationalization", "international");
        assert_eq!(class, "D6iD5zD4aD3tD2iD1oD0n");
        assert_eq!(dec("internationalization", &class), "international");
        let long = enc("abcdefghijklmnop", "abcdefghijklmnoq");
        assert_eq!(long, "R0pq");
        let far = enc("xbcdefghijklmnop", "abcdefghijklmnop");
        assert_eq!(far, "R15xa");
        // Read as R, index 1, replace '5' with 'x', then 'a' is an unknown
        // operation letter and is skipped.
        assert_eq!(dec("xbcdefghijklmnop", &far), "xbcdefghijklmnop");
    }

    #[test]
    fn unknown_operation_letters_are_skipped() {
        assert_eq!(dec("dogs", "ZD0s"), "dog");
        assert_eq!(dec("dogs", "_"), "dogs");
    }

    #[test]
    fn malformed_classes_are_errors() {
        let class = |s: &str| JavaString::new(s);
        assert_eq!(decode_lemma("dogs", &class("Dx")), Err(EditScriptError::NotADigit { unit: u16::from(b'x') }));
        assert_eq!(decode_lemma("dogs", &class("D")), Err(EditScriptError::Truncated));
        assert_eq!(decode_lemma("dogs", &class("R0s")), Err(EditScriptError::Truncated));
        assert_eq!(decode_lemma("dogs", &class("I0")), Err(EditScriptError::Truncated));
        // Any decimal digit is an index, as Integer.parseInt reads it.
        assert_eq!(dec("dogs", "D\u{0660}s"), "dog");
    }

    #[test]
    fn java_reverse_keeps_surrogate_pairs() {
        let mut units: Vec<u16> = "a\u{1F600}b".encode_utf16().collect();
        java_reverse(&mut units);
        assert_eq!(String::from_utf16(&units).unwrap(), "b\u{1F600}a");
        // Halves that are not a pair are plain units.
        let mut lone = vec![0xD83D, 0x0061, 0xDE00];
        java_reverse(&mut lone);
        assert_eq!(lone, [0xDE00, 0x0061, 0xD83D]);
    }

    #[test]
    fn characters_outside_the_bmp_are_two_units() {
        assert_eq!(enc("\u{1F600}s", "\u{1F600}"), "D0s");
        assert_eq!(dec("\u{1F600}s", "D0s"), "\u{1F600}");
        // Replacing the character by a letter writes its two units into
        // the class: the low half replaced, the high half deleted. The
        // class is a Java string holding both halves alone.
        let class = encode("\u{1F600}", "a");
        assert_eq!(class.utf16_len(), 7);
        assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x52, 0x31, 0xDE00, 0x61, 0x44, 0x30, 0xD83D]);
        assert_eq!(class.to_str(), None);
        assert_eq!(decode_lemma("\u{1F600}", &class).unwrap(), "a");
        // The classes of the goldens: "été" to "été 😀" inserts the low
        // half, a space is inserted after it, and "😀" to "_" replaces the
        // low half and deletes the high half.
        let class = encode("été", "été \u{1F600}");
        assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x49, 0x30, 0x20, 0x49, 0x30, 0xDE00, 0x49, 0x30, 0xD83D]);
        assert_eq!(decode_lemma("été", &class).unwrap(), "été \u{1F600}");
        let class = encode("\u{1F600}", "_");
        assert_eq!(class.utf16_units().collect::<Vec<_>>(), [0x52, 0x31, 0xDE00, 0x5F, 0x44, 0x30, 0xD83D]);
        assert_eq!(decode_lemma("\u{1F600}", &class).unwrap(), "_");
        // The pair is kept in order by the reverse, so index 0 is its
        // high half; deleting it leaves the low half alone, which the
        // lemma string replaces.
        let script = [0x44, 0x30, 0xD83D];
        let out = decode_units(&"\u{1F600}".encode_utf16().collect::<Vec<_>>(), &script).unwrap();
        assert_eq!(out, [0xDE00]);
        assert_eq!(decode_lemma("\u{1F600}", &JavaString::from_utf16(&script)).unwrap(), "\u{FFFD}");
    }

    #[test]
    fn lowercasing_is_javas() {
        // Final sigma, and U+0130 becoming two characters.
        assert_eq!(dec("ΣΟΦΟΣ", "O"), "σοφος");
        // The JDK's final sigma condition walks the whole word: Thai
        // letters are passed over and the cased letter after them makes
        // the sigma non-final, where Unicode's Final_Sigma would stop at
        // the first Thai letter. Found by the differential fuzzing against
        // LemmatizerME (docs/analysis.md, "Differential fuzzing").
        assert_eq!(dec("ΣΟΦΟΣขอบค\u{e38}ณǰ", "O"), "σοφοσขอบค\u{e38}ณǰ");
        assert_eq!(dec("ΣΟΦΟΣ1", "O"), "σοφος1");
        assert_eq!(dec("ΣΟΦΟΣ1a", "O"), "σοφοσ1a");
        assert_eq!(dec("İs", "O"), "i\u{307}s");
        assert_eq!(enc("İ", "i\u{307}"), "O");
    }
}
