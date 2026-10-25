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
    encode_with_case(word, lemma, false)
}

// The native format keeps case and uses delimited indices.
pub(super) fn encode_with_case(word: &str, lemma: &str, keep_case: bool) -> Vec<u16> {
    let mut w = if keep_case { word.encode_utf16().collect() } else { lowercase_units(word) };
    let mut l = if keep_case { lemma.encode_utf16().collect() } else { lowercase_units(lemma) };
    java_reverse(&mut w);
    java_reverse(&mut l);
    if w == l {
        return SAME.encode_utf16().collect();
    }
    if !keep_case && (w.is_empty() || l.is_empty()) {
        return Vec::new();
    }
    let distance = levenshtein(&w, &l);
    let mut script = Vec::new();
    shortest_edit_script(&w, &l, &distance, &mut script, keep_case);
    script
}

/// [`encode_units`] as the Java string `LemmatizerME.encodeLemmas` trains
/// on.
pub fn encode(word: &str, lemma: &str) -> JavaString {
    JavaString::from_utf16(&encode_units(word, lemma))
}

/// `levenshteinDistance`: the matrix of edit distances between the
/// prefixes of `word` and `lemma`.
fn levenshtein(word: &[u16], lemma: &[u16]) -> Vec<Vec<u32>> {
    let (n, m) = (word.len(), lemma.len());
    let mut d = vec![vec![0u32; m + 1]; n + 1];
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
fn shortest_edit_script(word: &[u16], lemma: &[u16], d: &[Vec<u32>], out: &mut Vec<u16>, delimited: bool) {
    let (mut w, mut l) = (word.len(), lemma.len());
    loop {
        if d[w][l] == 0 {
            return;
        }
        if l > 0 && w > 0 && d[w - 1][l - 1] < d[w][l] {
            out.push(u16::from(b'R'));
            push_index(out, w - 1, delimited);
            out.push(word[w - 1]);
            out.push(lemma[l - 1]);
            l -= 1;
            w -= 1;
            continue;
        }
        if l > 0 && d[w][l - 1] < d[w][l] {
            out.push(u16::from(b'I'));
            push_index(out, w, delimited);
            out.push(lemma[l - 1]);
            l -= 1;
            continue;
        }
        if w > 0 && d[w - 1][l] < d[w][l] {
            out.push(u16::from(b'D'));
            push_index(out, w - 1, delimited);
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
fn push_index(out: &mut Vec<u16>, index: usize, delimited: bool) {
    out.extend(index.to_string().encode_utf16());
    if delimited {
        out.push(u16::from(b':'));
    }
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
    decode_units_into(word, script, &mut lemma, false)?;
    Ok(lemma)
}

fn decode_units_into(
    word: &[u16],
    script: &[u16],
    lemma: &mut Vec<u16>,
    delimited: bool,
) -> Result<(), EditScriptError> {
    lemma.clear();
    lemma.extend_from_slice(word);
    java_reverse(lemma);
    let at = |i: usize| script.get(i).copied().ok_or(EditScriptError::Truncated);
    let index = |p: &mut usize| -> Result<usize, EditScriptError> {
        if !delimited {
            let unit = at(*p)?;
            *p += 1;
            return char::from_u32(u32::from(unit))
                .and_then(digit)
                .map(usize::from)
                .ok_or(EditScriptError::NotADigit { unit });
        }
        let mut value = 0usize;
        let start = *p;
        loop {
            let unit = at(*p)?;
            *p += 1;
            if unit == u16::from(b':') && *p > start + 1 {
                return Ok(value);
            }
            if !(u16::from(b'0')..=u16::from(b'9')).contains(&unit) {
                return Err(EditScriptError::NotADigit { unit });
            }
            value = value
                .checked_mul(10)
                .and_then(|v| v.checked_add(usize::from(unit - u16::from(b'0'))))
                .ok_or(EditScriptError::IndexOverflow)?;
        }
    };
    let mut p = 0;
    while p < script.len() {
        let op = script[p];
        p += 1;
        match op {
            0x52 => {
                let i = index(&mut p)?;
                let replace = at(p)?;
                let with = at(p + 1)?;
                if lemma.len() <= i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                if lemma[i] == replace {
                    lemma[i] = with;
                }
                p += 2;
            }
            0x49 => {
                let i = index(&mut p)?;
                let unit = at(p)?;
                if lemma.len() < i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                lemma.insert(i, unit);
                p += 1;
            }
            0x44 => {
                let i = index(&mut p)?;
                let _unit = at(p)?;
                if lemma.len() <= i {
                    lemma.clear();
                    lemma.extend_from_slice(word);
                    return Ok(());
                }
                lemma.remove(i);
                p += 1;
            }
            _ if delimited => return Err(EditScriptError::Operation { unit: op }),
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
        self.decode_with_case(word, class, false)
    }

    pub(crate) fn decode_with_case<'a>(
        &'a mut self,
        word: &'a str,
        class: &JavaString,
        native: bool,
    ) -> Result<&'a str, EditScriptError> {
        let (keep_case, skip) = if native {
            match class.as_bytes().get(..2) {
                Some(b"K:") => (true, 2),
                Some(b"L:") => (false, 2),
                _ => return Err(EditScriptError::Truncated),
            }
        } else {
            (false, 0)
        };
        if class.as_bytes().get(skip..) == Some(SAME.as_bytes()) {
            if word.is_empty() {
                return Ok("_");
            }
            if keep_case || word.bytes().all(|b| b.is_ascii() && !b.is_ascii_uppercase()) {
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
        if keep_case {
            self.lower.push_str(word);
        } else {
            to_lowercase(word, &mut self.lower);
        }
        self.word_units.clear();
        self.word_units.extend(self.lower.encode_utf16());
        self.script.clear();
        self.script.extend(class.utf16_units().skip(skip));
        decode_units_into(&self.word_units, &self.script, &mut self.lemma_units, native)?;
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
mod tests;
