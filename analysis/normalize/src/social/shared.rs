//! Emoticons and laughter, which both social media normalizers find the
//! same way.

use turbo_analysis::ViewBuilder;

pub(super) fn is_mouth(b: u8) -> bool {
    matches!(b, b'(' | b')' | b'd' | b'D' | b'o' | b'O' | b'p' | b'P')
}

/// The end of the emoticon at byte `i`, which holds its eyes. A nose is
/// tried first.
pub(super) fn emoticon_end(bytes: &[u8], i: usize) -> Option<usize> {
    if bytes.get(i + 1) == Some(&b'-') && bytes.get(i + 2).is_some_and(|&b| is_mouth(b)) {
        return Some(i + 3);
    }
    bytes.get(i + 1).is_some_and(|&b| is_mouth(b)).then_some(i + 2)
}

fn is_laugh_consonant(b: u8) -> bool {
    matches!(b, b'h' | b'H' | b'j' | b'J')
}

fn is_laugh_vowel(b: u8) -> bool {
    matches!(b, b'a' | b'e' | b'i' | b'o' | b'u' | b'A' | b'E' | b'I' | b'O' | b'U')
}

/// The end of a run of bytes from `from` that equal `b` ASCII
/// case-insensitively.
fn run_of(bytes: &[u8], from: usize, b: u8) -> usize {
    from + bytes[from..].iter().take_while(|x| x.eq_ignore_ascii_case(&b)).count()
}

/// Laughter at byte `i`, which holds `h` or `j`: where it ends and the
/// consonant and vowel it shrinks to.
///
/// When there is none, the error is where to look next: every position up
/// to the end of the run of consonants that starts at `i` leads to the
/// same runs and fails the same way, and no other pattern starts with a
/// consonant, so the search skips them. This keeps a long run of `h` from
/// costing quadratic time.
pub(super) fn laughter_at(bytes: &[u8], i: usize) -> Result<(usize, u8, u8), usize> {
    let p = i + 1 + bytes[i + 1..].iter().take_while(|&&b| is_laugh_consonant(b)).count();
    if !bytes.get(p).is_some_and(|&b| is_laugh_vowel(b)) {
        return Err(p);
    }
    let q = p + 1 + bytes[p + 1..].iter().take_while(|&&b| is_laugh_vowel(b)).count();
    let (consonant, vowel) = (bytes[p - 1], bytes[q - 1]);
    let mut end = q;
    loop {
        let consonants = run_of(bytes, end, consonant);
        if consonants == end {
            break;
        }
        let vowels = run_of(bytes, consonants, vowel);
        if vowels == consonants {
            break;
        }
        end = vowels;
    }
    if end > q { Ok((end, consonant, vowel)) } else { Err(p) }
}

/// Writes laughter `bytes[i..end]` shrunk to `shrunk`. The part both
/// share at the start is kept, so it maps to itself, and the rest is one
/// edit (often a deletion: `hahaha` keeps `haha` and deletes `ha`).
pub(super) fn shrink_laughter(bytes: &[u8], i: usize, end: usize, shrunk: [u8; 4], out: &mut ViewBuilder<'_>) {
    let same = bytes[i..end].iter().zip(&shrunk).take_while(|(a, b)| a == b).count();
    if same == shrunk.len() && end - i == same {
        return;
    }
    out.keep(i + same);
    out.replace(end, std::str::from_utf8(&shrunk[same..]).expect("ASCII letters"));
}
