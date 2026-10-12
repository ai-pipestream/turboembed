//! `FrenchMinimalStemmer`: Savoy, "A Stemming Procedure and Stopword List
//! for General French Corpora" (JASIS 50(10), 1999), from Lucene.

use super::util::{ch, is_letter, unit};

pub(crate) fn stem(s: &mut [u16], mut len: usize) -> usize {
    if len < 6 {
        return len;
    }
    if ch(s[len - 1]) == 'x' {
        if ch(s[len - 3]) == 'a' && ch(s[len - 2]) == 'u' {
            s[len - 2] = unit('l');
        }
        return len - 1;
    }
    for c in ['s', 'r', 'e', 'é'] {
        if ch(s[len - 1]) == c {
            len -= 1;
        }
    }
    if s[len - 1] == s[len - 2] && is_letter(s[len - 1]) {
        len -= 1;
    }
    len
}
