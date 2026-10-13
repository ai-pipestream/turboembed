//! `EnglishMinimalStemmer`: the "S-stemmer" of Harman, "How Effective Is
//! Suffixing?" (JASIS 42(1), 1991), from Lucene. It removes plural -s,
//! -es and -ies only.

use super::util::{ch, unit};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    if len < 3 || ch(s[len - 1]) != 's' {
        return len;
    }
    match ch(s[len - 2]) {
        'u' | 's' => len,
        'e' => {
            if len > 3 && ch(s[len - 3]) == 'i' && ch(s[len - 4]) != 'a' && ch(s[len - 4]) != 'e' {
                s[len - 3] = unit('y');
                return len - 2;
            }
            if matches!(ch(s[len - 3]), 'i' | 'a' | 'o' | 'e') {
                return len;
            }
            len - 1
        }
        _ => len - 1,
    }
}
