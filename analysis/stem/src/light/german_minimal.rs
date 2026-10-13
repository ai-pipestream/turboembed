//! `GermanMinimalStemmer`: Savoy, "Morphologie et recherche
//! d'information", from Lucene.

use super::util::{ch, fold};

pub(crate) fn stem(s: &mut [u16], mut len: usize) -> usize {
    if len < 5 {
        return len;
    }
    fold(&mut s[..len], u16s!("äöü"), u16s!("aou"));
    if len > 6 && ch(s[len - 3]) == 'n' && ch(s[len - 2]) == 'e' && ch(s[len - 1]) == 'n' {
        return len - 3;
    }
    if len > 5 {
        let before = ch(s[len - 2]);
        match ch(s[len - 1]) {
            'n' | 's' | 'r' if before == 'e' => return len - 2,
            'e' if before == 's' => return len - 2,
            _ => {}
        }
    }
    if matches!(ch(s[len - 1]), 'n' | 'e' | 's' | 'r') {
        len -= 1;
    }
    len
}
