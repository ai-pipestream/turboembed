//! `SwedishMinimalStemmer`: Lucene's minimal stemmer for Swedish, the light
//! stemmer cut down to the common noun plural and definite endings, the
//! genitive -s and a final -a, -e or -n.

use super::util::{ch, ends_any, ends_with};

pub(crate) fn stem(s: &[u16], mut len: usize) -> usize {
    if len > 4 && ch(s[len - 1]) == 's' {
        len -= 1;
    }
    if len > 6 && ends_any(s, len, &[u16s!("arne"), u16s!("erna"), u16s!("arna"), u16s!("orna"), u16s!("aren")]) {
        return len - 4;
    }
    if len > 5 && ends_with(s, len, u16s!("are")) {
        return len - 3;
    }
    let twos = [u16s!("ar"), u16s!("at"), u16s!("er"), u16s!("et"), u16s!("or"), u16s!("en")];
    if len > 4 && ends_any(s, len, &twos) {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}
