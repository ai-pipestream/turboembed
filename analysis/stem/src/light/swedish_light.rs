//! `SwedishLightStemmer`: Savoy, "Report on CLEF-2003 Monolingual Tracks",
//! from Lucene.

use super::util::{ch, ends_any};

pub(crate) fn stem(s: &[u16], mut len: usize) -> usize {
    if len > 4 && ch(s[len - 1]) == 's' {
        len -= 1;
    }
    if len > 7 && ends_any(s, len, &[u16s!("elser"), u16s!("heten")]) {
        return len - 5;
    }
    let fours =
        [u16s!("arne"), u16s!("erna"), u16s!("ande"), u16s!("else"), u16s!("aste"), u16s!("orna"), u16s!("aren")];
    if len > 6 && ends_any(s, len, &fours) {
        return len - 4;
    }
    if len > 5 && ends_any(s, len, &[u16s!("are"), u16s!("ast"), u16s!("het")]) {
        return len - 3;
    }
    let twos = [u16s!("ar"), u16s!("er"), u16s!("or"), u16s!("en"), u16s!("at"), u16s!("te"), u16s!("et")];
    if len > 4 && ends_any(s, len, &twos) {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 't' | 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}
