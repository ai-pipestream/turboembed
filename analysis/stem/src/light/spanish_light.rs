//! `SpanishLightStemmer`: Savoy, "Report on CLEF-2001 Experiments", from
//! Lucene.

use super::util::{ch, fold, unit};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    if len < 5 {
        return len;
    }
    fold(&mut s[..len], u16s!("àáâäòóôöèéêëùúûüìíîï"), u16s!("aaaaooooeeeeuuuuiiii"));
    match ch(s[len - 1]) {
        'o' | 'a' | 'e' => len - 1,
        's' => {
            let (a, b, c) = (ch(s[len - 2]), ch(s[len - 3]), ch(s[len - 4]));
            if a == 'e' && b == 's' && c == 'e' {
                return len - 2;
            }
            if a == 'e' && b == 'c' {
                s[len - 3] = unit('z');
                return len - 2;
            }
            if matches!(a, 'o' | 'a' | 'e') {
                return len - 2;
            }
            len
        }
        _ => len,
    }
}
