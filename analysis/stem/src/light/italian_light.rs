//! `ItalianLightStemmer`: Savoy, "Report on CLEF-2001 Experiments", from
//! Lucene.

use super::util::{ch, fold};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    if len < 6 {
        return len;
    }
    fold(&mut s[..len], u16s!("àáâäòóôöèéêëùúûüìíîï"), u16s!("aaaaooooeeeeuuuuiiii"));
    let before = ch(s[len - 2]);
    match ch(s[len - 1]) {
        'e' if before == 'i' || before == 'h' => len - 2,
        'i' if before == 'h' || before == 'i' => len - 2,
        'a' | 'o' if before == 'i' => len - 2,
        'e' | 'i' | 'a' | 'o' => len - 1,
        _ => len,
    }
}
