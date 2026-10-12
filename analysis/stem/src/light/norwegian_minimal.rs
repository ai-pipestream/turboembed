//! `NorwegianMinimalStemmer`: Lucene's minimal stemmer for Norwegian, for
//! Bokmål, Nynorsk or both: noun plural and definite endings and the
//! genitive -s only.

use super::NorwegianStandard;
use super::util::{ch, ends_with};

pub(crate) fn stem(s: &[u16], mut len: usize, standard: NorwegianStandard) -> usize {
    let nynorsk = standard.nynorsk();
    // A genitive -s goes, and the word goes on through the rules.
    if len > 4 && ch(s[len - 1]) == 's' {
        len -= 1;
    }
    if len > 5 && (ends_with(s, len, u16s!("ene")) || (nynorsk && ends_with(s, len, u16s!("ane")))) {
        return len - 3;
    }
    if len > 4
        && (ends_with(s, len, u16s!("er"))
            || ends_with(s, len, u16s!("en"))
            || ends_with(s, len, u16s!("et"))
            || (nynorsk && ends_with(s, len, u16s!("ar"))))
    {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 'a' | 'e') {
        return len - 1;
    }
    len
}
