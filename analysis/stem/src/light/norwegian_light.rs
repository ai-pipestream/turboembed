//! `NorwegianLightStemmer`: Lucene's light stemmer for Norwegian, for
//! Bokmål, Nynorsk or both. Parts of it derive from the Swedish light
//! stemmer.

use super::NorwegianStandard;
use super::util::{ch, ends_with};

pub(crate) fn stem(s: &[u16], mut len: usize, standard: NorwegianStandard) -> usize {
    let (bokmaal, nynorsk) = (standard.bokmaal(), standard.nynorsk());
    let ends = |len: usize, suffix: &[u16]| ends_with(s, len, suffix);
    // A possessive -s goes, and the word goes on through the rules.
    if len > 4 && ch(s[len - 1]) == 's' {
        len -= 1;
    }
    if len > 7
        && ((bokmaal && (ends(len, u16s!("heter")) || ends(len, u16s!("heten"))))
            || (nynorsk && ends(len, u16s!("heita"))))
    {
        return len - 5;
    }
    if len > 8 && nynorsk && (ends(len, u16s!("heiter")) || ends(len, u16s!("leiken")) || ends(len, u16s!("leikar"))) {
        return len - 6;
    }
    if len > 5 && (ends(len, u16s!("dom")) || (bokmaal && ends(len, u16s!("het")))) {
        return len - 3;
    }
    if len > 6 && nynorsk && (ends(len, u16s!("heit")) || ends(len, u16s!("semd")) || ends(len, u16s!("leik"))) {
        return len - 4;
    }
    if len > 7 && (ends(len, u16s!("elser")) || ends(len, u16s!("elsen"))) {
        return len - 5;
    }
    if len > 6
        && ((bokmaal && ends(len, u16s!("ende")))
            || (nynorsk && ends(len, u16s!("ande")))
            || ends(len, u16s!("else"))
            || (bokmaal && ends(len, u16s!("este")))
            || (nynorsk && ends(len, u16s!("aste")))
            || (bokmaal && ends(len, u16s!("eren")))
            || (nynorsk && ends(len, u16s!("aren"))))
    {
        return len - 4;
    }
    if len > 5
        && ((bokmaal && ends(len, u16s!("ere")))
            || (nynorsk && ends(len, u16s!("are")))
            || (bokmaal && ends(len, u16s!("est")))
            || (nynorsk && ends(len, u16s!("ast")))
            || ends(len, u16s!("ene"))
            || (nynorsk && ends(len, u16s!("ane"))))
    {
        return len - 3;
    }
    if len > 4
        && (ends(len, u16s!("er"))
            || ends(len, u16s!("en"))
            || ends(len, u16s!("et"))
            || (nynorsk && ends(len, u16s!("ar")))
            || (bokmaal && ends(len, u16s!("st")))
            || ends(len, u16s!("te")))
    {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 'a' | 'e' | 'n') {
        return len - 1;
    }
    len
}
