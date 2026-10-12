//! `PortugueseLightStemmer`: Savoy, "Light Stemming Approaches for the
//! French, Portuguese, German and Hungarian Languages" (ACM SAC 2006), from
//! Lucene.

use super::util::{ch, ends_with, fold, unit};

pub(crate) fn stem(s: &mut [u16], mut len: usize) -> usize {
    if len < 4 {
        return len;
    }
    len = remove_suffix(s, len);
    if len > 3 && ch(s[len - 1]) == 'a' {
        len = norm_feminine(s, len);
    }
    if len > 4 && matches!(ch(s[len - 1]), 'e' | 'a' | 'o') {
        len -= 1;
    }
    fold(&mut s[..len], u16s!("àáâäãòóôöõèéêëùúûüìíîïç"), u16s!("aaaaaoooooeeeeuuuuiiiic"));
    len
}

fn remove_suffix(s: &mut [u16], len: usize) -> usize {
    if len > 4 && ends_with(s, len, u16s!("es")) && matches!(ch(s[len - 3]), 'r' | 's' | 'l' | 'z') {
        return len - 2;
    }
    if len > 3 && ends_with(s, len, u16s!("ns")) {
        s[len - 2] = unit('m');
        return len - 1;
    }
    if len > 4 && (ends_with(s, len, u16s!("eis")) || ends_with(s, len, u16s!("éis"))) {
        s[len - 3] = unit('e');
        s[len - 2] = unit('l');
        return len - 1;
    }
    if len > 4 && ends_with(s, len, u16s!("ais")) {
        s[len - 2] = unit('l');
        return len - 1;
    }
    if len > 4 && ends_with(s, len, u16s!("óis")) {
        s[len - 3] = unit('o');
        s[len - 2] = unit('l');
        return len - 1;
    }
    if len > 4 && ends_with(s, len, u16s!("is")) {
        s[len - 1] = unit('l');
        return len;
    }
    if len > 3 && (ends_with(s, len, u16s!("ões")) || ends_with(s, len, u16s!("ães"))) {
        let len = len - 1;
        s[len - 2] = unit('ã');
        s[len - 1] = unit('o');
        return len;
    }
    if len > 6 && ends_with(s, len, u16s!("mente")) {
        return len - 5;
    }
    if len > 3 && ch(s[len - 1]) == 's' {
        return len - 1;
    }
    len
}

fn norm_feminine(s: &mut [u16], len: usize) -> usize {
    if len > 7
        && (ends_with(s, len, u16s!("inha")) || ends_with(s, len, u16s!("iaca")) || ends_with(s, len, u16s!("eira")))
    {
        s[len - 1] = unit('o');
        return len;
    }
    if len > 6 {
        let os = [u16s!("osa"), u16s!("ica"), u16s!("ida"), u16s!("ada"), u16s!("iva"), u16s!("ama")];
        if os.iter().any(|x| ends_with(s, len, x)) {
            s[len - 1] = unit('o');
            return len;
        }
        if ends_with(s, len, u16s!("ona")) {
            s[len - 3] = unit('ã');
            s[len - 2] = unit('o');
            return len - 1;
        }
        if ends_with(s, len, u16s!("ora")) {
            return len - 1;
        }
        if ends_with(s, len, u16s!("esa")) {
            s[len - 3] = unit('ê');
            return len - 1;
        }
        if ends_with(s, len, u16s!("na")) {
            s[len - 1] = unit('o');
            return len;
        }
    }
    len
}
