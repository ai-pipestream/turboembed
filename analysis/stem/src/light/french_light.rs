//! `FrenchLightStemmer`: the "UniNE" algorithm of Savoy, "Light Stemming
//! Approaches for the French, Portuguese, German and Hungarian Languages"
//! (ACM SAC 2006), from Lucene.

use super::util::{ch, delete, ends_with, fold, is_letter, unit};

pub(crate) fn stem(s: &mut [u16], mut len: usize) -> usize {
    if len > 5 && ch(s[len - 1]) == 'x' {
        if ch(s[len - 3]) == 'a' && ch(s[len - 2]) == 'u' && ch(s[len - 4]) != 'e' {
            s[len - 2] = unit('l');
        }
        len -= 1;
    }
    if len > 3 && ch(s[len - 1]) == 'x' {
        len -= 1;
    }
    if len > 3 && ch(s[len - 1]) == 's' {
        len -= 1;
    }
    if len > 9 && ends_with(s, len, u16s!("issement")) {
        len -= 6;
        s[len - 1] = unit('r');
        return norm(s, len);
    }
    if len > 8 && ends_with(s, len, u16s!("issant")) {
        len -= 4;
        s[len - 1] = unit('r');
        return norm(s, len);
    }
    if len > 6 && ends_with(s, len, u16s!("ement")) {
        len -= 4;
        if len > 3 && ends_with(s, len, u16s!("ive")) {
            len -= 1;
            s[len - 1] = unit('f');
        }
        return norm(s, len);
    }
    if len > 11 && ends_with(s, len, u16s!("ficatrice")) {
        len -= 5;
        set(s, len - 2, u16s!("er"));
        return norm(s, len);
    }
    if len > 10 && ends_with(s, len, u16s!("ficateur")) {
        len -= 4;
        set(s, len - 2, u16s!("er"));
        return norm(s, len);
    }
    if len > 9 && ends_with(s, len, u16s!("catrice")) {
        len -= 3;
        // The last unit is already r.
        set(s, len - 4, u16s!("que"));
        return norm(s, len);
    }
    if len > 8 && ends_with(s, len, u16s!("cateur")) {
        len -= 2;
        set(s, len - 4, u16s!("quer"));
        return norm(s, len);
    }
    if len > 8 && ends_with(s, len, u16s!("atrice")) {
        len -= 4;
        set(s, len - 2, u16s!("er"));
        return norm(s, len);
    }
    if len > 7 && ends_with(s, len, u16s!("ateur")) {
        len -= 3;
        set(s, len - 2, u16s!("er"));
        return norm(s, len);
    }
    if len > 6 && ends_with(s, len, u16s!("trice")) {
        // Not a return: the word goes on through the rules below.
        len -= 1;
        set(s, len - 3, u16s!("eur"));
    }
    if len > 5 && ends_with(s, len, u16s!("ième")) {
        return norm(s, len - 4);
    }
    if len > 7 && ends_with(s, len, u16s!("teuse")) {
        len -= 2;
        s[len - 1] = unit('r');
        return norm(s, len);
    }
    if len > 6 && ends_with(s, len, u16s!("teur")) {
        len -= 1;
        s[len - 1] = unit('r');
        return norm(s, len);
    }
    if len > 5 && ends_with(s, len, u16s!("euse")) {
        return norm(s, len - 2);
    }
    if len > 8 && ends_with(s, len, u16s!("ère")) {
        len -= 1;
        s[len - 2] = unit('e');
        return norm(s, len);
    }
    if len > 7 && ends_with(s, len, u16s!("ive")) {
        len -= 1;
        s[len - 1] = unit('f');
        return norm(s, len);
    }
    if len > 4 && (ends_with(s, len, u16s!("folle")) || ends_with(s, len, u16s!("molle"))) {
        len -= 2;
        s[len - 1] = unit('u');
        return norm(s, len);
    }
    if len > 9 && ends_with(s, len, u16s!("nnelle")) {
        return norm(s, len - 5);
    }
    if len > 9 && ends_with(s, len, u16s!("nnel")) {
        return norm(s, len - 3);
    }
    if len > 4 && ends_with(s, len, u16s!("ète")) {
        len -= 1;
        s[len - 2] = unit('e');
    }
    if len > 8 && ends_with(s, len, u16s!("ique")) {
        len -= 4;
    }
    if len > 8 && ends_with(s, len, u16s!("esse")) {
        return norm(s, len - 3);
    }
    if len > 7 && ends_with(s, len, u16s!("inage")) {
        return norm(s, len - 3);
    }
    if len > 9 && ends_with(s, len, u16s!("isation")) {
        len -= 7;
        if len > 5 && ends_with(s, len, u16s!("ual")) {
            s[len - 2] = unit('e');
        }
        return norm(s, len);
    }
    if len > 9 && ends_with(s, len, u16s!("isateur")) {
        return norm(s, len - 7);
    }
    if len > 8 && ends_with(s, len, u16s!("ation")) {
        return norm(s, len - 5);
    }
    if len > 8 && ends_with(s, len, u16s!("ition")) {
        return norm(s, len - 5);
    }
    norm(s, len)
}

/// Writes `units` over `s` from `at`.
fn set(s: &mut [u16], at: usize, units: &[u16]) {
    s[at..at + units.len()].copy_from_slice(units);
}

fn norm(s: &mut [u16], mut len: usize) -> usize {
    if len > 4 {
        fold(&mut s[..len], u16s!("àáâôèéêùûîç"), u16s!("aaaoeeeuuic"));
        // Removes the second of two equal letters in a row.
        let mut prev = s[0];
        let mut i = 1;
        while i < len {
            if s[i] == prev && is_letter(prev) {
                len = delete(s, i, len);
            } else {
                prev = s[i];
                i += 1;
            }
        }
    }
    if len > 4 && ends_with(s, len, u16s!("ie")) {
        len -= 2;
    }
    if len > 4 {
        if ch(s[len - 1]) == 'r' {
            len -= 1;
        }
        if ch(s[len - 1]) == 'e' {
            len -= 1;
        }
        if ch(s[len - 1]) == 'e' {
            len -= 1;
        }
        if s[len - 1] == s[len - 2] && is_letter(s[len - 1]) {
            len -= 1;
        }
    }
    len
}
