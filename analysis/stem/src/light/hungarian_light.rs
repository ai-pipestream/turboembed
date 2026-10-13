//! `HungarianLightStemmer`: Savoy, "Light Stemming Approaches for the
//! French, Portuguese, German and Hungarian Languages" (ACM SAC 2006), from
//! Lucene.

use super::util::{ch, ends_any, ends_with, fold, is_vowel};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    fold(&mut s[..len], u16s!("áëéíóőõöúűũûü"), u16s!("aeeioooouuuuu"));
    let len = remove_case(s, len);
    let len = remove_possessive(s, len);
    let len = remove_plural(s, len);
    normalize(s, len)
}

fn remove_case(s: &[u16], len: usize) -> usize {
    if len > 6 && ends_with(s, len, u16s!("kent")) {
        return len - 4;
    }
    if len > 5 {
        let threes = [
            u16s!("nak"),
            u16s!("nek"),
            u16s!("val"),
            u16s!("vel"),
            u16s!("ert"),
            u16s!("rol"),
            u16s!("ban"),
            u16s!("ben"),
            u16s!("bol"),
            u16s!("nal"),
            u16s!("nel"),
            u16s!("hoz"),
            u16s!("hez"),
            u16s!("tol"),
        ];
        if ends_any(s, len, &threes) {
            return len - 3;
        }
        if ends_any(s, len, &[u16s!("al"), u16s!("el")]) && !is_vowel(s[len - 3]) && s[len - 3] == s[len - 4] {
            return len - 3;
        }
    }
    if len > 4 {
        let twos = [
            u16s!("at"),
            u16s!("et"),
            u16s!("ot"),
            u16s!("va"),
            u16s!("ve"),
            u16s!("ra"),
            u16s!("re"),
            u16s!("ba"),
            u16s!("be"),
            u16s!("ul"),
            u16s!("ig"),
        ];
        if ends_any(s, len, &twos) {
            return len - 2;
        }
        if ends_any(s, len, &[u16s!("on"), u16s!("en")]) && !is_vowel(s[len - 3]) {
            return len - 2;
        }
        match ch(s[len - 1]) {
            't' | 'n' => return len - 1,
            'a' | 'e' if s[len - 2] == s[len - 3] && !is_vowel(s[len - 2]) => return len - 2,
            _ => {}
        }
    }
    len
}

fn remove_possessive(s: &[u16], len: usize) -> usize {
    if len > 6 {
        if !is_vowel(s[len - 5]) && ends_any(s, len, &[u16s!("atok"), u16s!("otok"), u16s!("etek")]) {
            return len - 4;
        }
        if ends_any(s, len, &[u16s!("itek"), u16s!("itok")]) {
            return len - 4;
        }
    }
    if len > 5 {
        if !is_vowel(s[len - 4]) && ends_any(s, len, &[u16s!("unk"), u16s!("tok"), u16s!("tek")]) {
            return len - 3;
        }
        if is_vowel(s[len - 4]) && ends_with(s, len, u16s!("juk")) {
            return len - 3;
        }
        if ends_with(s, len, u16s!("ink")) {
            return len - 3;
        }
    }
    if len > 4 {
        let twos = [u16s!("am"), u16s!("em"), u16s!("om"), u16s!("ad"), u16s!("ed"), u16s!("od"), u16s!("uk")];
        if !is_vowel(s[len - 3]) && ends_any(s, len, &twos) {
            return len - 2;
        }
        if is_vowel(s[len - 3]) && ends_any(s, len, &[u16s!("nk"), u16s!("ja"), u16s!("je")]) {
            return len - 2;
        }
        if ends_any(s, len, &[u16s!("im"), u16s!("id"), u16s!("ik")]) {
            return len - 2;
        }
    }
    if len > 3 {
        match ch(s[len - 1]) {
            'a' | 'e' if !is_vowel(s[len - 2]) => return len - 1,
            'm' | 'd' if is_vowel(s[len - 2]) => return len - 1,
            'i' => return len - 1,
            _ => {}
        }
    }
    len
}

fn remove_plural(s: &[u16], len: usize) -> usize {
    if len > 3 && ch(s[len - 1]) == 'k' {
        // After a, o or e the vowel goes too, when the word is long enough.
        if len > 4 && matches!(ch(s[len - 2]), 'a' | 'o' | 'e') {
            return len - 2;
        }
        return len - 1;
    }
    len
}

fn normalize(s: &[u16], len: usize) -> usize {
    if len > 3 && matches!(ch(s[len - 1]), 'a' | 'e' | 'i' | 'o') {
        return len - 1;
    }
    len
}
