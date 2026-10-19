//! `FinnishLightStemmer`: Savoy, "Report on CLEF-2003 Monolingual Tracks",
//! from Lucene.

use super::util::{ch, delete, ends_any, ends_with, fold, is_vowel, unit};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    if len < 4 {
        return len;
    }
    fold(&mut s[..len], u16s!("äåö"), u16s!("aao"));
    let len = step1(s, len);
    let len = step2(s, len);
    let len = step3(s, len);
    let len = norm1(s, len);
    norm2(s, len)
}

fn step1(s: &[u16], len: usize) -> usize {
    if len > 8 {
        if ends_with(s, len, u16s!("kin")) {
            return step1(s, len - 3);
        }
        if ends_with(s, len, u16s!("ko")) {
            return step1(s, len - 2);
        }
    }
    if len > 11 {
        if ends_with(s, len, u16s!("dellinen")) {
            return len - 8;
        }
        if ends_with(s, len, u16s!("dellisuus")) {
            return len - 9;
        }
    }
    len
}

fn step2(s: &[u16], len: usize) -> usize {
    if len > 5 {
        if ends_any(s, len, &[u16s!("lla"), u16s!("tse"), u16s!("sti")]) {
            return len - 3;
        }
        if ends_with(s, len, u16s!("ni")) {
            return len - 2;
        }
        if ends_with(s, len, u16s!("aa")) {
            return len - 1;
        }
    }
    len
}

fn step3(s: &mut [u16], len: usize) -> usize {
    if len > 8 {
        if ends_with(s, len, u16s!("nnen")) {
            s[len - 4] = unit('s');
            return len - 3;
        }
        if ends_with(s, len, u16s!("ntena")) {
            s[len - 5] = unit('s');
            return len - 4;
        }
        if ends_with(s, len, u16s!("tten")) {
            return len - 4;
        }
        if ends_with(s, len, u16s!("eiden")) {
            return len - 5;
        }
    }
    if len > 6 {
        if ends_any(s, len, &[u16s!("neen"), u16s!("niin"), u16s!("seen"), u16s!("teen"), u16s!("inen")]) {
            return len - 4;
        }
        if ch(s[len - 3]) == 'h' && is_vowel(s[len - 2]) && ch(s[len - 1]) == 'n' {
            return len - 3;
        }
        if ends_with(s, len, u16s!("den")) {
            s[len - 3] = unit('s');
            return len - 2;
        }
        if ends_with(s, len, u16s!("ksen")) {
            s[len - 4] = unit('s');
            return len - 3;
        }
        let threes = [u16s!("ssa"), u16s!("sta"), u16s!("lla"), u16s!("lta"), u16s!("tta"), u16s!("ksi"), u16s!("lle")];
        if ends_any(s, len, &threes) {
            return len - 3;
        }
    }
    if len > 5 {
        if ends_any(s, len, &[u16s!("na"), u16s!("ne")]) {
            return len - 2;
        }
        if ends_with(s, len, u16s!("nei")) {
            return len - 3;
        }
    }
    if len > 4 {
        if ends_any(s, len, &[u16s!("ja"), u16s!("ta")]) {
            return len - 2;
        }
        if ch(s[len - 1]) == 'a' {
            return len - 1;
        }
        if ch(s[len - 1]) == 'n' && is_vowel(s[len - 2]) {
            return len - 2;
        }
        if ch(s[len - 1]) == 'n' {
            return len - 1;
        }
    }
    len
}

fn norm1(s: &mut [u16], len: usize) -> usize {
    if len > 5 && ends_with(s, len, u16s!("hde")) {
        s[len - 3..len].copy_from_slice(u16s!("ksi"));
    }
    if len > 4 && ends_any(s, len, &[u16s!("ei"), u16s!("at")]) {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 't' | 's' | 'j' | 'e' | 'a' | 'i') {
        return len - 1;
    }
    len
}

fn norm2(s: &mut [u16], mut len: usize) -> usize {
    if len > 8 && matches!(ch(s[len - 1]), 'e' | 'o' | 'u') {
        len -= 1;
    }
    if len > 4 {
        if ch(s[len - 1]) == 'i' {
            len -= 1;
        }
        if len > 4 {
            // Removes the second of two equal k, p or t in a row.
            let mut prev = s[0];
            let mut i = 1;
            while i < len {
                if s[i] == prev && matches!(ch(prev), 'k' | 'p' | 't') {
                    len = delete(s, i, len);
                } else {
                    prev = s[i];
                    i += 1;
                }
            }
        }
    }
    len
}
