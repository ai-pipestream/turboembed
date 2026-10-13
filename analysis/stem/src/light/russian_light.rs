//! `RussianLightStemmer`: Dolamic and Savoy, "Indexing and Searching
//! Strategies for the Russian Language" (JASIST 60(12), 2009), from Lucene.

use super::util::{ch, ends_any};

pub(crate) fn stem(s: &[u16], len: usize) -> usize {
    let len = remove_case(s, len);
    normalize(s, len)
}

fn normalize(s: &[u16], len: usize) -> usize {
    if len > 3 {
        match ch(s[len - 1]) {
            'ь' | 'и' => return len - 1,
            'н' if ch(s[len - 2]) == 'н' => return len - 1,
            _ => {}
        }
    }
    len
}

fn remove_case(s: &[u16], len: usize) -> usize {
    if len > 6 && ends_any(s, len, &[u16s!("иями"), u16s!("оями")]) {
        return len - 4;
    }
    let threes = [
        u16s!("иям"),
        u16s!("иях"),
        u16s!("оях"),
        u16s!("ями"),
        u16s!("оям"),
        u16s!("оьв"),
        u16s!("ами"),
        u16s!("его"),
        u16s!("ему"),
        u16s!("ери"),
        u16s!("ими"),
        u16s!("ого"),
        u16s!("ому"),
        u16s!("ыми"),
        u16s!("оев"),
    ];
    if len > 5 && ends_any(s, len, &threes) {
        return len - 3;
    }
    let twos = [
        u16s!("ая"),
        u16s!("яя"),
        u16s!("ях"),
        u16s!("юю"),
        u16s!("ах"),
        u16s!("ею"),
        u16s!("их"),
        u16s!("ия"),
        u16s!("ию"),
        u16s!("ьв"),
        u16s!("ою"),
        u16s!("ую"),
        u16s!("ям"),
        u16s!("ых"),
        u16s!("ея"),
        u16s!("ам"),
        u16s!("ем"),
        u16s!("ей"),
        u16s!("ём"),
        u16s!("ев"),
        u16s!("ий"),
        u16s!("им"),
        u16s!("ое"),
        u16s!("ой"),
        u16s!("ом"),
        u16s!("ов"),
        u16s!("ые"),
        u16s!("ый"),
        u16s!("ым"),
        u16s!("ми"),
    ];
    if len > 4 && ends_any(s, len, &twos) {
        return len - 2;
    }
    if len > 3 && matches!(ch(s[len - 1]), 'а' | 'е' | 'и' | 'о' | 'у' | 'й' | 'ы' | 'я' | 'ь') {
        return len - 1;
    }
    len
}
