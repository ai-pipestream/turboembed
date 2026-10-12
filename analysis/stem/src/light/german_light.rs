//! `GermanLightStemmer`: the "UniNE" algorithm of Savoy, "Light Stemming
//! Approaches for the French, Portuguese, German and Hungarian Languages"
//! (ACM SAC 2006), from Lucene.

use super::util::{ch, fold};

pub(crate) fn stem(s: &mut [u16], len: usize) -> usize {
    fold(&mut s[..len], u16s!("äàáâöòóôïìíîüùúû"), u16s!("aaaaooooiiiiuuuu"));
    let len = step1(s, len);
    step2(s, len)
}

/// The letters that can come before an -st or -s that is removed.
fn st_ending(c: u16) -> bool {
    matches!(ch(c), 'b' | 'd' | 'f' | 'g' | 'h' | 'k' | 'l' | 'm' | 'n' | 't')
}

fn step1(s: &[u16], len: usize) -> usize {
    if len > 5 && ch(s[len - 3]) == 'e' && ch(s[len - 2]) == 'r' && ch(s[len - 1]) == 'n' {
        return len - 3;
    }
    if len > 4 && ch(s[len - 2]) == 'e' && matches!(ch(s[len - 1]), 'm' | 'n' | 'r' | 's') {
        return len - 2;
    }
    if len > 3 && ch(s[len - 1]) == 'e' {
        return len - 1;
    }
    if len > 3 && ch(s[len - 1]) == 's' && st_ending(s[len - 2]) {
        return len - 1;
    }
    len
}

fn step2(s: &[u16], len: usize) -> usize {
    if len > 5 && ch(s[len - 3]) == 'e' && ch(s[len - 2]) == 's' && ch(s[len - 1]) == 't' {
        return len - 3;
    }
    if len > 4 && ch(s[len - 2]) == 'e' && matches!(ch(s[len - 1]), 'r' | 'n') {
        return len - 2;
    }
    if len > 4 && ch(s[len - 2]) == 's' && ch(s[len - 1]) == 't' && st_ending(s[len - 3]) {
        return len - 2;
    }
    len
}
