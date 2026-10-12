//! Hangul syllable decomposition and composition (Unicode section 3.12).
//! A precomposed syllable is S_BASE + (L index * 21 + V index) * 28 + T
//! index, so it decomposes into a leading consonant L, a vowel V and an
//! optional trailing consonant T, and composes back by the same formula.

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

/// The jamo of syllable `c` (two or three), `None` when `c` is not a
/// precomposed Hangul syllable.
#[inline]
pub(super) fn decompose(c: char) -> Option<([char; 3], usize)> {
    let index = (c as u32).wrapping_sub(S_BASE);
    if index >= S_COUNT {
        return None;
    }
    let l = jamo(L_BASE + index / N_COUNT);
    let v = jamo(V_BASE + (index % N_COUNT) / T_COUNT);
    let t = index % T_COUNT;
    if t == 0 { Some(([l, v, l], 2)) } else { Some(([l, v, jamo(T_BASE + t)], 3)) }
}

/// The syllable composed of L + V or LV + T, when `first` and `second`
/// form one.
#[inline]
pub(super) fn compose(first: char, second: char) -> Option<char> {
    let (first, second) = (first as u32, second as u32);
    let l_index = first.wrapping_sub(L_BASE);
    let v_index = second.wrapping_sub(V_BASE);
    if l_index < L_COUNT && v_index < V_COUNT {
        return Some(jamo(S_BASE + (l_index * V_COUNT + v_index) * T_COUNT));
    }
    let s_index = first.wrapping_sub(S_BASE);
    let t_index = second.wrapping_sub(T_BASE);
    if s_index < S_COUNT && s_index % T_COUNT == 0 && t_index > 0 && t_index < T_COUNT {
        return Some(jamo(first + t_index));
    }
    None
}

fn jamo(cp: u32) -> char {
    char::from_u32(cp).expect("Hangul code points are not surrogates")
}
