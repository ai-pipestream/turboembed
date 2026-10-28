//! The Double Metaphone letter handlers: one function per case of the
//! dispatch, ported rule for rule from the reference; the parent
//! module documents the dialect.

use super::{Keys, at, contains, is_vowel};

const L_R_N_M_B_H_F_V_W_SPACE: &[&str] = &["L", "R", "N", "M", "B", "H", "F", "V", "W", " "];
const ES_EP_EB_EL_EY_IB_IL_IN_IE_EI_ER: &[&str] = &["ES", "EP", "EB", "EL", "EY", "IB", "IL", "IN", "IE", "EI", "ER"];
const L_T_K_S_N_M_B_Z: &[&str] = &["L", "T", "K", "S", "N", "M", "B", "Z"];

pub(super) fn condition_c0(w: &[char], index: usize) -> bool {
    if contains(w, index as isize, 4, &["CHIA"]) {
        return true;
    }
    if index <= 1 {
        return false;
    }
    if is_vowel(at(w, index as isize - 2)) {
        return false;
    }
    if !contains(w, index as isize - 1, 3, &["ACH"]) {
        return false;
    }
    let c = at(w, index as isize + 2);
    (c != 'I' && c != 'E') || contains(w, index as isize - 2, 6, &["BACHER", "MACHER"])
}

pub(super) fn condition_ch0(w: &[char], index: usize) -> bool {
    if index != 0 {
        return false;
    }
    if !contains(w, 1, 5, &["HARAC", "HARIS"]) && !contains(w, 1, 3, &["HOR", "HYM", "HIA", "HEM"]) {
        return false;
    }
    !contains(w, 0, 5, &["CHORE"])
}

pub(super) fn condition_ch1(w: &[char], index: usize) -> bool {
    let i = index as isize;
    contains(w, 0, 4, &["VAN ", "VON "])
        || contains(w, 0, 3, &["SCH"])
        || contains(w, i - 2, 6, &["ORCHES", "ARCHIT", "ORCHID"])
        || contains(w, i + 2, 1, &["T", "S"])
        || (contains(w, i - 1, 1, &["A", "O", "U", "E"]) || index == 0)
            && (contains(w, i + 2, 1, L_R_N_M_B_H_F_V_W_SPACE) || i + 1 == w.len() as isize - 1)
}

pub(super) fn condition_l0(w: &[char], index: usize) -> bool {
    if index == w.len() - 3 && contains(w, index as isize - 1, 4, &["ILLO", "ILLA", "ALLE"]) {
        return true;
    }
    (contains(w, w.len() as isize - 2, 2, &["AS", "OS"]) || contains(w, w.len() as isize - 1, 1, &["A", "O"]))
        && contains(w, index as isize - 1, 4, &["ALLE"])
}

pub(super) fn condition_m0(w: &[char], index: usize) -> bool {
    if at(w, index as isize + 1) == 'M' {
        return true;
    }
    contains(w, index as isize - 1, 3, &["UMB"])
        && (index as isize + 1 == w.len() as isize - 1 || contains(w, index as isize + 2, 2, &["ER"]))
}

pub(super) fn handle_c(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if condition_c0(w, index) {
        k.push(b"K");
        index + 2
    } else if index == 0 && contains(w, 0, 6, &["CAESAR"]) {
        k.push(b"S");
        index + 2
    } else if contains(w, i, 2, &["CH"]) {
        handle_ch(w, k, index)
    } else if contains(w, i, 2, &["CZ"]) && !contains(w, i - 2, 4, &["WICZ"]) {
        // Czerny.
        k.push2(b"S", b"X");
        index + 2
    } else if contains(w, i + 1, 3, &["CIA"]) {
        // Focaccia.
        k.push(b"X");
        index + 3
    } else if contains(w, i, 2, &["CC"]) && !(index == 1 && at(w, 0) == 'M') {
        // Double cc but not McClelland.
        handle_cc(w, k, index)
    } else if contains(w, i, 2, &["CK", "CG", "CQ"]) {
        k.push(b"K");
        index + 2
    } else if contains(w, i, 2, &["CI", "CE", "CY"]) {
        // Italian vs English.
        if contains(w, i, 3, &["CIO", "CIE", "CIA"]) {
            k.push2(b"S", b"X");
        } else {
            k.push(b"S");
        }
        index + 2
    } else {
        k.push(b"K");
        if contains(w, i + 1, 2, &[" C", " Q", " G"]) {
            // Mac Caffrey, Mac Gregor.
            index + 3
        } else if contains(w, i + 1, 1, &["C", "K", "Q"]) && !contains(w, i + 1, 2, &["CE", "CI"]) {
            index + 2
        } else {
            index + 1
        }
    }
}

pub(super) fn handle_cc(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if contains(w, i + 2, 1, &["I", "E", "H"]) && !contains(w, i + 2, 2, &["HU"]) {
        // Bellocchio but not bacchus.
        if index == 1 && at(w, 0) == 'A' || contains(w, i - 1, 5, &["UCCEE", "UCCES"]) {
            // Accident, accede, succeed.
            k.push(b"KS");
        } else {
            // Bacci, bertucci, other Italian.
            k.push(b"X");
        }
        index + 3
    } else {
        // Pierce's rule.
        k.push(b"K");
        index + 2
    }
}

pub(super) fn handle_ch(w: &[char], k: &mut Keys, index: usize) -> usize {
    if index > 0 && contains(w, index as isize, 4, &["CHAE"]) {
        // Michael.
        k.push2(b"K", b"X");
        return index + 2;
    }
    if condition_ch0(w, index) {
        // Greek roots: chemistry, chorus.
        k.push(b"K");
        return index + 2;
    }
    if condition_ch1(w, index) {
        // Germanic, Greek, or otherwise ch for kh sound.
        k.push(b"K");
        return index + 2;
    }
    if index > 0 {
        if contains(w, 0, 2, &["MC"]) {
            k.push(b"K");
        } else {
            k.push2(b"X", b"K");
        }
    } else {
        k.push(b"X");
    }
    index + 2
}

pub(super) fn handle_d(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if contains(w, i, 2, &["DG"]) {
        // Edge.
        if contains(w, i + 2, 1, &["I", "E", "Y"]) {
            k.push(b"J");
            index + 3
        } else {
            // Edgar.
            k.push(b"TK");
            index + 2
        }
    } else if contains(w, i, 2, &["DT", "DD"]) {
        k.push(b"T");
        index + 2
    } else {
        k.push(b"T");
        index + 1
    }
}

pub(super) fn handle_g(w: &[char], k: &mut Keys, index: usize, slavo: bool) -> usize {
    let i = index as isize;
    if at(w, i + 1) == 'H' {
        handle_gh(w, k, index)
    } else if at(w, i + 1) == 'N' {
        if index == 1 && is_vowel(at(w, 0)) && !slavo {
            k.push2(b"KN", b"N");
        } else if !contains(w, i + 2, 2, &["EY"]) && at(w, i + 1) != 'Y' && !slavo {
            k.push2(b"N", b"KN");
        } else {
            k.push(b"KN");
        }
        index + 2
    } else if contains(w, i + 1, 2, &["LI"]) && !slavo {
        k.push2(b"KL", b"L");
        index + 2
    } else if index == 0 && (at(w, i + 1) == 'Y' || contains(w, i + 1, 2, ES_EP_EB_EL_EY_IB_IL_IN_IE_EI_ER)) {
        // -ges-, -gep-, -gel-, -gie- at beginning.
        k.push2(b"K", b"J");
        index + 2
    } else if (contains(w, i + 1, 2, &["ER"]) || at(w, i + 1) == 'Y')
        && !contains(w, 0, 6, &["DANGER", "RANGER", "MANGER"])
        && !contains(w, i - 1, 1, &["E", "I"])
        && !contains(w, i - 1, 3, &["RGY", "OGY"])
    {
        // -ger-, -gy-.
        k.push2(b"K", b"J");
        index + 2
    } else if contains(w, i + 1, 1, &["E", "I", "Y"]) || contains(w, i - 1, 4, &["AGGI", "OGGI"]) {
        // Italian biaggi.
        if contains(w, 0, 4, &["VAN ", "VON "]) || contains(w, 0, 3, &["SCH"]) || contains(w, i + 1, 2, &["ET"]) {
            // Obvious germanic.
            k.push(b"K");
        } else if contains(w, i + 1, 3, &["IER"]) {
            k.push(b"J");
        } else {
            k.push2(b"J", b"K");
        }
        index + 2
    } else {
        k.push(b"K");
        index + usize::from(at(w, i + 1) == 'G') + 1
    }
}

pub(super) fn handle_gh(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if index > 0 && !is_vowel(at(w, i - 1)) {
        k.push(b"K");
    } else if index == 0 {
        if at(w, i + 2) == 'I' {
            k.push(b"J");
        } else {
            k.push(b"K");
        }
    } else if index > 1 && contains(w, i - 2, 1, &["B", "H", "D"])
        || index > 2 && contains(w, i - 3, 1, &["B", "H", "D"])
        || index > 3 && contains(w, i - 4, 1, &["B", "H"])
    {
        // Parker's rule (with some further refinements): hugh. Silent.
    } else {
        if index > 2 && at(w, i - 1) == 'U' && contains(w, i - 3, 1, &["C", "G", "L", "R", "T"]) {
            // Laugh, McLaughlin, cough, gough, rough, tough.
            k.push(b"F");
        } else if index > 0 && at(w, i - 1) != 'I' {
            k.push(b"K");
        }
    }
    index + 2
}

pub(super) fn handle_h(w: &[char], k: &mut Keys, index: usize) -> usize {
    // Only keep if first and before a vowel or between two vowels.
    if (index == 0 || is_vowel(at(w, index as isize - 1))) && is_vowel(at(w, index as isize + 1)) {
        k.push(b"H");
        index + 2
    } else {
        index + 1
    }
}

pub(super) fn handle_j(w: &[char], k: &mut Keys, index: usize, slavo: bool) -> usize {
    let i = index as isize;
    if contains(w, i, 4, &["JOSE"]) || contains(w, 0, 4, &["SAN "]) {
        // Obvious Spanish: Jose, San Jacinto.
        if index == 0 && at(w, i + 4) == ' ' || w.len() == 4 || contains(w, 0, 4, &["SAN "]) {
            k.push(b"H");
        } else {
            k.push2(b"J", b"H");
        }
        index + 1
    } else {
        if index == 0 && !contains(w, i, 4, &["JOSE"]) {
            k.push2(b"J", b"A");
        } else if is_vowel(at(w, i - 1)) && !slavo && (at(w, i + 1) == 'A' || at(w, i + 1) == 'O') {
            k.push2(b"J", b"H");
        } else if index == w.len() - 1 {
            k.push2(b"J", b" ");
        } else if !contains(w, i + 1, 1, L_T_K_S_N_M_B_Z) && !contains(w, i - 1, 1, &["S", "K", "L"]) {
            k.push(b"J");
        }
        index + if at(w, i + 1) == 'J' { 2 } else { 1 }
    }
}

pub(super) fn handle_l(w: &[char], k: &mut Keys, index: usize) -> usize {
    if at(w, index as isize + 1) == 'L' {
        if condition_l0(w, index) {
            k.push_pri(b"L");
        } else {
            k.push(b"L");
        }
        index + 2
    } else {
        k.push(b"L");
        index + 1
    }
}

pub(super) fn handle_p(w: &[char], k: &mut Keys, index: usize) -> usize {
    if at(w, index as isize + 1) == 'H' {
        k.push(b"F");
        index + 2
    } else {
        k.push(b"P");
        index + usize::from(contains(w, index as isize + 1, 1, &["P", "B"])) + 1
    }
}

pub(super) fn handle_r(w: &[char], k: &mut Keys, index: usize, slavo: bool) -> usize {
    let i = index as isize;
    if index == w.len() - 1 && !slavo && contains(w, i - 2, 2, &["IE"]) && !contains(w, i - 4, 2, &["ME", "MA"]) {
        k.push_alt(b"R");
    } else {
        k.push(b"R");
    }
    index + usize::from(at(w, i + 1) == 'R') + 1
}

pub(super) fn handle_s(w: &[char], k: &mut Keys, index: usize, slavo: bool) -> usize {
    let i = index as isize;
    if contains(w, i - 1, 3, &["ISL", "YSL"]) {
        // Special cases: island, isle, carlisle, carlysle.
        index + 1
    } else if index == 0 && contains(w, 0, 5, &["SUGAR"]) {
        // Special case sugar-.
        k.push2(b"X", b"S");
        index + 1
    } else if contains(w, i, 2, &["SH"]) {
        if contains(w, i + 1, 4, &["HEIM", "HOEK", "HOLM", "HOLZ"]) {
            // Germanic.
            k.push(b"S");
        } else {
            k.push(b"X");
        }
        index + 2
    } else if contains(w, i, 3, &["SIO", "SIA"]) || contains(w, i, 4, &["SIAN"]) {
        // Italian and Armenian.
        if slavo {
            k.push(b"S");
        } else {
            k.push2(b"S", b"X");
        }
        index + 3
    } else if index == 0 && contains(w, 1, 1, &["M", "N", "L", "W"]) || contains(w, i + 1, 1, &["Z"]) {
        // German and anglicisations, for example smith match schmidt,
        // snider match schneider; also -sz- in slavic language although in
        // hungarian it is pronounced s.
        k.push2(b"S", b"X");
        index + if contains(w, i + 1, 1, &["Z"]) { 2 } else { 1 }
    } else if contains(w, i, 2, &["SC"]) {
        handle_sc(w, k, index)
    } else {
        if index == w.len() - 1 && contains(w, i - 2, 2, &["AI", "OI"]) {
            // French, for example resnais, artois.
            k.push_alt(b"S");
        } else {
            k.push(b"S");
        }
        index + usize::from(contains(w, i + 1, 1, &["S", "Z"])) + 1
    }
}

pub(super) fn handle_sc(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if at(w, i + 2) == 'H' {
        // Schlesinger's rule.
        if contains(w, i + 3, 2, &["OO", "ER", "EN", "UY", "ED", "EM"]) {
            // Dutch origin, for example school, schooner.
            if contains(w, i + 3, 2, &["ER", "EN"]) {
                // Schermerhorn, schenker.
                k.push2(b"X", b"SK");
            } else {
                k.push(b"SK");
            }
        } else if index == 0 && !is_vowel(at(w, 3)) && at(w, 3) != 'W' {
            k.push2(b"X", b"S");
        } else {
            k.push(b"X");
        }
    } else if contains(w, i + 2, 1, &["I", "E", "Y"]) {
        k.push(b"S");
    } else {
        k.push(b"SK");
    }
    index + 3
}

pub(super) fn handle_t(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if contains(w, i, 4, &["TION"]) || contains(w, i, 3, &["TIA", "TCH"]) {
        k.push(b"X");
        index + 3
    } else if contains(w, i, 2, &["TH"]) || contains(w, i, 3, &["TTH"]) {
        if contains(w, i + 2, 2, &["OM", "AM"])
            // Special case thomas, thames or germanic.
            || contains(w, 0, 4, &["VAN ", "VON "])
            || contains(w, 0, 3, &["SCH"])
        {
            k.push(b"T");
        } else {
            k.push2(b"0", b"T");
        }
        index + 2
    } else {
        k.push(b"T");
        index + usize::from(contains(w, i + 1, 1, &["T", "D"])) + 1
    }
}

pub(super) fn handle_w(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if contains(w, i, 2, &["WR"]) {
        // Can also be in middle of word.
        k.push(b"R");
        index + 2
    } else if index == 0 && (is_vowel(at(w, i + 1)) || contains(w, i, 2, &["WH"])) {
        if is_vowel(at(w, i + 1)) {
            // Wasserman should match Vasserman.
            k.push2(b"A", b"F");
        } else {
            // Need Uomo to match Womo.
            k.push(b"A");
        }
        index + 1
    } else if index == w.len() - 1 && is_vowel(at(w, i - 1))
        || contains(w, i - 1, 5, &["EWSKI", "EWSKY", "OWSKI", "OWSKY"])
        || contains(w, 0, 3, &["SCH"])
    {
        // Arnow should match Arnoff.
        k.push_alt(b"F");
        index + 1
    } else if contains(w, i, 4, &["WICZ", "WITZ"]) {
        // Polish, for example filipowicz.
        k.push2(b"TS", b"FX");
        index + 4
    } else {
        index + 1
    }
}

pub(super) fn handle_x(w: &[char], k: &mut Keys, index: usize) -> usize {
    let i = index as isize;
    if index == 0 {
        k.push(b"S");
        index + 1
    } else {
        if !(index == w.len() - 1 && (contains(w, i - 3, 3, &["IAU", "EAU"]) || contains(w, i - 2, 2, &["AU", "OU"]))) {
            // French, for example breaux.
            k.push(b"KS");
        }
        index + usize::from(contains(w, i + 1, 1, &["C", "X"])) + 1
    }
}

pub(super) fn handle_z(w: &[char], k: &mut Keys, index: usize, slavo: bool) -> usize {
    let i = index as isize;
    if at(w, i + 1) == 'H' {
        // Chinese pinyin, for example zhao, or Angelina Zhang.
        k.push(b"J");
        index + 2
    } else {
        if contains(w, i + 1, 2, &["ZO", "ZI", "ZA"]) || slavo && index > 0 && at(w, i - 1) != 'T' {
            k.push2(b"S", b"TS");
        } else {
            k.push(b"S");
        }
        index + usize::from(at(w, i + 1) == 'Z') + 1
    }
}
