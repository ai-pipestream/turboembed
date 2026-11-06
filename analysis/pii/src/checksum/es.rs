//! Spanish national and tax identifier checksums: DNI, NIE (Ministerio del
//! Interior modulus-23 control table) and CIF (Real Decreto 1065/2007 / Orden
//! EHA/451/2008).

/// The 23 control letters of the Spanish NIF modulus-23 algorithm.
pub const NIF_LETTERS: &[u8; 23] = b"TRWAGMYFPDXBNJZSQVHLCKE";

/// Checks a 9-character normalized Spanish DNI: 8 ASCII digits (not all zero)
/// followed by the uppercase modulus-23 control letter.
pub fn dni(chars: &[u8; 9]) -> bool {
    if !chars[..8].iter().all(|b| b.is_ascii_digit()) || &chars[..8] == b"00000000" {
        return false;
    }
    let mut n = 0u32;
    for &b in &chars[..8] {
        n = n * 10 + (b - b'0') as u32;
    }
    chars[8] == NIF_LETTERS[(n % 23) as usize]
}

/// Checks a 9-character normalized Spanish NIE: leading `X` (`0`), `Y` (`1`),
/// or `Z` (`2`), 7 ASCII digits, and the uppercase modulus-23 control letter.
pub fn nie(chars: &[u8; 9]) -> bool {
    let prefix = match chars[0] {
        b'X' => 0u32,
        b'Y' => 1u32,
        b'Z' => 2u32,
        _ => return false,
    };
    if !chars[1..8].iter().all(|b| b.is_ascii_digit()) || &chars[1..8] == b"0000000" {
        return false;
    }
    let mut n = prefix;
    for &b in &chars[1..8] {
        n = n * 10 + (b - b'0') as u32;
    }
    chars[8] == NIF_LETTERS[(n % 23) as usize]
}

/// Checks a 9-character normalized Spanish CIF (Codigo de Identificacion
/// Fiscal): entity type letter in `ABCDEFGHJNPQRSUVW`, 7 ASCII digits (not
/// all zero), and the weighted digit or letter control character.
pub fn cif(chars: &[u8; 9]) -> bool {
    let org = chars[0];
    if !matches!(
        org,
        b'A' | b'B'
            | b'C'
            | b'D'
            | b'E'
            | b'F'
            | b'G'
            | b'H'
            | b'J'
            | b'N'
            | b'P'
            | b'Q'
            | b'R'
            | b'S'
            | b'U'
            | b'V'
            | b'W'
    ) {
        return false;
    }
    if !chars[1..8].iter().all(|b| b.is_ascii_digit()) || &chars[1..8] == b"0000000" {
        return false;
    }
    let mut even_sum = 0u32;
    let mut odd_sum = 0u32;
    for (i, &b) in chars[1..8].iter().enumerate() {
        let d = (b - b'0') as u32;
        if i % 2 == 1 {
            even_sum += d;
        } else {
            let doubled = d * 2;
            odd_sum += doubled / 10 + doubled % 10;
        }
    }
    let unit = (10 - ((even_sum + odd_sum) % 10)) % 10;
    let want_digit = b'0' + unit as u8;
    let want_letter = b"JABCDEFGHI"[unit as usize];
    let control = chars[8];
    let non_resident_prov = &chars[1..3] == b"00";
    if matches!(org, b'P' | b'Q' | b'R' | b'S' | b'W' | b'N') || non_resident_prov {
        control == want_letter
    } else if matches!(org, b'A' | b'B' | b'E' | b'H') {
        control == want_digit
    } else {
        control == want_digit || control == want_letter
    }
}
