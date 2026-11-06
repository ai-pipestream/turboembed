//! French INSEE national and business identifier validators: NIR (numero
//! d'inscription au repertoire, modulus-97 key), SIREN (9-digit Luhn), and
//! SIRET (14-digit Luhn with the La Poste modulus-5 rule).

/// Checks a 15-character normalized French NIR (13-character base with
/// optional Corsica `2A`/`2B` department code + 2-digit `01..=97` key).
pub fn nir(chars: &[u8; 15]) -> bool {
    if !matches!(chars[0], b'1' | b'2' | b'3' | b'4' | b'7' | b'8') {
        return false;
    }
    if !chars[1..5].iter().all(|b| b.is_ascii_digit()) {
        return false;
    }
    let month = (chars[3] - b'0') * 10 + (chars[4] - b'0');
    if !matches!(month, 1..=12 | 20 | 30..=42 | 50..=99) {
        return false;
    }
    let corsica = matches!(&chars[5..7], b"2A" | b"2B");
    if corsica {
        if !chars[7..10].iter().all(|b| b.is_ascii_digit()) || &chars[7..10] == b"000" {
            return false;
        }
    } else {
        if !chars[5..10].iter().all(|b| b.is_ascii_digit()) {
            return false;
        }
        let dept = (chars[5] - b'0') * 10 + (chars[6] - b'0');
        match dept {
            1..=19 | 21..=95 | 99 => {
                if &chars[7..10] == b"000" {
                    return false;
                }
            }
            97 => {
                if !matches!(chars[7], b'1'..=b'8') || &chars[8..10] == b"00" {
                    return false;
                }
            }
            98 => {
                if !matches!(chars[7], b'4'..=b'8') || &chars[8..10] == b"00" {
                    return false;
                }
            }
            _ => return false,
        }
    }
    if !chars[10..15].iter().all(|b| b.is_ascii_digit()) || &chars[10..13] == b"000" {
        return false;
    }
    let key = (chars[13] - b'0') as u64 * 10 + (chars[14] - b'0') as u64;
    if !(1..=97).contains(&key) {
        return false;
    }
    let mut numeric = [0u8; 13];
    numeric.copy_from_slice(&chars[..13]);
    if &chars[5..7] == b"2A" {
        numeric[5] = b'1';
        numeric[6] = b'9';
    } else if &chars[5..7] == b"2B" {
        numeric[5] = b'1';
        numeric[6] = b'8';
    }
    let mut n = 0u64;
    for &b in &numeric {
        n = n * 10 + (b - b'0') as u64;
    }
    key == 97 - (n % 97)
}

/// Checks a 9-digit French SIREN number: non-zero leading digit, not all
/// equal digits, and valid Luhn check digit.
pub fn siren(digits: &[u8; 9]) -> bool {
    if !digits.iter().all(|b| b.is_ascii_digit()) || digits[0] == b'0' {
        return false;
    }
    if digits.iter().all(|&b| b == digits[0]) {
        return false;
    }
    super::luhn::valid(digits)
}

/// Checks a 14-digit French SIRET number: valid 9-digit SIREN prefix, non-zero
/// 4-digit NIC establishment serial (`digits[9..13] != "0000"`), and either a
/// valid 14-digit Luhn checksum or the INSEE La Poste (`356000000`) sum-of-digits
/// modulus-5 exception.
pub fn siret(digits: &[u8; 14]) -> bool {
    let Ok(siren_part) = <&[u8; 9]>::try_from(&digits[..9]) else {
        return false;
    };
    if !siren(siren_part) || !digits[9..].iter().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if &digits[9..13] == b"0000" {
        return false;
    }
    if super::luhn::valid(digits) {
        return true;
    }
    if &digits[..9] == b"356000000" {
        let sum: u32 = digits.iter().map(|&b| (b - b'0') as u32).sum();
        return sum.is_multiple_of(5);
    }
    false
}
