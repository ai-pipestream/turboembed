//! Weighted modulus-11 check digits: United Kingdom NHS numbers and
//! Brazilian CPF and CNPJ tax identifiers (Receita Federal do Brasil).

/// Checks a 10-digit United Kingdom NHS number: weights `10..=2` over the
/// first nine digits, remainder `11 - (sum % 11)` (`11` maps to `0`, `10`
/// is invalid), rejecting all-equal digits.
pub fn uk_nhs(digits: &[u8; 10]) -> bool {
    if !digits.iter().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if digits.iter().all(|&b| b == digits[0]) {
        return false;
    }
    let mut sum = 0u32;
    for (i, &b) in digits[..9].iter().enumerate() {
        sum += (b - b'0') as u32 * (10 - i as u32);
    }
    let rem = 11 - (sum % 11);
    let check = if rem == 11 { 0 } else { rem };
    check < 10 && check == (digits[9] - b'0') as u32
}

/// Checks an 11-digit Brazilian CPF (Cadastro de Pessoas Fisicas) under the
/// Receita Federal double modulus-11 rule, rejecting all-equal digit runs and
/// the ascending placeholder `01234567890`.
pub fn br_cpf(digits: &[u8; 11]) -> bool {
    if !digits.iter().all(|b| b.is_ascii_digit()) {
        return false;
    }
    if digits.iter().all(|&b| b == digits[0]) || digits == b"01234567890" {
        return false;
    }
    let mut sum1 = 0u32;
    for (i, &b) in digits[..9].iter().enumerate() {
        sum1 += (b - b'0') as u32 * (10 - i as u32);
    }
    let rem1 = (sum1 * 10) % 11;
    let c1 = if rem1 == 10 { 0 } else { rem1 };
    if (digits[9] - b'0') as u32 != c1 {
        return false;
    }
    let mut sum2 = 0u32;
    for (i, &b) in digits[..10].iter().enumerate() {
        sum2 += (b - b'0') as u32 * (11 - i as u32);
    }
    let rem2 = (sum2 * 10) % 11;
    let c2 = if rem2 == 10 { 0 } else { rem2 };
    (digits[10] - b'0') as u32 == c2
}

/// Checks a 14-character Brazilian CNPJ (Cadastro Nacional da Pessoa Juridica)
/// under the Receita Federal double modulus-11 rule, supporting both numeric
/// CNPJs and the alphanumeric format (IN RFB 2.229/2024, where root and branch
/// characters `0..=9` or `A..=Z` carry weight `c - b'0'`).
pub fn br_cnpj(chars: &[u8; 14]) -> bool {
    if !chars[..12].iter().all(|b| b.is_ascii_digit() || b.is_ascii_uppercase())
        || !chars[12..].iter().all(|b| b.is_ascii_digit())
    {
        return false;
    }
    if chars.iter().all(|&b| b == chars[0]) || &chars[8..12] == b"0000" {
        return false;
    }
    const W1: [u32; 12] = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
    const W2: [u32; 13] = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2];
    let mut sum1 = 0u32;
    for i in 0..12 {
        sum1 += (chars[i] - b'0') as u32 * W1[i];
    }
    let r1 = sum1 % 11;
    let c1 = if r1 < 2 { 0 } else { 11 - r1 };
    if (chars[12] - b'0') as u32 != c1 {
        return false;
    }
    let mut sum2 = 0u32;
    for i in 0..13 {
        sum2 += (chars[i] - b'0') as u32 * W2[i];
    }
    let r2 = sum2 % 11;
    let c2 = if r2 < 2 { 0 } else { 11 - r2 };
    (chars[13] - b'0') as u32 == c2
}
